"""ML dataset pipeline compression encoder for HammerIO.

Provides streaming compression optimized for machine-learning dataset
files (CSV, Parquet, NPY, NPZ, PT, SafeTensors, HDF5, TFRecord) and
a ``StreamingDataset`` wrapper for on-the-fly decompression compatible
with PyTorch ``DataLoader`` patterns.
"""

from __future__ import annotations

import io
import logging
import os
import struct
import tarfile
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Iterator,
    Optional,
    Union,
)

from hammerio.core.hardware import HardwareProfile

logger = logging.getLogger("hammerio.encoders.dataset")

# Dataset file extensions recognised by this encoder
_DATASET_EXTENSIONS: frozenset[str] = frozenset({
    ".csv", ".tsv",
    ".parquet",
    ".npy", ".npz",
    ".pt", ".pth",
    ".safetensors",
    ".h5", ".hdf5",
    ".tfrecord", ".tfrecords",
})

# Default zstandard compression level per quality preset
_ZSTD_LEVEL_PRESETS: dict[str, int] = {
    "fast": 1,
    "balanced": 3,
    "quality": 9,
    "lossless": 19,
}

# Streaming chunk size (4 MB) — balances throughput and memory on
# both desktop and Jetson platforms.
_CHUNK_SIZE: int = 4 * 1024 * 1024

# Magic bytes written at the start of HammerIO dataset archives so
# ``StreamingDataset`` can identify them.
_MAGIC: bytes = b"HMRIO_DS"
_VERSION: int = 1


def _is_dataset_file(path: Path) -> bool:
    """Return ``True`` when *path* has a recognised dataset extension."""
    return path.suffix.lower() in _DATASET_EXTENSIONS


class DatasetEncoder:
    """Streaming compression encoder for ML datasets.

    For single files the encoder performs chunked zstandard
    compression, writing a thin header so that ``StreamingDataset``
    can decompress on the fly.

    For directories the encoder produces a ``.tar.zst`` archive that
    preserves the full directory structure while applying streaming
    zstandard compression.

    Args:
        hardware: Detected hardware profile used to tune compression
            parameters (thread count, chunk size).
    """

    def __init__(self, hardware: HardwareProfile) -> None:
        self.hardware = hardware
        self._threads = max(1, hardware.cpu_cores // 2)

        # Prefer zstandard (Python binding of zstd)
        self._has_zstd = False
        try:
            import zstandard  # type: ignore[import-untyped]  # noqa: F401

            self._has_zstd = True
            logger.info(
                "Dataset backend: zstandard (CPU, %d threads)", self._threads
            )
        except ImportError:
            logger.warning(
                "zstandard not installed — DatasetEncoder will use gzip fallback"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path, None],
        algorithm: str,
        quality: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Compress dataset file(s).

        Args:
            input_path: Source file or directory.
            output_path: Destination file or directory.  Auto-generated
                when ``None``.
            algorithm: Compression algorithm hint (``"zstd"`` is
                recommended; ``"gzip"`` accepted as fallback).
            quality: Quality preset — ``"fast"``, ``"balanced"``,
                ``"quality"``, or ``"lossless"``.
            progress_callback: Optional ``(job_id, percent)`` callable.
            job_id: Identifier forwarded to the progress callback.

        Returns:
            Absolute path to the compressed output.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            RuntimeError: If neither zstandard nor gzip is usable.
        """
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        jid = job_id or "dataset"
        level = _ZSTD_LEVEL_PRESETS.get(quality, _ZSTD_LEVEL_PRESETS["balanced"])

        if output_path is None:
            if input_path.is_dir():
                output_path = input_path.with_suffix(".tar.zst")
            else:
                output_path = input_path.with_suffix(input_path.suffix + ".zst")
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if input_path.is_dir():
            return self._compress_directory(
                input_path, output_path, level, progress_callback, jid
            )
        else:
            return self._compress_file(
                input_path, output_path, level, progress_callback, jid
            )

    # ------------------------------------------------------------------
    # Single-file streaming compression
    # ------------------------------------------------------------------

    def _compress_file(
        self,
        src: Path,
        dst: Path,
        level: int,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: str,
    ) -> str:
        """Compress a single dataset file with streaming zstd."""
        total_size = src.stat().st_size

        # Guard: verify output directory is writable
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(dst.parent, os.W_OK):
            raise PermissionError(f"Output directory is not writable: {dst.parent}")

        if progress_callback:
            progress_callback(job_id, 0.0)

        if self._has_zstd:
            self._compress_file_zstd(src, dst, level, total_size, progress_callback, job_id)
        else:
            self._compress_file_gzip(src, dst, level, total_size, progress_callback, job_id)

        out_size = dst.stat().st_size
        ratio = total_size / out_size if out_size > 0 else 0.0
        logger.info(
            "Compressed %s → %s (%.1f MB → %.1f MB, %.1fx)",
            src.name,
            dst.name,
            total_size / (1024 * 1024),
            out_size / (1024 * 1024),
            ratio,
        )

        if progress_callback:
            progress_callback(job_id, 100.0)

        return str(dst)

    def _compress_file_zstd(
        self,
        src: Path,
        dst: Path,
        level: int,
        total_size: int,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: str,
    ) -> None:
        """Streaming zstandard compression for a single file."""
        import zstandard as zstd  # type: ignore[import-untyped]

        params = zstd.ZstdCompressionParameters.from_level(
            level, threads=self._threads
        )
        cctx = zstd.ZstdCompressor(compression_params=params)

        bytes_read = 0
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            # Write HammerIO dataset header
            fout.write(_MAGIC)
            fout.write(struct.pack("<B", _VERSION))
            # Original filename length + name (for restore)
            name_bytes = src.name.encode("utf-8")
            fout.write(struct.pack("<H", len(name_bytes)))
            fout.write(name_bytes)

            with cctx.stream_writer(fout, closefd=False) as writer:
                while True:
                    chunk = fin.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    writer.write(chunk)
                    bytes_read += len(chunk)
                    if progress_callback and total_size > 0:
                        pct = min(bytes_read / total_size * 100.0, 99.9)
                        progress_callback(job_id, pct)

    def _compress_file_gzip(
        self,
        src: Path,
        dst: Path,
        level: int,
        total_size: int,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: str,
    ) -> None:
        """Fallback gzip streaming compression."""
        import gzip

        # Clamp zstd-range level to gzip range (1-9)
        gz_level = max(1, min(level, 9))

        bytes_read = 0
        with open(src, "rb") as fin, gzip.open(dst, "wb", compresslevel=gz_level) as fout:
            while True:
                chunk = fin.read(_CHUNK_SIZE)
                if not chunk:
                    break
                fout.write(chunk)
                bytes_read += len(chunk)
                if progress_callback and total_size > 0:
                    pct = min(bytes_read / total_size * 100.0, 99.9)
                    progress_callback(job_id, pct)

    # ------------------------------------------------------------------
    # Directory tar+compress
    # ------------------------------------------------------------------

    def _compress_directory(
        self,
        src_dir: Path,
        dst: Path,
        level: int,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: str,
    ) -> str:
        """Create a tar.zst archive preserving directory structure."""
        # Collect files to archive
        all_files = sorted(
            f for f in src_dir.rglob("*") if f.is_file()
        )
        total_size = sum(f.stat().st_size for f in all_files)
        if not all_files:
            logger.warning("No files found in %s", src_dir)
            return str(dst)

        logger.info(
            "Archiving %d files (%.1f MB) from %s",
            len(all_files),
            total_size / (1024 * 1024),
            src_dir,
        )

        if progress_callback:
            progress_callback(job_id, 0.0)

        if self._has_zstd:
            self._tar_zstd(src_dir, dst, all_files, total_size, level, progress_callback, job_id)
        else:
            self._tar_gzip(src_dir, dst, all_files, total_size, level, progress_callback, job_id)

        out_size = dst.stat().st_size
        ratio = total_size / out_size if out_size > 0 else 0.0
        logger.info(
            "Archive complete: %s (%.1f MB, %.1fx)",
            dst.name,
            out_size / (1024 * 1024),
            ratio,
        )

        if progress_callback:
            progress_callback(job_id, 100.0)

        return str(dst)

    def _tar_zstd(
        self,
        base_dir: Path,
        dst: Path,
        files: list[Path],
        total_size: int,
        level: int,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: str,
    ) -> None:
        """Create a tar archive piped through zstandard compression."""
        import zstandard as zstd  # type: ignore[import-untyped]

        params = zstd.ZstdCompressionParameters.from_level(
            level, threads=self._threads
        )
        cctx = zstd.ZstdCompressor(compression_params=params)

        bytes_added = 0
        with open(dst, "wb") as raw_fout:
            with cctx.stream_writer(raw_fout, closefd=False) as zst_writer:
                # Wrap the zst_writer so tarfile can write to it
                wrapper = _WriterWrapper(zst_writer)
                with tarfile.open(fileobj=wrapper, mode="w|") as tar:
                    for f in files:
                        arcname = str(f.relative_to(base_dir))
                        tar.add(str(f), arcname=arcname)
                        bytes_added += f.stat().st_size
                        if progress_callback and total_size > 0:
                            pct = min(bytes_added / total_size * 100.0, 99.9)
                            progress_callback(job_id, pct)

    def _tar_gzip(
        self,
        base_dir: Path,
        dst: Path,
        files: list[Path],
        total_size: int,
        level: int,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: str,
    ) -> None:
        """Fallback: tar + gzip archive."""
        gz_level = max(1, min(level, 9))
        # Change extension to .tar.gz for clarity
        if dst.name.endswith(".tar.zst"):
            dst = dst.with_name(dst.name.replace(".tar.zst", ".tar.gz"))

        bytes_added = 0
        with tarfile.open(str(dst), f"w:gz", compresslevel=gz_level) as tar:
            for f in files:
                arcname = str(f.relative_to(base_dir))
                tar.add(str(f), arcname=arcname)
                bytes_added += f.stat().st_size
                if progress_callback and total_size > 0:
                    pct = min(bytes_added / total_size * 100.0, 99.9)
                    progress_callback(job_id, pct)


class _WriterWrapper:
    """Thin file-like wrapper so ``tarfile`` can write to a zstd stream_writer."""

    def __init__(self, writer: Any) -> None:
        self._writer = writer

    def write(self, data: bytes) -> int:
        return self._writer.write(data)

    def tell(self) -> int:
        # tarfile streaming mode doesn't need accurate tell
        return 0

    def close(self) -> None:
        # Managed externally — do not close the underlying writer here
        pass


# ======================================================================
# StreamingDataset — PyTorch DataLoader compatible wrapper
# ======================================================================


def _get_iterable_dataset_base() -> type:
    """Return ``torch.utils.data.IterableDataset`` if PyTorch is installed,
    otherwise fall back to plain ``object`` so the class remains usable
    without a PyTorch dependency."""
    try:
        from torch.utils.data import IterableDataset  # type: ignore[import-untyped]
        return IterableDataset
    except ImportError:
        return object


class StreamingDataset(_get_iterable_dataset_base()):  # type: ignore[misc]
    """On-the-fly decompression wrapper for compressed ML datasets.

    Designed to be used with ``torch.utils.data.DataLoader`` for
    streaming access to zstandard-compressed dataset files without
    extracting them to disk first.

    When PyTorch is installed the class inherits from
    ``torch.utils.data.IterableDataset`` so it works directly with
    ``DataLoader``.  Without PyTorch the class still functions as a
    plain Python iterable.

    Supports two archive layouts:

    1. **Single compressed file** produced by ``DatasetEncoder`` for a
       single dataset file (has the ``HMRIO_DS`` magic header).  Each
       call to ``__getitem__`` reads a decompressed chunk.
    2. **tar.zst directory archive** — members are iterated lazily.

    For CSV files the ``iter_lines()`` helper yields decoded text lines.
    For ``.pt`` files the ``iter_tensors()`` helper yields deserialised
    PyTorch tensors.

    Args:
        path: Path to the compressed file.
        chunk_size: Decompressed chunk size returned per index.
            Defaults to 4 MB.

    Example::

        from torch.utils.data import DataLoader
        ds = StreamingDataset("train_data.npy.zst")
        loader = DataLoader(ds, batch_size=None, num_workers=0)
        for chunk in loader:
            process(chunk)
    """

    def __init__(self, path: Union[str, Path], chunk_size: int = _CHUNK_SIZE) -> None:
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        self.chunk_size = chunk_size
        self._is_tar = self.path.name.endswith(".tar.zst") or self.path.name.endswith(".tar.gz")
        self._length: Optional[int] = None
        self._decompressed_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Sequence protocol (len / getitem)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of chunks available.

        For tar archives this is the number of member files.
        For single compressed files this is ``ceil(decompressed_size / chunk_size)``.
        """
        if self._length is not None:
            return self._length

        if self._is_tar:
            self._length = self._count_tar_members()
        else:
            self._length = self._count_chunks()
        return self._length

    def __getitem__(self, index: int) -> bytes:
        """Return the *index*-th chunk as raw bytes.

        For tar archives returns the full content of the *index*-th
        member file.  For single files returns a decompressed chunk.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        if self._is_tar:
            return self._read_tar_member(index)
        else:
            return self._read_chunk(index)

    # ------------------------------------------------------------------
    # Iterator protocol (for DataLoader with batch_size=None)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[bytes]:
        """Yield decompressed chunks or tar member contents."""
        if self._is_tar:
            yield from self._iter_tar()
        else:
            yield from self._iter_chunks()

    # ------------------------------------------------------------------
    # Internal: single-file access
    # ------------------------------------------------------------------

    def _count_chunks(self) -> int:
        """Scan the file to determine the decompressed size."""
        try:
            import zstandard as zstd  # type: ignore[import-untyped]
        except ImportError:
            # Rough estimate from compressed size
            return max(1, self.path.stat().st_size // self.chunk_size)

        dctx = zstd.ZstdDecompressor()
        total = 0
        with open(self.path, "rb") as f:
            self._skip_header(f)
            with dctx.stream_reader(f, closefd=False) as reader:
                while True:
                    chunk = reader.read(self.chunk_size)
                    if not chunk:
                        break
                    total += len(chunk)
        self._decompressed_size = total
        return max(1, -(-total // self.chunk_size))  # ceil division

    def _read_chunk(self, index: int) -> bytes:
        """Read a specific decompressed chunk by index."""
        try:
            import zstandard as zstd  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError("zstandard required for streaming decompression")

        dctx = zstd.ZstdDecompressor()
        offset = index * self.chunk_size
        current = 0
        with open(self.path, "rb") as f:
            self._skip_header(f)
            with dctx.stream_reader(f, closefd=False) as reader:
                # Skip to desired offset
                while current < offset:
                    skip = min(self.chunk_size, offset - current)
                    data = reader.read(skip)
                    if not data:
                        raise IndexError(f"Chunk {index} past end of data")
                    current += len(data)
                return reader.read(self.chunk_size)

    def _iter_chunks(self) -> Iterator[bytes]:
        """Yield decompressed chunks sequentially."""
        try:
            import zstandard as zstd  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError("zstandard required for streaming decompression")

        dctx = zstd.ZstdDecompressor()
        with open(self.path, "rb") as f:
            self._skip_header(f)
            with dctx.stream_reader(f, closefd=False) as reader:
                while True:
                    chunk = reader.read(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk

    @staticmethod
    def _skip_header(f: BinaryIO) -> None:
        """Advance past the HMRIO_DS header if present."""
        magic = f.read(len(_MAGIC))
        if magic == _MAGIC:
            # Skip version byte
            f.read(1)
            # Skip original filename
            (name_len,) = struct.unpack("<H", f.read(2))
            f.read(name_len)
        else:
            # Not a HammerIO file — rewind
            f.seek(0)

    # ------------------------------------------------------------------
    # Internal: tar archive access
    # ------------------------------------------------------------------

    def _count_tar_members(self) -> int:
        """Count regular-file members in a tar archive."""
        count = 0
        for _ in self._open_tar_read():
            count += 1
        return count

    def _read_tar_member(self, index: int) -> bytes:
        """Extract the *index*-th member's content."""
        for i, (info, fileobj) in enumerate(self._open_tar_read()):
            if i == index:
                assert fileobj is not None
                return fileobj.read()
        raise IndexError(f"Index {index} out of range")

    def _iter_tar(self) -> Iterator[bytes]:
        """Yield contents of each tar member."""
        for info, fileobj in self._open_tar_read():
            if fileobj is not None:
                yield fileobj.read()

    def _open_tar_read(self) -> Iterator[tuple[tarfile.TarInfo, Optional[BinaryIO]]]:
        """Open the (possibly compressed) tar and yield ``(TarInfo, fileobj)``."""
        if self.path.name.endswith(".tar.zst"):
            try:
                import zstandard as zstd  # type: ignore[import-untyped]

                dctx = zstd.ZstdDecompressor()
                with open(self.path, "rb") as raw:
                    with dctx.stream_reader(raw, closefd=False) as reader:
                        # tarfile needs a seekable-ish wrapper in stream mode
                        with tarfile.open(fileobj=reader, mode="r|") as tar:  # type: ignore[arg-type]
                            for member in tar:
                                if member.isfile():
                                    yield member, tar.extractfile(member)
                return
            except ImportError:
                pass

        # Fallback: gzip tar
        mode = "r:gz" if self.path.name.endswith(".tar.gz") else "r:*"
        with tarfile.open(str(self.path), mode) as tar:
            for member in tar:
                if member.isfile():
                    yield member, tar.extractfile(member)

    # ------------------------------------------------------------------
    # High-level helpers for common dataset formats
    # ------------------------------------------------------------------

    def iter_lines(self, encoding: str = "utf-8") -> Iterator[str]:
        """Yield decoded text lines from a compressed CSV/TSV file.

        Handles line boundaries that span chunk boundaries correctly.
        """
        leftover = ""
        for chunk in self._iter_chunks():
            text = leftover + chunk.decode(encoding)
            lines = text.split("\n")
            # Last element may be incomplete — carry it over
            leftover = lines.pop()
            for line in lines:
                if line:
                    yield line
        if leftover:
            yield leftover

    def iter_tensors(self) -> Iterator[Any]:
        """Yield PyTorch tensors from a compressed ``.pt`` file.

        The compressed file is expected to contain a single pickled
        object (a tensor or list of tensors).  If the object is a list
        or tuple each element is yielded individually; otherwise the
        whole object is yielded once.
        """
        try:
            import torch  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError("PyTorch is required for iter_tensors()")

        # Reassemble all chunks into a single buffer for torch.load
        buf = io.BytesIO(b"".join(self._iter_chunks()))
        obj = torch.load(buf, weights_only=True)
        if isinstance(obj, (list, tuple)):
            yield from obj
        else:
            yield obj
