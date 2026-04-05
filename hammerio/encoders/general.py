"""General-purpose CPU compression encoder for HammerIO.

Supports zstd, gzip, bzip2, and lz4 compression with streaming I/O,
directory archiving, parallel batch processing, and decompression.
"""

from __future__ import annotations

import bz2
import gzip
import io
import logging
import os
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional, Union

from hammerio.core.hardware import HardwareProfile

logger = logging.getLogger("hammerio.encoders.general")

# Default streaming chunk size: 8 MB (16 MB for sequential reads of large files)
_CHUNK_SIZE = 8 * 1024 * 1024
_CHUNK_SIZE_LARGE = 16 * 1024 * 1024
_LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10 MB

# Quality preset to zstd compression level mapping
_ZSTD_QUALITY_MAP: dict[str, int] = {
    "fast": 1,
    "balanced": 3,
    "quality": 9,
    "lossless": 19,
}

# Quality preset to gzip/bz2 compression level mapping (1-9 range)
_GZIP_QUALITY_MAP: dict[str, int] = {
    "fast": 1,
    "balanced": 6,
    "quality": 9,
    "lossless": 9,
}

_BZ2_QUALITY_MAP: dict[str, int] = {
    "fast": 1,
    "balanced": 6,
    "quality": 9,
    "lossless": 9,
}

_LZ4_QUALITY_MAP: dict[str, int] = {
    "fast": 1,
    "balanced": 6,
    "quality": 9,
    "lossless": 12,
}

# File extension mapping per algorithm
_EXTENSION_MAP: dict[str, str] = {
    "zstd": ".zst",
    "gzip": ".gz",
    "bzip2": ".bz2",
    "lz4": ".lz4",
}

# Tar archive extension mapping per algorithm
_TAR_EXTENSION_MAP: dict[str, str] = {
    "zstd": ".tar.zst",
    "gzip": ".tar.gz",
    "bzip2": ".tar.bz2",
    "lz4": ".tar.lz4",
}


def _get_lz4_frame() -> Optional[object]:
    """Try to import lz4.frame, returning the module or None."""
    try:
        import lz4.frame  # type: ignore[import-untyped]
        return lz4.frame
    except ImportError:
        return None


def _lz4_subprocess_available() -> bool:
    """Check whether the lz4 CLI tool is available."""
    return shutil.which("lz4") is not None


# Magic bytes used for format auto-detection in decompress()
_MAGIC_BYTES: dict[str, list[bytes]] = {
    "zstd": [b"\x28\xb5\x2f\xfd"],
    "gzip": [b"\x1f\x8b"],
    "bzip2": [b"\x42\x5a\x68"],            # "BZh"
    "lz4": [b"\x04\x22\x4d\x18"],
}


def detect_format(path: Path) -> tuple[str, bool]:
    """Detect compression format from file extension, falling back to magic bytes.

    Returns:
        A ``(algorithm, is_tar)`` tuple.  *algorithm* is one of
        ``"zstd"``, ``"gzip"``, ``"bzip2"``, ``"lz4"``.

    Raises:
        ValueError: When the format cannot be determined.
    """
    name = path.name

    # 1. Try extension-based detection (fast path)
    for algo, ext in _TAR_EXTENSION_MAP.items():
        if name.endswith(ext):
            return algo, True

    for algo, ext in _EXTENSION_MAP.items():
        if name.endswith(ext):
            return algo, False

    # 2. Fallback: read magic bytes from the file header
    try:
        with open(path, "rb") as fh:
            header = fh.read(8)
    except OSError:
        header = b""

    for algo, magics in _MAGIC_BYTES.items():
        for magic in magics:
            if header[:len(magic)] == magic:
                return algo, False

    raise ValueError(
        f"Cannot determine compression format from extension or "
        f"magic bytes: {name}"
    )


class GeneralEncoder:
    """CPU-based general-purpose compression encoder.

    Supports zstd, gzip, bzip2, and lz4 algorithms with streaming I/O
    to keep memory usage bounded. Directories are automatically archived
    as tar bundles with the chosen compression applied.

    Args:
        hardware: The detected hardware profile used for tuning thread
            counts and chunk sizes.
    """

    def __init__(self, hardware: HardwareProfile) -> None:
        self.hardware = hardware
        # Cap worker threads to available CPU cores (min 1, max 8)
        self._max_workers = min(max(hardware.cpu_cores // 2, 1), 8)
        logger.debug(
            "GeneralEncoder initialised: %d CPU cores, %d workers",
            hardware.cpu_cores,
            self._max_workers,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path, None],
        algorithm: str = "zstd",
        quality: str = "balanced",
        progress_callback: Optional[Callable[[str, float], None]] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Compress a file or directory.

        Args:
            input_path: Path to the source file or directory.
            output_path: Destination path. If ``None`` an appropriate path
                is generated next to the input.
            algorithm: Compression algorithm — one of ``zstd``, ``gzip``,
                ``bzip2``, ``lz4``.
            quality: Quality preset — ``fast``, ``balanced``, ``quality``,
                or ``lossless``.
            progress_callback: Optional ``callback(job_id, pct)`` invoked
                with a float in ``[0, 100]``.
            job_id: Identifier passed through to *progress_callback*.

        Returns:
            Absolute path to the compressed output file as a string.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            ValueError: If *algorithm* is not supported.
        """
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        algorithm = algorithm.lower()
        if algorithm not in _EXTENSION_MAP:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Choose from: {', '.join(_EXTENSION_MAP)}"
            )

        is_dir = input_path.is_dir()

        # Resolve output path
        if output_path is None:
            if is_dir:
                ext = _TAR_EXTENSION_MAP[algorithm]
                output_path = input_path.parent / (input_path.name + ext)
            else:
                ext = _EXTENSION_MAP[algorithm]
                output_path = input_path.with_suffix(input_path.suffix + ext)
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Compressing %s → %s [algo=%s, quality=%s]",
            input_path,
            output_path,
            algorithm,
            quality,
        )

        if is_dir:
            self._compress_directory(
                input_path, output_path, algorithm, quality,
                progress_callback, job_id,
            )
        else:
            self._compress_file(
                input_path, output_path, algorithm, quality,
                progress_callback, job_id,
            )

        logger.info("Compression complete: %s", output_path)
        return str(output_path)

    def decompress(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path, None] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Decompress a previously compressed file.

        The algorithm is inferred from the file extension first, then
        from magic bytes if the extension is unrecognised. Tar archives
        (e.g. ``.tar.zst``) are extracted into a directory.

        Supported formats: ``.zst``, ``.gz``, ``.bz2``, ``.lz4`` (and
        their ``.tar.*`` variants).

        Args:
            input_path: Path to the compressed file.
            output_path: Where to write the decompressed output. Inferred
                from the input name when ``None``.
            progress_callback: Optional ``callback(job_id, pct)``.
            job_id: Identifier passed through to *progress_callback*.

        Returns:
            Absolute path to the decompressed output as a string.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            ValueError: If the format cannot be determined from the
                extension or magic bytes.
        """
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        # Auto-detect format from extension, falling back to magic bytes
        algorithm, is_tar = detect_format(input_path)
        name = input_path.name

        # Resolve output path
        if output_path is None:
            if is_tar:
                # Strip .tar.<ext> and create a directory
                stem = name
                for ext in _TAR_EXTENSION_MAP.values():
                    if stem.endswith(ext):
                        stem = stem[: -len(ext)]
                        break
                output_path = input_path.parent / stem
            else:
                # Strip the compression extension
                ext = _EXTENSION_MAP[algorithm]
                if name.endswith(ext):
                    output_path = input_path.parent / name[: -len(ext)]
                else:
                    output_path = input_path.with_suffix("")
        output_path = Path(output_path).resolve()

        logger.info(
            "Decompressing %s → %s [algo=%s, tar=%s]",
            input_path, output_path, algorithm, is_tar,
        )

        if is_tar:
            self._decompress_tar(
                input_path, output_path, algorithm,
                progress_callback, job_id,
            )
        else:
            self._decompress_file(
                input_path, output_path, algorithm,
                progress_callback, job_id,
            )

        logger.info("Decompression complete: %s", output_path)
        return str(output_path)

    def process_batch(
        self,
        file_paths: list[Union[str, Path]],
        output_dir: Union[str, Path],
        algorithm: str = "zstd",
        quality: str = "balanced",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> list[str]:
        """Compress multiple files in parallel using a thread pool.

        Args:
            file_paths: List of source file paths.
            output_dir: Directory to write compressed files into.
            algorithm: Compression algorithm.
            quality: Quality preset.
            progress_callback: Optional ``callback(job_id, pct)``.

        Returns:
            List of output file paths as strings, in the same order as
            *file_paths*.
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        results: dict[int, str] = {}

        def _worker(idx: int, src: Union[str, Path]) -> tuple[int, str]:
            src = Path(src)
            ext = _EXTENSION_MAP[algorithm]
            dst = output_dir / (src.name + ext)
            job_id = f"batch_{idx:04d}"
            out = self.process(
                src, dst, algorithm, quality, progress_callback, job_id,
            )
            return idx, out

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(_worker, i, p): i
                for i, p in enumerate(file_paths)
            }
            for fut in as_completed(futures):
                idx, out = fut.result()
                results[idx] = out

        return [results[i] for i in range(len(file_paths))]

    # ------------------------------------------------------------------
    # Compression internals
    # ------------------------------------------------------------------

    def _compress_file(
        self,
        input_path: Path,
        output_path: Path,
        algorithm: str,
        quality: str,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: Optional[str],
    ) -> None:
        """Stream-compress a single file."""
        total_size = input_path.stat().st_size

        # Handle empty files gracefully
        if total_size == 0:
            self._write_empty_compressed(output_path, algorithm, quality)
            self._report_progress(1, 1, progress_callback, job_id)
            return

        bytes_read = 0

        if algorithm == "zstd":
            self._compress_file_zstd(
                input_path, output_path, quality, total_size,
                progress_callback, job_id,
            )
        elif algorithm == "gzip":
            level = _GZIP_QUALITY_MAP.get(quality, 6)
            with open(input_path, "rb") as fin, gzip.open(
                output_path, "wb", compresslevel=level
            ) as fout:
                bytes_read = self._stream_copy(
                    fin, fout, total_size, progress_callback, job_id,
                )
        elif algorithm == "bzip2":
            level = _BZ2_QUALITY_MAP.get(quality, 6)
            with open(input_path, "rb") as fin, bz2.open(
                output_path, "wb", compresslevel=level
            ) as fout:
                bytes_read = self._stream_copy(
                    fin, fout, total_size, progress_callback, job_id,
                )
        elif algorithm == "lz4":
            self._compress_file_lz4(
                input_path, output_path, quality, total_size,
                progress_callback, job_id,
            )

    def _compress_file_zstd(
        self,
        input_path: Path,
        output_path: Path,
        quality: str,
        total_size: int,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: Optional[str],
    ) -> None:
        """Compress using the zstandard library with streaming.

        Uses multi-threaded mode and larger chunk sizes for files above
        the 10 MB threshold to improve throughput on multi-core systems.
        """
        import zstandard as zstd  # type: ignore[import-untyped]

        level = _ZSTD_QUALITY_MAP.get(quality, 3)
        # Use multiple threads for zstd on files > 10 MB; single-threaded
        # (threads=0) for small files to avoid thread-pool overhead.
        is_large = total_size > _LARGE_FILE_THRESHOLD
        threads = min(self._max_workers, self.hardware.cpu_cores) if is_large else 0
        cctx = zstd.ZstdCompressor(level=level, threads=threads)

        # Use larger chunk size for sequential reads on big files
        chunk_size = _CHUNK_SIZE_LARGE if is_large else _CHUNK_SIZE

        with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
            writer = cctx.stream_writer(fout)
            bytes_read = 0
            while True:
                chunk = fin.read(chunk_size)
                if not chunk:
                    break
                writer.write(chunk)
                bytes_read += len(chunk)
                self._report_progress(
                    bytes_read, total_size, progress_callback, job_id,
                )
            writer.close()

    def _compress_file_lz4(
        self,
        input_path: Path,
        output_path: Path,
        quality: str,
        total_size: int,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: Optional[str],
    ) -> None:
        """Compress using lz4.frame or falling back to the lz4 CLI."""
        level = _LZ4_QUALITY_MAP.get(quality, 6)
        lz4_frame = _get_lz4_frame()

        if lz4_frame is not None:
            with open(input_path, "rb") as fin, lz4_frame.open(
                output_path, "wb", compression_level=level,
            ) as fout:
                self._stream_copy(
                    fin, fout, total_size, progress_callback, job_id,
                )
        elif _lz4_subprocess_available():
            logger.info("lz4 Python library not found; using lz4 CLI")
            cmd = ["lz4", f"-{min(level, 12)}", "-f",
                   str(input_path), str(output_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"lz4 subprocess failed: {result.stderr.strip()}"
                )
            self._report_progress(
                total_size, total_size, progress_callback, job_id,
            )
        else:
            raise RuntimeError(
                "lz4 compression unavailable: install the 'lz4' Python "
                "package or the 'lz4' CLI tool"
            )

    def _compress_directory(
        self,
        input_path: Path,
        output_path: Path,
        algorithm: str,
        quality: str,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: Optional[str],
    ) -> None:
        """Create a compressed tar archive from a directory.

        The archive is streamed through the compressor so the full
        uncompressed tar is never materialised on disk.
        """
        total_size = sum(
            f.stat().st_size for f in input_path.rglob("*") if f.is_file()
        )

        if algorithm == "gzip":
            level = _GZIP_QUALITY_MAP.get(quality, 6)
            with open(output_path, "wb") as fout:
                gz = gzip.GzipFile(fileobj=fout, mode="wb", compresslevel=level)
                with tarfile.open(fileobj=gz, mode="w|") as tar:
                    self._add_dir_to_tar(
                        tar, input_path, total_size, progress_callback, job_id,
                    )
                gz.close()
        elif algorithm == "bzip2":
            level = _BZ2_QUALITY_MAP.get(quality, 6)
            with open(output_path, "wb") as fout:
                bz = bz2.BZ2File(fout, mode="wb", compresslevel=level)
                with tarfile.open(fileobj=bz, mode="w|") as tar:
                    self._add_dir_to_tar(
                        tar, input_path, total_size, progress_callback, job_id,
                    )
                bz.close()
        elif algorithm == "zstd":
            import zstandard as zstd  # type: ignore[import-untyped]

            level = _ZSTD_QUALITY_MAP.get(quality, 3)
            threads = min(self._max_workers, self.hardware.cpu_cores)
            cctx = zstd.ZstdCompressor(level=level, threads=threads)
            with open(output_path, "wb") as fout:
                writer = cctx.stream_writer(fout)
                with tarfile.open(fileobj=writer, mode="w|") as tar:
                    self._add_dir_to_tar(
                        tar, input_path, total_size, progress_callback, job_id,
                    )
                writer.close()
        elif algorithm == "lz4":
            lz4_frame = _get_lz4_frame()
            if lz4_frame is not None:
                level = _LZ4_QUALITY_MAP.get(quality, 6)
                with lz4_frame.open(
                    output_path, "wb", compression_level=level,
                ) as fout:
                    with tarfile.open(fileobj=fout, mode="w|") as tar:
                        self._add_dir_to_tar(
                            tar, input_path, total_size,
                            progress_callback, job_id,
                        )
            elif _lz4_subprocess_available():
                # Two-step: create tar then compress via CLI
                tar_path = output_path.with_suffix("")
                if not str(tar_path).endswith(".tar"):
                    tar_path = output_path.parent / (output_path.stem + ".tar")
                try:
                    with tarfile.open(tar_path, "w") as tar:
                        self._add_dir_to_tar(
                            tar, input_path, total_size,
                            progress_callback, job_id,
                        )
                    level = _LZ4_QUALITY_MAP.get(quality, 6)
                    cmd = ["lz4", f"-{min(level, 12)}", "-f",
                           str(tar_path), str(output_path)]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"lz4 subprocess failed: {result.stderr.strip()}"
                        )
                finally:
                    if tar_path.exists():
                        tar_path.unlink()
            else:
                raise RuntimeError(
                    "lz4 compression unavailable: install the 'lz4' Python "
                    "package or the 'lz4' CLI tool"
                )

    # ------------------------------------------------------------------
    # Decompression internals
    # ------------------------------------------------------------------

    def _decompress_file(
        self,
        input_path: Path,
        output_path: Path,
        algorithm: str,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: Optional[str],
    ) -> None:
        """Stream-decompress a single file."""
        total_size = input_path.stat().st_size
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if algorithm == "zstd":
            import zstandard as zstd  # type: ignore[import-untyped]

            dctx = zstd.ZstdDecompressor()
            with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
                reader = dctx.stream_reader(fin)
                bytes_written = 0
                while True:
                    chunk = reader.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    fout.write(chunk)
                    bytes_written += len(chunk)
                    # Progress based on compressed bytes consumed
                    compressed_pos = fin.tell()
                    self._report_progress(
                        compressed_pos, total_size, progress_callback, job_id,
                    )
                reader.close()
        elif algorithm == "gzip":
            with gzip.open(input_path, "rb") as fin, open(output_path, "wb") as fout:
                self._stream_copy(
                    fin, fout, total_size, progress_callback, job_id,
                    progress_from_output=True,
                )
        elif algorithm == "bzip2":
            with bz2.open(input_path, "rb") as fin, open(output_path, "wb") as fout:
                self._stream_copy(
                    fin, fout, total_size, progress_callback, job_id,
                    progress_from_output=True,
                )
        elif algorithm == "lz4":
            lz4_frame = _get_lz4_frame()
            if lz4_frame is not None:
                with lz4_frame.open(input_path, "rb") as fin, open(
                    output_path, "wb"
                ) as fout:
                    self._stream_copy(
                        fin, fout, total_size, progress_callback, job_id,
                        progress_from_output=True,
                    )
            elif _lz4_subprocess_available():
                cmd = ["lz4", "-d", "-f", str(input_path), str(output_path)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"lz4 decompress failed: {result.stderr.strip()}"
                    )
                self._report_progress(
                    total_size, total_size, progress_callback, job_id,
                )
            else:
                raise RuntimeError(
                    "lz4 decompression unavailable: install the 'lz4' Python "
                    "package or the 'lz4' CLI tool"
                )

    def _decompress_tar(
        self,
        input_path: Path,
        output_path: Path,
        algorithm: str,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: Optional[str],
    ) -> None:
        """Decompress and extract a tar archive."""
        output_path.mkdir(parents=True, exist_ok=True)
        total_size = input_path.stat().st_size

        if algorithm == "gzip":
            with tarfile.open(input_path, "r:gz") as tar:
                tar.extractall(path=output_path, filter="data")
        elif algorithm == "bzip2":
            with tarfile.open(input_path, "r:bz2") as tar:
                tar.extractall(path=output_path, filter="data")
        elif algorithm == "zstd":
            import zstandard as zstd  # type: ignore[import-untyped]

            dctx = zstd.ZstdDecompressor()
            with open(input_path, "rb") as fin:
                reader = dctx.stream_reader(fin)
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    tar.extractall(path=output_path, filter="data")
                reader.close()
        elif algorithm == "lz4":
            lz4_frame = _get_lz4_frame()
            if lz4_frame is not None:
                with lz4_frame.open(input_path, "rb") as fin:
                    with tarfile.open(fileobj=fin, mode="r|") as tar:
                        tar.extractall(path=output_path, filter="data")
            elif _lz4_subprocess_available():
                cmd = ["lz4", "-d", "-c", str(input_path)]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                    tar.extractall(path=output_path, filter="data")
                proc.wait()
                if proc.returncode != 0:
                    stderr = proc.stderr.read().decode() if proc.stderr else ""
                    raise RuntimeError(
                        f"lz4 decompress failed: {stderr.strip()}"
                    )
            else:
                raise RuntimeError(
                    "lz4 decompression unavailable: install the 'lz4' Python "
                    "package or the 'lz4' CLI tool"
                )

        self._report_progress(
            total_size, total_size, progress_callback, job_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_empty_compressed(output_path: Path, algorithm: str, quality: str) -> None:
        """Write a valid compressed file from empty input."""
        if algorithm == "zstd":
            import zstandard as zstd  # type: ignore[import-untyped]
            cctx = zstd.ZstdCompressor(level=1)
            with open(output_path, "wb") as fout:
                writer = cctx.stream_writer(fout)
                writer.close()
        elif algorithm == "gzip":
            with gzip.open(output_path, "wb") as fout:
                pass
        elif algorithm == "bzip2":
            with bz2.open(output_path, "wb") as fout:
                pass
        elif algorithm == "lz4":
            lz4_frame = _get_lz4_frame()
            if lz4_frame is not None:
                with lz4_frame.open(output_path, "wb") as fout:
                    pass
            else:
                # Write empty file for lz4 CLI compatibility
                output_path.write_bytes(b"")

    def _add_dir_to_tar(
        self,
        tar: tarfile.TarFile,
        root: Path,
        total_size: int,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: Optional[str],
    ) -> None:
        """Recursively add directory contents to an open tar archive."""
        bytes_added = 0
        for file_path in sorted(root.rglob("*")):
            arcname = str(file_path.relative_to(root))
            if file_path.is_file():
                tar.add(str(file_path), arcname=arcname)
                bytes_added += file_path.stat().st_size
                self._report_progress(
                    bytes_added, total_size, progress_callback, job_id,
                )
            elif file_path.is_dir():
                tar.add(str(file_path), arcname=arcname, recursive=False)

    @staticmethod
    def _stream_copy(
        src: io.RawIOBase | io.BufferedIOBase | gzip.GzipFile,
        dst: io.RawIOBase | io.BufferedIOBase | gzip.GzipFile,
        total_size: int,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: Optional[str],
        progress_from_output: bool = False,
    ) -> int:
        """Copy *src* to *dst* in chunks, reporting progress.

        Args:
            src: Readable binary stream.
            dst: Writable binary stream.
            total_size: Total expected bytes (for progress calculation).
            progress_callback: Optional progress callback.
            job_id: Job identifier for the callback.
            progress_from_output: When ``True``, progress is estimated
                as an approximation (decompression case where input size
                is known but decompressed size is not).

        Returns:
            Total bytes read from *src*.
        """
        bytes_processed = 0
        while True:
            chunk = src.read(_CHUNK_SIZE)
            if not chunk:
                break
            dst.write(chunk)
            bytes_processed += len(chunk)
            if progress_callback and job_id and total_size > 0:
                if progress_from_output:
                    # Approximate: we don't know final size, report
                    # based on compressed input consumed when possible
                    pct = min(bytes_processed / max(total_size, 1) * 100, 99.0)
                else:
                    pct = min(bytes_processed / total_size * 100, 100.0)
                progress_callback(job_id, round(pct, 1))

        # Final 100% tick
        if progress_callback and job_id:
            progress_callback(job_id, 100.0)

        return bytes_processed

    def _write_empty_compressed(
        self,
        output_path: Path,
        algorithm: str,
        quality: str,
    ) -> None:
        """Write a valid compressed representation of an empty file.

        Some compressors require explicit handling of zero-byte inputs to
        produce a valid (decompressible) output file.
        """
        if algorithm == "zstd":
            import zstandard as zstd  # type: ignore[import-untyped]

            level = _ZSTD_QUALITY_MAP.get(quality, 3)
            cctx = zstd.ZstdCompressor(level=level)
            output_path.write_bytes(cctx.compress(b""))
        elif algorithm == "gzip":
            level = _GZIP_QUALITY_MAP.get(quality, 6)
            with gzip.open(output_path, "wb", compresslevel=level) as f:
                pass  # writing nothing produces a valid empty gzip
        elif algorithm == "bzip2":
            level = _BZ2_QUALITY_MAP.get(quality, 6)
            with bz2.open(output_path, "wb", compresslevel=level) as f:
                pass
        elif algorithm == "lz4":
            lz4_frame = _get_lz4_frame()
            if lz4_frame is not None:
                level = _LZ4_QUALITY_MAP.get(quality, 6)
                with lz4_frame.open(output_path, "wb", compression_level=level) as f:
                    pass
            elif _lz4_subprocess_available():
                # Create an empty temp file and compress it
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    cmd = ["lz4", "-f", tmp_path, str(output_path)]
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    "lz4 compression unavailable: install the 'lz4' Python "
                    "package or the 'lz4' CLI tool"
                )

    @staticmethod
    def _report_progress(
        current: int,
        total: int,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: Optional[str],
    ) -> None:
        """Fire a progress callback if one is registered."""
        if progress_callback and job_id and total > 0:
            pct = min(current / total * 100, 100.0)
            progress_callback(job_id, round(pct, 1))
