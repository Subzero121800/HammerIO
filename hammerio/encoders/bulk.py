"""Bulk data encoder for HammerIO.

Provides GPU-accelerated compression via nvCOMP when available,
with automatic fallback to CPU-based implementations (zstandard, lz4).
Streams large files in configurable chunks to keep memory usage bounded
and writes a simple container format for later decompression.
"""

from __future__ import annotations

import logging
import os
import struct
import tarfile
import tempfile
from pathlib import Path
from typing import Callable, Optional, Union

from hammerio.core.hardware import HardwareProfile

logger = logging.getLogger("hammerio.encoders.bulk")

# ---------------------------------------------------------------------------
# Container format constants
# ---------------------------------------------------------------------------
MAGIC = b"HMIO"
FORMAT_VERSION = 2
ALGO_FIELD_LEN = 32  # algorithm name, null-padded
# Header: magic(4) + version(4) + algo(32) + original_size(8) = 48 bytes
HEADER_STRUCT = struct.Struct(f"<4sI{ALGO_FIELD_LEN}sQ")
# v2: 64-bit chunk lengths to support chunks > 4GB
CHUNK_SIZE_STRUCT_V2 = struct.Struct("<Q")
# v1 compat: 32-bit chunk lengths (read-only, for old files)
CHUNK_SIZE_STRUCT_V1 = struct.Struct("<I")

SUPPORTED_ALGORITHMS = ("lz4", "snappy", "zstd", "deflate")
DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024  # 64 MiB

# ---------------------------------------------------------------------------
# Dynamic GPU chunk sizing
# ---------------------------------------------------------------------------
# On Jetson (unified memory) the GPU can address most of system RAM.
# Larger chunks reduce synchronization overhead at the cost of memory.
# nvCOMP needs ~2x chunk_size of GPU memory (input + output buffers).

_1GB = 1024 * 1024 * 1024


def _compute_gpu_chunk_size(file_size: int, gpu_memory_mb: int = 0) -> int:
    """Choose an optimal GPU chunk size based on file size and GPU memory.

    Returns chunk size in bytes.  The goal is to keep the total number of
    chunks low (ideally ≤10) to minimise CPU-GPU sync overhead, while
    staying within GPU memory limits.
    """
    if gpu_memory_mb > 0:
        # nvCOMP needs ~3x chunk_size at peak (input + output + internal scratch).
        # On Jetson unified memory, pinned host buffers also consume from the
        # same pool.  Use at most 15% of total GPU memory per chunk to leave
        # headroom for double-buffered pinned allocations and system use.
        mem_limit = int(gpu_memory_mb * 0.15) * 1024 * 1024
    else:
        # Conservative default: 1 GB per chunk
        mem_limit = _1GB

    # Target ≤10 chunks for the file
    target_chunks = 10
    ideal_chunk = file_size // target_chunks if file_size > 0 else 256 * 1024 * 1024

    # Clamp: at least 256MB, at most mem_limit
    chunk = max(256 * 1024 * 1024, min(ideal_chunk, mem_limit))

    # Round to nearest 256MB boundary for alignment
    alignment = 256 * 1024 * 1024
    chunk = ((chunk + alignment - 1) // alignment) * alignment

    return chunk

# ---------------------------------------------------------------------------
# GPU compression helpers (nvCOMP)
# ---------------------------------------------------------------------------

_nvcomp_available: Optional[bool] = None


def _check_nvcomp() -> bool:
    """Return True if the nvCOMP Python bindings are importable."""
    global _nvcomp_available
    if _nvcomp_available is None:
        try:
            from nvidia.nvcomp import Codec  # noqa: F401
            _nvcomp_available = True
            logger.debug("nvCOMP available via nvidia-nvcomp.")
        except Exception:
            try:
                import kvikio.nvcomp as _nvc  # noqa: F401
                _nvcomp_available = True
                logger.debug("nvCOMP available via kvikio.")
            except Exception:
                _nvcomp_available = False
                logger.debug("nvCOMP not found; using CPU fallback.")
    return _nvcomp_available


def _to_pinned_array(data: bytes):
    """Copy *data* into a pinned (page-locked) host buffer and return as numpy array."""
    import cupy as cp
    import numpy as np

    pinned_mem = cp.cuda.alloc_pinned_memory(len(data))
    np_arr = np.frombuffer(pinned_mem, dtype=np.uint8, count=len(data))
    np_arr[:] = np.frombuffer(data, dtype=np.uint8)
    return np_arr


def _gpu_compress(data: bytes, algorithm: str) -> bytes:
    """Compress *data* on the GPU using nvCOMP (nvidia-nvcomp).

    Uses pinned memory for faster host-to-device transfers.

    Args:
        data: Raw bytes to compress.
        algorithm: One of: lz4, snappy, zstd, deflate, gdeflate.

    Returns:
        Compressed bytes.
    """
    import cupy as cp
    from nvidia.nvcomp import Codec, Array

    codec = Codec(algorithm=algorithm)
    np_pinned = _to_pinned_array(data)
    d_input = cp.asarray(np_pinned)
    compressed = codec.encode(Array(d_input))
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(cp.asarray(compressed)).tobytes()


def _gpu_decompress(data: bytes, algorithm: str) -> bytes:
    """Decompress *data* on the GPU using nvCOMP.

    Uses pinned memory for faster host-to-device transfers.

    Args:
        data: Compressed bytes (must have been compressed by nvCOMP).
        algorithm: Algorithm that was used for compression.

    Returns:
        Decompressed bytes.
    """
    import cupy as cp
    from nvidia.nvcomp import Codec, Array

    codec = Codec(algorithm=algorithm)
    np_pinned = _to_pinned_array(data)
    d_input = cp.asarray(np_pinned)
    decompressed = codec.decode(Array(d_input))
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(cp.asarray(decompressed)).tobytes()


# ---------------------------------------------------------------------------
# CPU compression helpers
# ---------------------------------------------------------------------------

def _cpu_compress(data: bytes, algorithm: str, quality: int) -> bytes:
    """Compress *data* on the CPU.

    Args:
        data: Raw bytes to compress.
        algorithm: Compression algorithm name.
        quality: Compression level (interpretation varies by algorithm).

    Returns:
        Compressed bytes.
    """
    if algorithm == "lz4":
        import lz4.frame
        return lz4.frame.compress(data, compression_level=quality)

    if algorithm == "zstd":
        import zstandard
        compressor = zstandard.ZstdCompressor(level=quality)
        return compressor.compress(data)

    if algorithm == "snappy":
        import snappy
        return snappy.compress(data)

    if algorithm == "deflate":
        import zlib
        return zlib.compress(data, quality)

    raise ValueError(f"Unsupported CPU algorithm: {algorithm}")


def _cpu_decompress(data: bytes, algorithm: str) -> bytes:
    """Decompress *data* on the CPU.

    Args:
        data: Compressed bytes.
        algorithm: Algorithm used during compression.

    Returns:
        Decompressed bytes.
    """
    if algorithm == "lz4":
        import lz4.frame
        return lz4.frame.decompress(data)

    if algorithm == "zstd":
        import zstandard
        decompressor = zstandard.ZstdDecompressor()
        # max_output_size must accommodate the largest possible uncompressed
        # chunk.  With dynamic GPU chunk sizing this can be up to 2GB+.
        # Use len(data) * 20 as a safe upper bound (20x expansion ratio).
        max_out = max(len(data) * 20, 2 * 1024 * 1024 * 1024)
        return decompressor.decompress(data, max_output_size=max_out)

    if algorithm == "snappy":
        import snappy
        return snappy.decompress(data)

    if algorithm == "deflate":
        import zlib
        return zlib.decompress(data)

    raise ValueError(f"Unsupported CPU algorithm: {algorithm}")


# ---------------------------------------------------------------------------
# BulkEncoder
# ---------------------------------------------------------------------------

class BulkEncoder:
    """GPU-accelerated bulk data compressor with CPU fallback.

    Streams input files in fixed-size chunks and writes a lightweight
    container format (``HMIO``) that stores each compressed chunk with
    a length prefix so the file can be decompressed without seeking.

    Args:
        hardware: A :class:`HardwareProfile` describing the current
            system capabilities.  Used to decide whether to route
            compression through nvCOMP or fall back to CPU.
    """

    # Maximum chunks before GPU overhead exceeds benefit — route to CPU
    GPU_MAX_CHUNKS = 20

    def __init__(self, hardware: HardwareProfile) -> None:
        self.hardware = hardware
        self.chunk_size: int = DEFAULT_CHUNK_SIZE

        # Decide backend: prefer GPU when nvCOMP is reported available *and*
        # the Python bindings can actually be imported.
        self._use_gpu: bool = hardware.has_nvcomp and _check_nvcomp()
        if self._use_gpu:
            logger.info("BulkEncoder initialised with GPU (nvCOMP) backend.")
        else:
            logger.info("BulkEncoder initialised with CPU backend.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        algorithm: str,
        quality: Union[int, str] = 3,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Compress a file using the specified algorithm.

        Args:
            input_path: Path to the source file.
            output_path: Destination path for the compressed container.
            algorithm: Compression algorithm (``lz4``, ``snappy``,
                ``zstd``, or ``deflate``).
            quality: Compression level hint (0-22 for zstd, 0-16 for
                lz4, 1-9 for deflate; ignored by snappy).
            progress_callback: Optional callable receiving a float in
                [0.0, 1.0] representing completion progress.
            job_id: Optional job identifier for logging context.

        Returns:
            The output path as a string.

        Raises:
            ValueError: If *algorithm* is not supported.
            FileNotFoundError: If *input_path* does not exist.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        algorithm = algorithm.lower().replace("nvcomp_", "")
        # Map quality preset strings to integer levels
        if isinstance(quality, str):
            _quality_map = {"fast": 1, "balanced": 3, "quality": 9, "lossless": 19}
            quality = _quality_map.get(quality, 3)
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Choose from: {', '.join(SUPPORTED_ALGORITHMS)}"
            )

        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        # Directory: create a temporary tar archive, compress it, clean up
        if input_path.is_dir():
            return self._process_directory(
                input_path, output_path, algorithm, quality,
                progress_callback, job_id,
            )

        original_size = input_path.stat().st_size
        prefix = f"[job={job_id}] " if job_id else ""
        logger.info(
            "%sCompressing %s (%d bytes) with %s (quality=%d)",
            prefix, input_path.name, original_size, algorithm, quality,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(output_path.parent, os.W_OK):
            raise PermissionError(
                f"Output directory is not writable: {output_path.parent}"
            )

        # Dynamic chunk sizing for GPU: pick chunk size based on file size
        use_gpu = self._use_gpu
        active_chunk_size = self.chunk_size  # local copy — don't mutate self
        if use_gpu:
            gpu_mem = self.hardware.gpu_memory_mb
            gpu_chunk = _compute_gpu_chunk_size(original_size, gpu_mem)
            est_chunks = (original_size + gpu_chunk - 1) // gpu_chunk
            if est_chunks > self.GPU_MAX_CHUNKS:
                logger.warning(
                    "Large file chunking overhead (%d chunks) may exceed GPU "
                    "benefit — routing to CPU zstd for %s",
                    est_chunks, input_path.name,
                )
                use_gpu = False
            else:
                active_chunk_size = gpu_chunk
                logger.info(
                    "GPU chunk size: %d MB (%d chunks for %d MB file)",
                    gpu_chunk // (1024 * 1024), est_chunks,
                    original_size // (1024 * 1024),
                )

        bytes_read = 0
        algo_bytes = algorithm.encode("ascii").ljust(ALGO_FIELD_LEN, b"\x00")

        with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
            # Write header (v2 format with 64-bit chunk lengths)
            header = HEADER_STRUCT.pack(MAGIC, FORMAT_VERSION, algo_bytes, original_size)
            fout.write(header)

            if use_gpu:
                bytes_read = self._gpu_process_chunks(
                    fin, fout, algorithm, original_size,
                    progress_callback, job_id,
                    chunk_size=active_chunk_size,
                )
            else:
                while True:
                    chunk = fin.read(active_chunk_size)
                    if not chunk:
                        break

                    compressed = _cpu_compress(chunk, algorithm, quality)
                    fout.write(CHUNK_SIZE_STRUCT_V2.pack(len(compressed)))
                    fout.write(compressed)

                    bytes_read += len(chunk)
                    if progress_callback is not None:
                        progress = min(bytes_read / original_size * 100, 100.0) if original_size else 100.0
                        progress_callback(job_id or "", round(progress, 1))

        compressed_size = output_path.stat().st_size
        ratio = compressed_size / original_size if original_size else 0.0
        logger.info(
            "%sCompression complete: %d -> %d bytes (ratio %.2f)",
            prefix, original_size, compressed_size, ratio,
        )
        return str(output_path)

    def decompress(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Decompress a HMIO container file.

        Args:
            input_path: Path to the ``.hmio`` container.
            output_path: Destination path for the decompressed data.
            progress_callback: Optional callable receiving a float in
                [0.0, 1.0] representing completion progress.
            job_id: Optional job identifier for logging context.

        Returns:
            The output path as a string.

        Raises:
            ValueError: If the container header is invalid.
            FileNotFoundError: If *input_path* does not exist.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        prefix = f"[job={job_id}] " if job_id else ""
        compressed_size = input_path.stat().st_size

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
            # Read header
            header_data = fin.read(HEADER_STRUCT.size)
            if len(header_data) < HEADER_STRUCT.size:
                raise ValueError("File too small to contain a valid HMIO header.")

            magic, version, algo_bytes, original_size = HEADER_STRUCT.unpack(header_data)

            if magic != MAGIC:
                raise ValueError(
                    f"Invalid magic bytes: expected {MAGIC!r}, got {magic!r}"
                )
            if version not in (1, 2):
                raise ValueError(
                    f"Unsupported format version: {version} (expected 1 or 2)"
                )

            # Select chunk size struct based on container version
            chunk_struct = CHUNK_SIZE_STRUCT_V2 if version >= 2 else CHUNK_SIZE_STRUCT_V1

            algorithm = algo_bytes.rstrip(b"\x00").decode("ascii")
            if algorithm not in SUPPORTED_ALGORITHMS:
                raise ValueError(f"Unknown algorithm in container: '{algorithm}'")

            logger.info(
                "%sDecompressing %s (v%d, algorithm=%s, original_size=%d)",
                prefix, input_path.name, version, algorithm, original_size,
            )

            bytes_written = 0

            if self._use_gpu:
                bytes_written = self._gpu_decompress_chunks(
                    fin, fout, algorithm, original_size,
                    progress_callback, job_id,
                    format_version=version,
                )
            else:
                while True:
                    size_data = fin.read(chunk_struct.size)
                    if not size_data:
                        break
                    if len(size_data) < chunk_struct.size:
                        raise ValueError("Truncated chunk size field.")

                    (chunk_len,) = chunk_struct.unpack(size_data)
                    compressed_chunk = fin.read(chunk_len)
                    if len(compressed_chunk) < chunk_len:
                        raise ValueError("Truncated compressed chunk data.")

                    decompressed = _cpu_decompress(compressed_chunk, algorithm)
                    fout.write(decompressed)

                    bytes_written += len(decompressed)
                    if progress_callback is not None:
                        progress = min(bytes_written / original_size * 100, 100.0) if original_size else 100.0
                        progress_callback(job_id or "", round(progress, 1))

        logger.info(
            "%sDecompression complete: %d bytes restored.", prefix, bytes_written,
        )
        return str(output_path)

    # ------------------------------------------------------------------
    # Directory support
    # ------------------------------------------------------------------

    def _process_directory(
        self,
        input_path: Path,
        output_path: Path,
        algorithm: str,
        quality: Union[int, str],
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: Optional[str],
    ) -> str:
        """Tar a directory, then compress the tar with :meth:`process`.

        The intermediate tar is written to a temporary file in the same
        parent directory as *output_path* so it stays on the same
        filesystem (no cross-device moves).
        """
        prefix = f"[job={job_id}] " if job_id else ""
        logger.info("%sPacking directory %s into tar before compression", prefix, input_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tar_tmp = tempfile.mkstemp(
            suffix=".tar", dir=str(output_path.parent),
        )
        os.close(fd)
        try:
            with tarfile.open(tar_tmp, "w") as tar:
                tar.add(str(input_path), arcname=input_path.name)
            return self.process(
                input_path=tar_tmp,
                output_path=output_path,
                algorithm=algorithm,
                quality=quality,
                progress_callback=progress_callback,
                job_id=job_id,
            )
        finally:
            try:
                os.unlink(tar_tmp)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Pipelined GPU helpers
    # ------------------------------------------------------------------

    def _gpu_process_chunks(
        self,
        fin,
        fout,
        algorithm: str,
        original_size: int,
        progress_callback: Optional[Callable] = None,
        job_id: Optional[str] = None,
        chunk_size: int = 0,
    ) -> int:
        """Compress chunks with double-buffered CUDA streams and pinned memory.

        Overlaps host-to-device transfer of chunk N+1 with GPU encoding of
        chunk N, roughly halving the wall-clock time spent on transfers.

        Returns total bytes read.
        """
        import cupy as cp
        import numpy as np
        from nvidia.nvcomp import Codec, Array

        codec = Codec(algorithm=algorithm)
        chunk_sz = chunk_size or self.chunk_size

        # Two CUDA streams for double-buffering
        streams = [cp.cuda.Stream(non_blocking=True),
                   cp.cuda.Stream(non_blocking=True)]

        # Two pinned host buffers
        pinned = [cp.cuda.alloc_pinned_memory(chunk_sz),
                  cp.cuda.alloc_pinned_memory(chunk_sz)]

        bytes_read = 0
        buf_idx = 0
        pending = None  # (stream, compressed_nvcomp_array)

        while True:
            chunk = fin.read(chunk_sz)
            if not chunk:
                break

            chunk_len = len(chunk)
            stream = streams[buf_idx]
            np_pinned = np.frombuffer(pinned[buf_idx], dtype=np.uint8, count=chunk_len)
            np_pinned[:] = np.frombuffer(chunk, dtype=np.uint8)

            # Launch async H2D + encode on this stream
            with stream:
                d_input = cp.asarray(np_pinned)
                compressed = codec.encode(Array(d_input))

            # While GPU works on current chunk, flush previous result
            if pending is not None:
                p_stream, p_comp = pending
                p_stream.synchronize()
                comp_bytes = cp.asnumpy(cp.asarray(p_comp)).tobytes()
                fout.write(CHUNK_SIZE_STRUCT_V2.pack(len(comp_bytes)))
                fout.write(comp_bytes)

            pending = (stream, compressed)
            buf_idx = 1 - buf_idx

            bytes_read += chunk_len
            if progress_callback is not None:
                progress = min(bytes_read / original_size * 100, 100.0) if original_size else 100.0
                progress_callback(job_id or "", round(progress, 1))

        # Flush last pending chunk
        if pending is not None:
            p_stream, p_comp = pending
            p_stream.synchronize()
            comp_bytes = cp.asnumpy(cp.asarray(p_comp)).tobytes()
            fout.write(CHUNK_SIZE_STRUCT_V2.pack(len(comp_bytes)))
            fout.write(comp_bytes)

        return bytes_read

    def _gpu_decompress_chunks(
        self,
        fin,
        fout,
        algorithm: str,
        original_size: int,
        progress_callback: Optional[Callable] = None,
        job_id: Optional[str] = None,
        format_version: int = FORMAT_VERSION,
    ) -> int:
        """Decompress chunks with double-buffered CUDA streams and pinned memory."""
        import cupy as cp
        import numpy as np
        from nvidia.nvcomp import Codec, Array

        codec = Codec(algorithm=algorithm)
        chunk_struct = CHUNK_SIZE_STRUCT_V2 if format_version >= 2 else CHUNK_SIZE_STRUCT_V1

        streams = [cp.cuda.Stream(non_blocking=True),
                   cp.cuda.Stream(non_blocking=True)]

        bytes_written = 0
        buf_idx = 0
        pending = None  # (stream, decompressed_array)

        while True:
            size_data = fin.read(chunk_struct.size)
            if not size_data:
                break
            if len(size_data) < chunk_struct.size:
                raise ValueError("Truncated chunk size field.")

            (chunk_len,) = chunk_struct.unpack(size_data)
            compressed_chunk = fin.read(chunk_len)
            if len(compressed_chunk) < chunk_len:
                raise ValueError("Truncated compressed chunk data.")

            stream = streams[buf_idx]
            np_pinned = _to_pinned_array(compressed_chunk)

            with stream:
                d_input = cp.asarray(np_pinned)
                decompressed = codec.decode(Array(d_input))

            # Flush previous result while GPU works
            if pending is not None:
                p_stream, p_dec = pending
                p_stream.synchronize()
                dec_bytes = cp.asnumpy(cp.asarray(p_dec)).tobytes()
                fout.write(dec_bytes)
                bytes_written += len(dec_bytes)

            pending = (stream, decompressed)
            buf_idx = 1 - buf_idx

            if progress_callback is not None:
                progress = min(bytes_written / original_size * 100, 100.0) if original_size else 100.0
                progress_callback(job_id or "", round(progress, 1))

        # Flush last pending
        if pending is not None:
            p_stream, p_comp = pending
            p_stream.synchronize()
            dec_bytes = cp.asnumpy(cp.asarray(p_comp)).tobytes()
            fout.write(dec_bytes)
            bytes_written += len(dec_bytes)

        return bytes_written

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compress_chunk(self, data: bytes, algorithm: str, quality: int) -> bytes:
        """Compress a single chunk, routing to GPU or CPU."""
        if self._use_gpu:
            try:
                return _gpu_compress(data, algorithm)
            except Exception as exc:
                logger.warning(
                    "GPU compression failed, falling back to CPU: %s", exc,
                )
        return _cpu_compress(data, algorithm, quality)

    def _decompress_chunk(self, data: bytes, algorithm: str) -> bytes:
        """Decompress a single chunk, routing to GPU or CPU."""
        if self._use_gpu:
            try:
                return _gpu_decompress(data, algorithm)
            except Exception as exc:
                logger.warning(
                    "GPU decompression failed, falling back to CPU: %s", exc,
                )
        return _cpu_decompress(data, algorithm)
