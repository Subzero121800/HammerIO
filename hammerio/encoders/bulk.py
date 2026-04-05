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
from pathlib import Path
from typing import Callable, Optional, Union

from hammerio.core.hardware import HardwareProfile

logger = logging.getLogger("hammerio.encoders.bulk")

# ---------------------------------------------------------------------------
# Container format constants
# ---------------------------------------------------------------------------
MAGIC = b"HMIO"
FORMAT_VERSION = 1
ALGO_FIELD_LEN = 32  # algorithm name, null-padded
# Header: magic(4) + version(4) + algo(32) + original_size(8) = 48 bytes
HEADER_STRUCT = struct.Struct(f"<4sI{ALGO_FIELD_LEN}sQ")
CHUNK_SIZE_STRUCT = struct.Struct("<I")

SUPPORTED_ALGORITHMS = ("lz4", "snappy", "zstd", "deflate")
DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024  # 64 MiB

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


def _gpu_compress(data: bytes, algorithm: str) -> bytes:
    """Compress *data* on the GPU using nvCOMP (nvidia-nvcomp).

    Uses the nvidia.nvcomp.Codec API (v4.x+) for GPU-accelerated compression.

    Args:
        data: Raw bytes to compress.
        algorithm: One of: lz4, snappy, zstd, deflate, gdeflate.

    Returns:
        Compressed bytes.
    """
    import cupy as cp
    from nvidia.nvcomp import Codec, Array

    codec = Codec(algorithm=algorithm)
    d_input = cp.frombuffer(bytearray(data), dtype=cp.uint8)
    arr = Array(d_input)
    compressed = codec.encode(arr)
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(cp.asarray(compressed)).tobytes()


def _gpu_decompress(data: bytes, algorithm: str) -> bytes:
    """Decompress *data* on the GPU using nvCOMP.

    Args:
        data: Compressed bytes (must have been compressed by nvCOMP).
        algorithm: Algorithm that was used for compression.

    Returns:
        Decompressed bytes.
    """
    import cupy as cp
    from nvidia.nvcomp import Codec, Array

    codec = Codec(algorithm=algorithm)
    d_input = cp.frombuffer(bytearray(data), dtype=cp.uint8)
    arr = Array(d_input)
    decompressed = codec.decode(arr)
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(cp.asarray(decompressed)).tobytes()

    d_input = cp.frombuffer(data, dtype=cp.uint8)
    manager = manager_cls()
    d_decompressed = manager.decompress(d_input)
    return cp.asnumpy(d_decompressed).tobytes()


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
        return decompressor.decompress(data, max_output_size=DEFAULT_CHUNK_SIZE * 2)

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

        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

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

        bytes_read = 0
        algo_bytes = algorithm.encode("ascii").ljust(ALGO_FIELD_LEN, b"\x00")

        with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
            # Write header
            header = HEADER_STRUCT.pack(MAGIC, FORMAT_VERSION, algo_bytes, original_size)
            fout.write(header)

            while True:
                chunk = fin.read(self.chunk_size)
                if not chunk:
                    break

                compressed = self._compress_chunk(chunk, algorithm, quality)
                fout.write(CHUNK_SIZE_STRUCT.pack(len(compressed)))
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
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported format version: {version} (expected {FORMAT_VERSION})"
                )

            algorithm = algo_bytes.rstrip(b"\x00").decode("ascii")
            if algorithm not in SUPPORTED_ALGORITHMS:
                raise ValueError(f"Unknown algorithm in container: '{algorithm}'")

            logger.info(
                "%sDecompressing %s (algorithm=%s, original_size=%d)",
                prefix, input_path.name, algorithm, original_size,
            )

            bytes_written = 0

            while True:
                size_data = fin.read(CHUNK_SIZE_STRUCT.size)
                if not size_data:
                    break
                if len(size_data) < CHUNK_SIZE_STRUCT.size:
                    raise ValueError("Truncated chunk size field.")

                (chunk_len,) = CHUNK_SIZE_STRUCT.unpack(size_data)
                compressed_chunk = fin.read(chunk_len)
                if len(compressed_chunk) < chunk_len:
                    raise ValueError("Truncated compressed chunk data.")

                decompressed = self._decompress_chunk(compressed_chunk, algorithm)
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
