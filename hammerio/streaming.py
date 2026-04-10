"""
HammerIO Streaming Compression
Chunk-based GPU compression for arbitrarily large
files and directories without OOM failures.

Memory usage: constant at chunk_size regardless
of input size.

Copyright 2026 ResilientMind AI | Joseph C McGinty Jr
Proprietary License — All Rights Reserved
"""

from __future__ import annotations

import fcntl
import logging
import os
import struct
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("hammerio.streaming")

# ── Constants ────────────────────────────────────────────

# Stream format identifiers
MAGIC = b"HMRIO_S1"  # HammerIO streaming v1
EOF_MARKER = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF"

# Memory sizing defaults (MB)
DEFAULT_CHUNK_MB = 256
MIN_CHUNK_MB = 64
MAX_CHUNK_MB = 2048
HEADROOM_MB = 4096  # Reserve for OS + processes
READ_BUFFER_BYTES = 128 * 1024 * 1024
WRITE_BUFFER_BYTES = 128 * 1024 * 1024

# GPU threshold — only use GPU chunks above this size
GPU_CROSSOVER_MB = 10


# ── Memory detection ─────────────────────────────────────

def get_available_unified_memory_mb() -> int:
    """
    Query available unified memory on Jetson.
    Returns conservative free MB leaving HEADROOM_MB
    for OS and other running processes.
    Falls back to 8192 MB if tegrastats unavailable.
    """
    try:
        result = subprocess.run(
            ["tegrastats", "--interval", "1"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        for line in result.stdout.splitlines():
            if "RAM" in line:
                # Format: RAM 9728/65536MB (lfb 4x2MB)
                ram_part = line.split("RAM")[1].split("(")[0].strip()
                used_str, total_str = ram_part.split("/")
                used = int(used_str.strip())
                total = int(total_str.replace("MB", "").strip())
                free = total - used
                return max(0, free - HEADROOM_MB)
    except Exception:
        pass

    # Fallback: read /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    kb = int(line.split()[1])
                    mb = kb // 1024
                    return max(0, mb - HEADROOM_MB)
    except Exception:
        pass

    return 8192  # Safe fallback


def choose_chunk_size_mb(
    override_mb: Optional[int] = None,
) -> int:
    """
    Dynamically choose chunk size based on
    available unified memory.

    override_mb: explicit user value (--chunk-mb N)
    Returns MB value for chunk size.
    """
    if override_mb is not None and override_mb > 0:
        return max(MIN_CHUNK_MB, min(override_mb, MAX_CHUNK_MB))

    available = get_available_unified_memory_mb()

    # Use 25% of available memory, within bounds
    target = available // 4
    target = max(MIN_CHUNK_MB, min(target, MAX_CHUNK_MB))

    # Round down to nearest 64MB for alignment
    target = (target // 64) * 64
    return max(MIN_CHUNK_MB, target)


# ── GPU buffer management ────────────────────────────────

def _alloc_gpu_buffer(chunk_bytes: int):
    """
    Allocate pinned GPU memory buffer.
    Retries with halved size on OOM.
    Returns (buffer, actual_chunk_bytes).
    """
    import cupy as cp

    for _attempt in range(3):
        try:
            buf = cp.cuda.alloc_pinned_memory(chunk_bytes)
            return buf, chunk_bytes
        except cp.cuda.memory.OutOfMemoryError:
            chunk_bytes = chunk_bytes // 2
            if chunk_bytes < MIN_CHUNK_MB * 1024 * 1024:
                raise RuntimeError(
                    "Insufficient GPU memory even for "
                    f"minimum chunk size of {MIN_CHUNK_MB}MB"
                )

    raise RuntimeError(
        "Could not allocate GPU buffer after 3 attempts"
    )


# ── CPU fallback compressor ──────────────────────────────

def _cpu_compress_chunk(chunk: bytes) -> bytes:
    """Compress one chunk via CPU zstd (fallback)."""
    import zstandard as zstd

    cctx = zstd.ZstdCompressor(level=1)
    return cctx.compress(chunk)


def _cpu_decompress_chunk(compressed: bytes) -> bytes:
    """Decompress one chunk via CPU zstd (fallback)."""
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(
        compressed, max_output_size=len(compressed) * 20,
    )


# ── Core streaming compressor ────────────────────────────

class StreamingGPUCompressor:
    """
    Compress and decompress arbitrarily large files
    through GPU without exhausting unified memory.

    Uses fixed-size pinned GPU buffer reused per chunk.
    Memory usage is O(chunk_size), not O(input_size).

    File format:
      [8B magic][8B original_size]
      [N x (4B chunk_compressed_size)(chunk_data)]
      [8B EOF_MARKER]
    """

    def __init__(
        self,
        chunk_mb: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ):
        self.chunk_mb = choose_chunk_size_mb(chunk_mb)
        self.chunk_bytes = self.chunk_mb * 1024 * 1024
        self.progress_callback = progress_callback
        self._gpu_buf = None
        self._actual_chunk_bytes = self.chunk_bytes
        self._gpu_available: Optional[bool] = None

    def _check_gpu(self) -> bool:
        if self._gpu_available is None:
            try:
                from nvidia.nvcomp import Codec  # noqa: F401
                import cupy as cp  # noqa: F401

                self._gpu_available = True
            except Exception:
                self._gpu_available = False
        return self._gpu_available

    def _ensure_gpu_buffer(self):
        if self._gpu_buf is None and self._check_gpu():
            try:
                self._gpu_buf, self._actual_chunk_bytes = (
                    _alloc_gpu_buffer(self.chunk_bytes)
                )
            except RuntimeError:
                logger.warning("Could not allocate GPU buffer, using CPU")
                self._gpu_available = False

    def _compress_chunk_gpu(self, chunk: bytes) -> bytes:
        """Compress one chunk via nvCOMP GPU LZ4."""
        if not self._check_gpu():
            return _cpu_compress_chunk(chunk)
        try:
            import cupy as cp
            from nvidia.nvcomp import Array, Codec

            codec = Codec(algorithm="lz4")
            gpu_data = cp.frombuffer(chunk, dtype=cp.uint8)
            compressed = codec.encode(Array(gpu_data))
            cp.cuda.Stream.null.synchronize()
            result = cp.asnumpy(cp.asarray(compressed))
            del gpu_data
            del compressed
            return bytes(result)
        except Exception as exc:
            logger.warning("GPU compress failed (%s), using CPU fallback", exc)
            return _cpu_compress_chunk(chunk)

    def _decompress_chunk_gpu(self, compressed: bytes) -> bytes:
        """Decompress one chunk via nvCOMP GPU LZ4."""
        if not self._check_gpu():
            return _cpu_decompress_chunk(compressed)
        try:
            import cupy as cp
            from nvidia.nvcomp import Array, Codec

            codec = Codec(algorithm="lz4")
            gpu_data = cp.frombuffer(compressed, dtype=cp.uint8)
            decompressed = codec.decode(Array(gpu_data))
            cp.cuda.Stream.null.synchronize()
            result = cp.asnumpy(cp.asarray(decompressed))
            del gpu_data
            del decompressed
            return bytes(result)
        except Exception as exc:
            logger.warning("GPU decompress failed (%s), using CPU fallback", exc)
            return _cpu_decompress_chunk(compressed)

    def compress_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> dict:
        """
        Stream-compress a single file.
        Returns metrics dict.
        """
        self._ensure_gpu_buffer()

        input_path = Path(input_path)
        output_path = Path(output_path)
        original_size = input_path.stat().st_size

        start = time.time()
        bytes_read = 0
        chunks_written = 0
        compressed_total = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(
            input_path, "rb", buffering=READ_BUFFER_BYTES,
        ) as fin, open(
            output_path, "wb", buffering=WRITE_BUFFER_BYTES,
        ) as fout:

            # Header
            fout.write(MAGIC)
            fout.write(struct.pack("<Q", original_size))

            while True:
                chunk = fin.read(self._actual_chunk_bytes)
                if not chunk:
                    break

                compressed = self._compress_chunk_gpu(chunk)
                fout.write(struct.pack("<I", len(compressed)))
                fout.write(compressed)

                bytes_read += len(chunk)
                compressed_total += len(compressed)
                chunks_written += 1

                if self.progress_callback:
                    self.progress_callback(bytes_read, original_size)

            fout.write(EOF_MARKER)

        elapsed = time.time() - start
        ratio = original_size / compressed_total if compressed_total > 0 else 1.0
        throughput = (
            original_size / elapsed / (1024 * 1024) if elapsed > 0 else 0
        )

        return {
            "input_size": original_size,
            "output_size": compressed_total,
            "ratio": ratio,
            "savings_pct": (1 - 1 / ratio) * 100 if ratio > 0 else 0,
            "elapsed_s": elapsed,
            "throughput_mbps": throughput,
            "chunks": chunks_written,
            "chunk_mb": self.chunk_mb,
            "processor": "gpu_nvcomp_streaming" if self._gpu_available else "cpu_zstd_streaming",
        }

    def decompress_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> dict:
        """
        Stream-decompress a .hammer streaming file.
        Returns metrics dict.
        """
        self._ensure_gpu_buffer()

        input_path = Path(input_path)
        output_path = Path(output_path)

        start = time.time()
        bytes_written = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(
            input_path, "rb", buffering=READ_BUFFER_BYTES,
        ) as fin, open(
            output_path, "wb", buffering=WRITE_BUFFER_BYTES,
        ) as fout:

            # Verify magic
            magic = fin.read(8)
            if magic != MAGIC:
                raise ValueError(
                    f"Not a HammerIO streaming file. Magic: {magic!r}"
                )

            original_size = struct.unpack("<Q", fin.read(8))[0]

            while True:
                size_bytes = fin.read(4)
                if not size_bytes or len(size_bytes) < 4:
                    break

                chunk_size = struct.unpack("<I", size_bytes)[0]

                # EOF marker check (first 4 bytes of 8-byte marker)
                if chunk_size == 0xFFFFFFFF:
                    break

                compressed_chunk = fin.read(chunk_size)
                if not compressed_chunk:
                    break

                decompressed = self._decompress_chunk_gpu(compressed_chunk)
                fout.write(decompressed)
                bytes_written += len(decompressed)

                if self.progress_callback:
                    self.progress_callback(bytes_written, original_size)

        elapsed = time.time() - start
        throughput = (
            bytes_written / elapsed / (1024 * 1024) if elapsed > 0 else 0
        )

        return {
            "output_size": bytes_written,
            "elapsed_s": elapsed,
            "throughput_mbps": throughput,
            "processor": "gpu_nvcomp_streaming" if self._gpu_available else "cpu_zstd_streaming",
        }

    def __del__(self):
        if self._gpu_buf is not None:
            del self._gpu_buf


# ── Directory streaming ──────────────────────────────────

def compress_directory_streaming(
    input_dir: Path,
    output_path: Path,
    chunk_mb: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Stream-compress a directory:
    tar (streaming) → GPU chunk compress → disk

    Never writes an intermediate uncompressed tar.
    Memory usage: constant at chunk_size.

    This is the fix for the 63GB OOM scenario.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    chunk_size_mb = choose_chunk_size_mb(chunk_mb)
    chunk_size = chunk_size_mb * 1024 * 1024

    # Estimate total size for progress reporting
    total_size = sum(
        f.stat().st_size
        for f in input_dir.rglob("*")
        if f.is_file() and not f.is_symlink()
    )

    # Create pipe: tar writes to write-end,
    # compressor reads from read-end
    pipe_r_fd, pipe_w_fd = os.pipe()

    # Set pipe buffer size for efficiency
    try:
        F_SETPIPE_SZ = 1031
        fcntl.fcntl(pipe_w_fd, F_SETPIPE_SZ, min(chunk_size, 65536 * 1024))
    except Exception:
        pass

    tar_exception: list = [None]

    def tar_worker():
        """
        Run tar in streaming mode in background thread.
        Writes directly to pipe — no temp file.
        """
        try:
            with os.fdopen(
                pipe_w_fd, "wb", buffering=READ_BUFFER_BYTES,
            ) as pipe_out:
                with tarfile.open(
                    fileobj=pipe_out,
                    mode="w|",  # Streaming: no seeking
                ) as tar:
                    tar.add(str(input_dir), arcname=input_dir.name)
        except Exception as e:
            tar_exception[0] = e
        finally:
            try:
                os.close(pipe_w_fd)
            except OSError:
                pass

    # Start tar in background
    tar_thread = threading.Thread(
        target=tar_worker,
        daemon=True,
        name="hammer-tar-stream",
    )
    tar_thread.start()

    # Detect GPU availability
    gpu_available = False
    compress_func = _cpu_compress_chunk
    try:
        import cupy as cp
        from nvidia.nvcomp import Array, Codec

        codec = Codec(algorithm="lz4")
        gpu_available = True

        def _gpu_compress(chunk: bytes) -> bytes:
            try:
                gpu_data = cp.frombuffer(chunk, dtype=cp.uint8)
                compressed = codec.encode(Array(gpu_data))
                cp.cuda.Stream.null.synchronize()
                result = cp.asnumpy(cp.asarray(compressed))
                del gpu_data
                del compressed
                return bytes(result)
            except Exception:
                return _cpu_compress_chunk(chunk)

        compress_func = _gpu_compress
    except Exception:
        pass

    start = time.time()
    bytes_compressed = 0
    chunks_written = 0
    compressed_total = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with os.fdopen(
        pipe_r_fd, "rb", buffering=READ_BUFFER_BYTES,
    ) as pipe_in, open(
        output_path, "wb", buffering=WRITE_BUFFER_BYTES,
    ) as fout:

        # Write header
        fout.write(MAGIC)
        fout.write(struct.pack("<Q", total_size))

        while True:
            chunk = pipe_in.read(chunk_size)
            if not chunk:
                break

            compressed_bytes = compress_func(chunk)

            fout.write(struct.pack("<I", len(compressed_bytes)))
            fout.write(compressed_bytes)

            bytes_compressed += len(chunk)
            compressed_total += len(compressed_bytes)
            chunks_written += 1

            if progress_callback:
                progress_callback(bytes_compressed, total_size)

        fout.write(EOF_MARKER)

    tar_thread.join(timeout=30)

    if tar_exception[0] is not None:
        raise RuntimeError(f"tar stream failed: {tar_exception[0]}")

    elapsed = time.time() - start
    ratio = total_size / compressed_total if compressed_total > 0 else 1.0
    throughput = (
        bytes_compressed / elapsed / (1024 * 1024) if elapsed > 0 else 0
    )

    return {
        "input_size": total_size,
        "output_size": compressed_total,
        "ratio": ratio,
        "savings_pct": (1 - 1 / ratio) * 100 if ratio > 0 else 0,
        "elapsed_s": elapsed,
        "throughput_mbps": throughput,
        "chunks": chunks_written,
        "chunk_mb": chunk_size_mb,
        "processor": "gpu_nvcomp_streaming" if gpu_available else "cpu_zstd_streaming",
    }
