"""Apple Silicon compression backend using the Accelerate/libcompression framework.

On macOS with Apple Silicon (M1/M2/M3/M4), this uses Apple's hardware-optimized
compression algorithms via libcompression.dylib:

- LZFSE — Apple-native, 2-3x faster than zstd on M-series, similar ratio to zstd-6
- LZ4   — Apple's SIMD-optimized LZ4 implementation
- ZLIB  — Hardware-accelerated zlib/deflate
- LZMA  — Maximum compression ratio

LZFSE is the default and recommended algorithm on Apple Silicon — it's the same
compression Apple uses internally for APFS, iOS assets, and Xcode archives.

Copyright 2026 ResilientMind AI | ResilientMindai.com | Joseph C McGinty Jr
Proprietary License — All Rights Reserved
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import platform
import struct
import time
from pathlib import Path
from typing import Callable, Optional, Union

logger = logging.getLogger("hammerio.encoders.apple")

# ─── libcompression bindings ──────────────────────────────────────────────────

# Apple compression algorithm constants
COMPRESSION_LZFSE = 0x801
COMPRESSION_LZ4 = 0x100
COMPRESSION_LZ4_RAW = 0x101
COMPRESSION_ZLIB = 0x205
COMPRESSION_LZMA = 0x306

ALGO_MAP = {
    "lzfse": COMPRESSION_LZFSE,
    "lz4": COMPRESSION_LZ4,
    "zlib": COMPRESSION_ZLIB,
    "lzma": COMPRESSION_LZMA,
}

ALGO_NAMES = {v: k for k, v in ALGO_MAP.items()}

# Container format: HMAC (HammerIO Apple Compression)
MAGIC = b"HMAC"
VERSION = 1
HEADER_FORMAT = "<4sBI"  # magic(4) + version(1) + algo_id(4) = 9 bytes
CHUNK_HEADER = "<II"  # compressed_size(4) + original_size(4) = 8 bytes

_libcompression = None


def _load_libcompression() -> ctypes.CDLL:
    """Load Apple's libcompression.dylib."""
    global _libcompression
    if _libcompression is not None:
        return _libcompression

    lib_path = ctypes.util.find_library("compression")
    if lib_path is None:
        lib_path = "/usr/lib/libcompression.dylib"

    if not Path(lib_path).exists() and not lib_path.startswith("/usr/lib"):
        raise OSError("libcompression.dylib not found — this backend requires macOS")

    _libcompression = ctypes.CDLL(lib_path)

    # size_t compression_encode_buffer(
    #     uint8_t *dst, size_t dst_size,
    #     const uint8_t *src, size_t src_size,
    #     void *scratch, compression_algorithm algo)
    _libcompression.compression_encode_buffer.restype = ctypes.c_size_t
    _libcompression.compression_encode_buffer.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_int,
    ]

    # size_t compression_decode_buffer(
    #     uint8_t *dst, size_t dst_size,
    #     const uint8_t *src, size_t src_size,
    #     void *scratch, compression_algorithm algo)
    _libcompression.compression_decode_buffer.restype = ctypes.c_size_t
    _libcompression.compression_decode_buffer.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_int,
    ]

    logger.info("Apple libcompression loaded: %s", lib_path)
    return _libcompression


def is_available() -> bool:
    """Check if Apple compression is available (macOS only)."""
    if platform.system() != "Darwin":
        return False
    try:
        _load_libcompression()
        return True
    except OSError:
        return False


def compress_buffer(data: bytes, algorithm: int = COMPRESSION_LZFSE) -> bytes:
    """Compress a buffer using Apple's libcompression.

    Args:
        data: Raw bytes to compress.
        algorithm: One of COMPRESSION_LZFSE, COMPRESSION_LZ4, etc.

    Returns:
        Compressed bytes.
    """
    lib = _load_libcompression()
    src_size = len(data)
    # Allocate worst-case output buffer
    dst_size = src_size + (src_size // 10) + 1024
    dst = ctypes.create_string_buffer(dst_size)
    src = ctypes.create_string_buffer(data)

    result_size = lib.compression_encode_buffer(
        dst, dst_size, src, src_size, None, algorithm
    )

    if result_size == 0:
        raise RuntimeError(
            f"Apple compression failed (algorithm={ALGO_NAMES.get(algorithm, algorithm)})"
        )

    return dst.raw[:result_size]


def decompress_buffer(
    data: bytes, original_size: int, algorithm: int = COMPRESSION_LZFSE
) -> bytes:
    """Decompress a buffer using Apple's libcompression.

    Args:
        data: Compressed bytes.
        original_size: Expected decompressed size.
        algorithm: Algorithm used for compression.

    Returns:
        Decompressed bytes.
    """
    lib = _load_libcompression()
    dst = ctypes.create_string_buffer(original_size)
    src = ctypes.create_string_buffer(data)

    result_size = lib.compression_decode_buffer(
        dst, original_size, src, len(data), None, algorithm
    )

    if result_size == 0:
        raise RuntimeError(
            f"Apple decompression failed (algorithm={ALGO_NAMES.get(algorithm, algorithm)})"
        )

    return dst.raw[:result_size]


# ─── AppleEncoder — file-level compress/decompress ───────────────────────────


class AppleEncoder:
    """Apple Silicon compression encoder using libcompression.

    Streams files in chunks through Apple's hardware-optimized compression,
    writing a HMAC container format that preserves chunk boundaries for
    parallel decompression.

    Args:
        hardware: HardwareProfile (optional, for compatibility with encoder interface).
    """

    DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks

    def __init__(self, hardware: object = None) -> None:
        self._hw = hardware
        if not is_available():
            raise OSError("Apple compression not available (requires macOS)")
        _load_libcompression()

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        algorithm: str = "lzfse",
        quality: str = "balanced",
        progress_callback: Optional[Callable[[str, float], None]] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Compress a file using Apple's Accelerate framework.

        Args:
            input_path: File to compress.
            output_path: Output path for compressed file.
            algorithm: One of 'lzfse', 'lz4', 'zlib', 'lzma'.
            quality: Quality preset (maps to algorithm selection).
            progress_callback: Optional callback(job_id, pct).
            job_id: Job identifier for callbacks.

        Returns:
            Output path as string.
        """
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        if not input_path.is_file():
            raise FileNotFoundError(f"Input not found: {input_path}")

        algo_name = algorithm.lower().replace("apple_", "")
        if algo_name not in ALGO_MAP:
            raise ValueError(
                f"Unsupported Apple algorithm '{algo_name}'. "
                f"Choose from: {', '.join(ALGO_MAP.keys())}"
            )
        algo_id = ALGO_MAP[algo_name]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_size = input_path.stat().st_size

        logger.info(
            "Apple compress: %s (%s, %.1f MB)",
            input_path.name, algo_name, file_size / (1024 * 1024),
        )

        bytes_read = 0
        with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
            # Write container header
            fout.write(struct.pack(HEADER_FORMAT, MAGIC, VERSION, algo_id))

            while True:
                chunk = fin.read(self.DEFAULT_CHUNK_SIZE)
                if not chunk:
                    break

                compressed = compress_buffer(chunk, algo_id)
                # Write chunk header: compressed_size, original_size
                fout.write(struct.pack(CHUNK_HEADER, len(compressed), len(chunk)))
                fout.write(compressed)

                bytes_read += len(chunk)
                if progress_callback and file_size > 0:
                    pct = min(bytes_read / file_size * 100, 100.0)
                    progress_callback(job_id or "", pct)

        out_size = output_path.stat().st_size
        ratio = file_size / out_size if out_size > 0 else 0
        logger.info(
            "Apple compress done: %s → %s (%.2fx, %s)",
            input_path.name, output_path.name, ratio, algo_name,
        )
        return str(output_path)

    def decompress(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Decompress a HMAC container file.

        Args:
            input_path: Path to .hmac compressed file.
            output_path: Output path (auto-generated if None).

        Returns:
            Output path as string.
        """
        input_path = Path(input_path).resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input not found: {input_path}")

        if output_path is None:
            # Strip .lzfse / .hmac extension
            name = input_path.name
            for ext in (".lzfse", ".hmac", ".apple"):
                if name.endswith(ext):
                    name = name[: -len(ext)]
                    break
            output_path = input_path.parent / name
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        file_size = input_path.stat().st_size
        bytes_read = 0

        with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
            # Read header
            header_size = struct.calcsize(HEADER_FORMAT)
            header_data = fin.read(header_size)
            if len(header_data) < header_size:
                raise ValueError("Truncated HMAC header")

            magic, version, algo_id = struct.unpack(HEADER_FORMAT, header_data)
            if magic != MAGIC:
                raise ValueError(f"Not an Apple-compressed file (magic={magic!r})")

            bytes_read += header_size

            # Read and decompress chunks
            chunk_header_size = struct.calcsize(CHUNK_HEADER)
            while True:
                chunk_hdr = fin.read(chunk_header_size)
                if len(chunk_hdr) < chunk_header_size:
                    break

                comp_size, orig_size = struct.unpack(CHUNK_HEADER, chunk_hdr)
                compressed = fin.read(comp_size)
                if len(compressed) < comp_size:
                    raise ValueError("Truncated chunk data")

                decompressed = decompress_buffer(compressed, orig_size, algo_id)
                fout.write(decompressed)

                bytes_read += chunk_header_size + comp_size
                if progress_callback and file_size > 0:
                    pct = min(bytes_read / file_size * 100, 100.0)
                    progress_callback(job_id or "", pct)

        logger.info("Apple decompress done: %s", output_path.name)
        return str(output_path)


# ─── Benchmark helper ────────────────────────────────────────────────────────


def benchmark_apple(data: bytes, algorithms: Optional[list[str]] = None) -> dict:
    """Benchmark Apple compression algorithms on the given data.

    Args:
        data: Raw bytes to benchmark.
        algorithms: List of algorithm names. Default: all.

    Returns:
        Dict of results per algorithm.
    """
    if algorithms is None:
        algorithms = ["lzfse", "lz4", "zlib", "lzma"]

    results = {}
    input_size = len(data)

    for algo_name in algorithms:
        algo_id = ALGO_MAP.get(algo_name)
        if algo_id is None:
            continue

        # Compress
        try:
            start = time.perf_counter()
            compressed = compress_buffer(data, algo_id)
            comp_time = time.perf_counter() - start
            comp_size = len(compressed)
            comp_throughput = (input_size / (1024 * 1024)) / comp_time if comp_time > 0 else 0

            # Decompress
            start = time.perf_counter()
            decompressed = decompress_buffer(compressed, input_size, algo_id)
            decomp_time = time.perf_counter() - start
            decomp_throughput = (input_size / (1024 * 1024)) / decomp_time if decomp_time > 0 else 0

            match = decompressed == data

            results[algo_name] = {
                "compress_time_ms": round(comp_time * 1000, 2),
                "compress_throughput_mbps": round(comp_throughput, 1),
                "decompress_time_ms": round(decomp_time * 1000, 2),
                "decompress_throughput_mbps": round(decomp_throughput, 1),
                "ratio": round(input_size / comp_size, 2) if comp_size > 0 else 0,
                "compressed_size": comp_size,
                "match": match,
            }
        except Exception as e:
            results[algo_name] = {"error": str(e)}

    return results
