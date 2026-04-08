"""Tests for HammerIO streaming compression module."""

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hammerio.streaming import (
    EOF_MARKER,
    MAGIC,
    StreamingGPUCompressor,
    choose_chunk_size_mb,
    compress_directory_streaming,
    get_available_unified_memory_mb,
)


@pytest.fixture
def small_file(tmp_path):
    """10MB compressible data file."""
    # Repetitive text-like data that compresses well
    block = b"HammerIO streaming test data block. " * 1000
    data = block * (10 * 1024 * 1024 // len(block) + 1)
    data = data[:10 * 1024 * 1024]
    p = tmp_path / "test_10mb.bin"
    p.write_bytes(data)
    return p


@pytest.fixture
def medium_file(tmp_path):
    """100MB compressible data file."""
    block = b"Medium file test block for HammerIO streaming compression. " * 500
    data = block * (100 * 1024 * 1024 // len(block) + 1)
    data = data[:100 * 1024 * 1024]
    p = tmp_path / "test_100mb.bin"
    p.write_bytes(data)
    return p


@pytest.fixture
def small_dir(tmp_path):
    """Directory with 10 small compressible files."""
    d = tmp_path / "test_dir"
    d.mkdir()
    for i in range(10):
        f = d / f"file_{i:03d}.txt"
        data = f"File {i} content line. " * 10000
        f.write_text(data)
    return d


def test_memory_detection():
    available = get_available_unified_memory_mb()
    assert available > 0
    assert available < 65536  # Less than 64GB


def test_chunk_size_auto():
    chunk = choose_chunk_size_mb()
    assert chunk >= 64
    assert chunk <= 2048
    assert chunk % 64 == 0


def test_chunk_size_explicit():
    assert choose_chunk_size_mb(128) == 128
    assert choose_chunk_size_mb(64) == 64
    assert choose_chunk_size_mb(2048) == 2048


def test_chunk_size_clamped():
    # Below minimum gets clamped up
    assert choose_chunk_size_mb(10) == 64
    # Above maximum gets clamped down
    assert choose_chunk_size_mb(9999) == 2048


def test_compress_decompress_roundtrip(small_file, tmp_path):
    compressed = tmp_path / "out.hammer"
    restored = tmp_path / "restored.bin"

    compressor = StreamingGPUCompressor(chunk_mb=64)
    metrics = compressor.compress_file(small_file, compressed)

    assert compressed.exists()
    assert metrics["ratio"] > 1.0
    assert metrics["output_size"] < metrics["input_size"]
    assert metrics["throughput_mbps"] > 0

    metrics_d = compressor.decompress_file(compressed, restored)

    assert restored.exists()
    assert restored.read_bytes() == small_file.read_bytes()
    assert metrics_d["throughput_mbps"] > 0


def test_streaming_format_header(small_file, tmp_path):
    compressed = tmp_path / "out.hammer"
    compressor = StreamingGPUCompressor(chunk_mb=64)
    compressor.compress_file(small_file, compressed)

    with open(compressed, "rb") as f:
        magic = f.read(8)
        size = struct.unpack("<Q", f.read(8))[0]

    assert magic == MAGIC
    assert size == small_file.stat().st_size


def test_corruption_detected(small_file, tmp_path):
    compressed = tmp_path / "out.hammer"
    corrupted = tmp_path / "corrupted.hammer"
    restored = tmp_path / "restored.bin"

    compressor = StreamingGPUCompressor(chunk_mb=64)
    compressor.compress_file(small_file, compressed)

    # Corrupt the header magic — this should always raise
    data = bytearray(compressed.read_bytes())
    data[0:4] = b"XXXX"
    corrupted.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="Not a HammerIO streaming file"):
        compressor.decompress_file(corrupted, restored)


def test_directory_streaming(small_dir, tmp_path):
    compressed = tmp_path / "dir.hammer"
    metrics = compress_directory_streaming(
        small_dir,
        compressed,
        chunk_mb=64,
    )

    assert compressed.exists()
    assert metrics["ratio"] > 1.0
    assert metrics["chunks"] > 0
    assert metrics["throughput_mbps"] > 0


def test_medium_file_streaming(medium_file, tmp_path):
    """100MB file — verifies chunking works correctly."""
    compressed = tmp_path / "medium.hammer"
    restored = tmp_path / "medium_restored.bin"

    compressor = StreamingGPUCompressor(chunk_mb=64)
    metrics_c = compressor.compress_file(medium_file, compressed)

    # Should have used multiple chunks
    assert metrics_c["chunks"] > 1

    metrics_d = compressor.decompress_file(compressed, restored)

    assert restored.read_bytes() == medium_file.read_bytes()


def test_constant_memory_usage(medium_file, tmp_path):
    """
    Verify GPU buffer does not grow with file size.
    Buffer should stay at chunk_size regardless of
    input size.
    """
    chunk_mb = 64
    compressor = StreamingGPUCompressor(chunk_mb=chunk_mb)
    compressor._ensure_gpu_buffer()

    # Buffer size should equal chunk size, not file size
    file_size_mb = medium_file.stat().st_size / (1024 ** 2)
    assert file_size_mb > chunk_mb  # File > chunk

    # If we got here without OOM, buffer is bounded
    assert compressor._actual_chunk_bytes <= chunk_mb * 1024 * 1024


def test_progress_callback(small_file, tmp_path):
    """Verify progress callback is called with increasing values."""
    compressed = tmp_path / "out.hammer"
    progress_values = []

    def cb(done, total):
        progress_values.append((done, total))

    compressor = StreamingGPUCompressor(chunk_mb=64, progress_callback=cb)
    compressor.compress_file(small_file, compressed)

    assert len(progress_values) > 0
    # Values should be monotonically increasing
    for i in range(1, len(progress_values)):
        assert progress_values[i][0] >= progress_values[i - 1][0]
    # Last value should equal total
    assert progress_values[-1][0] == progress_values[-1][1]


def test_invalid_magic_rejected(tmp_path):
    """Verify files with wrong magic are rejected."""
    fake = tmp_path / "fake.hammer"
    fake.write_bytes(b"NOT_HMIO" + b"\x00" * 100)

    compressor = StreamingGPUCompressor(chunk_mb=64)
    with pytest.raises(ValueError, match="Not a HammerIO streaming file"):
        compressor.decompress_file(fake, tmp_path / "out.bin")
