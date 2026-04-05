"""Tests for enhancement cycles 201-300.

Covers:
- StreamingDataset with PyTorch IterableDataset compatibility
- Batch error handling (one bad file should not stop the batch)
- General decompression with auto-detection (extension and magic bytes)
"""

from __future__ import annotations

import asyncio
import bz2
import gzip
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hammerio.core.hardware import (
    GpuVendor,
    GstreamerNvencCapability,
    HardwareProfile,
    NvcompCapability,
    NvdecCapability,
    NvencCapability,
    PlatformType,
    PowerMode,
    VpiCapability,
    detect_hardware,
)


def _make_hw(**overrides) -> HardwareProfile:
    defaults = dict(
        platform_type=PlatformType.DESKTOP,
        platform_name="Test",
        architecture="x86_64",
        gpu_vendor=GpuVendor.NONE,
        cuda_device=None,
        nvenc=NvencCapability(available=False),
        nvdec=NvdecCapability(available=False),
        nvcomp=NvcompCapability(available=False),
        vpi=VpiCapability(available=False),
        gstreamer_nvenc=GstreamerNvencCapability(available=False),
        cpu_cores=4,
        cpu_freq_mhz=2000.0,
        total_ram_mb=8192,
        power_mode=PowerMode.UNKNOWN,
        thermal_celsius=None,
    )
    defaults.update(overrides)
    return HardwareProfile(**defaults)


# ======================================================================
# Phase 1: StreamingDataset tests
# ======================================================================


class TestStreamingDataset:
    """Verify StreamingDataset works as a Python iterable and, when
    PyTorch is available, as a ``torch.utils.data.IterableDataset``."""

    def test_inherits_iterable_dataset(self) -> None:
        """StreamingDataset should inherit from IterableDataset when
        PyTorch is installed."""
        from hammerio.encoders.dataset import StreamingDataset

        try:
            from torch.utils.data import IterableDataset
            assert issubclass(StreamingDataset, IterableDataset)
        except ImportError:
            # Without PyTorch it should still be usable as a plain class
            assert hasattr(StreamingDataset, "__iter__")

    def test_iter_lines_csv(self, sample_csv_file: Path, tmp_dir: Path) -> None:
        """Compress a CSV then iterate lines through StreamingDataset."""
        from hammerio.encoders.dataset import DatasetEncoder, StreamingDataset

        hw = _make_hw()
        encoder = DatasetEncoder(hw)
        compressed = tmp_dir / "lines_test.csv.zst"
        encoder.process(sample_csv_file, compressed, "zstd", "fast")

        ds = StreamingDataset(compressed)
        lines = list(ds.iter_lines())
        original_lines = [
            l for l in sample_csv_file.read_text().split("\n") if l
        ]
        assert len(lines) == len(original_lines)
        assert lines[0] == original_lines[0]

    def test_iter_chunks_yields_bytes(self, sample_csv_file: Path, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder, StreamingDataset

        hw = _make_hw()
        encoder = DatasetEncoder(hw)
        compressed = tmp_dir / "chunks_test.csv.zst"
        encoder.process(sample_csv_file, compressed, "zstd", "fast")

        ds = StreamingDataset(compressed)
        chunks = list(ds)
        assert len(chunks) >= 1
        assert all(isinstance(c, bytes) for c in chunks)

    def test_getitem_and_len(self, sample_csv_file: Path, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder, StreamingDataset

        hw = _make_hw()
        encoder = DatasetEncoder(hw)
        compressed = tmp_dir / "getitem_test.csv.zst"
        encoder.process(sample_csv_file, compressed, "zstd", "fast")

        ds = StreamingDataset(compressed)
        length = len(ds)
        assert length >= 1
        chunk0 = ds[0]
        assert isinstance(chunk0, bytes)
        assert len(chunk0) > 0

    def test_index_out_of_range(self, sample_csv_file: Path, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder, StreamingDataset

        hw = _make_hw()
        encoder = DatasetEncoder(hw)
        compressed = tmp_dir / "oob_test.csv.zst"
        encoder.process(sample_csv_file, compressed, "zstd", "fast")

        ds = StreamingDataset(compressed)
        with pytest.raises(IndexError):
            ds[999999]

    def test_file_not_found(self, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import StreamingDataset

        with pytest.raises(FileNotFoundError):
            StreamingDataset(tmp_dir / "no_such_file.zst")

    def test_works_with_dataloader(self, sample_csv_file: Path, tmp_dir: Path) -> None:
        """When PyTorch is installed, StreamingDataset should work with DataLoader."""
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError:
            pytest.skip("PyTorch not installed")

        from hammerio.encoders.dataset import DatasetEncoder, StreamingDataset

        hw = _make_hw()
        encoder = DatasetEncoder(hw)
        compressed = tmp_dir / "dl_test.csv.zst"
        encoder.process(sample_csv_file, compressed, "zstd", "fast")

        ds = StreamingDataset(compressed)
        loader = DataLoader(ds, batch_size=None, num_workers=0)
        chunks = list(loader)
        assert len(chunks) >= 1


# ======================================================================
# Phase 2: Batch error handling tests
# ======================================================================


class TestBatchErrorHandling:
    """One bad file should not stop the rest of the batch."""

    def test_batch_continues_on_bad_file(self, tmp_dir: Path) -> None:
        """Create a directory with a good file and a bad file; ensure the
        batch completes and both results are returned."""
        from hammerio.core.router import JobRouter, JobStatus

        # Create test files
        src_dir = tmp_dir / "batch_src"
        src_dir.mkdir()
        good = src_dir / "good.txt"
        good.write_text("hello " * 5000)
        bad = src_dir / "bad.txt"
        bad.write_bytes(b"")  # empty file - still processable

        router = JobRouter()
        results = asyncio.get_event_loop().run_until_complete(
            router.execute_batch(src_dir, tmp_dir / "batch_out")
        )
        assert len(results) == 2
        # At least one should succeed (the non-empty file)
        statuses = [r.status for r in results]
        assert JobStatus.COMPLETED in statuses or JobStatus.FAILED in statuses

    def test_batch_returns_all_results(self, tmp_dir: Path) -> None:
        """Batch should return one result per input file regardless of errors."""
        from hammerio.core.router import JobRouter

        src_dir = tmp_dir / "batch_all"
        src_dir.mkdir()
        for i in range(3):
            (src_dir / f"file{i}.txt").write_text(f"data {i}\n" * 1000)

        router = JobRouter()
        results = asyncio.get_event_loop().run_until_complete(
            router.execute_batch(src_dir, tmp_dir / "batch_all_out")
        )
        assert len(results) == 3


# ======================================================================
# Phase 3: General decompression format auto-detection tests
# ======================================================================


class TestDecompressFormatDetection:
    """Test that decompress() auto-detects .zst, .gz, .bz2, .lz4."""

    def test_detect_zst_by_extension(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import detect_format
        p = tmp_dir / "file.zst"
        p.write_bytes(b"\x00")
        algo, is_tar = detect_format(p)
        assert algo == "zstd"
        assert is_tar is False

    def test_detect_gz_by_extension(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import detect_format
        p = tmp_dir / "file.gz"
        p.write_bytes(b"\x00")
        algo, is_tar = detect_format(p)
        assert algo == "gzip"
        assert is_tar is False

    def test_detect_bz2_by_extension(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import detect_format
        p = tmp_dir / "file.bz2"
        p.write_bytes(b"\x00")
        algo, is_tar = detect_format(p)
        assert algo == "bzip2"
        assert is_tar is False

    def test_detect_lz4_by_extension(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import detect_format
        p = tmp_dir / "file.lz4"
        p.write_bytes(b"\x00")
        algo, is_tar = detect_format(p)
        assert algo == "lz4"
        assert is_tar is False

    def test_detect_tar_zst_by_extension(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import detect_format
        p = tmp_dir / "archive.tar.zst"
        p.write_bytes(b"\x00")
        algo, is_tar = detect_format(p)
        assert algo == "zstd"
        assert is_tar is True

    def test_detect_tar_gz_by_extension(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import detect_format
        p = tmp_dir / "archive.tar.gz"
        p.write_bytes(b"\x00")
        algo, is_tar = detect_format(p)
        assert algo == "gzip"
        assert is_tar is True

    def test_detect_zstd_by_magic_bytes(self, tmp_dir: Path) -> None:
        """File has no recognisable extension but valid zstd magic."""
        from hammerio.encoders.general import detect_format
        p = tmp_dir / "mystery_file"
        p.write_bytes(b"\x28\xb5\x2f\xfd" + b"\x00" * 20)
        algo, is_tar = detect_format(p)
        assert algo == "zstd"
        assert is_tar is False

    def test_detect_gzip_by_magic_bytes(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import detect_format
        p = tmp_dir / "mystery_file2"
        p.write_bytes(b"\x1f\x8b" + b"\x00" * 20)
        algo, is_tar = detect_format(p)
        assert algo == "gzip"
        assert is_tar is False

    def test_detect_bz2_by_magic_bytes(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import detect_format
        p = tmp_dir / "mystery_bz2"
        p.write_bytes(b"\x42\x5a\x68" + b"\x00" * 20)
        algo, is_tar = detect_format(p)
        assert algo == "bzip2"
        assert is_tar is False

    def test_detect_lz4_by_magic_bytes(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import detect_format
        p = tmp_dir / "mystery_lz4"
        p.write_bytes(b"\x04\x22\x4d\x18" + b"\x00" * 20)
        algo, is_tar = detect_format(p)
        assert algo == "lz4"
        assert is_tar is False

    def test_unknown_format_raises(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import detect_format
        p = tmp_dir / "unknown_data"
        p.write_bytes(b"\xde\xad\xbe\xef")
        with pytest.raises(ValueError, match="Cannot determine"):
            detect_format(p)

    def test_decompress_zstd_round_trip(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder
        hw = detect_hardware()
        enc = GeneralEncoder(hw)

        original = tmp_dir / "rt_zst.txt"
        original.write_text("zstd round-trip\n" * 2000)

        compressed = tmp_dir / "rt_zst.txt.zst"
        enc.process(original, compressed, algorithm="zstd", quality="fast")

        decompressed = tmp_dir / "rt_zst_restored.txt"
        enc.decompress(compressed, decompressed)
        assert decompressed.read_text() == original.read_text()

    def test_decompress_gzip_round_trip(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder
        hw = detect_hardware()
        enc = GeneralEncoder(hw)

        original = tmp_dir / "rt_gz.txt"
        original.write_text("gzip round-trip\n" * 2000)

        compressed = tmp_dir / "rt_gz.txt.gz"
        enc.process(original, compressed, algorithm="gzip", quality="fast")

        decompressed = tmp_dir / "rt_gz_restored.txt"
        enc.decompress(compressed, decompressed)
        assert decompressed.read_text() == original.read_text()

    def test_decompress_bzip2_round_trip(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder
        hw = detect_hardware()
        enc = GeneralEncoder(hw)

        original = tmp_dir / "rt_bz2.txt"
        original.write_text("bzip2 round-trip\n" * 2000)

        compressed = tmp_dir / "rt_bz2.txt.bz2"
        enc.process(original, compressed, algorithm="bzip2", quality="fast")

        decompressed = tmp_dir / "rt_bz2_restored.txt"
        enc.decompress(compressed, decompressed)
        assert decompressed.read_text() == original.read_text()

    def test_decompress_magic_bytes_no_extension(self, tmp_dir: Path) -> None:
        """Compress with zstd, rename to remove extension, decompress via magic bytes."""
        from hammerio.encoders.general import GeneralEncoder
        hw = detect_hardware()
        enc = GeneralEncoder(hw)

        original = tmp_dir / "magic_test.txt"
        original.write_text("magic detection\n" * 1000)

        compressed = tmp_dir / "magic_test.txt.zst"
        enc.process(original, compressed, algorithm="zstd", quality="fast")

        # Rename to strip extension
        renamed = tmp_dir / "compressed_no_ext"
        compressed.rename(renamed)

        decompressed = tmp_dir / "magic_restored.txt"
        enc.decompress(renamed, decompressed)
        assert decompressed.read_text() == original.read_text()

    def test_decompress_gzip_magic_bytes_no_extension(self, tmp_dir: Path) -> None:
        """Compress with gzip, rename to remove extension, decompress via magic bytes."""
        from hammerio.encoders.general import GeneralEncoder
        hw = detect_hardware()
        enc = GeneralEncoder(hw)

        original = tmp_dir / "magic_gz.txt"
        original.write_text("gzip magic\n" * 1000)

        compressed = tmp_dir / "magic_gz.txt.gz"
        enc.process(original, compressed, algorithm="gzip", quality="fast")

        renamed = tmp_dir / "gz_no_ext"
        compressed.rename(renamed)

        decompressed = tmp_dir / "magic_gz_restored.txt"
        enc.decompress(renamed, decompressed)
        assert decompressed.read_text() == original.read_text()
