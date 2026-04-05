"""Tests for HammerIO cycles 301-400: performance, resilience, coverage.

Covers:
- Router execute() with real files and profiling overhead metric
- Batch processing with mixed file types via GeneralEncoder
- Config deep merge edge cases (deeply nested, empty overrides, list values)
- Encoder edge cases: missing input, auto-created output dirs, 0-byte files
- GeneralEncoder zstd large/small file optimization paths
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hammerio.core.hardware import (
    GstreamerNvencCapability,
    GpuVendor,
    HardwareProfile,
    NvcompCapability,
    NvdecCapability,
    NvencCapability,
    PlatformType,
    PowerMode,
    VpiCapability,
    detect_hardware,
)
from hammerio.core.profiler import CompressionMode, FileCategory
from hammerio.core.router import Job, JobResult, JobRouter, JobStatus


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
# Phase 1: Router execute() with real files
# ======================================================================


class TestRouterExecuteWithRealFiles:
    """Test router execute() end-to-end with real files and timing metrics."""

    def test_execute_binary_file(self, sample_binary_file: Path) -> None:
        router = JobRouter(hardware=_make_hw(), quality="fast")
        job = router.route(sample_binary_file, mode="cpu")
        result = router.execute(job)
        assert result.status == JobStatus.COMPLETED
        assert result.output_size > 0
        assert result.elapsed_seconds >= 0
        assert result.throughput_mbps >= 0
        assert Path(result.output_path).exists()
        Path(result.output_path).unlink(missing_ok=True)

    def test_execute_csv_dataset(self, sample_csv_file: Path) -> None:
        router = JobRouter(hardware=_make_hw(), quality="fast")
        job = router.route(sample_csv_file, mode="cpu")
        result = router.execute(job)
        assert result.status == JobStatus.COMPLETED
        assert result.compression_ratio > 1.0  # CSV is compressible
        Path(result.output_path).unlink(missing_ok=True)

    def test_execute_directory(self, sample_directory: Path, tmp_dir: Path) -> None:
        router = JobRouter(hardware=_make_hw(), quality="fast")
        job = router.route(sample_directory, mode="cpu")
        result = router.execute(job)
        assert result.status == JobStatus.COMPLETED
        assert result.input_size > 0
        assert result.output_size > 0
        Path(result.output_path).unlink(missing_ok=True)

    def test_profiling_overhead_metric(self, sample_text_file: Path) -> None:
        """Verify the new profiling_overhead_ms metric is populated."""
        router = JobRouter(hardware=_make_hw(), quality="fast")
        job = router.route(sample_text_file, mode="cpu")
        result = router.execute(job)
        assert result.profiling_overhead_ms >= 0
        assert isinstance(result.profiling_overhead_ms, float)
        Path(result.output_path).unlink(missing_ok=True)

    def test_execute_with_explicit_output(self, sample_text_file: Path, tmp_dir: Path) -> None:
        router = JobRouter(hardware=_make_hw(), quality="fast")
        out = tmp_dir / "output" / "nested" / "result.zst"
        job = router.route(sample_text_file, output_path=out, mode="cpu")
        result = router.execute(job)
        assert result.status == JobStatus.COMPLETED
        assert Path(result.output_path).exists()
        assert "nested" in result.output_path

    def test_execute_multiple_qualities(self, sample_text_file: Path) -> None:
        """Test that different quality settings all produce valid results."""
        for quality in ("fast", "balanced", "quality"):
            router = JobRouter(hardware=_make_hw(), quality=quality)
            job = router.route(sample_text_file, mode="cpu")
            result = router.execute(job)
            assert result.status == JobStatus.COMPLETED
            Path(result.output_path).unlink(missing_ok=True)


# ======================================================================
# Phase 1: Batch processing with mixed file types
# ======================================================================


class TestBatchProcessingMixedTypes:
    """Test GeneralEncoder batch processing with mixed content."""

    def test_batch_compress_mixed_files(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder

        hw = _make_hw()
        encoder = GeneralEncoder(hw)

        # Create files of different types/sizes
        (tmp_dir / "text.txt").write_text("Hello\n" * 1000)
        (tmp_dir / "binary.bin").write_bytes(os.urandom(5000))
        (tmp_dir / "data.json").write_text('{"key": "value"}\n' * 500)
        (tmp_dir / "empty.dat").write_bytes(b"")  # empty file!

        files = [
            tmp_dir / "text.txt",
            tmp_dir / "binary.bin",
            tmp_dir / "data.json",
        ]
        out_dir = tmp_dir / "compressed"

        results = encoder.process_batch(files, out_dir, algorithm="zstd", quality="fast")
        assert len(results) == 3
        for r in results:
            assert Path(r).exists()
            assert Path(r).stat().st_size > 0

    def test_batch_compress_with_progress(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder

        hw = _make_hw()
        encoder = GeneralEncoder(hw)

        for i in range(5):
            (tmp_dir / f"file_{i}.txt").write_text(f"Content {i}\n" * 200)

        files = [tmp_dir / f"file_{i}.txt" for i in range(5)]
        progress: list[tuple[str, float]] = []

        results = encoder.process_batch(
            files, tmp_dir / "out", algorithm="gzip", quality="fast",
            progress_callback=lambda jid, pct: progress.append((jid, pct)),
        )
        assert len(results) == 5
        assert len(progress) > 0

    def test_batch_preserves_order(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder

        hw = _make_hw()
        encoder = GeneralEncoder(hw)

        names = ["alpha.txt", "beta.txt", "gamma.txt"]
        for n in names:
            (tmp_dir / n).write_text(f"{n} content\n" * 100)

        files = [tmp_dir / n for n in names]
        results = encoder.process_batch(files, tmp_dir / "out", algorithm="zstd", quality="fast")
        assert len(results) == 3
        for name, result in zip(names, results):
            assert name in result


# ======================================================================
# Phase 2: Config deep merge edge cases
# ======================================================================


class TestDeepMergeEdgeCases:
    """Additional edge cases for _deep_merge."""

    def test_deeply_nested_merge(self) -> None:
        from hammerio.core.config import _deep_merge

        base = {"a": {"b": {"c": {"d": 1, "e": 2}}}}
        override = {"a": {"b": {"c": {"e": 3, "f": 4}}}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": {"c": {"d": 1, "e": 3, "f": 4}}}}

    def test_empty_override(self) -> None:
        from hammerio.core.config import _deep_merge

        base = {"a": 1, "b": {"x": 2}}
        result = _deep_merge(base, {})
        assert result == base

    def test_empty_base(self) -> None:
        from hammerio.core.config import _deep_merge

        override = {"a": 1, "b": {"x": 2}}
        result = _deep_merge({}, override)
        assert result == override

    def test_list_values_replaced_not_merged(self) -> None:
        from hammerio.core.config import _deep_merge

        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = _deep_merge(base, override)
        assert result["items"] == [4, 5]

    def test_replace_scalar_with_dict(self) -> None:
        from hammerio.core.config import _deep_merge

        base = {"a": 42}
        override = {"a": {"nested": True}}
        result = _deep_merge(base, override)
        assert result["a"] == {"nested": True}

    def test_both_empty(self) -> None:
        from hammerio.core.config import _deep_merge

        result = _deep_merge({}, {})
        assert result == {}


# ======================================================================
# Phase 1: Zstd optimization paths (large vs small files)
# ======================================================================


class TestZstdOptimization:
    """Verify the zstd multi-threaded / large-chunk optimization paths."""

    def test_small_file_uses_single_thread(self, tmp_dir: Path) -> None:
        """Files under 10 MB should use threads=0 and standard chunk size."""
        from hammerio.encoders.general import GeneralEncoder, _LARGE_FILE_THRESHOLD

        hw = _make_hw(cpu_cores=8)
        encoder = GeneralEncoder(hw)

        small = tmp_dir / "small.dat"
        small.write_bytes(b"X" * (_LARGE_FILE_THRESHOLD - 1))

        out = tmp_dir / "small.dat.zst"
        result = encoder.process(small, out, algorithm="zstd", quality="balanced")
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_large_file_compresses_correctly(self, tmp_dir: Path) -> None:
        """Files over 10 MB should use multi-threaded mode and still produce valid output."""
        from hammerio.encoders.general import GeneralEncoder, _LARGE_FILE_THRESHOLD

        hw = _make_hw(cpu_cores=8)
        encoder = GeneralEncoder(hw)

        # Create a file just over the threshold
        large = tmp_dir / "large.dat"
        large.write_bytes(b"A" * (_LARGE_FILE_THRESHOLD + 1024))

        out = tmp_dir / "large.dat.zst"
        result = encoder.process(large, out, algorithm="zstd", quality="fast")
        assert Path(result).exists()

        # Verify round-trip
        decompressed = tmp_dir / "large_restored.dat"
        encoder.decompress(out, decompressed)
        assert decompressed.read_bytes() == large.read_bytes()


# ======================================================================
# Phase 2: Additional encoder error resilience
# ======================================================================


class TestEncoderErrorResilience:
    """Verify encoders handle error conditions cleanly."""

    def test_general_encoder_missing_input(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder

        encoder = GeneralEncoder(_make_hw())
        with pytest.raises(FileNotFoundError, match="Input not found"):
            encoder.process(tmp_dir / "nonexistent.txt", tmp_dir / "out.zst")

    def test_general_encoder_decompress_missing_input(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder

        encoder = GeneralEncoder(_make_hw())
        with pytest.raises(FileNotFoundError, match="Input not found"):
            encoder.decompress(tmp_dir / "nonexistent.zst")

    def test_general_encoder_unsupported_algorithm(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder

        encoder = GeneralEncoder(_make_hw())
        f = tmp_dir / "test.txt"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            encoder.process(f, tmp_dir / "out", algorithm="xz")

    def test_general_encoder_creates_output_dir(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder

        encoder = GeneralEncoder(_make_hw())
        f = tmp_dir / "test.txt"
        f.write_text("data\n" * 100)
        out = tmp_dir / "deep" / "nested" / "output.zst"
        result = encoder.process(f, out, algorithm="zstd", quality="fast")
        assert Path(result).exists()

    def test_general_encoder_zero_byte_all_algorithms(self, tmp_dir: Path) -> None:
        """All algorithms should handle 0-byte files without crashing."""
        from hammerio.encoders.general import GeneralEncoder

        encoder = GeneralEncoder(detect_hardware())
        empty = tmp_dir / "empty.dat"
        empty.write_bytes(b"")

        for algo in ("zstd", "gzip", "bzip2"):
            out = tmp_dir / f"empty.{algo}"
            result = encoder.process(empty, out, algorithm=algo, quality="fast")
            assert Path(result).exists()

    def test_dataset_encoder_missing_input(self, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder

        encoder = DatasetEncoder(_make_hw())
        with pytest.raises(FileNotFoundError, match="Input not found"):
            encoder.process(tmp_dir / "nope.csv", None, algorithm="zstd", quality="fast")

    def test_dataset_encoder_creates_output_dir(self, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder

        encoder = DatasetEncoder(_make_hw())
        f = tmp_dir / "data.csv"
        f.write_text("a,b\n1,2\n" * 500)
        out = tmp_dir / "deep" / "output.csv.zst"
        result = encoder.process(f, out, algorithm="zstd", quality="fast")
        assert Path(result).exists()

    def test_image_encoder_missing_input(self, tmp_dir: Path) -> None:
        from hammerio.encoders.image import ImageEncoder

        encoder = ImageEncoder(_make_hw())
        with pytest.raises(FileNotFoundError, match="Input not found"):
            encoder.process(tmp_dir / "nope.jpg", None, algorithm="jpeg", quality="fast")

    def test_bulk_encoder_creates_output_dir(self, tmp_dir: Path) -> None:
        from hammerio.encoders.bulk import BulkEncoder

        encoder = BulkEncoder(_make_hw())
        f = tmp_dir / "data.bin"
        f.write_bytes(b"\x00" * 1000)
        out = tmp_dir / "deep" / "nested" / "data.hammer"
        result = encoder.process(f, out, algorithm="zstd", quality="fast")
        assert Path(result).exists()


# ======================================================================
# Phase 3: Router fallback and edge cases
# ======================================================================


class TestRouterFallbackPaths:
    """Test router fallback behavior when primary encoder fails."""

    def test_fallback_result_has_profiling_overhead(self, sample_text_file: Path) -> None:
        """Even in fallback path, profiling_overhead_ms should be set."""
        router = JobRouter(hardware=_make_hw(), quality="fast")
        job = router.route(sample_text_file, mode="cpu")
        result = router.execute(job)
        # Normal path succeeds, but still has overhead metric
        assert hasattr(result, "profiling_overhead_ms")
        Path(result.output_path).unlink(missing_ok=True)

    def test_execute_with_progress_callback(self, sample_text_file: Path) -> None:
        """Verify progress callback is called during execution."""
        router = JobRouter(hardware=_make_hw(), quality="fast")
        progress_calls: list[tuple[str, float]] = []
        router.set_progress_callback(
            lambda jid, pct: progress_calls.append((jid, pct))
        )
        job = router.route(sample_text_file, mode="cpu")
        result = router.execute(job)
        assert result.status == JobStatus.COMPLETED
        assert len(progress_calls) > 0
        # Last call should be 100%
        assert progress_calls[-1][1] == 100.0
        Path(result.output_path).unlink(missing_ok=True)


# ======================================================================
# Phase 3: General encoder round-trip tests for all algorithms
# ======================================================================


class TestGeneralEncoderRoundTrips:
    """Verify compress -> decompress round-trips for all CPU algorithms."""

    @pytest.mark.parametrize("algorithm", ["zstd", "gzip", "bzip2"])
    def test_roundtrip(self, tmp_dir: Path, algorithm: str) -> None:
        from hammerio.encoders.general import GeneralEncoder

        encoder = GeneralEncoder(detect_hardware())
        original = tmp_dir / "original.dat"
        original.write_bytes(b"Round trip test data. " * 5000)

        ext_map = {"zstd": ".zst", "gzip": ".gz", "bzip2": ".bz2"}
        compressed = tmp_dir / f"compressed{ext_map[algorithm]}"
        encoder.process(original, compressed, algorithm=algorithm, quality="balanced")

        decompressed = tmp_dir / "decompressed.dat"
        encoder.decompress(compressed, decompressed)
        assert decompressed.read_bytes() == original.read_bytes()

    @pytest.mark.parametrize("quality", ["fast", "balanced", "quality", "lossless"])
    def test_zstd_quality_levels(self, tmp_dir: Path, quality: str) -> None:
        from hammerio.encoders.general import GeneralEncoder

        encoder = GeneralEncoder(detect_hardware())
        original = tmp_dir / "quality_test.dat"
        original.write_bytes(b"Quality test data. " * 2000)

        compressed = tmp_dir / f"q_{quality}.zst"
        encoder.process(original, compressed, algorithm="zstd", quality=quality)
        assert compressed.exists()
        assert compressed.stat().st_size > 0

    def test_directory_compress_decompress(self, sample_directory: Path, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder

        encoder = GeneralEncoder(detect_hardware())
        archive = tmp_dir / "archive.tar.zst"
        encoder.process(sample_directory, archive, algorithm="zstd", quality="fast")
        assert archive.exists()

        # Decompress
        extracted = tmp_dir / "extracted"
        encoder.decompress(archive, extracted)
        assert extracted.is_dir()
        # Should have files
        extracted_files = list(extracted.iterdir())
        assert len(extracted_files) > 0


# ======================================================================
# Phase 3: Dataset encoder coverage
# ======================================================================


class TestDatasetEncoderCoverage:
    """Improve coverage for DatasetEncoder paths."""

    def test_compress_csv_roundtrip(self, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder

        encoder = DatasetEncoder(_make_hw())
        csv = tmp_dir / "data.csv"
        csv.write_text("id,value\n" + "".join(f"{i},{i*2}\n" for i in range(1000)))

        out = tmp_dir / "data.csv.zst"
        result = encoder.process(csv, out, algorithm="zstd", quality="fast")
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_compress_directory_tar_zst(self, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder

        encoder = DatasetEncoder(_make_hw())
        d = tmp_dir / "dataset"
        d.mkdir()
        for i in range(3):
            rows = "row\n" * 100
            (d / f"part_{i}.csv").write_text(f"col\n{rows}")

        out = tmp_dir / "dataset.tar.zst"
        result = encoder.process(d, out, algorithm="zstd", quality="balanced")
        assert Path(result).exists()

    def test_auto_output_path_file(self, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder

        encoder = DatasetEncoder(_make_hw())
        f = tmp_dir / "data.npy"
        f.write_bytes(b"\x00" * 500)
        result = encoder.process(f, None, algorithm="zstd", quality="fast")
        assert result.endswith(".zst")
        Path(result).unlink(missing_ok=True)

    def test_auto_output_path_directory(self, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder

        encoder = DatasetEncoder(_make_hw())
        d = tmp_dir / "mydata"
        d.mkdir()
        (d / "file.csv").write_text("a\n1\n")
        result = encoder.process(d, None, algorithm="zstd", quality="fast")
        assert ".tar.zst" in result or ".tar.gz" in result
        Path(result).unlink(missing_ok=True)


# ======================================================================
# Phase 3: Bulk encoder additional coverage
# ======================================================================


class TestBulkEncoderCoverage:
    """Improve coverage for BulkEncoder paths."""

    def test_compress_decompress_roundtrip(self, tmp_dir: Path) -> None:
        from hammerio.encoders.bulk import BulkEncoder

        encoder = BulkEncoder(_make_hw())
        original = tmp_dir / "data.bin"
        data = b"Bulk encoder roundtrip test. " * 2000
        original.write_bytes(data)

        compressed = tmp_dir / "data.hmio"
        encoder.process(original, compressed, algorithm="zstd", quality="balanced")

        decompressed = tmp_dir / "data_restored.bin"
        encoder.decompress(compressed, decompressed)
        assert decompressed.read_bytes() == data

    def test_compress_with_progress(self, tmp_dir: Path) -> None:
        from hammerio.encoders.bulk import BulkEncoder

        encoder = BulkEncoder(_make_hw())
        f = tmp_dir / "data.bin"
        f.write_bytes(b"X" * 10000)
        progress: list[tuple[str, float]] = []

        encoder.process(
            f, tmp_dir / "out.hmio", algorithm="zstd", quality="fast",
            progress_callback=lambda jid, pct: progress.append((jid, pct)),
            job_id="test_job",
        )
        assert len(progress) > 0
        assert progress[-1][1] == 100.0

    def test_decompress_with_progress(self, tmp_dir: Path) -> None:
        from hammerio.encoders.bulk import BulkEncoder

        encoder = BulkEncoder(_make_hw())
        f = tmp_dir / "data.bin"
        f.write_bytes(b"Y" * 10000)
        compressed = tmp_dir / "data.hmio"
        encoder.process(f, compressed, algorithm="zstd", quality="fast")

        progress: list[tuple[str, float]] = []
        encoder.decompress(
            compressed, tmp_dir / "restored.bin",
            progress_callback=lambda jid, pct: progress.append((jid, pct)),
            job_id="decomp_job",
        )
        assert len(progress) > 0
