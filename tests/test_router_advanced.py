"""Advanced router tests: job execution, fallback, output path generation."""

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
)
from hammerio.core.profiler import CompressionMode, FileCategory, FileProfile
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


class TestJobResultProperties:
    """Test JobResult computed properties."""

    def test_savings_pct_normal(self) -> None:
        r = JobResult(
            input_path="/in", output_path="/out",
            input_size=1000, output_size=200,
            compression_ratio=5.0, elapsed_seconds=1.0,
            throughput_mbps=1.0, processor_used="cpu",
            mode=CompressionMode.CPU_ZSTD, algorithm="zstd",
            routing_reason="test", status=JobStatus.COMPLETED,
        )
        assert r.savings_pct == 80.0

    def test_savings_pct_zero_input(self) -> None:
        r = JobResult(
            input_path="/in", output_path="/out",
            input_size=0, output_size=0,
            compression_ratio=0, elapsed_seconds=0,
            throughput_mbps=0, processor_used="cpu",
            mode=CompressionMode.CPU_ZSTD, algorithm="zstd",
            routing_reason="test", status=JobStatus.COMPLETED,
        )
        assert r.savings_pct == 0.0


class TestJobRouterOutputPath:
    """Test auto-generated output paths."""

    def test_auto_output_zstd(self, sample_text_file: Path) -> None:
        router = JobRouter(hardware=_make_hw())
        job = router.route(sample_text_file)
        assert job.output_path is not None
        assert str(job.output_path).endswith(".zst")

    def test_explicit_output_path(self, sample_text_file: Path, tmp_dir: Path) -> None:
        router = JobRouter(hardware=_make_hw())
        out = tmp_dir / "custom_output.zst"
        job = router.route(sample_text_file, output_path=out)
        assert job.output_path == out.resolve()

    def test_route_directory(self, sample_directory: Path) -> None:
        router = JobRouter(hardware=_make_hw())
        job = router.route(sample_directory)
        assert job.recommendation is not None
        assert job.status == JobStatus.ROUTING

    def test_route_empty_directory_raises(self, tmp_dir: Path) -> None:
        router = JobRouter(hardware=_make_hw())
        empty = tmp_dir / "empty_dir"
        empty.mkdir()
        with pytest.raises(ValueError, match="Empty directory"):
            router.route(empty)


class TestJobRouterExecution:
    """Test actual job execution with GeneralEncoder (CPU path)."""

    def test_execute_text_file(self, sample_text_file: Path) -> None:
        router = JobRouter(hardware=_make_hw(), quality="fast")
        job = router.route(sample_text_file, mode="cpu")
        result = router.execute(job)
        assert result.status == JobStatus.COMPLETED
        assert result.output_size > 0
        assert result.compression_ratio > 1.0
        assert result.elapsed_seconds >= 0
        assert Path(result.output_path).exists()
        # Clean up
        Path(result.output_path).unlink(missing_ok=True)

    def test_execute_compressible_file(self, sample_compressible_file: Path) -> None:
        router = JobRouter(hardware=_make_hw(), quality="fast")
        job = router.route(sample_compressible_file, mode="cpu")
        result = router.execute(job)
        assert result.status == JobStatus.COMPLETED
        assert result.compression_ratio > 5.0  # Highly compressible
        Path(result.output_path).unlink(missing_ok=True)

    def test_get_job_returns_correct(self, sample_text_file: Path) -> None:
        router = JobRouter(hardware=_make_hw())
        job = router.route(sample_text_file)
        retrieved = router.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_job_nonexistent(self) -> None:
        router = JobRouter(hardware=_make_hw())
        assert router.get_job("nonexistent_id") is None


class TestJobRouterForcing:
    """Test forced mode/algorithm overrides."""

    def test_force_algorithm(self, sample_text_file: Path) -> None:
        router = JobRouter(hardware=_make_hw())
        job = router.route(sample_text_file, mode="cpu", algorithm="gzip")
        assert job.recommendation is not None
        assert job.recommendation.algorithm == "gzip"

    def test_force_gpu_without_cuda_warns(self, sample_text_file: Path) -> None:
        """GPU forced on non-GPU hardware should still create a valid job."""
        router = JobRouter(hardware=_make_hw())
        job = router.route(sample_text_file, mode="gpu")
        assert job.recommendation is not None

    def test_progress_callback(self, sample_text_file: Path) -> None:
        router = JobRouter(hardware=_make_hw(), quality="fast")
        progress: list[tuple[str, float]] = []
        router.set_progress_callback(lambda jid, pct: progress.append((jid, pct)))
        job = router.route(sample_text_file, mode="cpu")
        result = router.execute(job)
        assert result.status == JobStatus.COMPLETED
        assert len(progress) > 0
        Path(result.output_path).unlink(missing_ok=True)


class TestJobRouterExplain:
    """Test explain_route for various file types."""

    def test_explain_binary(self, sample_binary_file: Path) -> None:
        router = JobRouter(hardware=_make_hw())
        explanation = router.explain_route(sample_binary_file)
        assert "File:" in explanation
        assert "Route:" in explanation
        assert "Algorithm:" in explanation

    def test_explain_csv(self, sample_csv_file: Path) -> None:
        router = JobRouter(hardware=_make_hw())
        explanation = router.explain_route(sample_csv_file)
        assert "File:" in explanation

    def test_explain_nonexistent_raises(self) -> None:
        router = JobRouter(hardware=_make_hw())
        with pytest.raises(FileNotFoundError):
            router.explain_route("/nonexistent/path.txt")
