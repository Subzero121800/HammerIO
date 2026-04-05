"""Tests for the smart job routing engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from hammerio.core.profiler import (
    CompressionMode,
    FileCategory,
    FileProfile,
    categorize_file,
    estimate_entropy,
    profile_file,
    profile_directory,
    recommend_compression,
)


class TestProfiler:
    def test_categorize_video(self, tmp_dir: Path) -> None:
        p = tmp_dir / "test.mp4"
        p.write_bytes(b"\x00" * 100)
        assert categorize_file(p) == FileCategory.VIDEO

    def test_categorize_image(self, tmp_dir: Path) -> None:
        p = tmp_dir / "test.jpg"
        p.write_bytes(b"\x00" * 100)
        assert categorize_file(p) == FileCategory.IMAGE

    def test_categorize_audio(self, tmp_dir: Path) -> None:
        p = tmp_dir / "test.mp3"
        p.write_bytes(b"\x00" * 100)
        assert categorize_file(p) == FileCategory.AUDIO

    def test_categorize_archive(self, tmp_dir: Path) -> None:
        p = tmp_dir / "test.zip"
        p.write_bytes(b"\x00" * 100)
        assert categorize_file(p) == FileCategory.ARCHIVE

    def test_categorize_dataset(self, tmp_dir: Path) -> None:
        p = tmp_dir / "data.csv"
        p.write_text("a,b,c\n1,2,3\n")
        assert categorize_file(p) == FileCategory.DATASET

    def test_categorize_text(self, sample_text_file: Path) -> None:
        # .txt should detect as text
        cat = categorize_file(sample_text_file)
        assert cat in (FileCategory.TEXT, FileCategory.DOCUMENT)

    def test_entropy_low(self, sample_compressible_file: Path) -> None:
        entropy = estimate_entropy(sample_compressible_file)
        assert 0 <= entropy < 8
        # Repeated pattern should have very low entropy
        assert entropy < 1

    def test_entropy_high(self, sample_binary_file: Path) -> None:
        entropy = estimate_entropy(sample_binary_file)
        # Random data should have high entropy
        assert entropy > 6

    def test_profile_file(self, sample_text_file: Path) -> None:
        fp = profile_file(sample_text_file)
        assert isinstance(fp, FileProfile)
        assert fp.size_bytes > 0
        assert fp.extension == ".txt"

    def test_profile_directory(self, sample_directory: Path) -> None:
        batch = profile_directory(sample_directory)
        assert batch.file_count > 0
        assert batch.total_size_bytes > 0
        assert len(batch.category_counts) > 0

    def test_recommend_video_compresses_not_transcodes(self, tmp_dir: Path) -> None:
        fp = FileProfile(
            path=tmp_dir / "test.mp4",
            size_bytes=1024 * 1024 * 100,
            category=FileCategory.VIDEO,
            mime_type="video/mp4",
            extension=".mp4",
            is_already_compressed=True,
        )
        rec = recommend_compression(fp, gpu_available=True, nvenc_available=True)
        # Compress should use zstd, NOT transcode with NVENC
        assert rec.mode == CompressionMode.CPU_ZSTD
        assert rec.algorithm in ("zstd", "lz4")
        assert rec.gpu_preferred is False

    def test_recommend_video_cpu_no_nvenc(self, tmp_dir: Path) -> None:
        fp = FileProfile(
            path=tmp_dir / "test.mp4",
            size_bytes=1024 * 1024 * 100,
            category=FileCategory.VIDEO,
            mime_type="video/mp4",
            extension=".mp4",
            is_already_compressed=True,
        )
        rec = recommend_compression(fp, gpu_available=False, nvenc_available=False)
        # Compress uses zstd regardless of GPU availability
        assert rec.mode == CompressionMode.CPU_ZSTD
        assert rec.algorithm in ("zstd", "lz4")
        assert rec.gpu_preferred is False

    def test_recommend_archive_passthrough(self, tmp_dir: Path) -> None:
        fp = FileProfile(
            path=tmp_dir / "test.zip",
            size_bytes=1024 * 100,
            category=FileCategory.ARCHIVE,
            mime_type="application/zip",
            extension=".zip",
            is_already_compressed=True,
        )
        rec = recommend_compression(fp)
        assert rec.mode == CompressionMode.PASSTHROUGH

    def test_recommend_text_cpu(self, tmp_dir: Path) -> None:
        fp = FileProfile(
            path=tmp_dir / "test.txt",
            size_bytes=1024 * 100,
            category=FileCategory.TEXT,
            mime_type="text/plain",
            extension=".txt",
            estimated_entropy=3.5,
        )
        rec = recommend_compression(fp)
        assert rec.mode == CompressionMode.CPU_ZSTD

    def test_recommend_large_dataset_gpu(self, tmp_dir: Path) -> None:
        fp = FileProfile(
            path=tmp_dir / "data.csv",
            size_bytes=1024 * 1024 * 500,  # 500MB
            category=FileCategory.DATASET,
            mime_type="text/csv",
            extension=".csv",
        )
        rec = recommend_compression(fp, gpu_available=True, nvcomp_available=True)
        assert rec.mode == CompressionMode.GPU_NVCOMP
        assert rec.gpu_preferred is True


class TestRouter:
    def test_router_creates_job(self, sample_text_file: Path) -> None:
        from hammerio.core.router import JobRouter, JobStatus
        router = JobRouter()
        job = router.route(sample_text_file)
        assert job.status == JobStatus.ROUTING
        assert job.recommendation is not None
        assert job.job_id.startswith("job_")

    def test_router_explain(self, sample_text_file: Path) -> None:
        from hammerio.core.router import JobRouter
        router = JobRouter()
        explanation = router.explain_route(sample_text_file)
        assert "File:" in explanation or "Route:" in explanation

    def test_router_explain_directory(self, sample_directory: Path) -> None:
        from hammerio.core.router import JobRouter
        router = JobRouter()
        explanation = router.explain_route(sample_directory)
        assert "Directory:" in explanation

    def test_router_force_cpu(self, sample_text_file: Path) -> None:
        from hammerio.core.router import JobRouter
        router = JobRouter()
        job = router.route(sample_text_file, mode="cpu")
        assert job.recommendation is not None
        assert job.recommendation.mode == CompressionMode.CPU_ZSTD

    def test_router_list_jobs(self, sample_text_file: Path) -> None:
        from hammerio.core.router import JobRouter
        router = JobRouter()
        router.route(sample_text_file)
        router.route(sample_text_file)
        assert len(router.list_jobs()) == 2

    def test_router_not_found(self) -> None:
        from hammerio.core.router import JobRouter
        router = JobRouter()
        with pytest.raises(FileNotFoundError):
            router.route("/nonexistent/file.txt")
