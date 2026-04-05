"""Edge-case tests for the profiler module.

Covers empty files, symlinks, entropy edge cases, special characters
in filenames, and recommendation variations.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hammerio.core.profiler import (
    BatchProfile,
    CompressionMode,
    CompressionRecommendation,
    FileCategory,
    FileProfile,
    categorize_file,
    estimate_entropy,
    is_already_compressed,
    profile_directory,
    profile_file,
    recommend_batch,
    recommend_compression,
)


class TestEntropy:
    """Test estimate_entropy edge cases."""

    def test_empty_file(self, tmp_dir: Path) -> None:
        f = tmp_dir / "empty.dat"
        f.write_bytes(b"")
        assert estimate_entropy(f) == 0.0

    def test_single_byte_file(self, tmp_dir: Path) -> None:
        f = tmp_dir / "single.dat"
        f.write_bytes(b"A")
        assert estimate_entropy(f) == 0.0  # 1 unique byte -> 0 entropy

    def test_all_same_bytes(self, tmp_dir: Path) -> None:
        f = tmp_dir / "uniform.dat"
        f.write_bytes(b"\x00" * 10000)
        assert estimate_entropy(f) == 0.0

    def test_two_equally_distributed(self, tmp_dir: Path) -> None:
        f = tmp_dir / "two_vals.dat"
        f.write_bytes(b"\x00\x01" * 50000)
        entropy = estimate_entropy(f)
        # Should be close to 1.0 (log2(2) = 1)
        assert 0.9 < entropy < 1.1

    def test_nonexistent_file(self, tmp_dir: Path) -> None:
        f = tmp_dir / "nope.dat"
        assert estimate_entropy(f) == 0.0

    def test_large_sample(self, tmp_dir: Path) -> None:
        """Test entropy sampling on a file larger than the sample size."""
        f = tmp_dir / "large.dat"
        # Write 200KB of random data
        f.write_bytes(os.urandom(200000))
        entropy = estimate_entropy(f, sample_size=65536)
        assert entropy > 7.0  # Random data has near-maximum entropy


class TestCategorizeFile:
    """Test file categorization edge cases."""

    def test_document_extension(self, tmp_dir: Path) -> None:
        f = tmp_dir / "test.pdf"
        f.write_bytes(b"\x00" * 50)
        assert categorize_file(f) == FileCategory.DOCUMENT

    def test_safetensors_is_dataset(self, tmp_dir: Path) -> None:
        f = tmp_dir / "model.safetensors"
        f.write_bytes(b"\x00" * 50)
        assert categorize_file(f) == FileCategory.DATASET

    def test_binary_detection(self, tmp_dir: Path) -> None:
        f = tmp_dir / "data.unknown_ext_xyz"
        f.write_bytes(os.urandom(1000))
        cat = categorize_file(f)
        assert cat == FileCategory.BINARY

    def test_text_detection_by_content(self, tmp_dir: Path) -> None:
        f = tmp_dir / "data.unknown_ext_xyz"
        f.write_text("This is plain text content\n" * 100)
        cat = categorize_file(f)
        assert cat == FileCategory.TEXT

    def test_empty_file_is_unknown(self, tmp_dir: Path) -> None:
        f = tmp_dir / "empty.unknown_ext_xyz"
        f.write_bytes(b"")
        cat = categorize_file(f)
        assert cat == FileCategory.UNKNOWN


class TestIsAlreadyCompressed:
    """Test the is_already_compressed function."""

    def test_zip_is_compressed(self, tmp_dir: Path) -> None:
        assert is_already_compressed(tmp_dir / "a.zip", FileCategory.ARCHIVE) is True

    def test_jpg_is_compressed(self, tmp_dir: Path) -> None:
        assert is_already_compressed(tmp_dir / "a.jpg", FileCategory.IMAGE) is True

    def test_wav_not_compressed(self, tmp_dir: Path) -> None:
        assert is_already_compressed(tmp_dir / "a.wav", FileCategory.AUDIO) is False

    def test_txt_not_compressed(self, tmp_dir: Path) -> None:
        assert is_already_compressed(tmp_dir / "a.txt", FileCategory.TEXT) is False


class TestProfileFile:
    """Test profile_file edge cases."""

    def test_empty_file(self, tmp_dir: Path) -> None:
        f = tmp_dir / "empty.txt"
        f.write_bytes(b"")
        profile = profile_file(f)
        assert profile.size_bytes == 0
        assert profile.estimated_entropy == 0.0

    def test_file_with_spaces(self, tmp_dir: Path) -> None:
        f = tmp_dir / "file with spaces.txt"
        f.write_text("test data\n" * 100)
        profile = profile_file(f)
        assert profile.size_bytes > 0
        assert "file with spaces" in str(profile.path)

    def test_size_human_bytes(self, tmp_dir: Path) -> None:
        f = tmp_dir / "tiny.bin"
        f.write_bytes(b"\x00" * 500)
        profile = profile_file(f)
        assert "B" in profile.size_human

    def test_size_human_kb(self, tmp_dir: Path) -> None:
        f = tmp_dir / "small.bin"
        f.write_bytes(b"\x00" * 5000)
        profile = profile_file(f)
        assert "KB" in profile.size_human

    def test_size_human_mb(self, tmp_dir: Path) -> None:
        profile = FileProfile(
            path=Path("/fake"), size_bytes=5 * 1024 * 1024,
            category=FileCategory.BINARY, mime_type="application/octet-stream",
            extension=".bin",
        )
        assert "MB" in profile.size_human

    def test_size_human_gb(self) -> None:
        profile = FileProfile(
            path=Path("/fake"), size_bytes=3 * 1024 ** 3,
            category=FileCategory.BINARY, mime_type="application/octet-stream",
            extension=".bin",
        )
        assert "GB" in profile.size_human

    def test_symlink(self, tmp_dir: Path) -> None:
        target = tmp_dir / "real.txt"
        target.write_text("real content\n" * 100)
        link = tmp_dir / "link.txt"
        link.symlink_to(target)
        profile = profile_file(link)
        assert profile.size_bytes == target.stat().st_size


class TestProfileDirectory:
    """Test profile_directory edge cases."""

    def test_empty_directory(self, tmp_dir: Path) -> None:
        d = tmp_dir / "empty_dir"
        d.mkdir()
        batch = profile_directory(d)
        assert batch.file_count == 0
        assert batch.total_size_bytes == 0

    def test_non_recursive(self, tmp_dir: Path) -> None:
        d = tmp_dir / "parent"
        d.mkdir()
        (d / "top.txt").write_text("top level")
        sub = d / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested file")
        batch = profile_directory(d, recursive=False)
        assert batch.file_count == 1  # only top.txt

    def test_batch_total_size(self, tmp_dir: Path) -> None:
        d = tmp_dir / "sized"
        d.mkdir()
        (d / "a.txt").write_bytes(b"A" * 1000)
        (d / "b.txt").write_bytes(b"B" * 2000)
        batch = profile_directory(d)
        assert batch.total_size_bytes == 3000
        assert abs(batch.total_size_mb - 3000 / (1024 * 1024)) < 0.001


class TestRecommendCompression:
    """Test recommendation logic edge cases."""

    def test_image_with_vpi(self) -> None:
        fp = FileProfile(
            path=Path("/test.jpg"), size_bytes=1024 * 100,
            category=FileCategory.IMAGE, mime_type="image/jpeg",
            extension=".jpg",
        )
        rec = recommend_compression(fp, gpu_available=True, vpi_available=True)
        assert rec.mode == CompressionMode.CPU_ZSTD

    def test_audio_no_gpu(self) -> None:
        fp = FileProfile(
            path=Path("/test.wav"), size_bytes=1024 * 1024,
            category=FileCategory.AUDIO, mime_type="audio/wav",
            extension=".wav",
        )
        rec = recommend_compression(fp, gpu_available=False)
        assert rec.mode == CompressionMode.CPU_ZSTD

    def test_large_binary_with_nvcomp(self) -> None:
        fp = FileProfile(
            path=Path("/test.bin"), size_bytes=1024 * 1024 * 600,
            category=FileCategory.BINARY, mime_type="application/octet-stream",
            extension=".bin",
        )
        rec = recommend_compression(fp, gpu_available=True, nvcomp_available=True)
        assert rec.mode == CompressionMode.GPU_NVCOMP

    def test_text_fast_uses_lz4(self) -> None:
        fp = FileProfile(
            path=Path("/test.txt"), size_bytes=1024 * 100,
            category=FileCategory.TEXT, mime_type="text/plain",
            extension=".txt", estimated_entropy=3.0,
        )
        rec = recommend_compression(fp, target_quality="fast")
        assert rec.algorithm == "lz4"

    def test_dataset_small_cpu(self) -> None:
        fp = FileProfile(
            path=Path("/data.csv"), size_bytes=1024 * 10,
            category=FileCategory.DATASET, mime_type="text/csv",
            extension=".csv",
        )
        rec = recommend_compression(fp, gpu_available=True, nvcomp_available=True)
        # Small dataset -> CPU (under 100MB threshold)
        assert rec.mode == CompressionMode.CPU_ZSTD

    def test_default_path_for_unknown(self) -> None:
        fp = FileProfile(
            path=Path("/test.xyz"), size_bytes=1024,
            category=FileCategory.UNKNOWN, mime_type="application/octet-stream",
            extension=".xyz",
        )
        rec = recommend_compression(fp)
        assert rec.mode == CompressionMode.CPU_ZSTD


class TestRecommendBatch:
    """Test batch recommendation."""

    def test_batch_with_mixed_files(self) -> None:
        files = [
            FileProfile(
                path=Path("/a.txt"), size_bytes=100,
                category=FileCategory.TEXT, mime_type="text/plain",
                extension=".txt", estimated_entropy=3.0,
            ),
            FileProfile(
                path=Path("/b.mp4"), size_bytes=1000000,
                category=FileCategory.VIDEO, mime_type="video/mp4",
                extension=".mp4",
            ),
        ]
        batch = BatchProfile(
            files=files, total_size_bytes=1000100,
            category_counts={FileCategory.TEXT: 1, FileCategory.VIDEO: 1},
            dominant_category=FileCategory.VIDEO,
        )
        recs = recommend_batch(batch)
        assert len(recs) == 2
        assert all(isinstance(r, CompressionRecommendation) for r in recs)
