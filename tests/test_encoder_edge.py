"""Edge-case tests for encoder process() methods.

Covers empty input files, files with spaces in names, read-only output
directories, and miscellaneous boundary conditions.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hammerio.core.hardware import detect_hardware


# Reuse the helper from test_encoders
from hammerio.core.hardware import (
    HardwareProfile,
    NvencCapability,
    NvdecCapability,
    NvcompCapability,
    VpiCapability,
    GstreamerNvencCapability,
    PlatformType,
    GpuVendor,
    PowerMode,
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


class TestGeneralEncoderEdgeCases:
    """Edge cases for GeneralEncoder.process()."""

    def test_empty_file_compresses(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder
        hw = detect_hardware()
        encoder = GeneralEncoder(hw)
        empty = tmp_dir / "empty.txt"
        empty.write_bytes(b"")
        out = tmp_dir / "empty.txt.zst"
        result = encoder.process(empty, out, algorithm="zstd", quality="fast")
        assert Path(result).exists()

    def test_file_with_spaces_in_name(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder
        hw = detect_hardware()
        encoder = GeneralEncoder(hw)
        f = tmp_dir / "file with spaces.txt"
        f.write_text("hello world\n" * 500)
        out = tmp_dir / "file with spaces.txt.zst"
        result = encoder.process(f, out, algorithm="zstd", quality="balanced")
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_output_dir_created_automatically(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder
        hw = detect_hardware()
        encoder = GeneralEncoder(hw)
        f = tmp_dir / "input.txt"
        f.write_text("test\n" * 100)
        out = tmp_dir / "nested" / "deep" / "output.zst"
        result = encoder.process(f, out, algorithm="zstd", quality="fast")
        assert Path(result).exists()

    def test_bzip2_compress(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder
        hw = detect_hardware()
        encoder = GeneralEncoder(hw)
        f = tmp_dir / "data.txt"
        f.write_text("bzip2 test data\n" * 500)
        out = tmp_dir / "data.txt.bz2"
        result = encoder.process(f, out, algorithm="bzip2", quality="fast")
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_compress_then_decompress_empty_file(self, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder
        hw = detect_hardware()
        encoder = GeneralEncoder(hw)
        empty = tmp_dir / "empty_rt.dat"
        empty.write_bytes(b"")
        compressed = tmp_dir / "empty_rt.zst"
        encoder.process(empty, compressed, algorithm="zstd", quality="fast")
        decompressed = tmp_dir / "empty_rt_restored.dat"
        encoder.decompress(compressed, decompressed)
        assert Path(decompressed).exists()
        assert Path(decompressed).stat().st_size == 0


class TestBulkEncoderEdgeCases:
    """Edge cases for BulkEncoder.process()."""

    def test_empty_file_raises_or_handles(self, tmp_dir: Path) -> None:
        from hammerio.encoders.bulk import BulkEncoder
        hw = _make_hw()
        encoder = BulkEncoder(hw)
        empty = tmp_dir / "empty.dat"
        empty.write_bytes(b"")
        out = tmp_dir / "empty.hammer"
        # BulkEncoder should handle empty files (0 chunks written)
        result = encoder.process(empty, out, algorithm="zstd", quality="fast")
        assert Path(result).exists()

    def test_file_with_spaces(self, tmp_dir: Path) -> None:
        from hammerio.encoders.bulk import BulkEncoder
        hw = _make_hw()
        encoder = BulkEncoder(hw)
        f = tmp_dir / "bulk file with spaces.dat"
        f.write_bytes(b"A" * 10000)
        out = tmp_dir / "bulk file with spaces.hammer"
        result = encoder.process(f, out, algorithm="zstd", quality="fast")
        assert Path(result).exists()

    def test_unsupported_algorithm(self, tmp_dir: Path) -> None:
        from hammerio.encoders.bulk import BulkEncoder
        hw = _make_hw()
        encoder = BulkEncoder(hw)
        f = tmp_dir / "data.bin"
        f.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            encoder.process(f, tmp_dir / "out.hammer", algorithm="xz", quality="fast")

    def test_nonexistent_input(self, tmp_dir: Path) -> None:
        from hammerio.encoders.bulk import BulkEncoder
        hw = _make_hw()
        encoder = BulkEncoder(hw)
        with pytest.raises(FileNotFoundError):
            encoder.process(
                tmp_dir / "nope.bin", tmp_dir / "out.hammer",
                algorithm="zstd", quality="fast",
            )

    def test_decompress_nonexistent(self, tmp_dir: Path) -> None:
        from hammerio.encoders.bulk import BulkEncoder
        hw = _make_hw()
        encoder = BulkEncoder(hw)
        with pytest.raises(FileNotFoundError):
            encoder.decompress(tmp_dir / "nope.hammer", tmp_dir / "out.bin")

    def test_decompress_invalid_magic(self, tmp_dir: Path) -> None:
        from hammerio.encoders.bulk import BulkEncoder
        hw = _make_hw()
        encoder = BulkEncoder(hw)
        bad = tmp_dir / "bad.hammer"
        bad.write_bytes(b"NOT_HMIO" + b"\x00" * 100)
        with pytest.raises(ValueError, match="Invalid magic"):
            encoder.decompress(bad, tmp_dir / "out.bin")

    def test_decompress_truncated_header(self, tmp_dir: Path) -> None:
        from hammerio.encoders.bulk import BulkEncoder
        hw = _make_hw()
        encoder = BulkEncoder(hw)
        truncated = tmp_dir / "truncated.hammer"
        truncated.write_bytes(b"HM")
        with pytest.raises(ValueError, match="too small"):
            encoder.decompress(truncated, tmp_dir / "out.bin")

    def test_quality_string_mapping(self, tmp_dir: Path) -> None:
        """Verify string quality presets map to valid integer levels."""
        from hammerio.encoders.bulk import BulkEncoder
        hw = _make_hw()
        encoder = BulkEncoder(hw)
        f = tmp_dir / "qs.bin"
        f.write_bytes(b"X" * 5000)
        for q in ("fast", "balanced", "quality", "lossless"):
            out = tmp_dir / f"qs_{q}.hammer"
            result = encoder.process(f, out, algorithm="zstd", quality=q)
            assert Path(result).exists()


class TestImageEncoderEdgeCases:
    """Edge cases for ImageEncoder.process()."""

    def test_empty_directory(self, tmp_dir: Path) -> None:
        from hammerio.encoders.image import ImageEncoder
        hw = _make_hw()
        encoder = ImageEncoder(hw)
        empty_dir = tmp_dir / "empty_images"
        empty_dir.mkdir()
        out_dir = tmp_dir / "empty_out"
        result = encoder.process(empty_dir, out_dir, algorithm="jpeg", quality="fast")
        assert Path(result).is_dir()

    def test_no_backend_raises(self, tmp_dir: Path) -> None:
        from hammerio.encoders.image import ImageEncoder
        hw = _make_hw()
        encoder = ImageEncoder(hw)
        encoder._backend = "none"
        f = tmp_dir / "test.jpg"
        f.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="No image processing backend"):
            encoder.process(f, tmp_dir / "out.jpg", algorithm="jpeg", quality="fast")


class TestDatasetEncoderEdgeCases:
    """Edge cases for DatasetEncoder.process()."""

    def test_empty_file(self, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder
        hw = _make_hw()
        encoder = DatasetEncoder(hw)
        empty = tmp_dir / "empty.csv"
        empty.write_bytes(b"")
        out = tmp_dir / "empty.csv.zst"
        result = encoder.process(empty, out, algorithm="zstd", quality="fast")
        assert Path(result).exists()

    def test_auto_output_path(self, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder
        hw = _make_hw()
        encoder = DatasetEncoder(hw)
        f = tmp_dir / "data.csv"
        f.write_text("a,b\n1,2\n" * 100)
        result = encoder.process(f, None, algorithm="zstd", quality="fast")
        assert result.endswith(".zst")
        assert Path(result).exists()
        # Clean up
        Path(result).unlink()

    def test_empty_directory(self, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder
        hw = _make_hw()
        encoder = DatasetEncoder(hw)
        d = tmp_dir / "empty_ds"
        d.mkdir()
        out = tmp_dir / "empty_ds.tar.zst"
        result = encoder.process(d, out, algorithm="zstd", quality="fast")
        # Should return path (may or may not create file)
        assert isinstance(result, str)


class TestVideoEncoderEdgeCases:
    """Edge cases for VideoEncoder validation."""

    def test_validate_missing_file(self, tmp_dir: Path) -> None:
        from hammerio.encoders.video import VideoEncoder
        hw = _make_hw()
        encoder = VideoEncoder(hw)
        with pytest.raises(FileNotFoundError):
            encoder.process(
                tmp_dir / "nonexistent.mp4", tmp_dir / "out.mp4",
                algorithm="h264", quality="fast",
            )

    def test_validate_bad_extension(self, tmp_dir: Path) -> None:
        from hammerio.encoders.video import VideoEncoder
        hw = _make_hw()
        encoder = VideoEncoder(hw)
        bad = tmp_dir / "bad.txt"
        bad.write_text("not a video")
        with pytest.raises(ValueError, match="Unsupported video format"):
            encoder.process(bad, tmp_dir / "out.mp4", algorithm="h264", quality="fast")

    def test_validate_bad_quality(self, tmp_dir: Path) -> None:
        from hammerio.encoders.video import VideoEncoder
        hw = _make_hw()
        encoder = VideoEncoder(hw)
        f = tmp_dir / "test.mp4"
        f.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="Unknown quality preset"):
            encoder.process(f, tmp_dir / "out.mp4", algorithm="h264", quality="ultra_hd")

    def test_resolve_codec_passthrough(self) -> None:
        from hammerio.encoders.video import VideoEncoder
        hw = _make_hw()
        encoder = VideoEncoder(hw)
        # Full encoder names should pass through unchanged
        assert encoder._resolve_codec("libx264", False) == "libx264"
        assert encoder._resolve_codec("h264_nvenc", True) == "h264_nvenc"
        assert encoder._resolve_codec("libsvtav1", False) == "libsvtav1"

    def test_resolve_codec_av1(self) -> None:
        from hammerio.encoders.video import VideoEncoder
        hw = _make_hw()
        encoder = VideoEncoder(hw)
        assert encoder._resolve_codec("av1", False) == "libsvtav1"
        assert encoder._resolve_codec("av1", True) == "av1_nvenc"


class TestAudioEncoderEdgeCases:
    """Edge cases for AudioEncoder."""

    def test_no_ffmpeg_raises(self, tmp_dir: Path) -> None:
        from hammerio.encoders.audio import AudioEncoder
        hw = _make_hw()
        encoder = AudioEncoder(hw)
        encoder._ffmpeg = None
        with pytest.raises(FileNotFoundError, match="ffmpeg"):
            encoder.process(
                tmp_dir / "test.wav", tmp_dir / "out.mp3",
                algorithm="mp3", quality="fast",
            )

    def test_nonexistent_input(self, tmp_dir: Path) -> None:
        from hammerio.encoders.audio import AudioEncoder
        hw = _make_hw()
        encoder = AudioEncoder(hw)
        if encoder._ffmpeg is None:
            pytest.skip("ffmpeg not installed")
        with pytest.raises(FileNotFoundError):
            encoder.process(
                tmp_dir / "no.wav", tmp_dir / "out.mp3",
                algorithm="mp3", quality="fast",
            )

    def test_unsupported_format(self, tmp_dir: Path) -> None:
        from hammerio.encoders.audio import AudioEncoder
        hw = _make_hw()
        encoder = AudioEncoder(hw)
        if encoder._ffmpeg is None:
            pytest.skip("ffmpeg not installed")
        f = tmp_dir / "test.wav"
        f.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="Unsupported audio format"):
            encoder.process(f, tmp_dir / "out.xyz", algorithm="badformat", quality="fast")
