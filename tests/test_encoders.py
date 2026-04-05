"""Tests for encoder modules."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
    detect_hardware,
)


def _make_hw_profile(**overrides) -> HardwareProfile:
    """Create a minimal HardwareProfile for testing without real detection."""
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
# VideoEncoder tests
# ======================================================================


class TestVideoEncoder:
    def test_init_and_route_detection(self) -> None:
        from hammerio.encoders.video import VideoEncoder

        hw = _make_hw_profile()
        encoder = VideoEncoder(hw)
        # Should fall back to CPU since no NVENC
        assert encoder._should_use_gpu("h264") is False
        assert encoder._resolve_codec("h264", False) == "libx264"
        assert encoder._resolve_codec("hevc", False) == "libx265"

    def test_init_with_nvenc(self) -> None:
        from hammerio.encoders.video import VideoEncoder

        hw = _make_hw_profile(
            nvenc=NvencCapability(available=True, codecs=["h264", "hevc"]),
        )
        encoder = VideoEncoder(hw)
        assert encoder._should_use_gpu("h264") is True
        assert encoder._resolve_codec("h264", True) == "h264_nvenc"
        assert encoder._resolve_codec("hevc", True) == "hevc_nvenc"

    def test_quality_presets_exposed(self) -> None:
        from hammerio.encoders.video import VideoEncoder

        presets = VideoEncoder.get_quality_presets()
        assert "fast" in presets
        assert "balanced" in presets
        assert "quality" in presets
        assert "lossless" in presets
        assert presets["balanced"]["cpu_crf"] == 23

    def test_process_missing_input(self, tmp_dir: Path) -> None:
        from hammerio.encoders.video import VideoEncoder

        hw = _make_hw_profile()
        encoder = VideoEncoder(hw)
        with pytest.raises(FileNotFoundError):
            encoder.process(
                tmp_dir / "nonexistent.mp4",
                tmp_dir / "out.mp4",
                algorithm="h264",
                quality="balanced",
            )

    def test_process_unsupported_extension(self, tmp_dir: Path) -> None:
        from hammerio.encoders.video import VideoEncoder

        hw = _make_hw_profile()
        encoder = VideoEncoder(hw)
        bad_file = tmp_dir / "file.xyz"
        bad_file.write_text("not a video")
        with pytest.raises(ValueError, match="Unsupported video format"):
            encoder.process(
                bad_file,
                tmp_dir / "out.mp4",
                algorithm="h264",
                quality="balanced",
            )

    def test_process_bad_quality_preset(self, tmp_dir: Path) -> None:
        from hammerio.encoders.video import VideoEncoder

        hw = _make_hw_profile()
        encoder = VideoEncoder(hw)
        vid = tmp_dir / "test.mp4"
        vid.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="Unknown quality preset"):
            encoder.process(vid, tmp_dir / "out.mp4", algorithm="h264", quality="ultra")

    def test_gstreamer_nvenc_detection(self) -> None:
        from hammerio.encoders.video import VideoEncoder

        hw = _make_hw_profile(
            gstreamer_nvenc=GstreamerNvencCapability(
                available=True, has_h264=True, has_h265=True,
                has_h264parse=True, has_h265parse=True,
                gst_launch_path="/usr/bin/gst-launch-1.0",
            ),
        )
        encoder = VideoEncoder(hw)
        assert encoder._use_gstreamer is True

    def test_gstreamer_disabled_when_unavailable(self) -> None:
        from hammerio.encoders.video import VideoEncoder

        hw = _make_hw_profile(
            gstreamer_nvenc=GstreamerNvencCapability(available=False),
        )
        encoder = VideoEncoder(hw)
        assert encoder._use_gstreamer is False

    def test_gstreamer_pipeline_builder_h264(self) -> None:
        from hammerio.encoders.video import VideoEncoder

        pipeline = VideoEncoder._build_gstreamer_pipeline(
            input_path=Path("/tmp/input.mp4"),
            output_path=Path("/tmp/output.mp4"),
            encoder="nvv4l2h264enc",
            parser="h264parse",
            muxer="mp4mux",
            bitrate=8_000_000,
        )
        assert "nvv4l2h264enc" in pipeline
        assert "h264parse" in pipeline
        assert "nvvidconv" in pipeline
        assert "video/x-raw(memory:NVMM),format=NV12" in pipeline
        assert "mp4mux" in pipeline
        assert "bitrate=8000000" in pipeline

    def test_gstreamer_pipeline_builder_h265(self) -> None:
        from hammerio.encoders.video import VideoEncoder

        pipeline = VideoEncoder._build_gstreamer_pipeline(
            input_path=Path("/tmp/input.mp4"),
            output_path=Path("/tmp/output.mkv"),
            encoder="nvv4l2h265enc",
            parser="h265parse",
            muxer="matroskamux",
            bitrate=15_000_000,
        )
        assert "nvv4l2h265enc" in pipeline
        assert "h265parse" in pipeline
        assert "matroskamux" in pipeline

    def test_gstreamer_fallback_to_ffmpeg(self) -> None:
        """When GStreamer is available but FFmpeg NVENC is not, the encoder
        should try GStreamer first and fall back gracefully."""
        from hammerio.encoders.video import VideoEncoder

        hw = _make_hw_profile(
            gstreamer_nvenc=GstreamerNvencCapability(
                available=True, has_h264=True, has_h265=False,
                has_h264parse=True, has_h265parse=False,
                gst_launch_path="/usr/bin/gst-launch-1.0",
            ),
        )
        encoder = VideoEncoder(hw)
        # When FFmpeg NVENC is False, GStreamer should be preferred
        assert encoder._use_gstreamer is True
        assert encoder._should_use_gpu("h264") is False  # no FFmpeg NVENC


# ======================================================================
# ImageEncoder tests
# ======================================================================


class TestImageEncoder:
    def test_init_backend_detection(self) -> None:
        from hammerio.encoders.image import ImageEncoder

        hw = _make_hw_profile()
        encoder = ImageEncoder(hw)
        # Should pick either opencv_cpu or pil as backend
        assert encoder._backend in ("opencv_cpu", "opencv_cuda", "pil", "none")

    def test_process_single_image(self, tmp_dir: Path) -> None:
        from hammerio.encoders.image import ImageEncoder

        hw = _make_hw_profile()
        encoder = ImageEncoder(hw)

        # Create a small test image using PIL or OpenCV
        input_img = tmp_dir / "test_input.jpg"
        try:
            from PIL import Image
            img = Image.new("RGB", (32, 32), color=(255, 0, 0))
            img.save(str(input_img), format="JPEG")
        except ImportError:
            import cv2
            import numpy as np
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            img[:, :, 2] = 255  # Red in BGR
            cv2.imwrite(str(input_img), img)

        output_img = tmp_dir / "test_output.jpg"
        result = encoder.process(
            input_path=input_img,
            output_path=output_img,
            algorithm="jpeg",
            quality="balanced",
        )
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_process_webp_output(self, tmp_dir: Path) -> None:
        from hammerio.encoders.image import ImageEncoder

        hw = _make_hw_profile()
        encoder = ImageEncoder(hw)

        input_img = tmp_dir / "test_input.png"
        try:
            from PIL import Image
            img = Image.new("RGB", (16, 16), color=(0, 128, 255))
            img.save(str(input_img), format="PNG")
        except ImportError:
            import cv2
            import numpy as np
            img = np.zeros((16, 16, 3), dtype=np.uint8)
            img[:, :, 0] = 255
            cv2.imwrite(str(input_img), img)

        output_img = tmp_dir / "test_output.webp"
        result = encoder.process(
            input_path=input_img,
            output_path=output_img,
            algorithm="webp",
            quality="quality",
        )
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_process_directory(self, tmp_dir: Path) -> None:
        from hammerio.encoders.image import ImageEncoder

        hw = _make_hw_profile()
        encoder = ImageEncoder(hw)

        img_dir = tmp_dir / "images"
        img_dir.mkdir()
        out_dir = tmp_dir / "images_out"

        # Create two small test images
        for name in ("a.jpg", "b.jpg"):
            path = img_dir / name
            try:
                from PIL import Image
                img = Image.new("RGB", (8, 8), color=(0, 0, 255))
                img.save(str(path), format="JPEG")
            except ImportError:
                import cv2
                import numpy as np
                img = np.zeros((8, 8, 3), dtype=np.uint8)
                cv2.imwrite(str(path), img)

        progress_values: list[float] = []
        result = encoder.process(
            input_path=img_dir,
            output_path=out_dir,
            algorithm="jpeg",
            quality="fast",
            progress_callback=lambda jid, pct: progress_values.append(pct),
            job_id="test_batch",
        )
        assert Path(result).is_dir()
        output_files = list(Path(result).iterdir())
        assert len(output_files) == 2
        assert len(progress_values) >= 2

    def test_process_missing_input(self, tmp_dir: Path) -> None:
        from hammerio.encoders.image import ImageEncoder

        hw = _make_hw_profile()
        encoder = ImageEncoder(hw)
        with pytest.raises(FileNotFoundError):
            encoder.process(
                tmp_dir / "nonexistent.jpg",
                tmp_dir / "out.jpg",
                algorithm="jpeg",
                quality="balanced",
            )


# ======================================================================
# AudioEncoder tests
# ======================================================================


class TestAudioEncoder:
    def test_init(self) -> None:
        from hammerio.encoders.audio import AudioEncoder

        hw = _make_hw_profile()
        encoder = AudioEncoder(hw)
        # ffmpeg may or may not be installed; just verify init succeeds
        assert encoder.hardware is hw

    def test_process_missing_input(self, tmp_dir: Path) -> None:
        from hammerio.encoders.audio import AudioEncoder

        hw = _make_hw_profile()
        encoder = AudioEncoder(hw)
        if encoder._ffmpeg is None:
            pytest.skip("ffmpeg not installed")
        with pytest.raises(FileNotFoundError):
            encoder.process(
                tmp_dir / "nonexistent.wav",
                tmp_dir / "out.mp3",
                algorithm="mp3",
                quality="balanced",
            )

    def test_process_bad_format(self, tmp_dir: Path) -> None:
        from hammerio.encoders.audio import AudioEncoder

        hw = _make_hw_profile()
        encoder = AudioEncoder(hw)
        if encoder._ffmpeg is None:
            pytest.skip("ffmpeg not installed")
        src = tmp_dir / "test.wav"
        src.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="Unsupported audio format"):
            encoder.process(src, tmp_dir / "out.xyz", algorithm="xyz", quality="fast")


# ======================================================================
# DatasetEncoder tests
# ======================================================================


class TestDatasetEncoder:
    def test_init(self) -> None:
        from hammerio.encoders.dataset import DatasetEncoder

        hw = _make_hw_profile()
        encoder = DatasetEncoder(hw)
        assert encoder.hardware is hw

    def test_compress_csv(self, sample_csv_file: Path, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder

        hw = _make_hw_profile()
        encoder = DatasetEncoder(hw)

        out = tmp_dir / "dataset.csv.zst"
        result = encoder.process(
            input_path=sample_csv_file,
            output_path=out,
            algorithm="zstd",
            quality="balanced",
        )
        result_path = Path(result)
        assert result_path.exists()
        assert result_path.stat().st_size > 0
        # CSV is highly compressible — output should be smaller
        assert result_path.stat().st_size < sample_csv_file.stat().st_size

    def test_compress_csv_with_progress(self, sample_csv_file: Path, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder

        hw = _make_hw_profile()
        encoder = DatasetEncoder(hw)

        progress_values: list[float] = []
        out = tmp_dir / "dataset_prog.csv.zst"
        encoder.process(
            input_path=sample_csv_file,
            output_path=out,
            algorithm="zstd",
            quality="fast",
            progress_callback=lambda jid, pct: progress_values.append(pct),
            job_id="ds_test",
        )
        # Should have received at least 0% and 100%
        assert len(progress_values) >= 2
        assert progress_values[-1] == 100.0

    def test_missing_input(self, tmp_dir: Path) -> None:
        from hammerio.encoders.dataset import DatasetEncoder

        hw = _make_hw_profile()
        encoder = DatasetEncoder(hw)
        with pytest.raises(FileNotFoundError):
            encoder.process(
                tmp_dir / "nofile.csv",
                tmp_dir / "out.zst",
                algorithm="zstd",
                quality="balanced",
            )

    def test_round_trip_csv(self, sample_csv_file: Path, tmp_dir: Path) -> None:
        """Compress a CSV with DatasetEncoder then decompress via StreamingDataset and verify content."""
        from hammerio.encoders.dataset import DatasetEncoder, StreamingDataset

        hw = _make_hw_profile()
        encoder = DatasetEncoder(hw)

        compressed = tmp_dir / "roundtrip_dataset.csv.zst"
        encoder.process(
            input_path=sample_csv_file,
            output_path=compressed,
            algorithm="zstd",
            quality="fast",
        )
        assert compressed.exists()
        assert compressed.stat().st_size > 0

        # Decompress via StreamingDataset and reassemble all chunks
        ds = StreamingDataset(compressed)
        assert len(ds) >= 1

        reassembled = b"".join(chunk for chunk in ds)
        original_bytes = sample_csv_file.read_bytes()
        assert reassembled == original_bytes, (
            f"Round-trip mismatch: got {len(reassembled)} bytes, expected {len(original_bytes)}"
        )

    def test_round_trip_large_csv(self, tmp_dir: Path) -> None:
        """Round-trip a larger CSV (100k rows) through DatasetEncoder."""
        from hammerio.encoders.dataset import DatasetEncoder, StreamingDataset

        hw = _make_hw_profile()
        encoder = DatasetEncoder(hw)

        csv_file = tmp_dir / "large_dataset.csv"
        csv_file.write_text("\n".join(f"{i},{i * 3.14}" for i in range(100000)))

        compressed = tmp_dir / "large_dataset.csv.zst"
        encoder.process(csv_file, compressed, "zstd", "fast")

        ds = StreamingDataset(compressed)
        reassembled = b"".join(ds)
        assert reassembled == csv_file.read_bytes()


# ======================================================================
# GeneralEncoder tests (pre-existing)
# ======================================================================


class TestGeneralEncoder:
    def test_compress_zstd(self, sample_text_file: Path, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder
        hw = detect_hardware()
        encoder = GeneralEncoder(hw)

        out = tmp_dir / "output.zst"
        result = encoder.process(
            input_path=sample_text_file,
            output_path=out,
            algorithm="zstd",
            quality="balanced",
        )
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0
        assert Path(result).stat().st_size < sample_text_file.stat().st_size

    def test_compress_gzip(self, sample_text_file: Path, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder
        hw = detect_hardware()
        encoder = GeneralEncoder(hw)

        out = tmp_dir / "output.gz"
        result = encoder.process(
            input_path=sample_text_file,
            output_path=out,
            algorithm="gzip",
            quality="balanced",
        )
        assert Path(result).exists()
        assert Path(result).stat().st_size < sample_text_file.stat().st_size

    def test_decompress_zstd(self, sample_compressible_file: Path, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder
        hw = detect_hardware()
        encoder = GeneralEncoder(hw)

        compressed = tmp_dir / "compressed.zst"
        encoder.process(
            input_path=sample_compressible_file,
            output_path=compressed,
            algorithm="zstd",
            quality="fast",
        )

        decompressed = tmp_dir / "decompressed.dat"
        result = encoder.decompress(compressed, decompressed)
        assert Path(result).exists()
        assert Path(result).stat().st_size == sample_compressible_file.stat().st_size

    def test_compress_directory(self, sample_directory: Path, tmp_dir: Path) -> None:
        from hammerio.encoders.general import GeneralEncoder
        hw = detect_hardware()
        encoder = GeneralEncoder(hw)

        out = tmp_dir / "archive.tar.zst"
        result = encoder.process(
            input_path=sample_directory,
            output_path=out,
            algorithm="zstd",
            quality="balanced",
        )
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0


class TestGeneralEncoderAdvanced:
    """Advanced tests: directory tar compression, round-trips, content verification."""

    def test_directory_tar_zstd_compress_decompress(self, sample_directory: Path, tmp_dir: Path) -> None:
        """Compress a directory to tar.zst, then decompress and verify all files match."""
        from hammerio.encoders.general import GeneralEncoder

        hw = detect_hardware()
        encoder = GeneralEncoder(hw)

        archive = tmp_dir / "archive.tar.zst"
        result = encoder.process(
            input_path=sample_directory,
            output_path=archive,
            algorithm="zstd",
            quality="balanced",
        )
        result_path = Path(result)
        assert result_path.exists()
        assert result_path.stat().st_size > 0

        # Decompress the tar.zst archive
        extract_dir = tmp_dir / "extracted"
        dec_result = encoder.decompress(result_path, extract_dir)
        dec_path = Path(dec_result)
        assert dec_path.is_dir()

        # Verify all original files exist in extracted output with matching content
        for orig_file in sorted(sample_directory.rglob("*")):
            if orig_file.is_file():
                rel = orig_file.relative_to(sample_directory)
                extracted_file = dec_path / rel
                assert extracted_file.exists(), f"Missing extracted file: {rel}"
                assert extracted_file.read_bytes() == orig_file.read_bytes(), (
                    f"Content mismatch for {rel}"
                )

    def test_directory_tar_gzip_compress_decompress(self, sample_directory: Path, tmp_dir: Path) -> None:
        """Compress a directory to tar.gz, then decompress and verify."""
        from hammerio.encoders.general import GeneralEncoder

        hw = detect_hardware()
        encoder = GeneralEncoder(hw)

        archive = tmp_dir / "archive.tar.gz"
        result = encoder.process(
            input_path=sample_directory,
            output_path=archive,
            algorithm="gzip",
            quality="fast",
        )
        assert Path(result).exists()

        extract_dir = tmp_dir / "extracted_gz"
        dec_result = encoder.decompress(Path(result), extract_dir)
        dec_path = Path(dec_result)
        assert dec_path.is_dir()

        for orig_file in sorted(sample_directory.rglob("*")):
            if orig_file.is_file():
                rel = orig_file.relative_to(sample_directory)
                extracted_file = dec_path / rel
                assert extracted_file.exists(), f"Missing extracted file: {rel}"
                assert extracted_file.read_bytes() == orig_file.read_bytes()

    def test_zstd_round_trip_content_matches(self, tmp_dir: Path) -> None:
        """Verify compress -> decompress round-trip preserves exact content."""
        from hammerio.encoders.general import GeneralEncoder
        import os

        hw = detect_hardware()
        encoder = GeneralEncoder(hw)

        # Create a file with mixed content (text + some binary-ish patterns)
        original = tmp_dir / "roundtrip_original.dat"
        content = b"Hello HammerIO! " * 10000 + os.urandom(1024) + b"\x00" * 5000
        original.write_bytes(content)

        compressed = tmp_dir / "roundtrip.zst"
        encoder.process(original, compressed, algorithm="zstd", quality="quality")

        decompressed = tmp_dir / "roundtrip_restored.dat"
        encoder.decompress(compressed, decompressed)

        assert Path(decompressed).read_bytes() == content

    def test_gzip_round_trip_content_matches(self, tmp_dir: Path) -> None:
        """Verify gzip compress -> decompress round-trip preserves exact content."""
        from hammerio.encoders.general import GeneralEncoder

        hw = detect_hardware()
        encoder = GeneralEncoder(hw)

        original = tmp_dir / "roundtrip_gz.txt"
        content = "Gzip round-trip test data.\n" * 5000
        original.write_text(content)

        compressed = tmp_dir / "roundtrip.gz"
        encoder.process(original, compressed, algorithm="gzip", quality="balanced")

        decompressed = tmp_dir / "roundtrip_gz_restored.txt"
        encoder.decompress(compressed, decompressed)

        assert Path(decompressed).read_text() == content

    def test_bzip2_round_trip_content_matches(self, tmp_dir: Path) -> None:
        """Verify bzip2 compress -> decompress round-trip preserves exact content."""
        from hammerio.encoders.general import GeneralEncoder

        hw = detect_hardware()
        encoder = GeneralEncoder(hw)

        original = tmp_dir / "roundtrip_bz2.txt"
        content = "Bzip2 round-trip test.\n" * 3000
        original.write_text(content)

        compressed = tmp_dir / "roundtrip.bz2"
        encoder.process(original, compressed, algorithm="bzip2", quality="balanced")

        decompressed = tmp_dir / "roundtrip_bz2_restored.txt"
        encoder.decompress(compressed, decompressed)

        assert Path(decompressed).read_text() == content

    def test_compress_with_progress_callback(self, sample_text_file: Path, tmp_dir: Path) -> None:
        """Verify progress callback fires with correct signature and reaches 100%."""
        from hammerio.encoders.general import GeneralEncoder

        hw = detect_hardware()
        encoder = GeneralEncoder(hw)

        progress_values: list[tuple[str, float]] = []

        def on_progress(job_id: str, pct: float) -> None:
            progress_values.append((job_id, pct))

        out = tmp_dir / "progress_test.zst"
        encoder.process(
            input_path=sample_text_file,
            output_path=out,
            algorithm="zstd",
            quality="fast",
            progress_callback=on_progress,
            job_id="test_progress",
        )

        assert len(progress_values) > 0
        # All callbacks should have the correct job_id
        assert all(jid == "test_progress" for jid, _ in progress_values)
        # Final progress should be 100%
        assert progress_values[-1][1] == 100.0

    def test_auto_output_path_generation(self, sample_text_file: Path) -> None:
        """Verify process generates correct output path when output_path=None."""
        from hammerio.encoders.general import GeneralEncoder

        hw = detect_hardware()
        encoder = GeneralEncoder(hw)

        result = encoder.process(
            input_path=sample_text_file,
            output_path=None,
            algorithm="zstd",
            quality="balanced",
        )
        assert result.endswith(".zst")
        assert Path(result).exists()
        # Clean up
        Path(result).unlink()

    def test_unsupported_algorithm_raises(self, sample_text_file: Path, tmp_dir: Path) -> None:
        """Verify ValueError for unsupported algorithm."""
        from hammerio.encoders.general import GeneralEncoder

        hw = detect_hardware()
        encoder = GeneralEncoder(hw)

        with pytest.raises(ValueError, match="Unsupported algorithm"):
            encoder.process(
                sample_text_file,
                tmp_dir / "out.xz",
                algorithm="xz",
                quality="balanced",
            )

    def test_missing_input_raises(self, tmp_dir: Path) -> None:
        """Verify FileNotFoundError for nonexistent input."""
        from hammerio.encoders.general import GeneralEncoder

        hw = detect_hardware()
        encoder = GeneralEncoder(hw)

        with pytest.raises(FileNotFoundError):
            encoder.process(
                tmp_dir / "does_not_exist.txt",
                tmp_dir / "out.zst",
                algorithm="zstd",
                quality="balanced",
            )

    def test_decompress_unknown_extension_raises(self, tmp_dir: Path) -> None:
        """Verify ValueError when decompressing a file with unknown extension."""
        from hammerio.encoders.general import GeneralEncoder

        hw = detect_hardware()
        encoder = GeneralEncoder(hw)

        bad_file = tmp_dir / "data.xyz"
        bad_file.write_bytes(b"some data")

        with pytest.raises(ValueError, match="Cannot determine"):
            encoder.decompress(bad_file)


class TestBulkEncoder:
    def test_compress_decompress(self, sample_compressible_file: Path, tmp_dir: Path) -> None:
        from hammerio.encoders.bulk import BulkEncoder
        hw = detect_hardware()
        encoder = BulkEncoder(hw)

        compressed = tmp_dir / "bulk.hammer"
        result = encoder.process(
            input_path=sample_compressible_file,
            output_path=compressed,
            algorithm="zstd",
            quality="balanced",
        )
        assert Path(result).exists()

        decompressed = tmp_dir / "decompressed.dat"
        dec_result = encoder.decompress(Path(result), decompressed)
        assert Path(dec_result).exists()
        assert Path(dec_result).stat().st_size == sample_compressible_file.stat().st_size


class TestTelemetry:
    def test_snapshot(self) -> None:
        from hammerio.core.telemetry import TelemetryCollector, SystemSnapshot
        collector = TelemetryCollector()
        snap = collector.get_snapshot()
        assert isinstance(snap, SystemSnapshot)
        assert snap.timestamp > 0
        assert snap.ram_total_mb > 0

    def test_snapshot_to_dict(self) -> None:
        from hammerio.core.telemetry import TelemetryCollector
        collector = TelemetryCollector()
        snap = collector.get_snapshot()
        d = snap.to_dict()
        assert "timestamp" in d
        assert "gpu_util_pct" in d
        assert "ram_pct" in d
        assert "emc_util_pct" in d
        assert "emc_freq_mhz" in d

    def test_snapshot_emc_fields_present(self) -> None:
        """Verify EMC fields exist on SystemSnapshot and to_dict."""
        from hammerio.core.telemetry import SystemSnapshot
        snap = SystemSnapshot(timestamp=1.0)
        assert snap.emc_utilization_pct == 0.0
        assert snap.emc_frequency_mhz == 0.0
        d = snap.to_dict()
        assert d["emc_util_pct"] == 0.0
        assert d["emc_freq_mhz"] == 0.0

    def test_collector_summary_empty(self) -> None:
        from hammerio.core.telemetry import TelemetryCollector
        collector = TelemetryCollector()
        summary = collector.get_summary()
        assert "error" in summary  # No data collected yet

    def test_jtop_snapshot_mock(self) -> None:
        """Verify JtopMonitor parses jtop data correctly with mock."""
        from unittest.mock import patch, MagicMock
        from hammerio.core.telemetry import JtopMonitor

        # Build a fake jtop context manager
        fake_jetson = MagicMock()
        fake_jetson.cpu = {
            "total": {"idle": 25.0},
            "cpu": [{"idle": 20.0}, {"idle": 30.0}],
        }
        fake_jetson.memory = {
            "RAM": {"used": 4096 * 1024, "tot": 8192 * 1024},  # KB
            "SWAP": {"used": 512 * 1024, "tot": 2048 * 1024},
            "EMC": {"val": 42, "cur": 2133000000},  # 42%, 2133 MHz in Hz
        }
        fake_jetson.temperature = {
            "CPU-therm": {"temp": 55.0, "online": True},
            "GPU-therm": {"temp": 52.0, "online": True},
        }
        fake_jetson.power = {
            "rail": {
                "VDD_CPU_GPU_CV": {"online": True, "power": 5000, "avg": 4500},
            },
            "tot": {},
        }

        # GPU object with val and frq attributes
        fake_gpu = MagicMock()
        fake_gpu.val = 65.0
        fake_gpu.frq = 1300000000  # 1.3 GHz in Hz
        fake_jetson.gpu = fake_gpu

        fake_fan = MagicMock()
        fake_fan.speed = 50.0
        fake_jetson.fan = fake_fan

        fake_jetson.nvpmodel = "MAXN"

        # Make it a context manager
        fake_jtop_class = MagicMock()
        fake_jtop_class.return_value.__enter__ = MagicMock(return_value=fake_jetson)
        fake_jtop_class.return_value.__exit__ = MagicMock(return_value=False)

        monitor = JtopMonitor.__new__(JtopMonitor)
        monitor._jtop = None
        monitor._available = True
        monitor._jtop_class = fake_jtop_class

        snap = monitor.snapshot()
        assert snap is not None
        assert snap.cpu.overall_pct == 75.0  # 100 - 25 idle
        assert len(snap.cpu.per_core_pct) == 2
        assert snap.ram_used_mb == 4096.0
        assert snap.ram_total_mb == 8192.0
        assert snap.emc_utilization_pct == 42.0
        assert snap.emc_frequency_mhz == 2133.0
        assert snap.gpu.utilization_pct == 65.0
        assert len(snap.thermal_zones) == 2
        assert len(snap.power_readings) == 1
        assert snap.fan_speed_pct == 50.0
        assert snap.power_mode == "MAXN"

    def test_fallback_gpu_load_paths(self) -> None:
        """Verify FallbackMonitor reads GPU load from sysfs."""
        from hammerio.core.telemetry import FallbackMonitor
        monitor = FallbackMonitor()
        snap = monitor.snapshot()
        # On Jetson Orin this should find the correct sysfs path
        # On non-Jetson it falls back to nvidia-smi
        assert hasattr(snap.gpu, "utilization_pct")
        assert snap.gpu.utilization_pct >= 0.0
