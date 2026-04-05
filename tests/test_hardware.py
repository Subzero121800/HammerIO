"""Tests for hardware detection module."""

from __future__ import annotations

from hammerio.core.hardware import (
    CudaDevice,
    GstreamerNvencCapability,
    HardwareProfile,
    NvcompCapability,
    NvdecCapability,
    NvencCapability,
    PlatformType,
    PowerMode,
    GpuVendor,
    VpiCapability,
    detect_cuda_device,
    detect_gstreamer_nvenc,
    detect_hardware,
    detect_nvenc,
    detect_nvdec,
    detect_thermal,
    format_hardware_report,
)


class TestHardwareDetection:
    def test_detect_hardware_returns_profile(self) -> None:
        profile = detect_hardware()
        assert isinstance(profile, HardwareProfile)
        assert profile.architecture in ("aarch64", "x86_64", "arm64")
        assert profile.cpu_cores > 0
        assert profile.total_ram_mb > 0

    def test_detect_hardware_caching(self) -> None:
        p1 = detect_hardware()
        p2 = detect_hardware()
        assert p1 is p2  # Should be cached

        detect_hardware.cache_clear()
        p3 = detect_hardware()
        assert isinstance(p3, HardwareProfile)

    def test_cuda_device_detection(self) -> None:
        device = detect_cuda_device()
        # May or may not have CUDA
        if device is not None:
            assert isinstance(device, CudaDevice)
            assert device.cuda_version != ""
            assert device.index == 0

    def test_nvenc_detection(self) -> None:
        nvenc = detect_nvenc()
        assert isinstance(nvenc, NvencCapability)
        if nvenc.available:
            assert len(nvenc.codecs) > 0

    def test_nvdec_detection(self) -> None:
        nvdec = detect_nvdec()
        assert isinstance(nvdec, NvdecCapability)

    def test_thermal_detection(self) -> None:
        temp = detect_thermal()
        if temp is not None:
            assert 0 < temp < 120

    def test_gstreamer_nvenc_detection(self) -> None:
        gst = detect_gstreamer_nvenc()
        assert isinstance(gst, GstreamerNvencCapability)
        # On Jetson with GStreamer this should find the encoders
        if gst.available:
            assert gst.has_h264 or gst.has_h265
            assert gst.gst_launch_path != ""

    def test_hardware_profile_properties(self) -> None:
        profile = detect_hardware()
        assert isinstance(profile.has_cuda, bool)
        assert isinstance(profile.has_nvenc, bool)
        assert isinstance(profile.has_gstreamer_nvenc, bool)
        assert isinstance(profile.has_nvcomp, bool)
        assert isinstance(profile.has_vpi, bool)
        assert isinstance(profile.gpu_memory_mb, int)

    def test_routing_summary(self) -> None:
        profile = detect_hardware()
        routes = profile.routing_summary()
        assert "general" in routes
        assert "large_files" in routes
        assert "datasets" in routes
        assert "general" in routes
        assert "text_logs" in routes

    def test_format_hardware_report(self) -> None:
        report = format_hardware_report()
        assert "HammerIO Hardware Profile" in report
        assert "Platform:" in report
        assert "Routing Profile:" in report
