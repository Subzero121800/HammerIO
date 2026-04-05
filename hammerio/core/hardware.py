"""Hardware detection and capability reporting for HammerIO.

Detects CUDA devices, NVENC/NVDEC support, nvCOMP availability,
VPI presence, and Jetson-specific features like unified memory
and power modes.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

import psutil


class GpuVendor(Enum):
    NVIDIA = "nvidia"
    NONE = "none"


class PlatformType(Enum):
    JETSON = "jetson"
    DESKTOP = "desktop"
    UNKNOWN = "unknown"


class PowerMode(Enum):
    MAXN = "MAXN"
    MODE_15W = "15W"
    MODE_30W = "30W"
    MODE_50W = "50W"
    UNKNOWN = "unknown"


@dataclass
class CudaDevice:
    """Represents a detected CUDA GPU device."""

    index: int
    name: str
    compute_capability: tuple[int, int]
    total_memory_mb: int
    driver_version: str
    cuda_version: str
    is_unified_memory: bool = False


@dataclass
class NvencCapability:
    """NVENC encoder capabilities."""

    available: bool = False
    codecs: list[str] = field(default_factory=list)  # e.g. ["h264", "hevc", "av1"]


@dataclass
class NvdecCapability:
    """NVDEC decoder capabilities."""

    available: bool = False
    codecs: list[str] = field(default_factory=list)


@dataclass
class NvcompCapability:
    """nvCOMP GPU compression library status."""

    available: bool = False
    algorithms: list[str] = field(default_factory=list)
    version: str = ""


@dataclass
class GstreamerNvencCapability:
    """GStreamer hardware encoder availability (Jetson V4L2 NVENC)."""

    available: bool = False
    has_h264: bool = False
    has_h265: bool = False
    has_h264parse: bool = False
    has_h265parse: bool = False
    gst_launch_path: str = ""


@dataclass
class VpiCapability:
    """NVIDIA VPI (Vision Programming Interface) status."""

    available: bool = False
    version: str = ""


@dataclass
class HardwareProfile:
    """Complete hardware profile for routing decisions."""

    platform_type: PlatformType
    platform_name: str
    architecture: str
    gpu_vendor: GpuVendor
    cuda_device: Optional[CudaDevice]
    nvenc: NvencCapability
    nvdec: NvdecCapability
    nvcomp: NvcompCapability
    vpi: VpiCapability
    gstreamer_nvenc: GstreamerNvencCapability
    cpu_cores: int
    cpu_freq_mhz: Optional[float]
    total_ram_mb: int
    power_mode: PowerMode
    thermal_celsius: Optional[float]
    jetpack_version: str = ""
    l4t_version: str = ""

    @property
    def has_cuda(self) -> bool:
        return self.cuda_device is not None

    @property
    def has_nvenc(self) -> bool:
        return self.nvenc.available

    @property
    def has_gstreamer_nvenc(self) -> bool:
        """GStreamer V4L2 NVENC available (Jetson preferred path)."""
        return self.gstreamer_nvenc.available

    @property
    def has_nvcomp(self) -> bool:
        return self.nvcomp.available

    @property
    def has_vpi(self) -> bool:
        return self.vpi.available

    @property
    def gpu_memory_mb(self) -> int:
        if self.cuda_device:
            return self.cuda_device.total_memory_mb
        return 0

    def routing_summary(self) -> dict[str, str]:
        """Return recommended compression processor for each file type."""
        routes: dict[str, str] = {}
        gpu_comp = "nvCOMP LZ4 (GPU)" if self.has_nvcomp else None
        routes["large_files"] = gpu_comp or "zstd parallel (CPU)"
        routes["datasets"] = gpu_comp or "zstd streaming (CPU)"
        routes["general"] = "zstd parallel (CPU)"
        routes["archives"] = "passthrough (already compressed)"
        routes["text_logs"] = "zstd (CPU, high ratio)"
        return routes


def _run_cmd(cmd: list[str], timeout: int = 10) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def detect_cuda_device() -> Optional[CudaDevice]:
    """Detect CUDA device using nvidia-smi and nvcc."""
    # Try pynvml first for accurate info
    try:
        import pynvml  # type: ignore[import-untyped]
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8")

        # Get CUDA version from nvcc
        cuda_ver = ""
        nvcc_out = _run_cmd(["nvcc", "--version"])
        if nvcc_out:
            m = re.search(r"release (\d+\.\d+)", nvcc_out)
            if m:
                cuda_ver = m.group(1)

        # Compute capability
        cc = (0, 0)
        try:
            cc_major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            if isinstance(cc_major, tuple):
                cc = cc_major
        except Exception:
            # Fallback: known Jetson compute capabilities
            if "Orin" in name:
                cc = (8, 7)
            elif "Xavier" in name:
                cc = (7, 2)
            elif "Nano" in name:
                cc = (5, 3)

        # Check unified memory (Jetson uses unified memory architecture)
        is_unified = _is_jetson_platform()

        total_mem_mb = mem_info.total // (1024 * 1024)

        # On Jetson unified memory, GPU shares system RAM
        # pynvml may report 0 — use system RAM as GPU memory
        if is_unified and total_mem_mb == 0:
            total_mem_mb = psutil.virtual_memory().total // (1024 * 1024)

        pynvml.nvmlShutdown()

        return CudaDevice(
            index=0,
            name=name,
            compute_capability=cc,
            total_memory_mb=total_mem_mb,
            driver_version=driver,
            cuda_version=cuda_ver,
            is_unified_memory=is_unified,
        )
    except Exception:
        pass

    # Fallback: parse nvidia-smi
    smi_out = _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if smi_out:
        parts = [p.strip() for p in smi_out.split(",")]
        if len(parts) >= 3:
            cuda_ver = ""
            nvcc_out = _run_cmd(["nvcc", "--version"])
            if nvcc_out:
                m = re.search(r"release (\d+\.\d+)", nvcc_out)
                if m:
                    cuda_ver = m.group(1)

            name = parts[0]
            cc = (0, 0)
            if "Orin" in name:
                cc = (8, 7)
            elif "Xavier" in name:
                cc = (7, 2)

            try:
                mem_mb = int(float(parts[1]))
            except ValueError:
                mem_mb = 0

            is_jetson = _is_jetson_platform()
            if is_jetson and mem_mb == 0:
                mem_mb = psutil.virtual_memory().total // (1024 * 1024)

            return CudaDevice(
                index=0,
                name=name,
                compute_capability=cc,
                total_memory_mb=mem_mb,
                driver_version=parts[2],
                cuda_version=cuda_ver,
                is_unified_memory=is_jetson,
            )

    # Final fallback: just check nvcc exists
    nvcc_out = _run_cmd(["nvcc", "--version"])
    if nvcc_out:
        cuda_ver = ""
        m = re.search(r"release (\d+\.\d+)", nvcc_out)
        if m:
            cuda_ver = m.group(1)
        is_jetson = _is_jetson_platform()
        mem_mb = psutil.virtual_memory().total // (1024 * 1024) if is_jetson else 0
        return CudaDevice(
            index=0,
            name="Unknown CUDA Device",
            compute_capability=(0, 0),
            total_memory_mb=mem_mb,
            driver_version="",
            cuda_version=cuda_ver,
            is_unified_memory=is_jetson,
        )

    return None


def _is_jetson_platform() -> bool:
    """Check if running on NVIDIA Jetson hardware."""
    # Check for Tegra release file
    if Path("/etc/nv_tegra_release").exists():
        return True
    # Check device tree
    dt_model = Path("/proc/device-tree/model")
    if dt_model.exists():
        try:
            model = dt_model.read_text(errors="ignore").lower()
            if "jetson" in model or "tegra" in model:
                return True
        except OSError:
            pass
    return False


def _get_jetson_model() -> str:
    """Get specific Jetson model name."""
    dt_model = Path("/proc/device-tree/model")
    if dt_model.exists():
        try:
            return dt_model.read_text(errors="ignore").strip().rstrip("\x00")
        except OSError:
            pass

    tegra = Path("/etc/nv_tegra_release")
    if tegra.exists():
        try:
            content = tegra.read_text()
            if "R36" in content:
                return "NVIDIA Jetson (JetPack 6.x)"
            if "R35" in content:
                return "NVIDIA Jetson (JetPack 5.x)"
            return "NVIDIA Jetson"
        except OSError:
            pass

    return "Unknown"


def _get_l4t_version() -> str:
    """Get L4T (Linux for Tegra) version."""
    tegra = Path("/etc/nv_tegra_release")
    if tegra.exists():
        try:
            content = tegra.read_text()
            m = re.search(r"R(\d+)\s+\(release\),\s+REVISION:\s+([\d.]+)", content)
            if m:
                return f"R{m.group(1)} rev {m.group(2)}"
        except OSError:
            pass
    return ""


def _get_jetpack_version() -> str:
    """Detect JetPack SDK version."""
    # Try apt
    dpkg_out = _run_cmd(["dpkg-query", "--showformat=${Version}", "--show", "nvidia-jetpack"])
    if dpkg_out:
        return dpkg_out.strip()
    # Infer from L4T
    l4t = _get_l4t_version()
    if "R36" in l4t:
        return "6.x"
    if "R35" in l4t:
        return "5.x"
    return ""


def _test_nvenc_works(ffmpeg_path: str, codec: str) -> bool:
    """Test if NVENC actually works by encoding a tiny test frame."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        result = _run_cmd([
            ffmpeg_path, "-y", "-f", "lavfi",
            "-i", "color=black:s=64x64:d=0.1:r=1",
            "-c:v", codec, "-frames:v", "1",
            tmp.name,
        ], timeout=15)
        return result is not None


def detect_nvenc() -> NvencCapability:
    """Detect NVENC hardware encoder availability via FFmpeg.

    Checks both that FFmpeg lists the encoder AND that it actually works
    at runtime (Jetson may list nvenc but lack runtime libraries).
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return NvencCapability()

    out = _run_cmd([ffmpeg_path, "-hide_banner", "-encoders"])
    if not out:
        return NvencCapability()

    # Check which codecs are listed
    listed_codecs: list[str] = []
    if "h264_nvenc" in out:
        listed_codecs.append("h264")
    if "hevc_nvenc" in out:
        listed_codecs.append("hevc")
    if "av1_nvenc" in out:
        listed_codecs.append("av1")

    if not listed_codecs:
        return NvencCapability()

    # Runtime verification — test that NVENC actually works
    working_codecs: list[str] = []
    for codec in listed_codecs:
        encoder_name = f"{codec}_nvenc"
        if _test_nvenc_works(ffmpeg_path, encoder_name):
            working_codecs.append(codec)

    if not working_codecs and listed_codecs:
        # NVENC listed but doesn't work (common on Jetson with distro FFmpeg)
        # Log this for debugging
        import logging
        logging.getLogger("hammerio.hardware").info(
            "NVENC encoders listed by FFmpeg (%s) but runtime test failed. "
            "Install NVIDIA-compiled FFmpeg for hardware encoding.",
            ", ".join(listed_codecs),
        )

    return NvencCapability(available=len(working_codecs) > 0, codecs=working_codecs)


def detect_nvdec() -> NvdecCapability:
    """Detect NVDEC hardware decoder availability via FFmpeg."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return NvdecCapability()

    out = _run_cmd([ffmpeg_path, "-hide_banner", "-decoders"])
    if not out:
        return NvdecCapability()

    codecs: list[str] = []
    for codec in ["h264_cuvid", "hevc_cuvid", "vp9_cuvid", "av1_cuvid", "mpeg4_cuvid"]:
        if codec in out:
            codecs.append(codec.replace("_cuvid", ""))

    return NvdecCapability(available=len(codecs) > 0, codecs=codecs)


def detect_nvcomp() -> NvcompCapability:
    """Detect nvCOMP GPU compression library."""
    # Check for nvcomp shared library
    lib_paths = [
        "/usr/local/lib",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
    ]
    found = False
    for lp in lib_paths:
        if list(Path(lp).glob("libnvcomp*")) if Path(lp).exists() else []:
            found = True
            break

    # Check for Python bindings
    try:
        import kvikio  # type: ignore[import-untyped]
        found = True
    except ImportError:
        pass

    if found:
        algorithms = ["lz4", "snappy", "zstd", "deflate", "gdeflate", "bitcomp", "ans"]
        return NvcompCapability(available=True, algorithms=algorithms)

    return NvcompCapability()


def detect_vpi() -> VpiCapability:
    """Detect NVIDIA VPI (Vision Programming Interface)."""
    try:
        import vpi  # type: ignore[import-untyped]
        version = getattr(vpi, "__version__", "unknown")
        return VpiCapability(available=True, version=str(version))
    except ImportError:
        pass

    # Check for VPI shared lib
    for lib_dir in ["/opt/nvidia/vpi2/lib64", "/opt/nvidia/vpi3/lib64", "/usr/lib/aarch64-linux-gnu"]:
        p = Path(lib_dir)
        if p.exists() and list(p.glob("libnvvpi*")):
            m = re.search(r"vpi(\d+)", lib_dir)
            ver = f"{m.group(1)}.x" if m else "unknown"
            return VpiCapability(available=True, version=ver)

    return VpiCapability()


def detect_gstreamer_nvenc() -> GstreamerNvencCapability:
    """Detect GStreamer V4L2 NVENC encoders (Jetson hardware encoding).

    On Jetson, the distro FFmpeg typically lacks working NVENC support.
    GStreamer ships with ``nvv4l2h264enc`` / ``nvv4l2h265enc`` which use
    the same V4L2-based hardware encoder and work out of the box.
    """
    gst_launch = shutil.which("gst-launch-1.0")
    gst_inspect = shutil.which("gst-inspect-1.0")
    if not gst_launch or not gst_inspect:
        return GstreamerNvencCapability()

    has_h264 = _run_cmd([gst_inspect, "nvv4l2h264enc"]) is not None
    has_h265 = _run_cmd([gst_inspect, "nvv4l2h265enc"]) is not None

    if not has_h264 and not has_h265:
        return GstreamerNvencCapability()

    has_h264parse = _run_cmd([gst_inspect, "h264parse"]) is not None
    has_h265parse = _run_cmd([gst_inspect, "h265parse"]) is not None

    return GstreamerNvencCapability(
        available=True,
        has_h264=has_h264,
        has_h265=has_h265,
        has_h264parse=has_h264parse,
        has_h265parse=has_h265parse,
        gst_launch_path=gst_launch,
    )


def detect_power_mode() -> PowerMode:
    """Detect Jetson power mode via nvpmodel."""
    out = _run_cmd(["nvpmodel", "-q"])
    if out:
        if "MAXN" in out.upper():
            return PowerMode.MAXN
        m = re.search(r"(\d+)W", out)
        if m:
            watts = m.group(1)
            for mode in PowerMode:
                if watts in mode.value:
                    return mode
    return PowerMode.UNKNOWN


def detect_thermal() -> Optional[float]:
    """Read current thermal zone temperature."""
    # Try Jetson thermal zones first
    for tz_path in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            temp_str = (tz_path / "temp").read_text().strip()
            temp = float(temp_str) / 1000.0
            if 10.0 < temp < 120.0:
                return temp
        except (OSError, ValueError):
            continue

    # Try nvidia-smi
    out = _run_cmd(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
    if out:
        try:
            return float(out.strip())
        except ValueError:
            pass

    return None


@lru_cache(maxsize=1)
def detect_hardware() -> HardwareProfile:
    """Perform full hardware detection and return a HardwareProfile.

    Results are cached for the process lifetime. Call
    detect_hardware.cache_clear() to force re-detection.
    """
    is_jetson = _is_jetson_platform()
    arch = platform.machine()

    if is_jetson:
        platform_type = PlatformType.JETSON
        platform_name = _get_jetson_model()
    elif arch == "x86_64":
        platform_type = PlatformType.DESKTOP
        platform_name = "Desktop/Server x86_64"
    else:
        platform_type = PlatformType.UNKNOWN
        platform_name = f"Unknown ({arch})"

    cuda_device = detect_cuda_device()
    gpu_vendor = GpuVendor.NVIDIA if cuda_device else GpuVendor.NONE

    # CPU info
    cpu_cores = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    try:
        freq = psutil.cpu_freq()
        cpu_freq = freq.current if freq else None
    except Exception:
        cpu_freq = None

    ram = psutil.virtual_memory()

    return HardwareProfile(
        platform_type=platform_type,
        platform_name=platform_name,
        architecture=arch,
        gpu_vendor=gpu_vendor,
        cuda_device=cuda_device,
        nvenc=detect_nvenc(),
        nvdec=detect_nvdec(),
        nvcomp=detect_nvcomp(),
        vpi=detect_vpi(),
        gstreamer_nvenc=detect_gstreamer_nvenc(),
        cpu_cores=cpu_cores,
        cpu_freq_mhz=cpu_freq,
        total_ram_mb=ram.total // (1024 * 1024),
        power_mode=detect_power_mode() if is_jetson else PowerMode.UNKNOWN,
        thermal_celsius=detect_thermal(),
        jetpack_version=_get_jetpack_version() if is_jetson else "",
        l4t_version=_get_l4t_version() if is_jetson else "",
    )


def format_hardware_report(profile: Optional[HardwareProfile] = None) -> str:
    """Format a rich text hardware report for terminal display."""
    if profile is None:
        profile = detect_hardware()

    lines: list[str] = []
    lines.append("HammerIO Hardware Profile")
    lines.append("━" * 45)

    lines.append(f"  Platform:     {profile.platform_name}")
    lines.append(f"  Architecture: {profile.architecture}")

    if profile.cuda_device:
        cd = profile.cuda_device
        cc_str = f"{cd.compute_capability[0]}.{cd.compute_capability[1]}"
        lines.append(f"  CUDA:         {cd.cuda_version} (compute {cc_str})")
        mem_label = "unified" if cd.is_unified_memory else "dedicated"
        lines.append(f"  GPU Memory:   {cd.total_memory_mb:,} MB {mem_label}")
    else:
        lines.append("  CUDA:         Not available")

    # NVENC
    if profile.nvenc.available:
        lines.append(f"  NVENC:        Available ({', '.join(profile.nvenc.codecs)})")
    else:
        lines.append("  NVENC:        Not available")

    # GStreamer NVENC
    gst = profile.gstreamer_nvenc
    if gst.available:
        codecs = []
        if gst.has_h264:
            codecs.append("h264")
        if gst.has_h265:
            codecs.append("h265")
        lines.append(f"  GStreamer HW:  Available ({', '.join(codecs)})")
    else:
        lines.append("  GStreamer HW:  Not available")

    # NVDEC
    if profile.nvdec.available:
        lines.append(f"  NVDEC:        Available ({', '.join(profile.nvdec.codecs)})")
    else:
        lines.append("  NVDEC:        Not available")

    # nvCOMP
    if profile.nvcomp.available:
        algos = ", ".join(profile.nvcomp.algorithms[:4])
        lines.append(f"  nvCOMP:       Available ({algos})")
    else:
        lines.append("  nvCOMP:       Not available")

    # VPI
    if profile.vpi.available:
        lines.append(f"  VPI:          Available (v{profile.vpi.version})")
    else:
        lines.append("  VPI:          Not available")

    # CPU
    freq_str = f" ({profile.cpu_freq_mhz:,.0f} MHz)" if profile.cpu_freq_mhz else ""
    lines.append(f"  CPU Cores:    {profile.cpu_cores}{freq_str}")
    lines.append(f"  RAM:          {profile.total_ram_mb:,} MB")

    # Jetson-specific
    if profile.platform_type == PlatformType.JETSON:
        lines.append(f"  Power Mode:   {profile.power_mode.value}")
        if profile.l4t_version:
            lines.append(f"  L4T:          {profile.l4t_version}")
        if profile.jetpack_version:
            lines.append(f"  JetPack:      {profile.jetpack_version}")

    if profile.thermal_celsius is not None:
        status = "nominal" if profile.thermal_celsius < 80 else "WARNING: HIGH"
        lines.append(f"  Thermal:      {profile.thermal_celsius:.1f}°C ({status})")

    lines.append("")
    lines.append("Routing Profile:")
    for workload, processor in profile.routing_summary().items():
        label = workload.replace("_", " ").title()
        lines.append(f"  {label:14s}→ {processor}")

    return "\n".join(lines)
