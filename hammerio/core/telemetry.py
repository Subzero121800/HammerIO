"""Tegrastats integration and system telemetry for HammerIO.

Provides real-time thermal, power, GPU utilization, and memory monitoring.
Integrates with jtop (jetson-stats) when available, falls back to
tegrastats and /sys/class/thermal for raw data.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("hammerio.telemetry")


@dataclass
class ThermalZone:
    """A single thermal sensor reading."""

    name: str
    temperature_c: float
    zone_path: str = ""


@dataclass
class PowerReading:
    """Power consumption reading."""

    rail_name: str
    current_mw: float
    average_mw: float = 0.0


@dataclass
class GpuMetrics:
    """GPU utilization and memory metrics."""

    utilization_pct: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    frequency_mhz: float = 0.0
    encoder_utilization_pct: float = 0.0
    decoder_utilization_pct: float = 0.0

    @property
    def memory_pct(self) -> float:
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100


@dataclass
class CpuMetrics:
    """CPU utilization metrics."""

    overall_pct: float = 0.0
    per_core_pct: list[float] = field(default_factory=list)
    frequency_mhz: float = 0.0


@dataclass
class SystemSnapshot:
    """Complete system telemetry snapshot."""

    timestamp: float
    thermal_zones: list[ThermalZone] = field(default_factory=list)
    power_readings: list[PowerReading] = field(default_factory=list)
    gpu: GpuMetrics = field(default_factory=GpuMetrics)
    cpu: CpuMetrics = field(default_factory=CpuMetrics)
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    swap_used_mb: float = 0.0
    swap_total_mb: float = 0.0
    emc_utilization_pct: float = 0.0
    emc_frequency_mhz: float = 0.0
    power_mode: str = ""
    fan_speed_pct: float = 0.0
    is_throttled: bool = False
    throttle_reason: str = ""

    @property
    def max_temperature(self) -> float:
        if not self.thermal_zones:
            return 0.0
        return max(tz.temperature_c for tz in self.thermal_zones)

    @property
    def total_power_mw(self) -> float:
        return sum(pr.current_mw for pr in self.power_readings)

    @property
    def ram_pct(self) -> float:
        if self.ram_total_mb == 0:
            return 0.0
        return (self.ram_used_mb / self.ram_total_mb) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "max_temp_c": self.max_temperature,
            "total_power_mw": self.total_power_mw,
            "gpu_util_pct": self.gpu.utilization_pct,
            "gpu_mem_pct": self.gpu.memory_pct,
            "cpu_pct": self.cpu.overall_pct,
            "ram_pct": self.ram_pct,
            "emc_util_pct": self.emc_utilization_pct,
            "emc_freq_mhz": self.emc_frequency_mhz,
            "is_throttled": self.is_throttled,
            "power_mode": self.power_mode,
            "thermal_zones": [
                {"name": tz.name, "temp_c": tz.temperature_c}
                for tz in self.thermal_zones
            ],
            "power_rails": [
                {"name": pr.rail_name, "mw": pr.current_mw}
                for pr in self.power_readings
            ],
        }


class JtopMonitor:
    """Integration with jtop (jetson-stats) for Jetson-specific monitoring.

    Wraps jtop's Python API when available, providing HammerIO-specific
    metrics views optimized for compression workload monitoring.
    """

    def __init__(self) -> None:
        self._jtop: Any = None
        self._available = False
        self._check_jtop()

    def _check_jtop(self) -> None:
        try:
            from jtop import jtop  # type: ignore[import-untyped]
            self._jtop_class = jtop
            self._available = True
            logger.info("jtop (jetson-stats) available")
        except ImportError:
            logger.info("jtop not available — using fallback telemetry")

    @property
    def available(self) -> bool:
        return self._available

    def snapshot(self) -> Optional[SystemSnapshot]:
        """Get a single snapshot using jtop.

        jtop API (jetson-stats 4.x):
        - jetson.cpu: dict with 'total' and 'cpu' (per-core list)
        - jetson.memory: Memory object
        - jetson.temperature: dict of {name: {temp, online}}
        - jetson.power: dict with 'rail' and 'tot'
        - jetson.gpu: GPU object
        - jetson.fan: Fan object
        """
        if not self._available:
            return None

        try:
            with self._jtop_class() as jetson:
                snap = SystemSnapshot(timestamp=time.time())

                # CPU — dict with 'total' (overall) and 'cpu' (per-core list)
                try:
                    cpu_info = jetson.cpu
                    if cpu_info and isinstance(cpu_info, dict):
                        total = cpu_info.get("total", {})
                        if isinstance(total, dict):
                            snap.cpu.overall_pct = 100.0 - total.get("idle", 0)
                        cores = cpu_info.get("cpu", [])
                        if isinstance(cores, list):
                            snap.cpu.per_core_pct = [
                                100.0 - c.get("idle", 0)
                                for c in cores if isinstance(c, dict)
                            ]
                except Exception:
                    pass

                # Memory — dict-like with "RAM", "SWAP", "EMC" keys
                # Values are in KB
                try:
                    mem = jetson.memory
                    if mem is not None:
                        ram = mem.get("RAM", mem.get("ram", {}))
                        if isinstance(ram, dict):
                            snap.ram_used_mb = ram.get("used", 0) / 1024  # KB to MB
                            snap.ram_total_mb = ram.get("tot", 0) / 1024
                        swap = mem.get("SWAP", mem.get("swap", {}))
                        if isinstance(swap, dict):
                            snap.swap_used_mb = swap.get("used", 0) / 1024
                            snap.swap_total_mb = swap.get("tot", 0) / 1024
                        # EMC (external memory controller) utilization
                        emc = mem.get("EMC", mem.get("emc", {}))
                        if isinstance(emc, dict):
                            snap.emc_utilization_pct = float(emc.get("val", 0))
                            # cur is frequency in Hz or kHz depending on jtop version
                            emc_freq = emc.get("cur", 0)
                            if emc_freq > 1e6:
                                snap.emc_frequency_mhz = emc_freq / 1e6
                            else:
                                snap.emc_frequency_mhz = float(emc_freq)
                except Exception:
                    pass

                # Temperature — dict of {name: {temp, online}}
                try:
                    temp = jetson.temperature
                    if temp and isinstance(temp, dict):
                        for name, val in temp.items():
                            temp_c = 0.0
                            if isinstance(val, dict):
                                if not val.get("online", True):
                                    continue
                                temp_c = val.get("temp", 0)
                            elif isinstance(val, (int, float)):
                                temp_c = float(val)
                            if 0 < temp_c < 120:
                                snap.thermal_zones.append(ThermalZone(
                                    name=name, temperature_c=temp_c,
                                ))
                except Exception:
                    pass

                # Power — dict with 'rail' (per-rail) and 'tot' (totals)
                try:
                    power = jetson.power
                    if power and isinstance(power, dict):
                        rails = power.get("rail", {})
                        for name, data in rails.items():
                            if isinstance(data, dict) and data.get("online", False):
                                snap.power_readings.append(PowerReading(
                                    rail_name=name,
                                    current_mw=data.get("power", 0),
                                    average_mw=data.get("avg", 0),
                                ))
                except Exception:
                    pass

                # GPU — GPU object
                try:
                    gpu = jetson.gpu
                    if gpu is not None:
                        if hasattr(gpu, "val"):
                            snap.gpu.utilization_pct = float(gpu.val)
                        if hasattr(gpu, "frq"):
                            f = gpu.frq
                            snap.gpu.frequency_mhz = f / 1e6 if f > 1e6 else float(f)
                except Exception:
                    pass

                # Fan
                try:
                    fan = jetson.fan
                    if fan is not None:
                        if hasattr(fan, "speed"):
                            snap.fan_speed_pct = float(fan.speed) if fan.speed else 0.0
                except Exception:
                    pass

                # Power mode
                try:
                    if hasattr(jetson, "nvpmodel") and jetson.nvpmodel:
                        snap.power_mode = str(jetson.nvpmodel)
                except Exception:
                    pass

                # Throttling
                snap.is_throttled = snap.max_temperature > 85.0
                if snap.is_throttled:
                    snap.throttle_reason = f"Thermal throttle: {snap.max_temperature:.1f}°C"

                return snap
        except Exception as e:
            logger.warning("jtop snapshot failed: %s", e)
            return None


class FallbackMonitor:
    """Fallback telemetry using /sys, tegrastats, and nvidia-smi."""

    def __init__(self) -> None:
        # Cache thermal zone paths and names (they don't change at runtime)
        self._thermal_zones: list[tuple[str, str]] = []  # (temp_path, name)
        thermal_base = Path("/sys/class/thermal")
        if thermal_base.exists():
            for tz_path in sorted(thermal_base.glob("thermal_zone*")):
                temp_path = str(tz_path / "temp")
                name = "unknown"
                type_file = tz_path / "type"
                try:
                    if type_file.exists():
                        with open(type_file) as f:
                            name = f.read().strip()
                except OSError:
                    pass
                self._thermal_zones.append((temp_path, name))

        # Cache GPU load sysfs path
        self._gpu_load_path: Optional[str] = None
        for candidate in (
            "/sys/devices/platform/bus@0/17000000.gpu/load",
            "/sys/devices/platform/17000000.gpu/load",
            "/sys/devices/platform/17000000.gv11b/load",
            "/sys/devices/platform/gpu.0/load",
        ):
            if Path(candidate).exists():
                self._gpu_load_path = candidate
                break

    def snapshot(self) -> SystemSnapshot:
        snap = SystemSnapshot(timestamp=time.time())

        # Thermal zones — use cached paths, explicit open/close
        for temp_path, name in self._thermal_zones:
            try:
                with open(temp_path, "rb") as f:
                    raw = f.read()
                temp_raw = raw.decode("utf-8", errors="ignore").strip()
                temp_c = float(temp_raw) / 1000.0
                if 0 < temp_c < 120:
                    snap.thermal_zones.append(ThermalZone(
                        name=name, temperature_c=temp_c, zone_path=temp_path,
                    ))
            except (OSError, ValueError):
                continue

        # Power (Jetson INA3221 sensors)
        power_base = Path("/sys/bus/i2c/drivers/ina3221")
        if power_base.exists():
            for sensor_path in power_base.glob("*/hwmon/hwmon*/"):
                try:
                    for power_file in sorted(sensor_path.glob("in*_input")):
                        name = power_file.stem
                        label_file = power_file.with_name(power_file.stem.replace("_input", "_label"))
                        if label_file.exists():
                            name = label_file.read_text().strip()
                        val = float(power_file.read_text().strip())
                        snap.power_readings.append(PowerReading(rail_name=name, current_mw=val))
                except (OSError, ValueError, TypeError, UnicodeDecodeError, AttributeError):
                    continue

        # RAM via /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            for line in meminfo.splitlines():
                if line.startswith("MemTotal:"):
                    snap.ram_total_mb = int(line.split()[1]) / 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1024
                    snap.ram_used_mb = snap.ram_total_mb - avail
                elif line.startswith("SwapTotal:"):
                    snap.swap_total_mb = int(line.split()[1]) / 1024
                elif line.startswith("SwapFree:"):
                    snap.swap_used_mb = snap.swap_total_mb - int(line.split()[1]) / 1024
        except (OSError, ValueError, TypeError, UnicodeDecodeError):
            pass

        # CPU utilization via /proc/stat
        try:
            stat = Path("/proc/stat").read_text()
            for line in stat.splitlines():
                if line.startswith("cpu "):
                    parts = line.split()
                    idle = int(parts[4])
                    total = sum(int(x) for x in parts[1:])
                    snap.cpu.overall_pct = (1 - idle / total) * 100 if total > 0 else 0
                    break
        except (OSError, ValueError, IndexError):
            pass

        # GPU via cached sysfs path or nvidia-smi
        if self._gpu_load_path is not None:
            try:
                with open(self._gpu_load_path) as f:
                    load = int(f.read().strip())
                snap.gpu.utilization_pct = load / 10.0  # Jetson reports 0-1000
            except (OSError, ValueError, TypeError, UnicodeDecodeError):
                pass
        else:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    parts = [p.strip() for p in result.stdout.strip().split(",")]
                    if len(parts) >= 3:
                        snap.gpu.utilization_pct = float(parts[0])
                        snap.gpu.memory_used_mb = float(parts[1])
                        snap.gpu.memory_total_mb = float(parts[2])
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                pass

        # Power mode (Jetson)
        try:
            result = subprocess.run(
                ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                snap.power_mode = result.stdout.strip().split("\n")[-1].strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Throttling
        snap.is_throttled = snap.max_temperature > 85.0
        if snap.is_throttled:
            snap.throttle_reason = f"Thermal: {snap.max_temperature:.1f}°C exceeds 85°C threshold"

        return snap


class TelemetryCollector:
    """Continuous telemetry collection with history and alerting.

    Designed for integration with HammerIO's web dashboard and CLI.
    Automatically chooses jtop or fallback based on availability.
    """

    def __init__(self, interval_seconds: float = 1.0, history_size: int = 300) -> None:
        self._jtop_monitor = JtopMonitor()
        self._fallback_monitor = FallbackMonitor()
        self._interval = interval_seconds
        self._history: list[SystemSnapshot] = []
        self._history_size = history_size
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[SystemSnapshot], None]] = []
        self._alert_callbacks: list[Callable[[str, SystemSnapshot], None]] = []

        # Thresholds for alerting
        self.thermal_warning_c = 80.0
        self.thermal_critical_c = 90.0
        self.gpu_util_warning_pct = 95.0
        self.ram_warning_pct = 90.0

    @property
    def using_jtop(self) -> bool:
        return self._jtop_monitor.available

    def get_snapshot(self) -> SystemSnapshot:
        """Get a single telemetry snapshot."""
        if self._jtop_monitor.available:
            snap = self._jtop_monitor.snapshot()
            if snap:
                return snap
        return self._fallback_monitor.snapshot()

    def add_callback(self, callback: Callable[[SystemSnapshot], None]) -> None:
        """Register a callback for each telemetry update."""
        self._callbacks.append(callback)

    def add_alert_callback(self, callback: Callable[[str, SystemSnapshot], None]) -> None:
        """Register a callback for alert conditions."""
        self._alert_callbacks.append(callback)

    def start(self) -> None:
        """Start continuous background telemetry collection."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True, name="hammerio-telemetry")
        self._thread.start()
        logger.info("Telemetry collection started (interval=%.1fs, jtop=%s)", self._interval, self.using_jtop)

    def stop(self) -> None:
        """Stop telemetry collection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Telemetry collection stopped")

    def _collect_loop(self) -> None:
        while self._running:
            try:
                snap = self.get_snapshot()
                with self._lock:
                    self._history.append(snap)
                    if len(self._history) > self._history_size:
                        self._history = self._history[-self._history_size:]

                # Fire callbacks
                for cb in self._callbacks:
                    try:
                        cb(snap)
                    except Exception as e:
                        logger.warning("Telemetry callback error: %s", e)

                # Check alerts
                self._check_alerts(snap)

            except Exception as e:
                logger.error("Telemetry collection error: %s", e)

            time.sleep(self._interval)

    def _check_alerts(self, snap: SystemSnapshot) -> None:
        alerts: list[str] = []

        if snap.max_temperature >= self.thermal_critical_c:
            alerts.append(f"CRITICAL: Temperature {snap.max_temperature:.1f}°C exceeds {self.thermal_critical_c}°C")
        elif snap.max_temperature >= self.thermal_warning_c:
            alerts.append(f"WARNING: Temperature {snap.max_temperature:.1f}°C approaching limit")

        if snap.gpu.utilization_pct >= self.gpu_util_warning_pct:
            alerts.append(f"WARNING: GPU utilization at {snap.gpu.utilization_pct:.0f}%")

        if snap.ram_pct >= self.ram_warning_pct:
            alerts.append(f"WARNING: RAM usage at {snap.ram_pct:.0f}%")

        for alert in alerts:
            logger.warning(alert)
            for cb in self._alert_callbacks:
                try:
                    cb(alert, snap)
                except Exception as e:
                    logger.warning("Alert callback error: %s", e)

    def get_history(self, last_n: Optional[int] = None) -> list[SystemSnapshot]:
        """Get telemetry history."""
        with self._lock:
            if last_n:
                return list(self._history[-last_n:])
            return list(self._history)

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics from collected history."""
        with self._lock:
            history = list(self._history)

        if not history:
            return {"error": "No telemetry data collected"}

        temps = [s.max_temperature for s in history if s.max_temperature > 0]
        gpu_utils = [s.gpu.utilization_pct for s in history]
        cpu_utils = [s.cpu.overall_pct for s in history]
        power = [s.total_power_mw for s in history if s.total_power_mw > 0]

        def _stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {"min": 0, "max": 0, "avg": 0}
            return {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }

        return {
            "samples": len(history),
            "duration_s": history[-1].timestamp - history[0].timestamp if len(history) > 1 else 0,
            "temperature_c": _stats(temps),
            "gpu_utilization_pct": _stats(gpu_utils),
            "cpu_utilization_pct": _stats(cpu_utils),
            "power_mw": _stats(power),
            "throttle_events": sum(1 for s in history if s.is_throttled),
            "using_jtop": self.using_jtop,
        }

    def format_live_display(self, snap: Optional[SystemSnapshot] = None) -> str:
        """Format telemetry for terminal display."""
        if snap is None:
            snap = self.get_snapshot()

        lines = ["HammerIO System Monitor"]
        lines.append("━" * 50)

        # Temperature
        if snap.thermal_zones:
            lines.append("Thermal:")
            for tz in snap.thermal_zones[:6]:
                bar = "█" * int(tz.temperature_c / 5) + "░" * (20 - int(tz.temperature_c / 5))
                status = " ⚠" if tz.temperature_c > self.thermal_warning_c else ""
                lines.append(f"  {tz.name:20s} {tz.temperature_c:5.1f}°C {bar}{status}")

        # GPU
        lines.append(f"\nGPU: {snap.gpu.utilization_pct:5.1f}%  Mem: {snap.gpu.memory_pct:5.1f}%")

        # CPU
        lines.append(f"CPU: {snap.cpu.overall_pct:5.1f}%  Cores: {len(snap.cpu.per_core_pct)}")

        # RAM
        lines.append(f"RAM: {snap.ram_used_mb:,.0f} / {snap.ram_total_mb:,.0f} MB ({snap.ram_pct:.1f}%)")

        # Power
        if snap.power_readings:
            total = snap.total_power_mw
            lines.append(f"Power: {total:,.0f} mW")

        if snap.power_mode:
            lines.append(f"Mode: {snap.power_mode}")

        if snap.is_throttled:
            lines.append(f"\n⚠ THROTTLING: {snap.throttle_reason}")

        return "\n".join(lines)
