"""Tests for telemetry collector start/stop, history, alerts, and summary."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from hammerio.core.telemetry import (
    CpuMetrics,
    GpuMetrics,
    PowerReading,
    SystemSnapshot,
    TelemetryCollector,
    ThermalZone,
)


class TestTelemetryCollectorStartStop:
    """Verify the background collection thread starts and stops cleanly."""

    def test_start_stop(self) -> None:
        collector = TelemetryCollector(interval_seconds=0.05, history_size=10)
        assert not collector._running
        collector.start()
        assert collector._running
        # Let it collect a few snapshots
        time.sleep(0.2)
        collector.stop()
        assert not collector._running
        assert len(collector.get_history()) > 0

    def test_double_start_ignored(self) -> None:
        collector = TelemetryCollector(interval_seconds=0.05)
        collector.start()
        first_thread = collector._thread
        collector.start()  # should be no-op
        assert collector._thread is first_thread
        collector.stop()

    def test_stop_without_start(self) -> None:
        collector = TelemetryCollector()
        # Should not raise
        collector.stop()

    def test_history_limited_by_size(self) -> None:
        collector = TelemetryCollector(interval_seconds=0.02, history_size=5)
        collector.start()
        time.sleep(0.25)
        collector.stop()
        assert len(collector.get_history()) <= 5

    def test_get_history_last_n(self) -> None:
        collector = TelemetryCollector(interval_seconds=0.02, history_size=50)
        collector.start()
        time.sleep(0.2)
        collector.stop()
        full = collector.get_history()
        partial = collector.get_history(last_n=2)
        assert len(partial) <= 2
        if len(full) >= 2:
            assert partial[-1].timestamp == full[-1].timestamp

    def test_callback_fires(self) -> None:
        received: list[SystemSnapshot] = []
        collector = TelemetryCollector(interval_seconds=0.05, history_size=10)
        collector.add_callback(lambda snap: received.append(snap))
        collector.start()
        time.sleep(0.2)
        collector.stop()
        assert len(received) > 0
        assert all(isinstance(s, SystemSnapshot) for s in received)

    def test_alert_callback_fires_on_high_temp(self) -> None:
        alerts: list[str] = []
        collector = TelemetryCollector(interval_seconds=1.0)
        collector.add_alert_callback(lambda msg, snap: alerts.append(msg))
        # Manually trigger alert check with a hot snapshot
        snap = SystemSnapshot(timestamp=time.time())
        snap.thermal_zones = [ThermalZone(name="test", temperature_c=95.0)]
        collector._check_alerts(snap)
        assert any("CRITICAL" in a for a in alerts)

    def test_alert_callback_warning_temp(self) -> None:
        alerts: list[str] = []
        collector = TelemetryCollector(interval_seconds=1.0)
        collector.add_alert_callback(lambda msg, snap: alerts.append(msg))
        snap = SystemSnapshot(timestamp=time.time())
        snap.thermal_zones = [ThermalZone(name="test", temperature_c=82.0)]
        collector._check_alerts(snap)
        assert any("WARNING" in a and "Temperature" in a for a in alerts)

    def test_alert_callback_high_ram(self) -> None:
        alerts: list[str] = []
        collector = TelemetryCollector(interval_seconds=1.0)
        collector.add_alert_callback(lambda msg, snap: alerts.append(msg))
        snap = SystemSnapshot(timestamp=time.time())
        snap.ram_used_mb = 9500.0
        snap.ram_total_mb = 10000.0
        collector._check_alerts(snap)
        assert any("RAM" in a for a in alerts)


class TestTelemetrySummary:
    """Test get_summary with actual collected data."""

    def test_summary_with_data(self) -> None:
        collector = TelemetryCollector(interval_seconds=0.02, history_size=50)
        collector.start()
        time.sleep(0.15)
        collector.stop()
        summary = collector.get_summary()
        assert "samples" in summary
        assert summary["samples"] > 0
        assert "gpu_utilization_pct" in summary
        assert "temperature_c" in summary

    def test_using_jtop_property(self) -> None:
        collector = TelemetryCollector()
        # On most non-Jetson systems this will be False; on Jetson True.
        assert isinstance(collector.using_jtop, bool)


class TestSystemSnapshotProperties:
    """Test computed properties on SystemSnapshot."""

    def test_max_temperature_empty(self) -> None:
        snap = SystemSnapshot(timestamp=1.0)
        assert snap.max_temperature == 0.0

    def test_max_temperature(self) -> None:
        snap = SystemSnapshot(timestamp=1.0)
        snap.thermal_zones = [
            ThermalZone(name="a", temperature_c=55.0),
            ThermalZone(name="b", temperature_c=70.0),
            ThermalZone(name="c", temperature_c=60.0),
        ]
        assert snap.max_temperature == 70.0

    def test_total_power_mw(self) -> None:
        snap = SystemSnapshot(timestamp=1.0)
        snap.power_readings = [
            PowerReading(rail_name="vdd", current_mw=3000),
            PowerReading(rail_name="gpu", current_mw=2000),
        ]
        assert snap.total_power_mw == 5000.0

    def test_ram_pct_zero_total(self) -> None:
        snap = SystemSnapshot(timestamp=1.0)
        snap.ram_total_mb = 0
        assert snap.ram_pct == 0.0

    def test_ram_pct_normal(self) -> None:
        snap = SystemSnapshot(timestamp=1.0)
        snap.ram_used_mb = 4000.0
        snap.ram_total_mb = 8000.0
        assert snap.ram_pct == 50.0

    def test_gpu_memory_pct_zero_total(self) -> None:
        gpu = GpuMetrics(memory_total_mb=0)
        assert gpu.memory_pct == 0.0

    def test_gpu_memory_pct_normal(self) -> None:
        gpu = GpuMetrics(memory_used_mb=512, memory_total_mb=2048)
        assert gpu.memory_pct == 25.0

    def test_to_dict_completeness(self) -> None:
        snap = SystemSnapshot(timestamp=123.0)
        snap.thermal_zones = [ThermalZone(name="cpu", temperature_c=50.0)]
        snap.power_readings = [PowerReading(rail_name="vdd", current_mw=1000)]
        d = snap.to_dict()
        assert d["timestamp"] == 123.0
        assert len(d["thermal_zones"]) == 1
        assert len(d["power_rails"]) == 1
        assert "emc_util_pct" in d
        assert "is_throttled" in d


class TestTelemetryFormatDisplay:
    """Test the format_live_display method."""

    def test_format_with_snapshot(self) -> None:
        collector = TelemetryCollector()
        snap = SystemSnapshot(timestamp=time.time())
        snap.ram_used_mb = 4000.0
        snap.ram_total_mb = 8000.0
        snap.thermal_zones = [ThermalZone(name="cpu", temperature_c=55.0)]
        output = collector.format_live_display(snap)
        assert "HammerIO System Monitor" in output
        assert "RAM:" in output

    def test_format_without_snapshot(self) -> None:
        collector = TelemetryCollector()
        output = collector.format_live_display()
        assert "HammerIO System Monitor" in output
