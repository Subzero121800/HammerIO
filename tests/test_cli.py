"""Smoke tests for the HammerIO CLI."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _run_cli(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a CLI command via ``python -m hammerio`` and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "hammerio", *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=check,
    )


class TestCLISmoke:
    """Basic smoke tests — each command runs without a raw traceback."""

    def test_version(self) -> None:
        result = _run_cli("version")
        assert result.returncode == 0
        assert "HammerIO" in result.stdout

    def test_help(self) -> None:
        result = _run_cli("--help")
        assert result.returncode == 0
        assert "compress" in result.stdout

    def test_compress_help(self) -> None:
        result = _run_cli("compress", "--help")
        assert result.returncode == 0
        assert "input_path" in result.stdout.lower() or "INPUT_PATH" in result.stdout

    def test_decompress_help(self) -> None:
        result = _run_cli("decompress", "--help")
        assert result.returncode == 0

    def test_batch_help(self) -> None:
        result = _run_cli("batch", "--help")
        assert result.returncode == 0

    def test_info_hardware(self) -> None:
        result = _run_cli("info", "--hardware")
        assert result.returncode == 0
        assert "Platform" in result.stdout or "platform" in result.stdout.lower()

    def test_compress_missing_file(self) -> None:
        """Compressing a non-existent file should give a clean error, not a traceback."""
        result = _run_cli("compress", "/nonexistent/path/file.txt")
        assert result.returncode != 0
        # Must not leak a raw Python traceback
        assert "Traceback" not in result.stderr

    def test_decompress_missing_file(self) -> None:
        result = _run_cli("decompress", "/nonexistent/path/file.zst")
        assert result.returncode != 0
        assert "Traceback" not in result.stderr

    def test_batch_not_a_directory(self) -> None:
        result = _run_cli("batch", "/nonexistent/dir")
        assert result.returncode != 0
        assert "Traceback" not in result.stderr

    def test_config_show(self) -> None:
        result = _run_cli("config", "--show")
        assert result.returncode == 0

    def test_monitor_single_tick(self) -> None:
        """Monitor with --count 1 should exit cleanly."""
        result = _run_cli("monitor", "--count", "1")
        assert result.returncode == 0


class TestWebDashboard:
    """Test the web dashboard endpoints using Flask test client."""

    @pytest.fixture(autouse=True)
    def _setup_app(self) -> None:
        try:
            from hammerio.web.app import create_app
            self.app = create_app()
            self.client = self.app.test_client()
            self._available = True
        except ImportError:
            self._available = False

    def test_api_hardware(self) -> None:
        if not self._available:
            pytest.skip("Web dependencies not installed")
        r = self.client.get("/api/hardware")
        assert r.status_code == 200
        data = r.get_json()
        assert "platform" in data

    def test_api_telemetry(self) -> None:
        if not self._available:
            pytest.skip("Web dependencies not installed")
        r = self.client.get("/api/telemetry")
        assert r.status_code == 200

    def test_api_jtop(self) -> None:
        if not self._available:
            pytest.skip("Web dependencies not installed")
        r = self.client.get("/api/jtop")
        assert r.status_code == 200
        data = r.get_json()
        assert "gpu" in data
        assert "cpu" in data

    def test_api_telemetry_summary(self) -> None:
        if not self._available:
            pytest.skip("Web dependencies not installed")
        r = self.client.get("/api/telemetry/summary")
        assert r.status_code == 200

    def test_api_route(self) -> None:
        if not self._available:
            pytest.skip("Web dependencies not installed")
        # Create a temporary file so the router can actually inspect it
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("Hello HammerIO\n" * 100)
            tmp_path = f.name
        try:
            r = self.client.post("/api/route", json={"input_path": tmp_path})
            assert r.status_code == 200
            data = r.get_json()
            assert "explanation" in data
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_api_route_missing(self) -> None:
        if not self._available:
            pytest.skip("Web dependencies not installed")
        r = self.client.post("/api/route", json={"input_path": "/nonexistent/file.txt"})
        # Should return 500 with an error message, not crash
        assert r.status_code in (400, 500)

    def test_api_jobs(self) -> None:
        if not self._available:
            pytest.skip("Web dependencies not installed")
        r = self.client.get("/api/jobs")
        assert r.status_code == 200

    def test_health_endpoint(self) -> None:
        """Verify /health returns status, version, and uptime."""
        if not self._available:
            pytest.skip("Web dependencies not installed")
        r = self.client.get("/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert "uptime" in data
        assert isinstance(data["uptime"], float)
        assert data["uptime"] >= 0

    def test_cors_headers_present(self) -> None:
        """Verify CORS headers are set on API responses."""
        if not self._available:
            pytest.skip("Web dependencies not installed")
        r = self.client.get("/api/hardware")
        assert r.headers.get("Access-Control-Allow-Origin") == "*"
        assert "GET" in r.headers.get("Access-Control-Allow-Methods", "")
