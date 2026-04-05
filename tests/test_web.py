"""Comprehensive tests for the HammerIO web dashboard and API endpoints.

Tests API endpoints, JSON validity, error handling, and edge cases.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def web_client():
    """Create a Flask test client for the web dashboard."""
    try:
        from hammerio.web.app import create_app
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client
    except ImportError:
        pytest.skip("Web dependencies not installed")


class TestDashboardPage:
    """Verify the main dashboard HTML loads."""

    def test_dashboard_loads(self, web_client) -> None:
        r = web_client.get("/")
        assert r.status_code == 200
        assert len(r.data) > 100
        assert b"HammerIO" in r.data

    def test_dashboard_content_type(self, web_client) -> None:
        r = web_client.get("/")
        assert "text/html" in r.content_type


class TestApiHardware:
    """Test /api/hardware endpoint."""

    def test_returns_json(self, web_client) -> None:
        r = web_client.get("/api/hardware")
        assert r.status_code == 200
        data = r.get_json()
        assert isinstance(data, dict)

    def test_required_fields(self, web_client) -> None:
        r = web_client.get("/api/hardware")
        data = r.get_json()
        assert "platform" in data
        assert "architecture" in data
        assert "cuda" in data
        assert "nvenc" in data
        assert "cpu_cores" in data
        assert "ram_mb" in data
        assert "routes" in data

    def test_cuda_subfields(self, web_client) -> None:
        r = web_client.get("/api/hardware")
        cuda = r.get_json()["cuda"]
        assert "available" in cuda
        assert "memory_mb" in cuda


class TestApiTelemetry:
    """Test /api/telemetry endpoint."""

    def test_returns_json(self, web_client) -> None:
        r = web_client.get("/api/telemetry")
        assert r.status_code == 200
        data = r.get_json()
        assert isinstance(data, dict)

    def test_required_fields(self, web_client) -> None:
        r = web_client.get("/api/telemetry")
        data = r.get_json()
        assert "timestamp" in data
        assert "gpu_util_pct" in data
        assert "cpu_pct" in data
        assert "ram_pct" in data

    def test_telemetry_values_reasonable(self, web_client) -> None:
        r = web_client.get("/api/telemetry")
        data = r.get_json()
        assert data["timestamp"] > 0
        assert 0 <= data["gpu_util_pct"] <= 100
        assert 0 <= data["ram_pct"] <= 100


class TestApiTelemetryHistory:
    """Test /api/telemetry/history endpoint."""

    def test_returns_list(self, web_client) -> None:
        r = web_client.get("/api/telemetry/history")
        assert r.status_code == 200
        data = r.get_json()
        assert isinstance(data, list)

    def test_with_n_param(self, web_client) -> None:
        r = web_client.get("/api/telemetry/history?n=5")
        assert r.status_code == 200
        data = r.get_json()
        assert isinstance(data, list)


class TestApiTelemetrySummary:
    """Test /api/telemetry/summary endpoint."""

    def test_returns_json(self, web_client) -> None:
        r = web_client.get("/api/telemetry/summary")
        assert r.status_code == 200
        data = r.get_json()
        assert isinstance(data, dict)


class TestApiJtop:
    """Test /api/jtop endpoint."""

    def test_returns_json(self, web_client) -> None:
        r = web_client.get("/api/jtop")
        assert r.status_code == 200
        data = r.get_json()
        assert isinstance(data, dict)

    def test_required_sections(self, web_client) -> None:
        r = web_client.get("/api/jtop")
        data = r.get_json()
        assert "gpu" in data
        assert "cpu" in data
        assert "memory" in data
        assert "thermal" in data
        assert "power" in data
        assert "is_throttled" in data

    def test_gpu_subfields(self, web_client) -> None:
        data = web_client.get("/api/jtop").get_json()
        gpu = data["gpu"]
        assert "utilization" in gpu
        assert "memory_used_mb" in gpu
        assert "frequency_mhz" in gpu


class TestApiJobs:
    """Test /api/jobs endpoint."""

    def test_returns_list(self, web_client) -> None:
        r = web_client.get("/api/jobs")
        assert r.status_code == 200
        data = r.get_json()
        assert isinstance(data, list)


class TestApiRoute:
    """Test /api/route endpoint."""

    def test_route_valid_file(self, web_client, tmp_dir: Path) -> None:
        f = tmp_dir / "test.txt"
        f.write_text("test data\n" * 100)
        r = web_client.post("/api/route", json={"input_path": str(f)})
        assert r.status_code == 200
        data = r.get_json()
        assert "explanation" in data

    def test_route_missing_input_path(self, web_client) -> None:
        r = web_client.post("/api/route", json={})
        assert r.status_code == 400
        data = r.get_json()
        assert "error" in data

    def test_route_nonexistent_file(self, web_client) -> None:
        r = web_client.post("/api/route", json={"input_path": "/nonexistent/file.txt"})
        assert r.status_code in (400, 500)

    def test_route_empty_body(self, web_client) -> None:
        r = web_client.post(
            "/api/route",
            data="",
            content_type="application/json",
        )
        # Should return 400 or handle gracefully
        assert r.status_code in (400, 500)


class TestApiCompress:
    """Test /api/compress endpoint."""

    def test_compress_missing_input_path(self, web_client) -> None:
        r = web_client.post("/api/compress", json={})
        assert r.status_code == 400
        data = r.get_json()
        assert "error" in data

    def test_compress_nonexistent_file(self, web_client) -> None:
        r = web_client.post("/api/compress", json={"input_path": "/nonexistent/file.txt"})
        assert r.status_code == 500
        data = r.get_json()
        assert "error" in data

    def test_compress_valid_file(self, web_client, tmp_dir: Path) -> None:
        f = tmp_dir / "compress_test.txt"
        f.write_text("compressible data\n" * 5000)
        out = tmp_dir / "compress_test.txt.zst"
        r = web_client.post("/api/compress", json={
            "input_path": str(f),
            "output_path": str(out),
            "quality": "fast",
        })
        assert r.status_code == 200
        data = r.get_json()
        assert "output_path" in data
        assert data["status"] == "completed"


class TestApiBrowse:
    """Test /api/browse endpoint."""

    def test_browse_default(self, web_client) -> None:
        r = web_client.get("/api/browse")
        assert r.status_code == 200
        data = r.get_json()
        assert "path" in data
        assert "entries" in data

    def test_browse_specific_path(self, web_client, tmp_dir: Path) -> None:
        (tmp_dir / "a.txt").write_text("hello")
        (tmp_dir / "subdir").mkdir()
        r = web_client.get(f"/api/browse?path={tmp_dir}")
        assert r.status_code == 200
        data = r.get_json()
        names = [e["name"] for e in data["entries"]]
        assert "a.txt" in names

    def test_browse_path_traversal_blocked(self, web_client) -> None:
        r = web_client.get("/api/browse?path=/tmp/../etc/passwd")
        assert r.status_code == 400
        data = r.get_json()
        assert "error" in data

    def test_browse_nonexistent(self, web_client) -> None:
        r = web_client.get("/api/browse?path=/nonexistent_dir_12345")
        assert r.status_code == 404


class TestApiSystem:
    """Test /api/system endpoint."""

    def test_returns_json(self, web_client) -> None:
        r = web_client.get("/api/system")
        assert r.status_code == 200
        data = r.get_json()
        assert isinstance(data, dict)

    def test_required_fields(self, web_client) -> None:
        r = web_client.get("/api/system")
        data = r.get_json()
        assert "uptime" in data
        assert "disk" in data
        assert "hostname" in data
        assert "process_count" in data
        assert data["process_count"] > 0


class TestApiExportTelemetry:
    """Test /api/export/telemetry endpoint."""

    def test_returns_json_file(self, web_client) -> None:
        r = web_client.get("/api/export/telemetry")
        assert r.status_code == 200
        assert "application/json" in r.content_type
        assert "Content-Disposition" in r.headers
        # Should be valid JSON
        data = json.loads(r.data)
        assert isinstance(data, list)


class TestApiArchitecture:
    """Test /api/architecture.svg endpoint."""

    def test_returns_svg(self, web_client) -> None:
        r = web_client.get("/api/architecture.svg")
        assert r.status_code == 200
        assert "svg" in r.content_type.lower()


class TestAllEndpointsReturnValidJson:
    """Verify every GET API endpoint returns valid JSON or valid content."""

    ENDPOINTS = [
        "/api/hardware",
        "/api/telemetry",
        "/api/telemetry/history",
        "/api/telemetry/summary",
        "/api/jtop",
        "/api/jobs",
        "/api/system",
        "/api/browse",
    ]

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_endpoint_valid_json(self, web_client, endpoint: str) -> None:
        r = web_client.get(endpoint)
        assert r.status_code == 200
        # Must be parseable JSON
        data = json.loads(r.data)
        assert data is not None
