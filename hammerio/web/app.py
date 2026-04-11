"""HammerIO Web Dashboard — Flask-based monitoring and control.

Copyright 2026 ResilientMind AI | ResilientMindai.com | Joseph C McGinty Jr
Proprietary License — All Rights Reserved
"""

from __future__ import annotations

import json
import logging
import os
import platform
import secrets
import shlex
import shutil
import socket
import time
from pathlib import Path
from typing import Any

import psutil
from flask import Flask, Response as FlaskResponse, jsonify, render_template_string, request
from flask_socketio import SocketIO  # type: ignore[import-untyped]

from hammerio.core.hardware import detect_hardware, format_hardware_report
from hammerio.core.telemetry import TelemetryCollector, SystemSnapshot

logger = logging.getLogger("hammerio.web")

socketio = SocketIO()
telemetry_collector = TelemetryCollector(interval_seconds=1.0)

# Store job history for the dashboard
_job_history: list[dict[str, Any]] = []
_JOB_HISTORY_FILE = Path.home() / ".config" / "hammerio" / "job_history.json"


def _load_job_history() -> None:
    """Load persisted job history from disk."""
    global _job_history
    try:
        if _JOB_HISTORY_FILE.exists():
            _job_history = json.loads(_JOB_HISTORY_FILE.read_text())[-500:]
    except Exception:
        pass


def _save_job_history() -> None:
    """Persist job history to disk."""
    try:
        _JOB_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _JOB_HISTORY_FILE.write_text(json.dumps(_job_history[-500:], default=str))
    except Exception:
        pass

# Track active batch progress
_batch_progress: dict[str, Any] = {
    "active": False,
    "batch_id": "",
    "total_files": 0,
    "completed_files": 0,
    "current_file": "",
    "current_file_pct": 0,
    "overall_pct": 0.0,
    "files": [],
}

# Track when the app module was first loaded (used by /health uptime)
_app_start_time: float = time.time()


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get(
        "HAMMERIO_SECRET_KEY", secrets.token_hex(32)
    )

    _allowed_origins = os.environ.get("HAMMERIO_CORS_ORIGINS", "").strip()
    _cors = _allowed_origins.split(",") if _allowed_origins else ["http://localhost:*", "http://127.0.0.1:*"]
    socketio.init_app(app, cors_allowed_origins=_cors, async_mode="threading")

    # Load persisted job history
    _load_job_history()

    # Start telemetry collection
    telemetry_collector.add_callback(_broadcast_telemetry)
    telemetry_collector.add_alert_callback(_broadcast_alert)
    telemetry_collector.start()

    _register_routes(app)
    _register_socket_events()

    return app


def _broadcast_telemetry(snap: SystemSnapshot) -> None:
    """Broadcast telemetry data to all connected WebSocket clients."""
    try:
        socketio.emit("telemetry", snap.to_dict())
    except Exception:
        pass


def emit_batch_progress(
    batch_id: str,
    total_files: int,
    completed_files: int,
    current_file: str = "",
    current_file_pct: int = 0,
    files: list[dict[str, Any]] | None = None,
) -> None:
    """Update and broadcast batch processing progress to WebSocket clients.

    Call this from batch processing code to keep the dashboard updated.
    """
    overall_pct = (completed_files / total_files * 100) if total_files > 0 else 0.0
    _batch_progress.update({
        "active": completed_files < total_files,
        "batch_id": batch_id,
        "total_files": total_files,
        "completed_files": completed_files,
        "current_file": current_file,
        "current_file_pct": current_file_pct,
        "overall_pct": round(overall_pct, 1),
        "files": files or [],
    })
    try:
        socketio.emit("batch_progress", _batch_progress)
    except Exception:
        pass


def _broadcast_alert(message: str, snap: SystemSnapshot) -> None:
    """Broadcast alert to all connected clients."""
    try:
        socketio.emit("alert", {"message": message, "timestamp": snap.timestamp})
    except Exception:
        pass


def _register_socket_events() -> None:
    @socketio.on("connect")
    def handle_connect() -> None:
        logger.info("WebSocket client connected")
        # Send initial hardware profile
        hw = detect_hardware()
        socketio.emit("hardware_profile", {
            "platform": hw.platform_name,
            "architecture": hw.architecture,
            "has_cuda": hw.has_cuda,
            "cuda_version": hw.cuda_device.cuda_version if hw.cuda_device else "",
            "gpu_name": hw.cuda_device.name if hw.cuda_device else "N/A",
            "gpu_memory_mb": hw.gpu_memory_mb,
            "has_nvenc": hw.has_nvenc,
            "has_nvcomp": hw.has_nvcomp,
            "has_vpi": hw.has_vpi,
            "cpu_cores": hw.cpu_cores,
            "ram_mb": hw.total_ram_mb,
            "routes": hw.routing_summary(),
        })

    @socketio.on("request_snapshot")
    def handle_snapshot_request() -> None:
        snap = telemetry_collector.get_snapshot()
        socketio.emit("telemetry", snap.to_dict())

    @socketio.on("request_history")
    def handle_history_request(data: dict[str, Any]) -> None:
        n = data.get("last_n", 60)
        history = telemetry_collector.get_history(last_n=n)
        socketio.emit("telemetry_history", [s.to_dict() for s in history])

    @socketio.on("request_batch_status")
    def handle_batch_status_request() -> None:
        socketio.emit("batch_progress", _batch_progress)


def _register_routes(app: Flask) -> None:

    @app.errorhandler(404)
    def handle_not_found(e: Exception) -> Any:
        """Return empty response for 404s (like favicon.ico)."""
        return "", 404

    @app.errorhandler(Exception)
    def handle_uncaught_exception(e: Exception) -> Any:
        """Catch-all handler so uncaught exceptions return JSON, not HTML."""
        logger.exception("Unhandled exception in API endpoint")
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

    @app.after_request
    def add_security_headers(response: Any) -> Any:
        origin = request.headers.get("Origin", "")
        if origin and any(
            origin == o or (o.endswith(":*") and origin.startswith(o[:-1]))
            for o in _cors
        ):
            response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.route("/")
    def index() -> str:
        return render_template_string(DASHBOARD_HTML)

    @app.route("/console")
    def console_page() -> str:
        return render_template_string(CONSOLE_HTML)

    @app.route("/architecture")
    def architecture_page() -> str:
        return render_template_string(ARCHITECTURE_HTML)

    @app.route("/favicon.ico")
    def favicon() -> Any:
        """Return a minimal favicon to avoid 404 spam."""
        return "", 204

    @app.route("/health")
    def health() -> Any:
        import time as _time
        from hammerio import __version__
        return jsonify({
            "status": "ok",
            "version": __version__,
            "service": "hammerio",
            "timestamp": _time.time(),
            "uptime": _time.time() - _app_start_time,
        })

    @app.route("/api/version")
    def api_version() -> Any:
        from hammerio import __version__, __author__, __license__
        return jsonify({
            "version": __version__,
            "name": "hammerio",
            "author": __author__,
            "license": __license__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        })

    @app.route("/api/batch")
    def api_batch() -> Any:
        """Return current batch processing status."""
        return jsonify(_batch_progress)

    @app.route("/api/hardware")
    def api_hardware() -> Any:
        try:
            hw = detect_hardware()
            return jsonify({
                "platform": hw.platform_name,
                "architecture": hw.architecture,
                "cuda": {
                    "available": hw.has_cuda,
                    "version": hw.cuda_device.cuda_version if hw.cuda_device else None,
                    "device": hw.cuda_device.name if hw.cuda_device else None,
                    "memory_mb": hw.gpu_memory_mb,
                    "unified": hw.cuda_device.is_unified_memory if hw.cuda_device else False,
                },
                "nvenc": {"available": hw.has_nvenc, "codecs": hw.nvenc.codecs},
                "nvcomp": {"available": hw.has_nvcomp},
                "vpi": {"available": hw.has_vpi},
                "cpu_cores": hw.cpu_cores,
                "ram_mb": hw.total_ram_mb,
                "routes": hw.routing_summary(),
            })
        except Exception as e:
            logger.exception("Error in /api/hardware")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/telemetry")
    def api_telemetry() -> Any:
        try:
            snap = telemetry_collector.get_snapshot()
            return jsonify(snap.to_dict())
        except Exception as e:
            logger.exception("Error in /api/telemetry")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/telemetry/history")
    def api_telemetry_history() -> Any:
        n = request.args.get("n", 60, type=int)
        history = telemetry_collector.get_history(last_n=n)
        return jsonify([s.to_dict() for s in history])

    @app.route("/api/export/telemetry")
    def api_export_telemetry() -> Any:
        """Return full telemetry history as a downloadable JSON file."""
        n = request.args.get("n", 3600, type=int)
        history = telemetry_collector.get_history(last_n=n)
        payload = json.dumps([s.to_dict() for s in history], indent=2)
        return FlaskResponse(
            payload,
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=hammerio_telemetry.json"},
        )

    @app.route("/api/telemetry/summary")
    def api_telemetry_summary() -> Any:
        return jsonify(telemetry_collector.get_summary())

    @app.route("/api/jtop")
    def api_jtop() -> Any:
        """Enhanced jtop-style monitoring for HammerIO workloads."""
        try:
            snap = telemetry_collector.get_snapshot()
            hw = detect_hardware()
            data: dict[str, Any] = {
                "gpu": {
                    "utilization": snap.gpu.utilization_pct,
                    "memory_used_mb": snap.gpu.memory_used_mb,
                    "memory_total_mb": snap.gpu.memory_total_mb or hw.gpu_memory_mb,
                    "frequency_mhz": snap.gpu.frequency_mhz,
                    "encoder_util": snap.gpu.encoder_utilization_pct,
                    "decoder_util": snap.gpu.decoder_utilization_pct,
                },
                "cpu": {
                    "overall": snap.cpu.overall_pct,
                    "per_core": snap.cpu.per_core_pct,
                    "frequency_mhz": snap.cpu.frequency_mhz,
                    "cores": hw.cpu_cores,
                },
                "memory": {
                    "ram_used_mb": snap.ram_used_mb,
                    "ram_total_mb": snap.ram_total_mb,
                    "swap_used_mb": snap.swap_used_mb,
                    "swap_total_mb": snap.swap_total_mb,
                    "unified": hw.cuda_device.is_unified_memory if hw.cuda_device else False,
                },
                "thermal": [
                    {"name": tz.name, "temp_c": tz.temperature_c}
                    for tz in snap.thermal_zones
                ],
                "power": {
                    "total_mw": snap.total_power_mw,
                    "rails": [
                        {"name": pr.rail_name, "current_mw": pr.current_mw, "avg_mw": pr.average_mw}
                        for pr in snap.power_readings
                    ],
                    "mode": snap.power_mode,
                },
                "fan_speed_pct": snap.fan_speed_pct,
                "is_throttled": snap.is_throttled,
                "throttle_reason": snap.throttle_reason,
                "platform": hw.platform_name,
                "jetpack": hw.jetpack_version,
                "l4t": hw.l4t_version,
            }
            return jsonify(data)
        except Exception as e:
            logger.exception("Error in /api/jtop")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/architecture.svg")
    def api_architecture_svg() -> Any:
        try:
            from hammerio.web.architecture import generate_architecture_svg
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
                generate_architecture_svg(f.name)
                svg_content = Path(f.name).read_text()
            from flask import Response
            return Response(svg_content, mimetype="image/svg+xml")
        except Exception as e:
            logger.exception("Error in /api/architecture.svg")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/jobs")
    def api_jobs() -> Any:
        return jsonify(_job_history[-100:])

    @app.route("/api/jobs/all")
    def api_jobs_all() -> Any:
        return jsonify(_job_history)

    @app.route("/api/jobs/clear", methods=["POST"])
    def api_jobs_clear() -> Any:
        """Clear job history."""
        data = request.json or {}
        scope = data.get("scope", "recent")  # "recent" or "all"
        if scope == "all":
            _job_history.clear()
        else:
            # Clear last 100 (recent view)
            del _job_history[-100:]
        _save_job_history()
        return jsonify({"status": "cleared", "scope": scope, "remaining": len(_job_history)})

    @app.route("/api/jobs/log", methods=["POST"])
    def api_jobs_log() -> Any:
        """Log an externally completed job (from CLI, right-click actions, etc)."""
        data = request.json or {}
        if not data.get("input_path"):
            return jsonify({"error": "input_path required"}), 400
        entry = {
            "input_path": data.get("input_path", ""),
            "output_path": data.get("output_path", ""),
            "input_size": data.get("input_size", 0),
            "output_size": data.get("output_size", 0),
            "ratio": data.get("ratio", 0),
            "savings_pct": data.get("savings_pct", 0),
            "elapsed_s": data.get("elapsed_s", 0),
            "throughput_mbps": data.get("throughput_mbps", 0),
            "processor": data.get("processor", "cli"),
            "algorithm": data.get("algorithm", ""),
            "reason": data.get("reason", "Right-click action"),
            "status": data.get("status", "completed"),
            "timestamp": data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
        }
        _job_history.append(entry)
        _save_job_history()
        socketio.emit("job_complete", entry)
        return jsonify({"status": "logged"})

    @app.route("/api/compress", methods=["POST"])
    def api_compress() -> Any:
        data = request.json or {}
        input_path = data.get("input_path", "")
        output_path = data.get("output_path")
        mode = data.get("mode", "auto")
        quality = data.get("quality", "balanced")

        if not input_path:
            return jsonify({"error": "input_path required"}), 400

        try:
            from hammerio.core.router import JobRouter
            router = JobRouter(quality=quality)
            job = router.route(input_path, output_path, mode=mode if mode != "auto" else None)
            result = router.execute(job)

            savings = (1 - result.output_size / result.input_size) * 100 if result.input_size > 0 else 0
            result_dict = {
                "input_path": result.input_path,
                "output_path": result.output_path,
                "input_size": result.input_size,
                "output_size": result.output_size,
                "ratio": result.compression_ratio,
                "savings_pct": round(savings, 1),
                "elapsed_s": result.elapsed_seconds,
                "throughput_mbps": result.throughput_mbps,
                "processor": result.processor_used,
                "algorithm": result.algorithm,
                "reason": result.routing_reason,
                "status": result.status.value,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _job_history.append(result_dict)
            _save_job_history()
            socketio.emit("job_complete", result_dict)
            return jsonify(result_dict)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/decompress", methods=["POST"])
    def api_decompress() -> Any:
        data = request.json or {}
        input_path = data.get("input_path", "")
        output_path = data.get("output_path")

        if not input_path:
            return jsonify({"error": "input_path required"}), 400

        try:
            from hammerio.encoders.general import GeneralEncoder
            from hammerio.encoders.bulk import BulkEncoder
            input_p = Path(input_path)
            if not input_p.exists():
                return jsonify({"error": f"File not found: {input_path}"}), 404

            hw = detect_hardware()
            ext = input_p.suffix.lower()
            import time as _t
            start = _t.time()

            if ext == ".hammer":
                enc = BulkEncoder(hw)
                out = enc.decompress(input_p, Path(output_path) if output_path else None)
            elif ext in (".zst", ".gz", ".bz2", ".lz4"):
                enc = GeneralEncoder(hw)
                out = enc.decompress(input_p, Path(output_path) if output_path else None)
            else:
                return jsonify({"error": f"Unknown compressed format: {ext}"}), 400

            elapsed = _t.time() - start
            in_size = input_p.stat().st_size
            out_size = Path(out).stat().st_size if Path(out).exists() else 0
            throughput = (out_size / elapsed / (1024 * 1024)) if elapsed > 0 else 0
            expansion = out_size / in_size if in_size > 0 else 0
            result_dict = {
                "input_path": str(input_p),
                "output_path": out,
                "input_size": in_size,
                "output_size": out_size,
                "ratio": round(expansion, 2),
                "savings_pct": 0,
                "elapsed_s": round(elapsed, 3),
                "throughput_mbps": round(throughput, 1),
                "processor": "decompress",
                "algorithm": ext.lstrip("."),
                "reason": "Decompression",
                "status": "completed",
                "timestamp": _t.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _job_history.append(result_dict)
            _save_job_history()
            return jsonify(result_dict)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/route", methods=["POST"])
    def api_route() -> Any:
        data = request.json or {}
        input_path = data.get("input_path", "")
        if not input_path:
            return jsonify({"error": "input_path required"}), 400

        try:
            from hammerio.core.router import JobRouter
            router = JobRouter()
            explanation = router.explain_route(input_path)
            return jsonify({"explanation": explanation})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/console", methods=["POST"])
    def api_console() -> Any:
        """Execute a HammerIO CLI command and return output.

        Only allows 'hammer' / 'python3 -m hammerio' commands and a
        strict allowlist of read-only system commands. Commands are
        executed without shell=True to prevent injection.
        """
        import re as _re
        import subprocess as _sp

        data = request.json or {}
        raw_cmd = data.get("command", "").strip()

        if not raw_cmd:
            return jsonify({"error": "No command provided"}), 400

        # Reject shell metacharacters outright
        if any(c in raw_cmd for c in ";|&`$(){}!<>\n\\"):
            return jsonify({"error": "Shell metacharacters are not allowed"}), 403

        try:
            args = shlex.split(raw_cmd)
        except ValueError as e:
            return jsonify({"error": f"Invalid command syntax: {e}"}), 400

        if not args:
            return jsonify({"error": "No command provided"}), 400

        # Strict allowlist: only these exact executables may run
        allowed_executables = {
            "hammer", "pwd", "ls", "cat", "head", "tail", "wc",
            "df", "free", "uname", "uptime", "nvidia-smi",
            "nvpmodel", "tegrastats", "jtop", "gst-inspect-1.0",
        }
        # Also allow "python3 -m hammerio" / "python -m hammerio"
        is_python_hammerio = (
            args[0] in ("python3", "python")
            and len(args) >= 3
            and args[1] == "-m"
            and args[2] == "hammerio"
        )

        if args[0] not in allowed_executables and not is_python_hammerio:
            return jsonify({
                "error": "Command not allowed. Use 'hammer <command>' or allowed system commands.",
                "allowed": sorted(allowed_executables),
            }), 403

        # Bare "hammer" with no subcommand
        if args == ["hammer"]:
            args = ["hammer", "--help"]

        try:
            result = _sp.run(
                args,
                shell=False,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                env={**os.environ, "COLUMNS": "120", "TERM": "dumb", "NO_COLOR": "1"},
            )
            # Strip ANSI escape codes for clean display
            ansi_escape = _re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            stdout = ansi_escape.sub('', result.stdout)
            stderr = ansi_escape.sub('', result.stderr)

            return jsonify({
                "stdout": stdout,
                "stderr": stderr,
                "returncode": result.returncode,
                "command": raw_cmd,
            })
        except _sp.TimeoutExpired:
            return jsonify({"error": "Command timed out (120s limit)", "command": raw_cmd}), 408
        except Exception as e:
            return jsonify({"error": str(e), "command": raw_cmd}), 500

    @app.route("/api/browse")
    def api_browse() -> Any:
        """File browser endpoint with path traversal protection."""
        default_path = str(Path.home())
        target = request.args.get("path", default_path)

        # Security: reject path traversal attempts
        if ".." in target:
            return jsonify({"error": "Path traversal not allowed"}), 400

        target_path = Path(target).resolve()

        # Verify resolved path does not escape via symlinks with ..
        if ".." in str(target_path):
            return jsonify({"error": "Path traversal not allowed"}), 400

        if not target_path.exists():
            return jsonify({"error": "Path does not exist"}), 404

        if not target_path.is_dir():
            return jsonify({"error": "Path is not a directory"}), 400

        entries = []
        try:
            for item in sorted(target_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                try:
                    # Skip symlinks to prevent traversal
                    if item.is_symlink():
                        continue
                    stat = item.stat(follow_symlinks=False)
                    entries.append({
                        "name": item.name,
                        "size": stat.st_size if item.is_file() else 0,
                        "type": "dir" if item.is_dir() else "file",
                        "extension": item.suffix.lstrip(".") if item.is_file() and item.suffix else "",
                    })
                except (PermissionError, OSError):
                    continue
        except PermissionError:
            return jsonify({"error": "Permission denied"}), 403

        return jsonify({
            "path": str(target_path),
            "parent": str(target_path.parent) if target_path != target_path.parent else None,
            "entries": entries,
        })

    @app.route("/api/system")
    def api_system() -> Any:
        """System info: uptime, disk, network, process count."""
        # Uptime
        uptime_seconds = time.time() - psutil.boot_time()
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        uptime_str = f"{days}d {hours}h {minutes}m"

        # Disk usage for key mount points
        disk_info = []
        checked: set[str] = set()
        for mount in ["/", "/home"]:
            try:
                usage = shutil.disk_usage(mount)
                disk_info.append({
                    "mount": mount,
                    "total_gb": round(usage.total / (1024 ** 3), 2),
                    "used_gb": round(usage.used / (1024 ** 3), 2),
                    "free_gb": round(usage.free / (1024 ** 3), 2),
                    "percent": round(usage.used / usage.total * 100, 1) if usage.total else 0,
                })
                checked.add(mount)
            except (OSError, FileNotFoundError):
                pass

        # Detect NVMe mounts
        try:
            for part in psutil.disk_partitions(all=False):
                if "nvme" in part.device and part.mountpoint not in checked:
                    try:
                        usage = shutil.disk_usage(part.mountpoint)
                        disk_info.append({
                            "mount": part.mountpoint,
                            "total_gb": round(usage.total / (1024 ** 3), 2),
                            "used_gb": round(usage.used / (1024 ** 3), 2),
                            "free_gb": round(usage.free / (1024 ** 3), 2),
                            "percent": round(usage.used / usage.total * 100, 1) if usage.total else 0,
                        })
                        checked.add(part.mountpoint)
                    except (OSError, FileNotFoundError):
                        pass
        except Exception:
            pass

        # Network interfaces
        interfaces = []
        try:
            addrs = psutil.net_if_addrs()
            for iface, addr_list in addrs.items():
                ips = [a.address for a in addr_list if a.family == socket.AF_INET]
                if ips:
                    interfaces.append({"name": iface, "ipv4": ips})
        except Exception:
            pass

        # Process count
        process_count = len(psutil.pids())

        return jsonify({
            "uptime": uptime_str,
            "uptime_seconds": int(uptime_seconds),
            "disk": disk_info,
            "network": interfaces,
            "process_count": process_count,
            "hostname": platform.node(),
        })


# ─── Dashboard HTML ────────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%230d1117'/%3E%3Cpath d='M8 8v16M24 8v16M8 16h16' stroke='%2339d2c0' stroke-width='3' stroke-linecap='round'/%3E%3Ccircle cx='16' cy='10' r='2.5' fill='%233fb950'/%3E%3C/svg%3E">
    <meta http-equiv="Pragma" content="no-cache">
    <title>HammerIO Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js" async></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js" async></script>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-card: #1c2333;
            --border: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-green: #3fb950;
            --accent-blue: #58a6ff;
            --accent-orange: #d29922;
            --accent-red: #f85149;
            --accent-purple: #bc8cff;
            --accent-cyan: #39d2c0;
        }

        /* Light theme overrides */
        [data-theme="light"] {
            --bg-primary: #f0f2f5;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --border: #d0d7de;
            --text-primary: #1f2328;
            --text-secondary: #656d76;
            --accent-green: #1a7f37;
            --accent-blue: #0969da;
            --accent-orange: #bf8700;
            --accent-red: #cf222e;
            --accent-purple: #8250df;
            --accent-cyan: #0891b2;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
        }

        .header {
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
            border-bottom: 1px solid var(--border);
            padding: 16px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            font-size: 24px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header .tagline {
            color: var(--text-secondary);
            font-size: 12px;
        }

        .header .creator {
            color: var(--text-secondary);
            font-size: 11px;
            text-align: right;
        }

        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px;
            padding: 16px 24px;
            max-width: 1600px;
            margin: 0 auto;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }

        .card h2 {
            font-size: 14px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card h2 .indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }

        .indicator.green { background: var(--accent-green); }
        .indicator.red { background: var(--accent-red); }
        .indicator.orange { background: var(--accent-orange); }
        .indicator.pulse-green {
            background: var(--accent-green);
            animation: pulse-dot 2s ease-in-out infinite;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid var(--border);
            gap: 12px;
        }

        .metric .label { min-width: 80px; flex-shrink: 0; }

        .metric:last-child { border-bottom: none; }
        .metric .label { color: var(--text-secondary); }
        .metric .value { font-weight: 600; font-family: 'JetBrains Mono', monospace; }

        .gauge-container {
            display: flex;
            gap: 24px;
            justify-content: center;
            padding: 12px 0;
        }

        .gauge {
            text-align: center;
        }

        .gauge canvas { display: block; margin: 0 auto; }
        .gauge .gauge-label { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }
        .gauge .gauge-value { font-size: 20px; font-weight: 700; }

        .bar {
            height: 8px;
            background: var(--bg-primary);
            border-radius: 4px;
            overflow: hidden;
            margin: 4px 0;
        }

        .bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }

        .thermal-row {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 4px 0;
            font-size: 13px;
        }

        .thermal-row .name { width: 120px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; }
        .thermal-row .bar { flex: 1; }
        .thermal-row .temp { width: 60px; text-align: right; font-family: monospace; }

        .route-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            margin: 4px 0;
            background: var(--bg-primary);
            border-radius: 6px;
            font-size: 13px;
        }

        .route-item .workload { color: var(--accent-cyan); }
        .route-item .processor { color: var(--accent-green); }

        .chart-container {
            position: relative;
            height: 200px;
            width: 100%;
        }

        .full-width { grid-column: 1 / -1; }

        .alert-banner {
            background: var(--accent-red);
            color: white;
            padding: 8px 24px;
            text-align: center;
            font-weight: 600;
            display: none;
        }

        .status-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }

        .status-badge.available { background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }
        .status-badge.unavailable { background: rgba(248, 81, 73, 0.2); color: var(--accent-red); }

        .compress-form {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .compress-form input, .compress-form select {
            background: var(--bg-primary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 13px;
        }

        .compress-form input { flex: 1; min-width: 200px; }

        .compress-form button {
            background: var(--accent-blue);
            color: white;
            border: none;
            padding: 8px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
        }

        .compress-form button:hover { opacity: 0.9; }

        .job-table { width: 100%; border-collapse: collapse; font-size: 12px; }
        .job-table th { color: var(--text-secondary); text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
        .job-table td { padding: 6px 8px; border-bottom: 1px solid var(--border); font-family: monospace; }

        .footer {
            text-align: center;
            padding: 16px;
            color: var(--text-secondary);
            font-size: 11px;
            border-top: 1px solid var(--border);
        }

        .footer a { color: var(--accent-blue); text-decoration: none; }

        /* File Browser */
        .breadcrumb { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; font-size: 13px; }
        .breadcrumb a { color: var(--accent-blue); text-decoration: none; cursor: pointer; }
        .breadcrumb a:hover { text-decoration: underline; }
        .breadcrumb span.sep { color: var(--text-secondary); }

        .file-list { max-height: 350px; overflow-y: auto; }
        .file-entry {
            display: flex; align-items: center; gap: 8px;
            padding: 5px 8px; border-bottom: 1px solid var(--border);
            font-size: 13px; cursor: pointer;
        }
        .file-entry:hover { background: var(--bg-primary); }
        .file-entry .file-icon { width: 20px; text-align: center; flex-shrink: 0; }
        .file-entry .file-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .file-entry .file-size { width: 80px; text-align: right; color: var(--text-secondary); font-family: monospace; font-size: 12px; }
        .file-entry .file-type { width: 50px; text-align: right; color: var(--text-secondary); font-size: 11px; text-transform: uppercase; }
        .file-entry.dir .file-name { color: var(--accent-blue); }
        .file-entry.file .file-name { color: var(--text-primary); }

        /* System Info */
        .disk-bar-container { margin: 6px 0; }
        .disk-label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px; }
        .disk-label .mount-name { color: var(--text-secondary); }
        .disk-label .disk-pct { font-family: monospace; }
        .net-item { font-size: 13px; padding: 4px 0; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; }
        .net-item .iface-name { color: var(--accent-cyan); font-weight: 600; }
        .net-item .iface-ip { font-family: monospace; }

        /* Responsive: single column on narrow screens */
        @media (max-width: 768px) {
            .dashboard { grid-template-columns: 1fr; padding: 8px; gap: 10px; }
            .header { flex-direction: column; gap: 8px; text-align: center; }
            .header .creator { text-align: center; }
            .gauge-container { flex-wrap: wrap; }
            .compress-form { flex-direction: column; }
            .compress-form input { min-width: 100%; }
            .full-width { grid-column: 1 / -1; }
        }

        /* Auto-refresh live indicator */
        .live-dot {
            width: 10px; height: 10px; border-radius: 50%;
            background: var(--accent-green);
            display: inline-block; margin-right: 6px;
            animation: pulse-dot 2s ease-in-out infinite;
        }
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(63,185,80,0.7); }
            50% { opacity: 0.6; box-shadow: 0 0 0 6px rgba(63,185,80,0); }
        }
        .live-label { font-size: 12px; color: var(--accent-green); font-weight: 600; display: flex; align-items: center; }

        /* Theme toggle button */
        .theme-toggle {
            background: var(--bg-primary); border: 1px solid var(--border);
            color: var(--text-primary); padding: 6px 14px; border-radius: 6px;
            cursor: pointer; font-size: 13px; font-weight: 600;
        }
        .theme-toggle:hover { border-color: var(--accent-blue); }

        /* Header controls row */
        .header-controls { display: flex; align-items: center; gap: 12px; }

        /* Toast notifications */
        .toast-container {
            position: fixed; bottom: 20px; right: 20px; z-index: 9999;
            display: flex; flex-direction: column-reverse; gap: 8px;
            pointer-events: none;
        }
        .toast {
            background: var(--bg-card); border: 1px solid var(--border);
            border-left: 4px solid var(--accent-blue); border-radius: 8px;
            padding: 12px 16px; min-width: 300px; max-width: 420px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3); pointer-events: auto;
            animation: toast-in 0.3s ease-out;
            font-size: 13px; color: var(--text-primary);
        }
        .toast.alert { border-left-color: var(--accent-red); }
        .toast.success { border-left-color: var(--accent-green); }
        .toast .toast-title { font-weight: 700; margin-bottom: 2px; font-size: 12px; text-transform: uppercase; color: var(--text-secondary); }
        .toast .toast-msg { line-height: 1.4; }
        @keyframes toast-in { from { opacity: 0; transform: translateX(40px); } to { opacity: 1; transform: translateX(0); } }
        @keyframes toast-out { from { opacity: 1; transform: translateX(0); } to { opacity: 0; transform: translateX(40px); } }

        /* Export button */
        .export-btn {
            background: var(--bg-primary); border: 1px solid var(--border);
            color: var(--text-secondary); padding: 3px 10px; border-radius: 4px;
            cursor: pointer; font-size: 11px; margin-left: auto;
        }
        .export-btn:hover { color: var(--accent-blue); border-color: var(--accent-blue); }

        /* Keyboard shortcut hint */
        .kbd-hint { font-size: 10px; color: var(--text-secondary); margin-left: 4px; background: var(--bg-primary); padding: 1px 5px; border-radius: 3px; border: 1px solid var(--border); }
    </style>
</head>
<body>
    <div id="alert-banner" class="alert-banner"></div>

    <div class="header">
        <div>
            <h1>HammerIO Dashboard</h1>
            <div class="tagline">GPU where it matters. CPU where it doesn't. Zero configuration.</div>
        </div>
        <div style="display:flex;align-items:center;gap:16px">
            <a href="/" style="color:var(--accent-cyan);text-decoration:none;font-weight:600;font-size:13px;padding:4px 12px;border:1px solid var(--accent-cyan);border-radius:6px">Dashboard</a>
            <a href="/console" style="color:var(--accent-green);text-decoration:none;font-weight:600;font-size:13px;padding:4px 12px;border:1px solid var(--accent-green);border-radius:6px">Console</a>
            <a href="/architecture" style="color:var(--accent-orange);text-decoration:none;font-weight:600;font-size:13px;padding:4px 12px;border:1px solid var(--accent-orange);border-radius:6px">Architecture</a>
        </div>
        <div class="header-controls">
            <span class="live-label"><span class="live-dot" id="live-dot"></span> LIVE</span>
            <button class="theme-toggle" id="theme-toggle" title="Toggle dark/light theme (shortcut: d)">Light Mode</button>
            <div class="creator">
                <div>ResilientMind AI</div>
                <div>Joseph C McGinty Jr</div>
            </div>
        </div>
    </div>

    <div class="dashboard">
        <!-- Performance History (top of dashboard) -->
        <div class="card full-width">
            <h2>Performance History <button class="export-btn" onclick="exportTelemetry()">Export JSON</button></h2>
            <div class="chart-container">
                <canvas id="history-chart"></canvas>
            </div>
        </div>

        <!-- Hardware Profile -->
        <div class="card">
            <h2><span class="indicator green" id="hw-indicator"></span> Hardware Profile</h2>
            <div id="hw-info">
                <div class="metric"><span class="label">Platform</span><span class="value" id="hw-platform">Detecting...</span></div>
                <div class="metric"><span class="label">Architecture</span><span class="value" id="hw-arch">-</span></div>
                <div class="metric"><span class="label">GPU</span><span class="value" id="hw-gpu">-</span></div>
                <div class="metric"><span class="label">CUDA</span><span class="value" id="hw-cuda">-</span></div>
                <div class="metric"><span class="label">GPU Memory</span><span class="value" id="hw-gpu-mem">-</span></div>
                <!-- NVENC removed — compression focus -->
                <div class="metric"><span class="label">nvCOMP</span><span class="value" id="hw-nvcomp">-</span></div>
                <div class="metric"><span class="label">VPI</span><span class="value" id="hw-vpi">-</span></div>
                <div class="metric"><span class="label">CPU Cores</span><span class="value" id="hw-cpu">-</span></div>
                <div class="metric"><span class="label">RAM</span><span class="value" id="hw-ram">-</span></div>
            </div>
        </div>

        <!-- Live Gauges -->
        <div class="card">
            <h2><span class="indicator green" id="telemetry-indicator"></span> System Utilization</h2>
            <div class="gauge-container">
                <div class="gauge">
                    <canvas id="gpu-gauge" width="100" height="100"></canvas>
                    <div class="gauge-value" id="gpu-pct">0%</div>
                    <div class="gauge-label">GPU</div>
                </div>
                <div class="gauge">
                    <canvas id="cpu-gauge" width="100" height="100"></canvas>
                    <div class="gauge-value" id="cpu-pct">0%</div>
                    <div class="gauge-label">CPU</div>
                </div>
                <div class="gauge">
                    <canvas id="ram-gauge" width="100" height="100"></canvas>
                    <div class="gauge-value" id="ram-pct">0%</div>
                    <div class="gauge-label">RAM</div>
                </div>
            </div>
            <div class="metric"><span class="label">Power</span><span class="value" id="power-val">- mW</span></div>
            <div class="metric"><span class="label">Mode</span><span class="value" id="power-mode">-</span></div>
        </div>

        <!-- Thermal -->
        <div class="card">
            <h2><span class="indicator" id="thermal-indicator"></span> Thermal Zones</h2>
            <div id="thermal-zones">
                <div class="thermal-row"><span class="name">Loading...</span></div>
            </div>
        </div>

        <!-- Routing -->
        <div class="card">
            <h2>Smart Routing</h2>
            <div id="routes"></div>
        </div>

        <!-- Per-Core CPU -->
        <div class="card">
            <h2>CPU Cores</h2>
            <div id="cpu-cores"></div>
        </div>

        <!-- Power Rails -->
        <div class="card">
            <h2>Power Rails</h2>
            <div id="power-rails">
                <div class="metric"><span class="label">Loading...</span></div>
            </div>
            <div class="metric" style="margin-top:8px;border-top:2px solid var(--border);padding-top:8px">
                <span class="label" style="font-weight:600">Total Power</span>
                <span class="value" id="total-power" style="color:var(--accent-orange)">- mW</span>
            </div>
        </div>

        <!-- Architecture moved to /architecture tab -->

        <!-- File Browser Modal -->
        <div id="browser-modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:1000;justify-content:center;align-items:center">
            <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;width:90%;max-width:700px;max-height:80vh;display:flex;flex-direction:column;overflow:hidden">
                <div style="display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid var(--border)">
                    <h2 style="margin:0;font-size:14px;color:var(--text-secondary)">Select File or Directory</h2>
                    <button onclick="closeBrowser()" style="background:none;border:none;color:var(--text-secondary);font-size:20px;cursor:pointer">&times;</button>
                </div>
                <div style="padding:8px 16px">
                    <div id="file-breadcrumb" class="breadcrumb"></div>
                </div>
                <div id="file-list" class="file-list" style="flex:1;overflow-y:auto;padding:0 16px 16px;max-height:60vh">
                    <div class="metric"><span class="label">Loading...</span></div>
                </div>
                <div style="padding:8px 16px;border-top:1px solid var(--border);display:flex;gap:8px;justify-content:flex-end">
                    <button onclick="selectCurrentDir()" style="background:var(--accent-cyan);color:#000;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-weight:600;font-size:12px">Select This Directory</button>
                    <button onclick="closeBrowser()" style="background:var(--border);color:var(--text-primary);border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-size:12px">Cancel</button>
                </div>
            </div>
        </div>

        <!-- System Info -->
        <div class="card">
            <h2><span class="indicator green"></span> System Info</h2>
            <div class="metric"><span class="label">Hostname</span><span class="value" id="sys-hostname">-</span></div>
            <div class="metric"><span class="label">Uptime</span><span class="value" id="sys-uptime">-</span></div>
            <div class="metric"><span class="label">Processes</span><span class="value" id="sys-procs">-</span></div>
            <h2 style="margin-top:12px">Disk Usage</h2>
            <div id="sys-disk"></div>
            <h2 style="margin-top:12px">Network</h2>
            <div id="sys-network"></div>
        </div>

        <!-- Compress -->
        <div class="card full-width">
            <h2>Quick Compress</h2>
            <div class="compress-form">
                <input type="text" id="compress-input" placeholder="/path/to/file/or/directory">
                <button onclick="openBrowser('compress-input')" style="background:var(--border);color:var(--text-primary);padding:8px 12px;border:none;border-radius:6px;cursor:pointer;font-size:13px" title="Browse files">Browse</button>
                <select id="compress-mode">
                    <option value="auto">Auto</option>
                    <option value="gpu">GPU Prefer</option>
                    <option value="cpu">CPU Only</option>
                </select>
                <select id="compress-quality">
                    <option value="balanced">Balanced</option>
                    <option value="fast">Fast</option>
                    <option value="quality">Quality</option>
                    <option value="lossless">Lossless</option>
                </select>
                <button onclick="analyzeRoute()" style="background:var(--accent-cyan)">Analyze</button>
                <button onclick="submitCompress()">Compress</button>
            </div>
            <div id="compress-result" style="margin-top: 12px;"></div>
        </div>

        <!-- Decompress -->
        <div class="card full-width">
            <h2>Quick Decompress</h2>
            <div class="compress-form">
                <input type="text" id="decompress-input" placeholder="/path/to/compressed/file (.zst, .gz, .bz2, .lz4)">
                <button onclick="openBrowser('decompress-input')" style="background:var(--border);color:var(--text-primary);padding:8px 12px;border:none;border-radius:6px;cursor:pointer;font-size:13px" title="Browse files">Browse</button>
                <button onclick="submitDecompress()" style="background:var(--accent-orange)">Decompress</button>
            </div>
            <div id="decompress-result" style="margin-top: 12px;"></div>
        </div>

        <!-- Batch Progress -->
        <div class="card full-width" id="batch-card" style="display:none">
            <h2><span class="indicator" id="batch-indicator"></span> Batch Progress</h2>
            <div id="batch-info">
                <div class="metric"><span class="label">Batch ID</span><span class="value" id="batch-id">-</span></div>
                <div class="metric"><span class="label">Files</span><span class="value" id="batch-files">0 / 0</span></div>
                <div class="metric"><span class="label">Current File</span><span class="value" id="batch-current">-</span></div>
                <div class="metric"><span class="label">Current File Progress</span><span class="value" id="batch-file-pct">0%</span></div>
                <div style="margin-top:8px">
                    <div class="disk-label"><span class="mount-name">Overall Completion</span><span class="disk-pct" id="batch-overall-pct">0%</span></div>
                    <div class="bar" style="height:12px"><div class="bar-fill" id="batch-bar" style="width:0%;background:var(--accent-blue)"></div></div>
                </div>
                <div id="batch-file-list" style="margin-top:8px;max-height:200px;overflow-y:auto"></div>
            </div>
        </div>

        <!-- Compression Stats Summary -->
        <div class="card full-width" id="stats-card">
            <h2 onclick="toggleSection('stats-section')" style="cursor:pointer">
                <span id="stats-section-arrow">&#9660;</span> Compression Stats
            </h2>
            <div id="stats-section" style="">
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px">
                    <div class="metric"><span class="label">Total Jobs</span><span class="value" id="stat-total">0</span></div>
                    <div class="metric"><span class="label">Compressed</span><span class="value" id="stat-compress">0</span></div>
                    <div class="metric"><span class="label">Decompressed</span><span class="value" id="stat-decompress">0</span></div>
                    <div class="metric"><span class="label">Total Input</span><span class="value" id="stat-input-size">0 B</span></div>
                    <div class="metric"><span class="label">Total Output</span><span class="value" id="stat-output-size">0 B</span></div>
                    <div class="metric"><span class="label">Space Saved</span><span class="value" style="color:var(--accent-green)" id="stat-saved">0 B</span></div>
                    <div class="metric"><span class="label">Avg Ratio</span><span class="value" id="stat-avg-ratio">-</span></div>
                    <div class="metric"><span class="label">Avg Savings</span><span class="value" style="color:var(--accent-green)" id="stat-avg-savings">-</span></div>
                    <div class="metric"><span class="label">Avg Throughput</span><span class="value" id="stat-avg-speed">-</span></div>
                    <div class="metric"><span class="label">Total Time</span><span class="value" id="stat-total-time">0s</span></div>
                    <div class="metric"><span class="label">GPU Jobs</span><span class="value" style="color:var(--accent-cyan)" id="stat-gpu">0</span></div>
                    <div class="metric"><span class="label">CPU Jobs</span><span class="value" id="stat-cpu">0</span></div>
                </div>
            </div>
        </div>

        <!-- Recent Jobs (collapsible) -->
        <div class="card full-width">
            <h2 onclick="toggleSection('recent-jobs')" style="cursor:pointer">
                <span id="recent-jobs-arrow">&#9660;</span> Recent Jobs
                <span style="float:right;display:flex;gap:6px">
                    <button class="export-btn" onclick="event.stopPropagation();clearJobs('recent')">Clear</button>
                    <button class="export-btn" onclick="event.stopPropagation();exportJobs()">Export JSON</button>
                </span>
            </h2>
            <div id="recent-jobs" style="">
                <table class="job-table">
                    <thead><tr>
                        <th>File</th><th>Input</th><th>Output</th><th>Ratio</th><th>Savings</th>
                        <th>Time</th><th>Speed</th><th>Processor</th><th>Algorithm</th><th>Status</th><th>When</th>
                    </tr></thead>
                    <tbody id="job-tbody"></tbody>
                </table>
            </div>
        </div>

        <!-- All Jobs (collapsible, starts collapsed) -->
        <div class="card full-width">
            <h2 onclick="toggleSection('all-jobs')" style="cursor:pointer">
                <span id="all-jobs-arrow">&#9654;</span> All Jobs
                <span style="float:right;display:flex;gap:6px">
                    <button class="export-btn" onclick="event.stopPropagation();clearJobs('all')">Clear</button>
                    <button class="export-btn" onclick="event.stopPropagation();exportAllJobs()">Export JSON</button>
                </span>
            </h2>
            <div id="all-jobs" style="display:none">
                <table class="job-table">
                    <thead><tr>
                        <th>File</th><th>Input</th><th>Output</th><th>Ratio</th><th>Savings</th>
                        <th>Time</th><th>Speed</th><th>Processor</th><th>Algorithm</th><th>Status</th><th>When</th>
                    </tr></thead>
                    <tbody id="all-job-tbody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="toast-container" id="toast-container"></div>

    <div class="footer">
        Copyright &copy; 2026 <a href="https://resilientmindai.com">ResilientMind AI</a> | Joseph C McGinty Jr<br>
        HammerIO is proprietary software. All rights reserved
    </div>

    <script>
        // Socket.IO — graceful fallback if CDN fails
        let socket = null;
        try {
            socket = (typeof io !== 'undefined') ? io() : null;
        } catch(e) { console.warn('WebSocket unavailable:', e); }

        let historyChart = null;
        const historyData = { timestamps: [], gpu: [], cpu: [], temp: [] };

        // Safe socket wrapper — no-op if WebSocket unavailable
        const on = (evt, fn) => { if (socket) socket.on(evt, fn); };

        console.log('[HammerIO] Dashboard loading...');
        console.log('[HammerIO] Socket.IO available:', typeof io !== 'undefined');
        console.log('[HammerIO] Socket connected:', socket ? 'initializing' : 'unavailable');

        // WebSocket handlers
        on('connect', () => {
            const ti = document.getElementById('telemetry-indicator');
            ti.classList.add('green', 'pulse-green');
        });

        on('disconnect', () => {
            const ti = document.getElementById('telemetry-indicator');
            ti.classList.remove('green', 'pulse-green');
            ti.classList.add('red');
        });

        on('hardware_profile', (data) => {
            document.getElementById('hw-platform').textContent = data.platform;
            document.getElementById('hw-arch').textContent = data.architecture;
            document.getElementById('hw-gpu').textContent = data.gpu_name;
            document.getElementById('hw-cuda').textContent = data.cuda_version || 'N/A';
            document.getElementById('hw-gpu-mem').textContent = formatMB(data.gpu_memory_mb);
            document.getElementById('hw-nvenc').innerHTML = badge(data.has_nvenc);
            document.getElementById('hw-nvcomp').innerHTML = badge(data.has_nvcomp);
            document.getElementById('hw-vpi').innerHTML = badge(data.has_vpi);
            document.getElementById('hw-cpu').textContent = data.cpu_cores;
            document.getElementById('hw-ram').textContent = formatMB(data.ram_mb);
            document.getElementById('hw-indicator').className = 'indicator ' + (data.has_cuda ? 'green' : 'orange');

            // Routes
            const routesDiv = document.getElementById('routes');
            routesDiv.innerHTML = '';
            for (const [workload, processor] of Object.entries(data.routes)) {
                routesDiv.innerHTML += `<div class="route-item">
                    <span class="workload">${workload.replace('_', ' ')}</span>
                    <span class="processor">${processor}</span>
                </div>`;
            }
        });

        on('telemetry', (data) => {
            // Gauges
            drawGauge('gpu-gauge', data.gpu_util_pct, '#58a6ff');
            drawGauge('cpu-gauge', data.cpu_pct, '#3fb950');
            drawGauge('ram-gauge', data.ram_pct, '#bc8cff');
            document.getElementById('gpu-pct').textContent = data.gpu_util_pct.toFixed(0) + '%';
            document.getElementById('cpu-pct').textContent = data.cpu_pct.toFixed(0) + '%';
            document.getElementById('ram-pct').textContent = data.ram_pct.toFixed(0) + '%';
            document.getElementById('power-val').textContent = data.total_power_mw.toFixed(0) + ' mW';
            document.getElementById('power-mode').textContent = data.power_mode || 'N/A';

            // Thermal
            const thermalDiv = document.getElementById('thermal-zones');
            thermalDiv.innerHTML = '';
            if (data.thermal_zones) {
                data.thermal_zones.forEach(tz => {
                    const pct = Math.min(tz.temp_c / 100, 1) * 100;
                    const color = tz.temp_c > 85 ? '#f85149' : tz.temp_c > 70 ? '#d29922' : '#3fb950';
                    thermalDiv.innerHTML += `<div class="thermal-row">
                        <span class="name">${tz.name}</span>
                        <div class="bar"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
                        <span class="temp">${tz.temp_c.toFixed(1)}°C</span>
                    </div>`;
                });
            }

            // Thermal indicator
            const maxTemp = data.max_temp_c || 0;
            const ti = document.getElementById('thermal-indicator');
            ti.className = 'indicator ' + (maxTemp > 85 ? 'red' : maxTemp > 70 ? 'orange' : 'green');

            // Throttle alert
            const alertBanner = document.getElementById('alert-banner');
            if (data.is_throttled) {
                alertBanner.textContent = 'THERMAL THROTTLING DETECTED — Performance degraded';
                alertBanner.style.display = 'block';
            } else {
                alertBanner.style.display = 'none';
            }

            // History chart
            updateHistory(data);

            // Fetch enhanced jtop data for per-core CPU and power rails
            fetch('/api/jtop').then(r => r.json()).then(jtop => {
                // Per-core CPU bars
                const coresDiv = document.getElementById('cpu-cores');
                if (jtop.cpu && jtop.cpu.per_core && jtop.cpu.per_core.length > 0) {
                    coresDiv.innerHTML = '';
                    jtop.cpu.per_core.forEach((pct, i) => {
                        const color = pct > 90 ? '#f85149' : pct > 70 ? '#d29922' : '#3fb950';
                        coresDiv.innerHTML += `<div class="thermal-row">
                            <span class="name" style="width:50px">Core ${i}</span>
                            <div class="bar"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
                            <span class="temp" style="width:45px">${pct.toFixed(0)}%</span>
                        </div>`;
                    });
                } else {
                    coresDiv.innerHTML = `<div class="metric"><span class="label">Overall</span><span class="value">${jtop.cpu.overall.toFixed(1)}%</span></div>`;
                }

                // Power rails
                const railsDiv = document.getElementById('power-rails');
                if (jtop.power && jtop.power.rails && jtop.power.rails.length > 0) {
                    railsDiv.innerHTML = '';
                    jtop.power.rails.forEach(rail => {
                        railsDiv.innerHTML += `<div class="metric">
                            <span class="label">${rail.name}</span>
                            <span class="value">${rail.current_mw.toFixed(0)} mW</span>
                        </div>`;
                    });
                }
                document.getElementById('total-power').textContent = jtop.power.total_mw.toFixed(0) + ' mW';
            }).catch(() => {});
        });

        on('alert', (data) => {
            const banner = document.getElementById('alert-banner');
            banner.textContent = data.message;
            banner.style.display = 'block';
            setTimeout(() => { banner.style.display = 'none'; }, 10000);
        });

        on('job_complete', (data) => {
            addJobRow(data);
        });

        on('batch_progress', (data) => {
            const card = document.getElementById('batch-card');
            if (data.active || data.completed_files > 0) {
                card.style.display = '';
                document.getElementById('batch-id').textContent = data.batch_id || '-';
                document.getElementById('batch-files').textContent = data.completed_files + ' / ' + data.total_files;
                document.getElementById('batch-current').textContent = data.current_file ? data.current_file.split('/').pop() : '-';
                document.getElementById('batch-file-pct').textContent = data.current_file_pct + '%';
                document.getElementById('batch-overall-pct').textContent = data.overall_pct + '%';
                document.getElementById('batch-bar').style.width = data.overall_pct + '%';
                const ind = document.getElementById('batch-indicator');
                ind.className = 'indicator ' + (data.active ? 'pulse-green' : 'green');
                // Per-file list
                const fl = document.getElementById('batch-file-list');
                if (data.files && data.files.length > 0) {
                    let html = '';
                    data.files.forEach(f => {
                        const color = f.status === 'done' ? 'var(--accent-green)' : f.status === 'active' ? 'var(--accent-blue)' : 'var(--text-secondary)';
                        const name = f.name ? f.name.split('/').pop() : '?';
                        html += '<div class="metric"><span class="label">' + esc(name) + '</span><span class="value" style="color:' + color + '">' + (f.pct || 0) + '%</span></div>';
                    });
                    fl.innerHTML = html;
                }
            } else {
                card.style.display = 'none';
            }
        });

        // Request batch status on connect
        on('connect', () => { if(socket) socket.emit('request_batch_status'); });

        // Drawing
        function drawGauge(canvasId, pct, color) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const cx = 50, cy = 50, r = 40;
            ctx.clearRect(0, 0, 100, 100);

            // Background arc
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0.75 * Math.PI, 2.25 * Math.PI);
            ctx.strokeStyle = '#30363d';
            ctx.lineWidth = 8;
            ctx.lineCap = 'round';
            ctx.stroke();

            // Value arc
            const angle = 0.75 * Math.PI + (pct / 100) * 1.5 * Math.PI;
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0.75 * Math.PI, angle);
            ctx.strokeStyle = color;
            ctx.lineWidth = 8;
            ctx.lineCap = 'round';
            ctx.stroke();
        }

        function updateHistory(data) {
            const now = new Date().toLocaleTimeString();
            historyData.timestamps.push(now);
            historyData.gpu.push(data.gpu_util_pct);
            historyData.cpu.push(data.cpu_pct);
            historyData.temp.push(data.max_temp_c);

            if (historyData.timestamps.length > 120) {
                historyData.timestamps.shift();
                historyData.gpu.shift();
                historyData.cpu.shift();
                historyData.temp.shift();
            }

            if (!historyChart) {
                const ctx = document.getElementById('history-chart').getContext('2d');
                historyChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: historyData.timestamps,
                        datasets: [
                            { label: 'GPU %', data: historyData.gpu, borderColor: '#58a6ff', fill: false, tension: 0.3, pointRadius: 0 },
                            { label: 'CPU %', data: historyData.cpu, borderColor: '#3fb950', fill: false, tension: 0.3, pointRadius: 0 },
                            { label: 'Temp °C', data: historyData.temp, borderColor: '#f85149', fill: false, tension: 0.3, pointRadius: 0, yAxisID: 'y1' },
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: { duration: 0 },
                        plugins: { legend: { labels: { color: '#8b949e' } } },
                        scales: {
                            x: { ticks: { color: '#8b949e', maxTicksLimit: 10 }, grid: { color: '#30363d' } },
                            y: { min: 0, max: 100, ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
                            y1: { position: 'right', min: 20, max: 100, ticks: { color: '#f85149' }, grid: { display: false } }
                        }
                    }
                });
            } else {
                historyChart.update();
            }
        }

        function badge(available) {
            return available
                ? '<span class="status-badge available">Available</span>'
                : '<span class="status-badge unavailable">N/A</span>';
        }

        function formatMB(mb) {
            if (!mb) return 'N/A';
            return mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : mb + ' MB';
        }

        function formatBytes(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
            if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + ' MB';
            return (bytes / 1073741824).toFixed(2) + ' GB';
        }

        function esc(s) {
            const d = document.createElement('div');
            d.appendChild(document.createTextNode(s));
            return d.innerHTML;
        }

        // ─── File Browser Modal ──────────────────────────────────────────
        let browserTargetInput = '';
        let currentBrowserPath = '';

        function openBrowser(inputId) {
            browserTargetInput = inputId;
            document.getElementById('browser-modal').style.display = 'flex';
            browseTo('');
        }

        function closeBrowser() {
            document.getElementById('browser-modal').style.display = 'none';
        }

        function selectCurrentDir() {
            if (browserTargetInput && currentBrowserPath) {
                document.getElementById(browserTargetInput).value = currentBrowserPath;
            }
            closeBrowser();
        }

        function selectFileForCompress(path) {
            if (browserTargetInput) {
                document.getElementById(browserTargetInput).value = path;
            }
            closeBrowser();
        }

        // Close modal on Escape or clicking outside
        document.getElementById('browser-modal').addEventListener('click', function(e) {
            if (e.target === this) closeBrowser();
        });
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeBrowser();
        });

        // ─── Decompress ─────────────────────────────────────────────────
        function submitDecompress() {
            const input = document.getElementById('decompress-input').value;
            const resultDiv = document.getElementById('decompress-result');
            if (!input) { resultDiv.innerHTML = '<span style="color:#f85149">Enter a file path</span>'; return; }
            resultDiv.innerHTML = '<span style="color:#58a6ff">Decompressing...</span>';
            fetch('/api/decompress', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({input_path: input})
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    resultDiv.innerHTML = '<span style="color:#f85149">' + esc(data.error) + '</span>';
                } else {
                    resultDiv.innerHTML = '<span style="color:#3fb950">Done!</span> ' +
                        formatBytes(data.input_size) + ' \\u2192 ' + formatBytes(data.output_size) +
                        ' in ' + data.elapsed_s.toFixed(2) + 's<br>' +
                        '<span style="color:var(--text-secondary);font-size:11px">Output: ' + esc(data.output_path) + '</span>';
                    addJobRow({...data, processor: 'decompress', ratio: 0, throughput_mbps: 0, algorithm: 'decompress'});
                }
            })
            .catch(e => { resultDiv.innerHTML = '<span style="color:#f85149">' + esc(String(e)) + '</span>'; });
        }

        function analyzeRoute() {
            const input = document.getElementById('compress-input').value;
            const resultDiv = document.getElementById('compress-result');
            if (!input) { resultDiv.innerHTML = '<span style="color:#f85149">Enter a path</span>'; return; }
            resultDiv.innerHTML = '<span style="color:#58a6ff">Analyzing...</span>';
            fetch('/api/route', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ input_path: input })
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    resultDiv.innerHTML = `<span style="color:#f85149">${esc(data.error)}</span>`;
                } else {
                    resultDiv.innerHTML = `<pre style="color:var(--accent-cyan);white-space:pre-wrap;font-size:12px;margin:0">${esc(data.explanation)}</pre>`;
                }
            })
            .catch(e => { resultDiv.innerHTML = `<span style="color:#f85149">${esc(String(e))}</span>`; });
        }

        function submitCompress() {
            const input = document.getElementById('compress-input').value;
            const mode = document.getElementById('compress-mode').value;
            const quality = document.getElementById('compress-quality').value;
            const resultDiv = document.getElementById('compress-result');

            if (!input) { resultDiv.innerHTML = '<span style="color:#f85149">Enter a path</span>'; return; }

            resultDiv.innerHTML = '<span style="color:#58a6ff">Processing...</span>';

            fetch('/api/compress', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ input_path: input, mode: mode, quality: quality })
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    resultDiv.innerHTML = `<span style="color:#f85149">${esc(data.error)}</span>`;
                } else {
                    resultDiv.innerHTML = `<span style="color:#3fb950">Done!</span> ${formatBytes(data.input_size)} → ${formatBytes(data.output_size)} (${data.ratio.toFixed(2)}x) in ${data.elapsed_s.toFixed(2)}s via <b>${esc(data.processor)}</b>` +
                        (data.reason ? `<br><span style="color:var(--text-secondary);font-size:11px">${esc(data.reason)}</span>` : '');
                    addJobRow(data);
                }
            })
            .catch(e => { resultDiv.innerHTML = `<span style="color:#f85149">${esc(String(e))}</span>`; });
        }

        function makeJobRow(data) {
            const filename = data.input_path ? data.input_path.split('/').pop() : '?';
            const savings = data.savings_pct != null ? data.savings_pct.toFixed(1) + '%' : (data.input_size && data.output_size && data.processor !== 'decompress' ? ((1 - data.output_size / data.input_size) * 100).toFixed(1) + '%' : '-');
            const savingsColor = data.processor === 'decompress' ? 'var(--accent-blue)' : (parseFloat(savings) > 0 ? 'var(--accent-green)' : 'var(--text-secondary)');
            const when = data.timestamp || '-';
            return '<tr>' +
                '<td title="' + esc(data.input_path || '') + '">' + esc(filename) + '</td>' +
                '<td>' + formatBytes(data.input_size || 0) + '</td>' +
                '<td>' + formatBytes(data.output_size || 0) + '</td>' +
                '<td>' + (data.ratio ? data.ratio.toFixed(2) + 'x' : '-') + '</td>' +
                '<td style="color:' + savingsColor + '">' + savings + '</td>' +
                '<td>' + (data.elapsed_s ? data.elapsed_s.toFixed(2) + 's' : '-') + '</td>' +
                '<td>' + (data.throughput_mbps ? data.throughput_mbps.toFixed(1) + ' MB/s' : '-') + '</td>' +
                '<td style="color:#3fb950">' + esc(data.processor || '-') + '</td>' +
                '<td>' + esc(data.algorithm || '-') + '</td>' +
                '<td>' + esc(data.status || '-') + '</td>' +
                '<td style="color:var(--text-secondary);font-size:11px">' + esc(when) + '</td>' +
                '</tr>';
        }

        function addJobRow(data) {
            const tbody = document.getElementById('job-tbody');
            tbody.insertAdjacentHTML('afterbegin', makeJobRow(data));
            if (tbody.children.length > 50) tbody.removeChild(tbody.lastChild);
            // Also add to all-jobs
            const allTbody = document.getElementById('all-job-tbody');
            allTbody.insertAdjacentHTML('afterbegin', makeJobRow(data));
            // Refresh stats
            fetch('/api/jobs/all').then(r => r.json()).then(updateStats);
        }

        function toggleSection(id) {
            const el = document.getElementById(id);
            const arrow = document.getElementById(id + '-arrow');
            if (el.style.display === 'none') {
                el.style.display = '';
                arrow.innerHTML = '&#9660;';
            } else {
                el.style.display = 'none';
                arrow.innerHTML = '&#9654;';
            }
        }

        function clearJobs(scope) {
            if (!confirm('Clear ' + scope + ' job history?')) return;
            fetch('/api/jobs/clear', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({scope: scope})
            }).then(r => r.json()).then(data => {
                if (scope === 'all') {
                    document.getElementById('job-tbody').innerHTML = '';
                    document.getElementById('all-job-tbody').innerHTML = '';
                } else {
                    document.getElementById('job-tbody').innerHTML = '';
                }
            });
        }

        function exportAllJobs() {
            fetch('/api/jobs/all').then(r => r.json()).then(jobs => {
                const blob = new Blob([JSON.stringify(jobs, null, 2)], {type: 'application/json'});
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'hammerio-all-jobs.json';
                a.click();
            });
        }

        // ─── REST API Fallback Loading ───────────────────────────────────
        // Load data via REST on page load so cards populate even if
        // WebSocket is slow or unavailable.
        console.log('[HammerIO] Loading data via REST API...');

        // Hardware profile
        fetch('/api/hardware').then(r => r.json()).then(data => {
            console.log('[HammerIO] Hardware loaded:', data.platform);
            document.getElementById('hw-platform').textContent = data.platform || '-';
            document.getElementById('hw-arch').textContent = data.architecture || '-';
            document.getElementById('hw-gpu').textContent = data.cuda?.device || '-';
            document.getElementById('hw-cuda').textContent = data.cuda?.version || '-';
            const memMb = data.cuda?.memory_mb || data.ram_mb;
            document.getElementById('hw-gpu-mem').textContent = memMb ? formatMB(memMb) : '-';
            document.getElementById('hw-nvenc').innerHTML = badge(data.nvenc?.available);
            document.getElementById('hw-nvcomp').innerHTML = badge(data.nvcomp?.available);
            document.getElementById('hw-vpi').innerHTML = badge(data.vpi?.available);
            document.getElementById('hw-cpu').textContent = data.cpu_cores || '-';
            document.getElementById('hw-ram').textContent = data.ram_mb ? formatMB(data.ram_mb) : '-';
            document.getElementById('hw-indicator').className = 'indicator ' + (data.cuda?.available ? 'green' : 'orange');
            // Routes
            if (data.routes) {
                const routesDiv = document.getElementById('routes');
                routesDiv.innerHTML = '';
                for (const [workload, processor] of Object.entries(data.routes)) {
                    routesDiv.innerHTML += '<div class="route-item"><span class="workload">' +
                        workload.replace('_', ' ') + '</span><span class="processor">' + processor + '</span></div>';
                }
            }
        }).catch(() => {});

        // Telemetry snapshot
        fetch('/api/telemetry').then(r => r.json()).then(data => {
            console.log('[HammerIO] Telemetry loaded:', data.cpu_pct?.toFixed(1) + '% CPU, ' + data.max_temp_c?.toFixed(1) + 'C');
            drawGauge('gpu-gauge', data.gpu_util_pct || 0, '#58a6ff');
            drawGauge('cpu-gauge', data.cpu_pct || 0, '#3fb950');
            drawGauge('ram-gauge', data.ram_pct || 0, '#bc8cff');
            document.getElementById('gpu-pct').textContent = (data.gpu_util_pct || 0).toFixed(0) + '%';
            document.getElementById('cpu-pct').textContent = (data.cpu_pct || 0).toFixed(0) + '%';
            document.getElementById('ram-pct').textContent = (data.ram_pct || 0).toFixed(0) + '%';
            document.getElementById('power-val').textContent = (data.total_power_mw || 0).toFixed(0) + ' mW';
            document.getElementById('power-mode').textContent = data.power_mode || 'N/A';
            // Thermal zones
            if (data.thermal_zones) {
                const thermalDiv = document.getElementById('thermal-zones');
                thermalDiv.innerHTML = '';
                data.thermal_zones.forEach(tz => {
                    const pct = Math.min(tz.temp_c / 100, 1) * 100;
                    const color = tz.temp_c > 85 ? '#f85149' : tz.temp_c > 70 ? '#d29922' : '#3fb950';
                    thermalDiv.innerHTML += '<div class="thermal-row"><span class="name">' + tz.name +
                        '</span><div class="bar"><div class="bar-fill" style="width:' + pct + '%;background:' + color +
                        '"></div></div><span class="temp">' + tz.temp_c.toFixed(1) + '\u00B0C</span></div>';
                });
            }
        }).catch(() => {});

        // System info
        fetch('/api/system').then(r => r.json()).then(data => {
            if (document.getElementById('sys-hostname')) {
                document.getElementById('sys-hostname').textContent = data.hostname || '-';
                document.getElementById('sys-uptime').textContent = data.uptime || '-';
                document.getElementById('sys-procs').textContent = data.process_count || '-';
            }
        }).catch(() => {});

        // jtop data for per-core CPU and power
        fetch('/api/jtop').then(r => r.json()).then(jtop => {
            const coresDiv = document.getElementById('cpu-cores');
            if (jtop.cpu && jtop.cpu.per_core && jtop.cpu.per_core.length > 0) {
                coresDiv.innerHTML = '';
                jtop.cpu.per_core.forEach((pct, i) => {
                    const color = pct > 90 ? '#f85149' : pct > 70 ? '#d29922' : '#3fb950';
                    coresDiv.innerHTML += '<div class="thermal-row"><span class="name" style="width:50px">Core ' + i +
                        '</span><div class="bar"><div class="bar-fill" style="width:' + pct + '%;background:' + color +
                        '"></div></div><span class="temp" style="width:45px">' + pct.toFixed(0) + '%</span></div>';
                });
            }
            const railsDiv = document.getElementById('power-rails');
            if (jtop.power && jtop.power.rails && jtop.power.rails.length > 0) {
                railsDiv.innerHTML = '';
                jtop.power.rails.forEach(rail => {
                    railsDiv.innerHTML += '<div class="metric"><span class="label">' + rail.name +
                        '</span><span class="value">' + rail.current_mw.toFixed(0) + ' mW</span></div>';
                });
            }
            document.getElementById('total-power').textContent = (jtop.power?.total_mw || 0).toFixed(0) + ' mW';
        }).catch(() => {});

        function updateStats(jobs) {
            const compress = jobs.filter(j => j.processor !== 'decompress');
            const decomp = jobs.filter(j => j.processor === 'decompress');
            const totalIn = compress.reduce((s, j) => s + (j.input_size || 0), 0);
            const totalOut = compress.reduce((s, j) => s + (j.output_size || 0), 0);
            const saved = totalIn - totalOut;
            const ratios = compress.filter(j => j.ratio > 0).map(j => j.ratio);
            const avgRatio = ratios.length ? (ratios.reduce((a, b) => a + b, 0) / ratios.length) : 0;
            const savingsPcts = compress.filter(j => j.input_size > 0).map(j => (1 - j.output_size / j.input_size) * 100);
            const avgSavings = savingsPcts.length ? (savingsPcts.reduce((a, b) => a + b, 0) / savingsPcts.length) : 0;
            const speeds = jobs.filter(j => j.throughput_mbps > 0).map(j => j.throughput_mbps);
            const avgSpeed = speeds.length ? (speeds.reduce((a, b) => a + b, 0) / speeds.length) : 0;
            const totalTime = jobs.reduce((s, j) => s + (j.elapsed_s || 0), 0);
            const gpuJobs = jobs.filter(j => (j.processor || '').toLowerCase().includes('gpu') || (j.processor || '').toLowerCase().includes('nvcomp')).length;
            const cpuJobs = jobs.filter(j => j.processor !== 'decompress' && !((j.processor || '').toLowerCase().includes('gpu') || (j.processor || '').toLowerCase().includes('nvcomp'))).length;

            document.getElementById('stat-total').textContent = jobs.length;
            document.getElementById('stat-compress').textContent = compress.length;
            document.getElementById('stat-decompress').textContent = decomp.length;
            document.getElementById('stat-input-size').textContent = formatBytes(totalIn);
            document.getElementById('stat-output-size').textContent = formatBytes(totalOut);
            document.getElementById('stat-saved').textContent = formatBytes(Math.max(0, saved));
            document.getElementById('stat-avg-ratio').textContent = avgRatio > 0 ? avgRatio.toFixed(2) + 'x' : '-';
            document.getElementById('stat-avg-savings').textContent = avgSavings > 0 ? avgSavings.toFixed(1) + '%' : '-';
            document.getElementById('stat-avg-speed').textContent = avgSpeed > 0 ? avgSpeed.toFixed(1) + ' MB/s' : '-';
            document.getElementById('stat-total-time').textContent = totalTime > 60 ? (totalTime / 60).toFixed(1) + 'm' : totalTime.toFixed(1) + 's';
            document.getElementById('stat-gpu').textContent = gpuJobs;
            document.getElementById('stat-cpu').textContent = cpuJobs;
        }

        // Fetch existing jobs on load — recent (last 100) + all
        fetch('/api/jobs').then(r => r.json()).then(jobs => {
            jobs.reverse().forEach(j => {
                document.getElementById('job-tbody').insertAdjacentHTML('beforeend', makeJobRow(j));
            });
        });
        fetch('/api/jobs/all').then(r => r.json()).then(jobs => {
            const allTbody = document.getElementById('all-job-tbody');
            jobs.reverse().forEach(j => {
                allTbody.insertAdjacentHTML('beforeend', makeJobRow(j));
            });
            updateStats(jobs);
        });

        // ─── Polling — refresh telemetry every 2 seconds ─────────────────
        function pollTelemetry() {
            fetch('/api/telemetry').then(r => r.json()).then(data => {
                drawGauge('gpu-gauge', data.gpu_util_pct || 0, '#58a6ff');
                drawGauge('cpu-gauge', data.cpu_pct || 0, '#3fb950');
                drawGauge('ram-gauge', data.ram_pct || 0, '#bc8cff');
                document.getElementById('gpu-pct').textContent = (data.gpu_util_pct || 0).toFixed(0) + '%';
                document.getElementById('cpu-pct').textContent = (data.cpu_pct || 0).toFixed(0) + '%';
                document.getElementById('ram-pct').textContent = (data.ram_pct || 0).toFixed(0) + '%';
                document.getElementById('power-val').textContent = (data.total_power_mw || 0).toFixed(0) + ' mW';
                document.getElementById('power-mode').textContent = data.power_mode || 'N/A';
                // Update thermal
                if (data.thermal_zones) {
                    const thermalDiv = document.getElementById('thermal-zones');
                    thermalDiv.innerHTML = '';
                    data.thermal_zones.forEach(tz => {
                        const pct = Math.min(tz.temp_c / 100, 1) * 100;
                        const color = tz.temp_c > 85 ? '#f85149' : tz.temp_c > 70 ? '#d29922' : '#3fb950';
                        thermalDiv.innerHTML += '<div class="thermal-row"><span class="name">' + tz.name +
                            '</span><div class="bar"><div class="bar-fill" style="width:' + pct + '%;background:' + color +
                            '"></div></div><span class="temp">' + tz.temp_c.toFixed(1) + '\\u00B0C</span></div>';
                    });
                    const ti = document.getElementById('thermal-indicator');
                    const maxT = data.max_temp_c || 0;
                    ti.className = 'indicator ' + (maxT > 85 ? 'red' : maxT > 70 ? 'orange' : 'green');
                }
                // Update history chart
                updateHistory(data);
            }).catch(() => {});

            // Update per-core CPU and power rails
            fetch('/api/jtop').then(r => r.json()).then(jtop => {
                const coresDiv = document.getElementById('cpu-cores');
                if (jtop.cpu && jtop.cpu.per_core && jtop.cpu.per_core.length > 0) {
                    coresDiv.innerHTML = '';
                    jtop.cpu.per_core.forEach((pct, i) => {
                        const color = pct > 90 ? '#f85149' : pct > 70 ? '#d29922' : '#3fb950';
                        coresDiv.innerHTML += '<div class="thermal-row"><span class="name" style="width:50px">Core ' + i +
                            '</span><div class="bar"><div class="bar-fill" style="width:' + pct + '%;background:' + color +
                            '"></div></div><span class="temp" style="width:45px">' + pct.toFixed(0) + '%</span></div>';
                    });
                }
                const railsDiv = document.getElementById('power-rails');
                if (jtop.power && jtop.power.rails && jtop.power.rails.length > 0) {
                    railsDiv.innerHTML = '';
                    jtop.power.rails.forEach(rail => {
                        railsDiv.innerHTML += '<div class="metric"><span class="label">' + rail.name +
                            '</span><span class="value">' + rail.current_mw.toFixed(0) + ' mW</span></div>';
                    });
                }
                document.getElementById('total-power').textContent = (jtop.power?.total_mw || 0).toFixed(0) + ' mW';
            }).catch(() => {});
        }
        setInterval(pollTelemetry, 2000);
        console.log('[HammerIO] Polling started (2s interval)');

        // ─── File Browser ────────────────────────────────────────────────
        // currentBrowserPath is defined in the modal section above

        function fileIcon(entry) {
            if (entry.type === 'dir') return '&#128193;';
            const ext = entry.extension.toLowerCase();
            const videoExts = ['mp4','mkv','avi','mov','webm','flv','wmv','ts','m4v'];
            const imageExts = ['jpg','jpeg','png','gif','bmp','webp','svg','tiff'];
            const audioExts = ['mp3','wav','flac','aac','ogg','wma','m4a'];
            const archiveExts = ['zip','tar','gz','bz2','xz','7z','rar','zst','lz4'];
            if (videoExts.includes(ext)) return '&#127910;';
            if (imageExts.includes(ext)) return '&#128247;';
            if (audioExts.includes(ext)) return '&#127925;';
            if (archiveExts.includes(ext)) return '&#128230;';
            return '&#128196;';
        }

        function browseTo(dirPath) {
            const url = dirPath ? '/api/browse?path=' + encodeURIComponent(dirPath) : '/api/browse';
            fetch(url).then(r => r.json()).then(data => {
                if (data.error) {
                    document.getElementById('file-list').innerHTML =
                        '<div class="metric"><span class="label" style="color:var(--accent-red)">' + data.error + '</span></div>';
                    return;
                }
                currentBrowserPath = data.path;

                // Build breadcrumb
                const bc = document.getElementById('file-breadcrumb');
                const parts = data.path.split('/').filter(Boolean);
                let html = '<a data-path="/" class="bc-link">/</a>';
                let accumulated = '';
                parts.forEach((p, i) => {
                    accumulated += '/' + p;
                    html += '<span class="sep">/</span>';
                    if (i === parts.length - 1) {
                        html += '<span style="color:var(--text-primary);font-weight:600">' + p + '</span>';
                    } else {
                        html += '<a data-path="' + accumulated.replace(/"/g, '&quot;') + '" class="bc-link">' + p + '</a>';
                    }
                });
                bc.innerHTML = html;
                bc.querySelectorAll('.bc-link').forEach(a => {
                    a.style.cursor = 'pointer';
                    a.style.color = 'var(--accent-cyan)';
                    a.addEventListener('click', () => browseTo(a.dataset.path));
                });

                // Build file list
                const fl = document.getElementById('file-list');
                if (data.entries.length === 0) {
                    fl.innerHTML = '<div class="metric"><span class="label">Empty directory</span></div>';
                    return;
                }

                let listHtml = '';
                // Parent directory link
                if (data.parent) {
                    listHtml += '<div class="file-entry dir" data-browse="' + data.parent.replace(/"/g, '&quot;') + '">'
                        + '<span class="file-icon">&#128194;</span>'
                        + '<span class="file-name">..</span>'
                        + '<span class="file-size"></span>'
                        + '<span class="file-type">dir</span></div>';
                }

                data.entries.forEach(e => {
                    const fullPath = data.path + (data.path.endsWith('/') ? '' : '/') + e.name;
                    if (e.type === 'dir') {
                        listHtml += '<div class="file-entry dir" data-browse="' + fullPath.replace(/"/g, '&quot;') + '">'
                            + '<span class="file-icon">' + fileIcon(e) + '</span>'
                            + '<span class="file-name">' + e.name + '</span>'
                            + '<span class="file-size"></span>'
                            + '<span class="file-type">dir</span></div>';
                    } else {
                        listHtml += '<div class="file-entry file" data-select="' + fullPath.replace(/"/g, '&quot;') + '">'
                            + '<span class="file-icon">' + fileIcon(e) + '</span>'
                            + '<span class="file-name">' + e.name + '</span>'
                            + '<span class="file-size">' + formatBytes(e.size) + '</span>'
                            + '<span class="file-type">' + (e.extension || '-') + '</span></div>';
                    }
                });
                fl.innerHTML = listHtml;
                // Attach click handlers via event delegation
                fl.querySelectorAll('[data-browse]').forEach(el => {
                    el.style.cursor = 'pointer';
                    el.addEventListener('click', () => browseTo(el.dataset.browse));
                });
                fl.querySelectorAll('[data-select]').forEach(el => {
                    el.style.cursor = 'pointer';
                    el.addEventListener('click', () => selectFileForCompress(el.dataset.select));
                });
            }).catch(err => {
                document.getElementById('file-list').innerHTML =
                    '<div class="metric"><span class="label" style="color:var(--accent-red)">Error: ' + err + '</span></div>';
            });
        }

        function selectFileForCompress(filePath) {
            document.getElementById('compress-input').value = filePath;
            // Scroll to compress card
            document.getElementById('compress-input').scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        // File browser loads on demand via openBrowser() — no auto-load

        // ─── System Info ─────────────────────────────────────────────────
        function loadSystemInfo() {
            fetch('/api/system').then(r => r.json()).then(data => {
                document.getElementById('sys-hostname').textContent = data.hostname || '-';
                document.getElementById('sys-uptime').textContent = data.uptime || '-';
                document.getElementById('sys-procs').textContent = data.process_count || '-';

                // Disk usage
                const diskDiv = document.getElementById('sys-disk');
                let diskHtml = '';
                (data.disk || []).forEach(d => {
                    const color = d.percent > 90 ? 'var(--accent-red)' : d.percent > 75 ? 'var(--accent-orange)' : 'var(--accent-green)';
                    diskHtml += '<div class="disk-bar-container">'
                        + '<div class="disk-label"><span class="mount-name">' + d.mount + '</span>'
                        + '<span class="disk-pct">' + d.used_gb + ' / ' + d.total_gb + ' GB (' + d.percent + '%)</span></div>'
                        + '<div class="bar"><div class="bar-fill" style="width:' + d.percent + '%;background:' + color + '"></div></div>'
                        + '</div>';
                });
                diskDiv.innerHTML = diskHtml || '<div class="metric"><span class="label">No data</span></div>';

                // Network
                const netDiv = document.getElementById('sys-network');
                let netHtml = '';
                (data.network || []).forEach(n => {
                    netHtml += '<div class="net-item"><span class="iface-name">' + n.name + '</span>'
                        + '<span class="iface-ip">' + n.ipv4.join(', ') + '</span></div>';
                });
                netDiv.innerHTML = netHtml || '<div class="metric"><span class="label">No interfaces</span></div>';
            }).catch(() => {});
        }

        // Load system info on startup and refresh every 30s
        loadSystemInfo();
        setInterval(loadSystemInfo, 30000);

        // ─── Dark/Light Theme Toggle ────────────────────────────────────
        (function() {
            const saved = localStorage.getItem('hammerio-theme') || 'dark';
            if (saved === 'light') {
                document.documentElement.setAttribute('data-theme', 'light');
                document.getElementById('theme-toggle').textContent = 'Dark Mode';
            }
        })();

        document.getElementById('theme-toggle').addEventListener('click', function() {
            const current = document.documentElement.getAttribute('data-theme');
            if (current === 'light') {
                document.documentElement.removeAttribute('data-theme');
                this.textContent = 'Light Mode';
                localStorage.setItem('hammerio-theme', 'dark');
            } else {
                document.documentElement.setAttribute('data-theme', 'light');
                this.textContent = 'Dark Mode';
                localStorage.setItem('hammerio-theme', 'light');
            }
        });

        // ─── Toast Notification System ──────────────────────────────────
        function showToast(title, message, type) {
            type = type || '';
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = 'toast' + (type ? ' ' + type : '');
            toast.innerHTML = '<div class="toast-title">' + title + '</div><div class="toast-msg">' + message + '</div>';
            container.appendChild(toast);
            setTimeout(function() {
                toast.style.animation = 'toast-out 0.3s ease-in forwards';
                setTimeout(function() { toast.remove(); }, 300);
            }, 5000);
        }

        // Hook into existing alert handler to also show toast
        on('alert', function(data) {
            showToast('Alert', data.message, 'alert');
        });

        // Hook into job_complete to show toast
        on('job_complete', function(data) {
            const filename = data.input_path ? data.input_path.split('/').pop() : 'Job';
            showToast('Job Complete', filename + ' — ' + (data.ratio ? data.ratio.toFixed(2) + 'x' : '') + ' via ' + (data.processor || '?'), 'success');
        });

        // ─── Auto-refresh live indicator ────────────────────────────────
        let lastTelemetryTime = 0;
        on('telemetry', function() {
            lastTelemetryTime = Date.now();
            const dot = document.getElementById('live-dot');
            if (dot) dot.style.background = 'var(--accent-green)';
        });
        setInterval(function() {
            const dot = document.getElementById('live-dot');
            if (!dot) return;
            if (Date.now() - lastTelemetryTime > 5000) {
                dot.style.background = 'var(--accent-orange)';
                dot.style.animation = 'none';
            } else {
                dot.style.background = '';
                dot.style.animation = '';
            }
        }, 2000);

        // ─── Keyboard Shortcuts ─────────────────────────────────────────
        document.addEventListener('keydown', function(e) {
            // Ignore when typing in input fields
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
            switch (e.key.toLowerCase()) {
                case 'h':
                    document.getElementById('hw-platform').scrollIntoView({ behavior: 'smooth', block: 'center' });
                    break;
                case 't':
                    document.getElementById('history-chart').scrollIntoView({ behavior: 'smooth', block: 'center' });
                    break;
                case 'c':
                    e.preventDefault();
                    document.getElementById('compress-input').focus();
                    document.getElementById('compress-input').scrollIntoView({ behavior: 'smooth', block: 'center' });
                    break;
                case 'd':
                    document.getElementById('theme-toggle').click();
                    break;
            }
        });

        // ─── Export Functions ────────────────────────────────────────────
        function exportTelemetry() {
            window.open('/api/export/telemetry', '_blank');
        }

        function exportJobs() {
            fetch('/api/jobs').then(function(r) { return r.json(); }).then(function(data) {
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'hammerio_jobs.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            });
        }
    </script>
</body>
</html>
"""

# ─── Console Page HTML ─────────────────────────────────────────────────────────

CONSOLE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%230d1117'/%3E%3Cpath d='M8 8v16M24 8v16M8 16h16' stroke='%2339d2c0' stroke-width='3' stroke-linecap='round'/%3E%3Ccircle cx='16' cy='10' r='2.5' fill='%233fb950'/%3E%3C/svg%3E">
    <title>HammerIO Console</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-card: #1c2333;
            --border: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-green: #3fb950;
            --accent-blue: #58a6ff;
            --accent-orange: #d29922;
            --accent-red: #f85149;
            --accent-cyan: #39d2c0;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 20px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .nav-links a {
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 13px;
            font-weight: 600;
            padding: 4px 12px;
            border: 1px solid var(--border);
            border-radius: 6px;
            margin-left: 8px;
        }
        .nav-links a:hover, .nav-links a.active {
            color: var(--accent-green);
            border-color: var(--accent-green);
        }
        .main {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        /* Command Sidebar */
        .sidebar {
            width: 240px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            overflow-y: auto;
            padding: 12px 0;
            flex-shrink: 0;
        }
        .sidebar h3 {
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 8px 16px 4px;
        }
        .cmd-item {
            display: block;
            width: 100%;
            text-align: left;
            background: none;
            border: none;
            color: var(--text-primary);
            padding: 6px 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            cursor: pointer;
            border-left: 3px solid transparent;
        }
        .cmd-item:hover {
            background: var(--bg-card);
            border-left-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }
        .cmd-item .desc {
            display: block;
            font-size: 10px;
            color: var(--text-secondary);
            margin-top: 1px;
        }
        /* Console Area */
        .console-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        #output {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .line-cmd { color: var(--accent-green); }
        .line-out { color: var(--text-primary); }
        .line-err { color: var(--accent-orange); }
        .line-sys { color: var(--accent-cyan); font-style: italic; }
        /* Input Area */
        .input-area {
            border-top: 1px solid var(--border);
            background: var(--bg-secondary);
            display: flex;
            align-items: flex-end;
            padding: 8px 12px;
            gap: 8px;
        }
        .prompt {
            color: var(--accent-green);
            font-family: monospace;
            font-weight: bold;
            font-size: 14px;
            padding-bottom: 6px;
        }
        #cmd-input {
            flex: 1;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 12px 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            border-radius: 6px;
            resize: vertical;
            min-height: 48px;
            max-height: 200px;
            outline: none;
            line-height: 1.5;
        }
        #cmd-input:focus { border-color: var(--accent-cyan); }
        .run-btn {
            background: var(--accent-green);
            color: #000;
            border: none;
            padding: 8px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 700;
            font-size: 13px;
            margin-bottom: 1px;
        }
        .run-btn:hover { opacity: 0.9; }
        .hint {
            font-size: 11px;
            color: var(--text-secondary);
            padding: 4px 12px 8px;
            background: var(--bg-secondary);
        }
        /* Autocomplete */
        .autocomplete {
            position: absolute;
            bottom: 100%;
            left: 0;
            right: 0;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            max-height: 200px;
            overflow-y: auto;
            display: none;
            z-index: 100;
        }
        .autocomplete.show { display: block; }
        .ac-item {
            padding: 6px 12px;
            font-family: monospace;
            font-size: 12px;
            cursor: pointer;
            color: var(--text-primary);
        }
        .ac-item:hover, .ac-item.selected {
            background: var(--accent-cyan);
            color: #000;
        }
        .ac-item .ac-desc {
            color: var(--text-secondary);
            font-size: 10px;
            margin-left: 8px;
        }
        .ac-item:hover .ac-desc, .ac-item.selected .ac-desc { color: #333; }
        .input-wrapper { position: relative; flex: 1; }
        .footer {
            text-align: center; padding: 8px; color: var(--text-secondary);
            font-size: 10px; border-top: 1px solid var(--border);
        }
        .footer a { color: var(--accent-blue); text-decoration: none; }
        @media (max-width: 768px) {
            .sidebar { display: none; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>HammerIO Console</h1>
        <div class="nav-links">
            <a href="/">Dashboard</a>
            <a href="/console" class="active">Console</a>
            <a href="/architecture">Architecture</a>
        </div>
        <div style="color:var(--text-secondary);font-size:11px;text-align:right">
            ResilientMind AI<br>Joseph C McGinty Jr
        </div>
    </div>

    <div class="main">
        <div class="sidebar">
            <h3>Compression</h3>
            <button class="cmd-item" onclick="insertCmd('hammer compress ')">
                compress <span class="desc">Compress file or directory</span>
            </button>
            <button class="cmd-item" onclick="insertCmd('hammer decompress ')">
                decompress <span class="desc">Decompress a file</span>
            </button>
            <button class="cmd-item" onclick="insertCmd('hammer batch ')">
                batch <span class="desc">Batch process directory</span>
            </button>

            <h3>Information</h3>
            <button class="cmd-item" onclick="insertCmd('hammer info --hardware')">
                info --hardware <span class="desc">Show GPU/CPU profile</span>
            </button>
            <button class="cmd-item" onclick="insertCmd('hammer info --telemetry')">
                info --telemetry <span class="desc">Live system metrics</span>
            </button>
            <button class="cmd-item" onclick="insertCmd('hammer info --routes ')">
                info --routes <span class="desc">Show routing for file</span>
            </button>
            <button class="cmd-item" onclick="insertCmd('hammer version')">
                version <span class="desc">Version and credits</span>
            </button>

            <h3>Tools</h3>
            <button class="cmd-item" onclick="insertCmd('hammer config --show')">
                config --show <span class="desc">Show configuration</span>
            </button>
            <button class="cmd-item" onclick="insertCmd('hammer config --generate')">
                config --generate <span class="desc">Generate config file</span>
            </button>
            <button class="cmd-item" onclick="insertCmd('hammer benchmark --quick')">
                benchmark --quick <span class="desc">Run quick benchmark</span>
            </button>
            <button class="cmd-item" onclick="insertCmd('hammer monitor --count 5')">
                monitor --count 5 <span class="desc">5 telemetry snapshots</span>
            </button>

            <h3>System</h3>
            <button class="cmd-item" onclick="insertCmd('nvidia-smi')">
                nvidia-smi <span class="desc">GPU status</span>
            </button>
            <button class="cmd-item" onclick="insertCmd('free -h')">
                free -h <span class="desc">Memory usage</span>
            </button>
            <button class="cmd-item" onclick="insertCmd('df -h')">
                df -h <span class="desc">Disk usage</span>
            </button>
            <button class="cmd-item" onclick="insertCmd('uptime')">
                uptime <span class="desc">System uptime</span>
            </button>
            <button class="cmd-item" onclick="insertCmd('ls -la')">
                ls -la <span class="desc">List files</span>
            </button>

            <h3>Help</h3>
            <button class="cmd-item" onclick="insertCmd('hammer --help')">
                --help <span class="desc">All commands</span>
            </button>
            <button class="cmd-item" onclick="insertCmd('hammer compress --help')">
                compress --help <span class="desc">Compress options</span>
            </button>
            <button class="cmd-item" onclick="insertCmd('hammer batch --help')">
                batch --help <span class="desc">Batch options</span>
            </button>
        </div>

        <div class="console-area">
            <div id="output"><span class="line-sys">HammerIO CLI Console — Type commands below or click sidebar.
Enter = Run | Shift+Enter = New line | Up/Down = History | Tab = Autocomplete
</span></div>
            <div class="hint">Allowed: hammer *, ls, pwd, df, free, nvidia-smi, nvpmodel, uptime, cat, head, tail, wc</div>
            <div class="input-area">
                <span class="prompt">$</span>
                <div class="input-wrapper">
                    <div class="autocomplete" id="autocomplete"></div>
                    <textarea id="cmd-input" rows="1" placeholder="hammer info --hardware" spellcheck="false"></textarea>
                </div>
                <button class="run-btn" onclick="runCmd()">Run</button>
                <button onclick="clearConsole()" style="background:var(--border);color:var(--text-primary);border:none;padding:8px 12px;border-radius:6px;cursor:pointer;font-size:12px;margin-bottom:1px" title="Clear console">Clear</button>
                <button onclick="exportMd()" style="background:var(--accent-blue);color:#fff;border:none;padding:8px 12px;border-radius:6px;cursor:pointer;font-weight:600;font-size:12px;margin-bottom:1px" title="Export as Markdown">Export .md</button>
            </div>
        </div>
    </div>

    <div class="footer">
        Copyright &copy; 2026 <a href="https://resilientmindai.com">ResilientMind AI</a> | Joseph C McGinty Jr | All Rights Reserved
    </div>

    <script>
        const cmdInput = document.getElementById('cmd-input');
        const output = document.getElementById('output');
        const acBox = document.getElementById('autocomplete');
        const history = [];
        let histIdx = -1;
        let acIdx = -1;
        let running = false;

        // All known commands for autocomplete
        const commands = [
            {cmd: 'hammer compress ', desc: 'Compress file/directory'},
            {cmd: 'hammer decompress ', desc: 'Decompress a file'},
            {cmd: 'hammer batch ', desc: 'Batch process directory'},
            {cmd: 'hammer info --hardware', desc: 'Hardware profile'},
            {cmd: 'hammer info --telemetry', desc: 'System telemetry'},
            {cmd: 'hammer info --telemetry --json', desc: 'Telemetry as JSON'},
            {cmd: 'hammer info --routes ', desc: 'Show routing for file'},
            {cmd: 'hammer info --hardware --json', desc: 'Hardware as JSON'},
            {cmd: 'hammer version', desc: 'Version info'},
            {cmd: 'hammer config --show', desc: 'Show config'},
            {cmd: 'hammer config --generate', desc: 'Generate config file'},
            {cmd: 'hammer benchmark --quick', desc: 'Quick benchmark'},
            {cmd: 'hammer benchmark', desc: 'Full benchmark'},
            {cmd: 'hammer monitor --count 5', desc: '5 telemetry snapshots'},
            {cmd: 'hammer webui', desc: 'Start web dashboard'},
            {cmd: 'hammer --help', desc: 'All commands'},
            {cmd: 'hammer compress --help', desc: 'Compress options'},
            {cmd: 'hammer decompress --help', desc: 'Decompress options'},
            {cmd: 'hammer batch --help', desc: 'Batch options'},
            {cmd: 'nvidia-smi', desc: 'GPU status'},
            {cmd: 'free -h', desc: 'Memory usage'},
            {cmd: 'df -h', desc: 'Disk usage'},
            {cmd: 'uptime', desc: 'System uptime'},
            {cmd: 'ls -la', desc: 'List files'},
            {cmd: 'pwd', desc: 'Current directory'},
            {cmd: 'uname -a', desc: 'System info'},
        ];

        function appendLine(text, cls) {
            const span = document.createElement('span');
            span.className = cls || 'line-out';
            span.textContent = text;
            output.appendChild(span);
            output.scrollTop = output.scrollHeight;
        }

        function insertCmd(cmd) {
            cmdInput.value = cmd;
            cmdInput.focus();
            hideAc();
            // Place cursor at end
            cmdInput.setSelectionRange(cmd.length, cmd.length);
        }

        function runCmd() {
            const cmd = cmdInput.value.trim();
            if (!cmd || running) return;
            running = true;

            history.unshift(cmd);
            histIdx = -1;
            cmdInput.value = '';
            autoResize();
            hideAc();

            appendLine('\\n$ ' + cmd + '\\n', 'line-cmd');

            fetch('/api/console', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: cmd})
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    appendLine(data.error + '\\n', 'line-err');
                } else {
                    if (data.stdout) appendLine(data.stdout, 'line-out');
                    if (data.stderr) appendLine(data.stderr, 'line-err');
                    if (!data.stdout && !data.stderr) appendLine('(no output)\\n', 'line-sys');
                    if (data.returncode !== 0) appendLine('[exit code: ' + data.returncode + ']\\n', 'line-err');
                }
                running = false;
            })
            .catch(e => {
                appendLine('Error: ' + e + '\\n', 'line-err');
                running = false;
            });
        }

        // Keyboard handling
        cmdInput.addEventListener('keydown', function(e) {
            // Enter = run (unless Shift held for newline)
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (acBox.classList.contains('show') && acIdx >= 0) {
                    // Select autocomplete item
                    const items = acBox.querySelectorAll('.ac-item');
                    if (items[acIdx]) items[acIdx].click();
                } else {
                    runCmd();
                }
                return;
            }

            // Up/Down for history
            if (e.key === 'ArrowUp' && !acBox.classList.contains('show')) {
                if (history.length > 0) {
                    histIdx = Math.min(histIdx + 1, history.length - 1);
                    cmdInput.value = history[histIdx];
                    e.preventDefault();
                }
                return;
            }
            if (e.key === 'ArrowDown' && !acBox.classList.contains('show')) {
                histIdx = Math.max(histIdx - 1, -1);
                cmdInput.value = histIdx >= 0 ? history[histIdx] : '';
                e.preventDefault();
                return;
            }

            // Tab for autocomplete
            if (e.key === 'Tab') {
                e.preventDefault();
                if (acBox.classList.contains('show')) {
                    const items = acBox.querySelectorAll('.ac-item');
                    acIdx = (acIdx + 1) % items.length;
                    items.forEach((it, i) => it.classList.toggle('selected', i === acIdx));
                } else {
                    showAc();
                }
                return;
            }

            // Escape to close autocomplete
            if (e.key === 'Escape') {
                hideAc();
                return;
            }

            // Up/Down in autocomplete
            if (acBox.classList.contains('show')) {
                const items = acBox.querySelectorAll('.ac-item');
                if (e.key === 'ArrowDown') {
                    acIdx = Math.min(acIdx + 1, items.length - 1);
                    items.forEach((it, i) => it.classList.toggle('selected', i === acIdx));
                    e.preventDefault();
                    return;
                }
                if (e.key === 'ArrowUp') {
                    acIdx = Math.max(acIdx - 1, 0);
                    items.forEach((it, i) => it.classList.toggle('selected', i === acIdx));
                    e.preventDefault();
                    return;
                }
            }
        });

        // Auto-resize textarea
        cmdInput.addEventListener('input', function() {
            autoResize();
            // Show autocomplete as you type
            const val = this.value.trim().toLowerCase();
            if (val.length >= 2) showAc();
            else hideAc();
        });

        function autoResize() {
            cmdInput.style.height = 'auto';
            cmdInput.style.height = Math.min(cmdInput.scrollHeight, 150) + 'px';
        }

        // Autocomplete
        function showAc() {
            const val = cmdInput.value.trim().toLowerCase();
            const matches = commands.filter(c =>
                c.cmd.toLowerCase().includes(val) || c.desc.toLowerCase().includes(val)
            );
            if (matches.length === 0) { hideAc(); return; }

            acBox.innerHTML = '';
            matches.forEach((m, i) => {
                const div = document.createElement('div');
                div.className = 'ac-item';
                div.innerHTML = m.cmd + '<span class="ac-desc">' + m.desc + '</span>';
                div.onclick = () => {
                    cmdInput.value = m.cmd;
                    cmdInput.focus();
                    hideAc();
                };
                acBox.appendChild(div);
            });
            acIdx = -1;
            acBox.classList.add('show');
        }

        function hideAc() {
            acBox.classList.remove('show');
            acIdx = -1;
        }

        // Click outside to close autocomplete
        document.addEventListener('click', function(e) {
            if (!e.target.closest('.input-wrapper')) hideAc();
        });

        // Clear console
        function clearConsole() {
            output.innerHTML = '<span class="line-sys">Console cleared.\\n</span>';
        }

        // Export console output as Markdown
        function exportMd() {
            const lines = [];
            const now = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            lines.push('# HammerIO Console Session');
            lines.push('');
            lines.push('**Date:** ' + new Date().toLocaleString());
            lines.push('**Platform:** HammerIO v0.1.0');
            lines.push('**Creator:** ResilientMind AI | Joseph C McGinty Jr');
            lines.push('');
            lines.push('---');
            lines.push('');

            // Walk through output children and convert to markdown
            const children = output.querySelectorAll('span');
            let inCodeBlock = false;
            children.forEach(span => {
                const text = span.textContent;
                if (!text.trim()) return;

                if (span.classList.contains('line-cmd')) {
                    // Command line — close any open code block, add as heading
                    if (inCodeBlock) { lines.push('```'); inCodeBlock = false; }
                    const cmd = text.replace(/^\\n?\\$\\s*/, '').trim();
                    if (cmd) {
                        lines.push('## `' + cmd + '`');
                        lines.push('');
                        lines.push('```');
                        inCodeBlock = true;
                    }
                } else if (span.classList.contains('line-err')) {
                    if (inCodeBlock) { lines.push('```'); inCodeBlock = false; }
                    lines.push('> **Error:** ' + text.trim());
                    lines.push('');
                } else if (span.classList.contains('line-sys')) {
                    // System messages as italic
                    if (inCodeBlock) { lines.push('```'); inCodeBlock = false; }
                    if (text.trim()) lines.push('*' + text.trim() + '*');
                    lines.push('');
                } else {
                    // Normal output — inside code block
                    if (!inCodeBlock) { lines.push('```'); inCodeBlock = true; }
                    lines.push(text.replace(/\\n$/, ''));
                }
            });
            if (inCodeBlock) lines.push('```');

            lines.push('');
            lines.push('---');
            lines.push('*Exported from HammerIO Console — Copyright 2026 ResilientMind AI*');

            const md = lines.join('\\n');
            const blob = new Blob([md], {type: 'text/markdown'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'hammerio-console-' + now + '.md';
            a.click();
            URL.revokeObjectURL(url);
        }

        // Focus input on page load
        cmdInput.focus();

        // Run initial help command
        insertCmd('hammer --help');
        runCmd();
    </script>
</body>
</html>
"""

# ─── Architecture Page HTML ────────────────────────────────────────────────────

ARCHITECTURE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HammerIO Architecture</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, monospace; background: #0d1117; color: #e6edf3; min-height: 100vh; }
        .header { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 20px; background: linear-gradient(135deg, #39d2c0, #58a6ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .nav a { color: #8b949e; text-decoration: none; font-size: 13px; font-weight: 600; padding: 4px 12px; border: 1px solid #30363d; border-radius: 6px; margin-left: 8px; }
        .nav a:hover, .nav a.active { color: #d29922; border-color: #d29922; }
        .content { max-width: 900px; margin: 40px auto; padding: 0 20px; }
        .svg-box { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; text-align: center; }
        .svg-box object { max-width: 100%; height: auto; }
        .info { margin-top: 24px; color: #8b949e; font-size: 13px; line-height: 1.8; }
        .info h2 { color: #e6edf3; font-size: 16px; margin-bottom: 8px; }
        .footer { text-align: center; padding: 24px; color: #8b949e; font-size: 11px; }
        .footer a { color: #58a6ff; text-decoration: none; }
    </style>
</head>
<body>
    <div class="header">
        <h1>HammerIO Architecture</h1>
        <div class="nav">
            <a href="/">Dashboard</a>
            <a href="/console">Console</a>
            <a href="/architecture" class="active">Architecture</a>
        </div>
        <div style="color:#8b949e;font-size:11px;text-align:right">ResilientMind AI<br>Joseph C McGinty Jr</div>
    </div>
    <div class="content">
        <div class="svg-box">
            <object data="/api/architecture.svg" type="image/svg+xml" width="100%"></object>
        </div>
        <div class="info">
            <h2>Compression Pipeline</h2>
            <p>Every file is profiled (size, type, entropy) and routed to the optimal compressor:</p>
            <ul style="margin:8px 0 0 20px">
                <li><strong>Large files (Jetson/CUDA)</strong> - nvCOMP GPU LZ4 (10+ GB/s decompress)</li>
                <li><strong>General files (macOS)</strong> - Apple LZFSE via Accelerate framework</li>
                <li><strong>Default</strong> - CPU zstd with parallel threading</li>
                <li><strong>Already compressed</strong> - Passthrough (no re-compression)</li>
                <li><strong>GPU failure</strong> - Automatic CPU fallback with logged reason</li>
            </ul>
        </div>
    </div>
    <div class="footer">
        Copyright 2026 <a href="https://resilientmindai.com">ResilientMind AI</a> | Joseph C McGinty Jr
    </div>
</body>
</html>
"""
