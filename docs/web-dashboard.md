# Web Dashboard

Copyright 2026 ResilientMind AI | Joseph C McGinty Jr

## Overview

The HammerIO Web Dashboard provides a real-time monitoring and control interface
for media processing operations. It displays hardware status, system telemetry,
batch progress, and allows submitting compression jobs from a browser.

## Starting the Dashboard

### From the CLI

```bash
# Default: http://localhost:5000
hammer webui

# Custom port
hammer webui --port 8080

# Bind to all interfaces (accessible from other machines)
hammer webui --host 0.0.0.0 --port 5000
```

### From Python

```python
from hammerio.web.app import create_app, socketio

app = create_app()
socketio.run(app, host="0.0.0.0", port=5000)
```

### From Docker

```bash
docker run --runtime nvidia -p 5000:5000 hammerio:jetson hammer webui --host 0.0.0.0
```

## Dashboard Cards

The dashboard is organized into cards, each showing a specific aspect of the
system:

- **Hardware Profile** -- Detected platform, GPU, CUDA version, NVENC/nvCOMP/VPI
  availability, CPU cores, and RAM. A green indicator means CUDA is available;
  orange means CPU-only mode.

- **System Utilization** -- Three live gauges showing GPU utilization, CPU
  utilization, and RAM usage in real time. Also shows current power draw and
  power mode.

- **Thermal Zones** -- Bar chart of all thermal sensor readings. Bars turn
  orange above 70C and red above 85C. A throttle banner appears when thermal
  throttling is detected.

- **Smart Routing** -- Shows how each workload type (video, bulk, images, audio,
  dataset, general) is routed based on detected hardware.

- **CPU Cores** -- Per-core utilization bars updated in real time.

- **Power Rails** -- Individual power rail readings (e.g., VDD_GPU_SOC,
  VDD_CPU_CV) with total power consumption.

- **Batch Progress** -- Appears when a batch job is active. Shows the batch ID,
  completed vs total files, current file name and progress, overall completion
  bar, and per-file status list.

- **Architecture** -- Embedded SVG diagram of the HammerIO processing pipeline.

- **Performance History** -- Chart.js line chart tracking GPU %, CPU %, and
  temperature over time (last 120 data points). Includes an "Export JSON" button
  to download the full telemetry history.

- **File Browser** -- Navigate the filesystem, click directories to open them,
  click files to populate the compress input field. Breadcrumb navigation with
  path traversal protection.

- **System Info** -- Hostname, system uptime, process count, disk usage bars for
  key mount points (including NVMe), and network interface IP addresses.

- **Quick Compress** -- Input a file or directory path, select mode (Auto/GPU/CPU)
  and quality (Balanced/Fast/Quality/Lossless), then Analyze routing or Compress
  directly. Results appear inline.

- **Recent Jobs** -- Table of completed jobs showing filename, input/output
  sizes, compression ratio, elapsed time, throughput, processor used, and status.
  Includes "Export JSON" for the job history.

## API Endpoints

All endpoints return JSON unless noted otherwise. CORS headers are set to allow
requests from any origin.

### GET /

Returns the dashboard HTML page.

### GET /health

Health check endpoint.

```json
{"status": "ok", "version": "0.1.0", "service": "hammerio", "timestamp": 1700000000.0, "uptime": 3600.0}
```

### GET /api/version

Version and environment information.

```json
{"version": "0.1.0", "name": "hammerio", "author": "Joseph C McGinty Jr", "license": "Apache-2.0", "python": "3.10.12", "platform": "Linux-5.15.x-aarch64"}
```

### GET /api/hardware

Full hardware profile including CUDA, NVENC, nvCOMP, VPI, CPU, and RAM details
plus the routing summary.

### GET /api/telemetry

Current system snapshot: GPU utilization, CPU usage, RAM, thermal zones, power
readings, fan speed, and throttle status.

### GET /api/telemetry/history?n=60

List of the last `n` telemetry snapshots (default 60).

### GET /api/telemetry/summary

Aggregated telemetry statistics (min/max/avg for key metrics).

### GET /api/export/telemetry?n=3600

Download telemetry history as a JSON file attachment.

### GET /api/jtop

Enhanced jtop-style monitoring data: GPU details, per-core CPU, memory
(including unified memory flag), thermal zones, power rails, fan speed,
throttle state, and Jetson platform/JetPack/L4T versions.

### GET /api/jobs

List of the last 100 completed job results.

### GET /api/batch

Current batch processing status: active flag, batch ID, file counts,
current file progress, and overall completion percentage.

### POST /api/compress

Submit a compression job.

Request body:
```json
{"input_path": "/path/to/file", "output_path": "/path/to/output", "mode": "auto", "quality": "balanced"}
```

Response includes input/output sizes, compression ratio, elapsed time,
throughput, processor used, algorithm, routing reason, and status.

### POST /api/route

Analyze routing for a file without executing.

Request body:
```json
{"input_path": "/path/to/file"}
```

Returns an explanation of how the file would be routed.

### GET /api/browse?path=/some/dir

File browser endpoint. Returns directory contents with name, size, type, and
extension for each entry. Path traversal is blocked.

### GET /api/system

System information: hostname, uptime, disk usage (including NVMe), network
interfaces, and process count.

### GET /api/architecture.svg

Returns the HammerIO architecture diagram as an SVG image.

## WebSocket Events

The dashboard uses Flask-SocketIO for real-time communication. Connect using
the Socket.IO client library.

### Server-emitted events

| Event | Payload | Description |
|-------|---------|-------------|
| `hardware_profile` | Hardware details object | Sent on client connect with full hardware profile |
| `telemetry` | Snapshot dict | Periodic system telemetry (every 1 second) |
| `telemetry_history` | List of snapshot dicts | Response to `request_history` |
| `alert` | `{message, timestamp}` | Thermal or system alert |
| `job_complete` | Job result dict | Fired when a compression job finishes |
| `batch_progress` | Batch status dict | Batch processing progress updates |

### Client-emitted events

| Event | Payload | Description |
|-------|---------|-------------|
| `request_snapshot` | (none) | Request an immediate telemetry snapshot |
| `request_history` | `{last_n: 60}` | Request telemetry history |
| `request_batch_status` | (none) | Request current batch progress |

### Connecting from JavaScript

```javascript
const socket = io("http://localhost:5000");

socket.on("connect", () => {
    console.log("Connected to HammerIO dashboard");
});

socket.on("telemetry", (data) => {
    console.log("GPU:", data.gpu_util_pct, "% CPU:", data.cpu_pct, "%");
});

socket.on("batch_progress", (data) => {
    console.log("Batch:", data.completed_files, "/", data.total_files);
});

socket.on("job_complete", (data) => {
    console.log("Done:", data.input_path, data.ratio + "x");
});
```

### Connecting from Python

```python
import socketio

sio = socketio.Client()

@sio.on("telemetry")
def on_telemetry(data):
    print(f"GPU: {data['gpu_util_pct']}%  CPU: {data['cpu_pct']}%")

@sio.on("batch_progress")
def on_batch(data):
    print(f"Batch: {data['completed_files']}/{data['total_files']} ({data['overall_pct']}%)")

sio.connect("http://localhost:5000")
sio.wait()
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `h` | Scroll to Hardware Profile card |
| `t` | Scroll to Performance History chart |
| `c` | Focus the Quick Compress input field |
| `d` | Toggle dark/light theme |

## Theme Support

The dashboard supports dark mode (default) and light mode. Toggle with the
button in the header or press `d`. The preference is saved in localStorage.
