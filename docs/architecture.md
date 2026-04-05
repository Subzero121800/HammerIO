# Architecture

Copyright 2026 ResilientMind AI | Joseph C McGinty Jr

## Overview

HammerIO uses a **smart routing architecture** that automatically selects the optimal processor for each job based on hardware capabilities, file characteristics, and system state.

```
Input (CLI / API / Web UI)
         │
         ▼
    ┌─────────┐     ┌──────────────────┐
    │ Profiler │────►│ Hardware Detector │
    └────┬────┘     └────────┬─────────┘
         │                   │
         ▼                   ▼
    ┌─────────────────────────────┐
    │        Job Router           │
    │  (route + fallback chain)   │
    └──────────┬──────────────────┘
               │
    ┌──────────┼──────────────────┐
    │          │                  │
    ▼          ▼                  ▼
 ┌──────┐  ┌──────┐  ┌────────────────┐
 │ GPU  │  │ GPU  │  │     CPU        │
 │Video │  │Bulk  │  │ General/Audio  │
 │NVENC │  │nvCOMP│  │ zstd/gzip     │
 └──┬───┘  └──┬───┘  └───────┬────────┘
    │         │               │
    ▼         ▼               ▼
 ┌────────────────────────────────────┐
 │     Result (metrics + output)      │
 └────────────────────────────────────┘
```

## Components

### Input Layer
- **CLI** (`hammerio/cli/main.py`): Typer-based command-line interface
- **Python API** (`hammerio/core/router.py`): `JobRouter` class for programmatic use
- **Web Dashboard** (`hammerio/web/app.py`): Flask + WebSocket monitoring UI

### Core Engine
- **Hardware Detector** (`hammerio/core/hardware.py`): Detects CUDA, NVENC, nvCOMP, VPI, Jetson features
- **Profiler** (`hammerio/core/profiler.py`): Analyzes files — type, size, entropy, compressibility
- **Job Router** (`hammerio/core/router.py`): Routes to optimal encoder with fallback chain
- **Config** (`hammerio/core/config.py`): TOML configuration management

### Encoder Layer
| Encoder | File | GPU Path | CPU Fallback |
|---------|------|----------|-------------|
| Video | `encoders/video.py` | NVENC via FFmpeg | libx264/libx265 |
| Bulk | `encoders/bulk.py` | nvCOMP | zstd/lz4 |
| General | `encoders/general.py` | N/A | zstd/gzip/bzip2 |
| Image | `encoders/image.py` | OpenCV CUDA | PIL/OpenCV CPU |
| Audio | `encoders/audio.py` | FFmpeg CUDA | FFmpeg CPU |
| Dataset | `encoders/dataset.py` | nvCOMP stream | zstd streaming |

### Monitoring Layer
- **Telemetry** (`hammerio/core/telemetry.py`): jtop integration, thermal/power monitoring
- **Web Dashboard**: Real-time WebSocket telemetry, Chart.js visualization

## Routing Decision Flow

1. **Profile** the input file (type, size, entropy)
2. **Check** hardware capabilities (CUDA, NVENC, nvCOMP, VPI)
3. **Select** optimal encoder based on file type + hardware
4. **Execute** with progress tracking
5. **Fallback** to CPU if GPU fails (logged with reason)
6. **Report** metrics: size, ratio, time, throughput, processor, reason

## Data Flow

Files are processed in **streaming mode** — never fully loaded into RAM:
- General compression: 8MB chunks
- Bulk compression: 64MB chunks (configurable)
- Video: FFmpeg handles streaming internally
- Images: One at a time or batch via directory

## Concurrency Model

- **Batch processing**: `asyncio` + `ThreadPoolExecutor` with configurable worker count
- **Telemetry**: Background daemon thread with configurable interval
- **Web dashboard**: Flask-SocketIO with threading async mode
