# HammerIO Documentation

**GPU where it matters. CPU where it doesn't. Zero configuration.**

Copyright 2026 ResilientMind AI | ResilientMindai.com | Joseph C McGinty Jr

## Contents

- [Quick Start](quickstart.md) — Get running in 60 seconds
- [API Reference](api.md) — Python API documentation
- [Benchmarks](benchmarks.md) — Performance numbers and methodology
- [Jetson Guide](jetson.md) — NVIDIA Jetson deployment guide
- [Configuration](configuration.md) — TOML config reference
- [Web Dashboard](web-dashboard.md) — Dashboard, API endpoints, and WebSocket events
- [Architecture](architecture.md) — System design and routing engine

## What is HammerIO?

HammerIO is a smart compression and media processing library that automatically routes every job — video, bulk data, images, archives — to the most efficient processor available on your hardware.

On an NVIDIA Jetson AGX Orin, that means:
- **Video** → NVENC hardware encoder (6-10x faster than CPU)
- **Bulk data** → nvCOMP GPU compression (5-9x faster)
- **Images** → VPI/CUDA batch processing (7x faster)
- **Everything else** → Parallel CPU zstd

If the GPU is unavailable or busy, HammerIO falls back cleanly and tells you why.

## Quick Install

```bash
pip install hammerio
```

## Quick Example

```python
from hammerio.core.router import JobRouter

router = JobRouter()
job = router.route("input.mp4")
result = router.execute(job)

print(f"Compressed: {result.compression_ratio:.1f}x in {result.elapsed_seconds:.1f}s")
print(f"Processor: {result.processor_used} — {result.routing_reason}")
```

## CLI

```bash
hammer compress video.mp4
hammer compress ./dataset/ --mode bulk --algo lz4
hammer batch ./images/ --output ./compressed/ --workers 4
hammer info --hardware
hammer webui
```
