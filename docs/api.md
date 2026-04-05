# API Reference

Copyright 2026 ResilientMind AI | Joseph C McGinty Jr

## Core Modules

### `hammerio.core.router.JobRouter`

The main entry point for compression and processing jobs.

```python
from hammerio.core.router import JobRouter

router = JobRouter(
    quality="balanced",     # fast, balanced, quality, lossless
    force_mode=None,        # None, "gpu", "cpu"
    max_workers=4,          # Parallel workers for batch
)

# Route a job (analyze + recommend)
job = router.route("input.mp4", output="output.mp4", mode="auto")

# Execute synchronously
result = router.execute(job)

# Execute batch asynchronously
import asyncio
results = asyncio.run(router.execute_batch("./dir/", workers=4))

# Explain routing without executing
print(router.explain_route("input.mp4"))
```

### `hammerio.core.hardware.detect_hardware()`

Detect available hardware and capabilities.

```python
from hammerio.core.hardware import detect_hardware

hw = detect_hardware()
print(hw.platform_name)     # "NVIDIA Jetson AGX Orin Developer Kit"
print(hw.has_cuda)           # True
print(hw.has_nvenc)          # True
print(hw.gpu_memory_mb)      # 62828
print(hw.routing_summary())  # {"video": "NVENC (GPU)", ...}
```

### `hammerio.core.profiler`

Analyze files for optimal compression routing.

```python
from hammerio.core.profiler import profile_file, recommend_compression

fp = profile_file("dataset.csv")
print(fp.category)           # FileCategory.DATASET
print(fp.size_human)         # "1.2 GB"
print(fp.estimated_entropy)  # 4.2

rec = recommend_compression(fp, gpu_available=True)
print(rec.mode)              # CompressionMode.GPU_NVCOMP
print(rec.algorithm)         # "nvcomp_zstd"
print(rec.reason)            # "Large dataset → nvCOMP GPU compression"
```

### `hammerio.core.telemetry.TelemetryCollector`

System monitoring with jtop integration.

```python
from hammerio.core.telemetry import TelemetryCollector

collector = TelemetryCollector(interval_seconds=1.0)

# Single snapshot
snap = collector.get_snapshot()
print(snap.max_temperature)
print(snap.gpu.utilization_pct)
print(snap.total_power_mw)

# Continuous collection
collector.add_callback(lambda s: print(f"GPU: {s.gpu.utilization_pct}%"))
collector.start()
# ... do work ...
print(collector.get_summary())
collector.stop()
```

### `hammerio.core.config`

Configuration management.

```python
from hammerio.core.config import load_config, generate_default_config

config = load_config()  # Auto-discover config file
print(config.quality)   # "balanced"
config.set("video", "gpu_codec", "hevc_nvenc")
config.save()

# Generate default config file
generate_default_config("hammerio.toml")
```

## Encoder Modules

All encoders follow the same interface:

```python
encoder.process(
    input_path,         # Path to input file or directory
    output_path,        # Path for output
    algorithm,          # Algorithm name
    quality,            # Quality preset string
    progress_callback,  # Optional: callback(job_id, pct)
    job_id,             # Optional: job identifier
) -> str               # Returns output path
```

### Available Encoders

| Encoder | Module | GPU | CPU Fallback |
|---------|--------|-----|-------------|
| VideoEncoder | `hammerio.encoders.video` | NVENC | libx264/libx265 |
| BulkEncoder | `hammerio.encoders.bulk` | nvCOMP | zstd/lz4 |
| GeneralEncoder | `hammerio.encoders.general` | - | zstd/gzip/bzip2 |
| ImageEncoder | `hammerio.encoders.image` | OpenCV CUDA | PIL/OpenCV CPU |
| AudioEncoder | `hammerio.encoders.audio` | FFmpeg CUDA | FFmpeg CPU |
| DatasetEncoder | `hammerio.encoders.dataset` | nvCOMP | zstd streaming |
