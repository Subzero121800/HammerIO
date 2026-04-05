# Quick Start

Copyright 2026 ResilientMind AI | Joseph C McGinty Jr

## Installation

```bash
# Basic install
pip install hammerio

# With GPU support
pip install hammerio[gpu]

# With web dashboard
pip install hammerio[web]

# Everything
pip install hammerio[all]

# From source
git clone https://github.com/resilientmindai/hammerio.git
cd hammerio
pip install -e ".[all]"
```

## Verify Installation

```bash
hammer info --hardware
```

This shows your hardware profile and how HammerIO will route jobs.

## Example 1: Compress a Video (60 seconds)

```bash
# Auto-detect: uses NVENC on GPU, libx264 on CPU
hammer compress video.mp4

# Force GPU
hammer compress video.mp4 --mode gpu

# High quality HEVC
hammer compress video.mp4 --quality quality
```

## Example 2: Compress a Dataset

```bash
# Compress a directory of CSV/Parquet files
hammer compress ./training_data/ --mode bulk --algo lz4

# Python API
from hammerio.core.router import JobRouter

router = JobRouter(quality="fast")
job = router.route("./training_data/")
result = router.execute(job)
print(f"{result.compression_ratio:.1f}x compression, {result.throughput_mbps:.0f} MB/s")
```

## Example 3: Batch Image Processing

```bash
# Resize and compress 1000 images
hammer batch ./images/ --output ./compressed/ --workers 8

# Python API
import asyncio
from hammerio.core.router import JobRouter

router = JobRouter()
results = asyncio.run(router.execute_batch("./images/", output_dir="./out/", workers=8))
for r in results:
    print(f"{r.input_path}: {r.compression_ratio:.1f}x via {r.processor_used}")
```

## Web Dashboard

```bash
hammer webui --port 5000
# Open http://localhost:5000
```

## Configuration

Generate a config file:

```bash
python -c "from hammerio.core.config import generate_default_config; generate_default_config()"
```

Edit `hammerio.toml` to customize defaults.
