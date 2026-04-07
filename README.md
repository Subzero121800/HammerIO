<p align="center">
  <img src="docs/architecture.svg" alt="HammerIO Architecture" width="800">
</p>

<h1 align="center">HammerIO</h1>

<p align="center">
  <strong>GPU-accelerated compression for edge AI. GPU where it matters. CPU where it doesn't.</strong><br>
  <em>Created by <strong>ResilientMind AI</strong> | <a href="https://resilientmindai.com">ResilientMindai.com</a> | <strong>Joseph C McGinty Jr</strong></em>
</p>

<p align="center">
  <a href="https://pypi.org/project/hammerio/"><img src="https://img.shields.io/pypi/v/hammerio?style=flat-square&color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/hammerio/"><img src="https://img.shields.io/pypi/pyversions/hammerio?style=flat-square" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/platform-Linux%20%7C%20Jetson%20%7C%20macOS-lightgrey?style=flat-square" alt="Platform"></a>
  <a href="#"><img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia" alt="CUDA"></a>
  <a href="#"><img src="https://img.shields.io/badge/nvCOMP-GPU%20LZ4-76B900?style=flat-square" alt="nvCOMP"></a>
</p>

---

**HammerIO** is a GPU-accelerated compression toolkit that automatically routes files to the fastest available hardware. Large files go to **nvCOMP GPU LZ4** (6+ GB/s decompression). On macOS, it uses **Apple LZFSE** via the Accelerate framework. Everything else goes to **CPU zstd** with parallel threading. No flags, no configuration.

## Installation

```bash
pip install hammerio
```

**GPU acceleration** (NVIDIA Jetson or desktop GPU):
```bash
pip install hammerio[gpu]
```

**Jetson with jtop monitoring:**
```bash
pip install hammerio[jetson]
```

**From source:**
```bash
git clone https://github.com/Subzero121800/HammerIO.git
cd HammerIO && pip install -e .
```

Requires Python 3.10+. GPU features require CUDA 12.x and NVIDIA nvCOMP.

## Quick Start

```bash
# Compress anything -- routing is automatic
hammer compress data.csv
hammer compress ./dataset/ --quality fast
hammer compress archive.tar --algo lz4

# Decompress
hammer decompress data.csv.zst
hammer decompress archive.tar.lz4

# Batch compress a directory
hammer batch ./logs/ --workers 8

# See what HammerIO would do
hammer info --routes ./my_file.csv

# Hardware profile
hammer info --hardware
```

**Python API:**

```python
import hammerio

router = hammerio.JobRouter(quality="fast")
job = router.route("dataset.csv")
result = router.execute(job)
print(f"{result.compression_ratio:.1f}x via {result.processor_used}")
```

## Benchmarks

Measured on **Jetson AGX Orin 64GB**, JetPack 6.2.2, CUDA 12.6, MAXN mode. Realistic mixed data (25% CSV, 25% patterns, 25% semi-random, 25% true entropy).

### Roundtrip Performance (10 GB)

| Method | Compress | Decompress | Ratio | Roundtrip |
|--------|--------:|----------:|------:|----------:|
| CPU zstd-1 | 1,094 MB/s | 733 MB/s | 2.00x | 24.1s |
| CPU zstd-3 | 1,014 MB/s | 741 MB/s | 2.00x | 24.0s |
| **GPU nvCOMP LZ4** | 517 MB/s | **4,258 MB/s** | 1.98x | **22.2s** |

> GPU wins the roundtrip: **1.1x faster** than CPU best. GPU decompression is **5.7x faster** than CPU.

### In-Memory (Pure Algorithm Speed, No Disk I/O)

| Method | Compress | Decompress | Ratio |
|--------|--------:|----------:|------:|
| CPU zstd-1 | 1,747 MB/s | 2,001 MB/s | 2.00x |
| CPU zstd-3 | 1,615 MB/s | 2,088 MB/s | 2.00x |
| **GPU nvCOMP LZ4** | 705 MB/s | **8,537 MB/s** | 1.98x |
| **GPU nvCOMP Snappy** | **1,615 MB/s** | **5,756 MB/s** | 1.68x |

### Random I/O (64KB Blocks)

| Pattern | IOPS | Bandwidth | Latency p99 |
|---------|-----:|----------:|------------:|
| Random Read | 109,299 | 6,831 MB/s | 0.02 ms |
| Random Write | 29,405 | 1,838 MB/s | 0.05 ms |
| Mixed 70/30 R/W | 57,146 | 3,572 MB/s | 0.05 ms |

### Scalability (GPU vs CPU Crossover)

| Size | CPU zstd-3 | GPU LZ4 | GPU Wins? |
|-----:|-----------:|--------:|:---------:|
| 1 MB | 260 MB/s | 29 MB/s | No |
| 10 MB | 341 MB/s | 378 MB/s | Yes |
| 100 MB | 678 MB/s | 693 MB/s | Yes |
| 1 GB | 380 MB/s | 464 MB/s | Yes |

> GPU crossover at ~10 MB. Below that, kernel launch overhead dominates.

Run benchmarks yourself:
```bash
hammer benchmark              # Standard (500MB)
hammer benchmark --quick      # Quick (100MB)
hammer benchmark --10gb       # Stress test (10GB)
hammer benchmark --type memory     # In-memory only
hammer benchmark --type random-io  # Random I/O patterns
hammer benchmark --type scale      # Scalability sweep
```

## Smart Routing

HammerIO profiles every file and routes to the optimal compressor:

```
$ hammer info --hardware

Routing Profile:
  Large Files   -> nvCOMP LZ4 (GPU)      # Files > 500MB
  Datasets      -> nvCOMP LZ4 (GPU)      # CSV, Parquet, NPY, etc.
  General       -> zstd parallel (CPU)    # Default path
  Archives      -> passthrough            # Already compressed
  Text Logs     -> zstd (CPU, high ratio) # Best ratio
```

On macOS Apple Silicon: `General -> Apple LZFSE (Accelerate)` -- hardware-optimized, faster than zstd on M-series.

If the GPU path fails (OOM, driver issue), HammerIO falls back to CPU and logs why.

## Watch Daemon

Drop-folder automation for edge pipelines:

```bash
hammer watch --watch-root ./pipeline --threshold-mb 500

# ./pipeline/
#   compress/        <- drop files here
#   decompress/      <- drop .zst/.lz4 files here
#   compressed/      <- output appears here
#   decompressed/    <- output appears here
#   processed/       <- originals moved here after success
```

GPU routing kicks in for files above the threshold. CPU handles the rest.

## Web Dashboard

```bash
hammer webui             # http://localhost:5000
```

Real-time monitoring: GPU/CPU utilization, thermal zones, power rails, per-core CPU bars, compression job history, file browser, quick compress/decompress, and a built-in CLI console.

## Hardware Compatibility

| Device | Compression Engine | Status |
|--------|-------------------|--------|
| **Jetson AGX Orin** | nvCOMP GPU LZ4 | Primary target, fully tested |
| Jetson Orin NX / Nano | nvCOMP GPU LZ4 | Supported |
| RTX 3000 / 4000 / 5000 | nvCOMP GPU LZ4 | Supported |
| Any CUDA 12.x GPU | nvCOMP GPU LZ4 | Supported |
| **macOS Apple Silicon** | Apple LZFSE (Accelerate) | M1/M2/M3/M4 |
| CPU-only (no GPU) | zstd / gzip / lz4 | Universal fallback |

## CLI Reference

| Command | Description |
|---------|-------------|
| `hammer compress` | Compress file or directory (auto GPU/CPU routing) |
| `hammer decompress` | Decompress .zst, .lz4, .lzfse, .gz, .bz2 files |
| `hammer batch` | Batch compress a directory with parallel workers |
| `hammer watch` | Watch folders and auto-process dropped files |
| `hammer benchmark` | Run GPU vs CPU benchmark suite |
| `hammer info` | Hardware profile, routing decisions, telemetry |
| `hammer config` | Show, save, or generate configuration |
| `hammer monitor` | Live terminal telemetry (jtop-style) |
| `hammer webui` | Launch web dashboard |
| `hammer version` | Version and system info |

## License

```
Copyright 2026 ResilientMind AI | ResilientMindai.com | Joseph C McGinty Jr
Licensed under the Apache License, Version 2.0
```

**Open source** for personal, educational, research, and internal business use.

**Commercial licenses** available for redistribution, SaaS, and OEM embedding.

| License | Price |
|---------|-------|
| Individual commercial | $199 one-time |
| Organization | $999/year |
| OEM / Embedded | Custom |

See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for details.

**Contact:** [joe@resilientmindai.com](mailto:joe@resilientmindai.com) | [resilientmindai.com](https://resilientmindai.com)

---

<p align="center">
  <strong>ResilientMind AI</strong> | <a href="https://resilientmindai.com">resilientmindai.com</a> | <strong>Joseph C McGinty Jr</strong>
</p>
