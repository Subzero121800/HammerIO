<p align="center">
  <img src="docs/architecture.svg" alt="HammerIO Architecture" width="800">
</p>

<h1 align="center">HammerIO</h1>

<p align="center">
  <strong>GPU where it matters. CPU where it doesn't. Zero configuration.</strong><br>
  <em>Created by <strong>ResilientMind AI</strong> | <a href="https://resilientmindai.com">ResilientMindai.com</a> | <strong>Joseph C McGinty Jr</strong></em>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/platform-Linux%20%7C%20Jetson%20%7C%20macOS%20%7C%20Windows-lightgrey?style=flat-square" alt="Platform"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia" alt="CUDA"></a>
  <a href="#"><img src="https://img.shields.io/badge/nvCOMP-GPU%20LZ4-76B900?style=flat-square" alt="nvCOMP"></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-325%20passing-brightgreen?style=flat-square" alt="Tests"></a>
</p>

---

**HammerIO** is a GPU-accelerated compression tool that automatically routes files to the fastest available hardware. Large files go to **nvCOMP GPU LZ4** (10+ GB/s decompression). On macOS, it uses **Apple LZFSE** via the Accelerate framework. Everything else goes to **CPU zstd** with parallel threading. No flags, no configuration.

## Benchmarks

Measured on Jetson AGX Orin 64GB, JetPack 6.2.2, CUDA 12.6, MAXN mode. Mixed realistic data — not synthetic patterns.

| Workload | Method | Throughput | Ratio | Notes |
|---|---|---:|---:|---|
| 100MB mixed data | CPU zstd-1 | 842 MB/s | 2.0x | Baseline |
| 100MB mixed data | CPU zstd-9 | 349 MB/s | 2.0x | Higher effort, same data |
| 100MB mixed data | CPU gzip-6 | 50 MB/s | 2.0x | Compatibility mode |
| 96MB ML dataset | **GPU nvCOMP LZ4** | **2,639 MB/s** | **3.7x** | CUDA-accelerated |
| GPU LZ4 decompress | **GPU nvCOMP** | **10,533 MB/s** | — | 4.3x faster than CPU |
| 3.2MB server logs | CPU zstd-3 | 410 MB/s | 3.5x | Realistic varied logs |
| 5.4MB sensor CSV | CPU zstd-3 | 415 MB/s | 3.6x | 100K rows, varied values |
| 20MB ML weights | CPU zstd-1 | 820 MB/s | 1.1x | Float32 tensors |
| 40MB video (archive) | CPU zstd | 332 MB/s | 1.0x | Already compressed, passthrough |

> Real numbers from `hammer benchmark --quick`. Ratios depend on data entropy — text/CSV compresses well, random/pre-compressed data does not.

## Smart Routing

HammerIO profiles every file and routes to the optimal compressor:

```
$ hammer info --hardware

Routing Profile:
  Large Files   → nvCOMP LZ4 (GPU)      # Files > 500MB
  Datasets      → nvCOMP LZ4 (GPU)      # CSV, Parquet, NPY, etc.
  General       → zstd parallel (CPU)    # Default path
  Archives      → passthrough            # Already compressed
  Text Logs     → zstd (CPU, high ratio) # Best ratio
```

On macOS Apple Silicon: `General → Apple LZFSE (Accelerate)` — hardware-optimized, faster than zstd on M-series.

If the GPU path fails (OOM, driver issue), HammerIO falls back to CPU and logs why.

## Install

```bash
pip install hammerio

# Or from source
git clone https://github.com/Subzero121800/HammerIO.git
cd HammerIO
./setup_venv.sh
```

Works on Jetson (ARM64), macOS (Apple Silicon), Linux (x86), and Windows.

## Quick Start

```bash
# Compress anything — routing is automatic
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

## Watch Daemon

Drop-folder automation for edge pipelines. Files dropped into `compress/` get compressed automatically.

```bash
hammer watch --watch-root ./pipeline --threshold-mb 500

# ./pipeline/
#   compress/        ← drop files here
#   decompress/      ← drop .zst/.lz4 files here
#   compressed/      ← output appears here
#   decompressed/    ← output appears here
#   processed/       ← originals moved here after success
```

GPU routing kicks in for files above the threshold. CPU handles the rest. Startup scan catches files dropped while the daemon was offline.

## Hardware Compatibility

| Device | Compression Engine | Status |
|---|---|---|
| **Jetson AGX Orin** | nvCOMP GPU LZ4 | Primary target, fully tested |
| Jetson Orin NX / Nano | nvCOMP GPU LZ4 | Supported |
| RTX 3000 / 4000 / 5000 | nvCOMP GPU LZ4 | Supported |
| Any CUDA GPU | nvCOMP GPU LZ4 | Supported |
| **macOS Apple Silicon** | Apple LZFSE (Accelerate) | M1/M2/M3/M4 |
| CPU-only (no GPU) | zstd / gzip / lz4 | Universal fallback |

## Web Dashboard

```bash
hammer webui             # http://localhost:5000
```

Real-time monitoring: GPU/CPU utilization, thermal zones, power rails, per-core CPU bars, compression job history, file browser, quick compress/decompress, and a built-in CLI console.

## CLI Commands

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

**Contact:** [contact@resilientmindai.com](mailto:contact@resilientmindai.com) | [resilientmindai.com](https://resilientmindai.com)

---

<p align="center">
  <a href="docs/quickstart.md">Quick Start</a> · <a href="docs/api.md">API Reference</a> · <a href="docs/jetson.md">Jetson Guide</a> · <a href="docs/configuration.md">Configuration</a> · <a href="CONTRIBUTING.md">Contributing</a> · <a href="CHANGELOG.md">Changelog</a>
</p>

<p align="center">
  <strong>ResilientMind AI</strong> | <a href="https://resilientmindai.com">resilientmindai.com</a> | <strong>Joseph C McGinty Jr</strong>
</p>
