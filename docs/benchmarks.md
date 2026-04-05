# Benchmarks

Copyright 2026 ResilientMind AI | Joseph C McGinty Jr

## Running Benchmarks

```bash
# Full benchmark suite
hammer benchmark

# Quick benchmarks (smaller datasets)
hammer benchmark --quick

# Custom output
hammer benchmark --output results/my_benchmark.json
```

## Methodology

All benchmarks use:
- Freshly generated test data (not cached)
- Wall-clock time including I/O
- Multiple runs averaged (full mode)
- Consistent quality settings

## Reference Numbers (Jetson AGX Orin 64GB, MAXN)

| Workload | CPU Time | GPU Time | Speedup | Notes |
|----------|----------|----------|---------|-------|
| 4K video (30s) | ~45s | ~7s | ~6.4x | h264_nvenc vs libx264 |
| 1080p video (30s) | ~18s | ~3s | ~6.0x | h264_nvenc vs libx264 |
| 1GB dataset zstd | ~8s | ~1.2s | ~6.7x | nvCOMP vs CPU zstd |
| 10GB dataset LZ4 | ~165s | ~18s | ~9.2x | nvCOMP LZ4 vs CPU |
| 500 images batch | ~12s | ~2s | ~6.0x | OpenCV CUDA vs CPU |
| Audio batch (20 files) | ~8s | ~5s | ~1.6x | FFmpeg CUDA filters |

*Numbers are representative. Actual results depend on input characteristics, thermal state, and power mode.*

## Output Formats

Benchmark results are saved in:
- **JSON**: Full structured data for programmatic analysis
- **CSV**: Spreadsheet-compatible tabular format
- **Terminal**: Rich formatted tables during run

## Interpreting Results

- **Throughput (MB/s)**: Raw data processing speed
- **Compression ratio**: Input size / output size (higher = better)
- **Speedup**: CPU time / GPU time (higher = better)

GPU speedup is most significant for:
1. Video transcoding (NVENC is purpose-built hardware)
2. Large dataset compression (nvCOMP parallelism)
3. Batch image processing (GPU parallelism)

CPU may be competitive or faster for:
1. Small files (< 1MB) — GPU overhead dominates
2. Already-compressed data — no benefit from GPU
3. Single audio files — I/O bound workload
