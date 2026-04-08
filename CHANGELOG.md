# Changelog

All notable changes to HammerIO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Streaming GPU Compression

### Added
- hammerio/streaming.py — new streaming module
- StreamingGPUCompressor class for single file
  streaming compression and decompression
- compress_directory_streaming() for directory
  compression via tar pipe without temp files
- --stream flag on compress and decompress commands
- --chunk-mb flag for explicit chunk size control
  (0 = auto-detect from available memory)
- Auto-streaming: directories >8GB and files
  exceeding 75% of available memory automatically
  use streaming mode without requiring --stream flag
- Dynamic chunk sizing based on unified memory
  availability via tegrastats + /proc/meminfo fallback

### Fixed
- GPU OOM on large files and directories
  (previously: 63GB directory failed with
  "GPU out of memory at 0 MB read")
- Memory usage now constant at chunk_size regardless
  of input size

### Performance
- Streaming throughput within 5-10% of non-streaming
  on same hardware due to pipelined IO
- tar → GPU compress pipeline eliminates intermediate
  disk write for directory compression
- Pinned GPU buffer allocated once per compressor
  instance, reused per chunk (no repeated alloc/free)

## [1.0.0] - 2026-04-05

### Added
- **nvCOMP GPU compression** — LZ4 at 2,639 MB/s compress, 10,533 MB/s decompress
- **Smart routing engine** — auto-selects GPU (nvCOMP) or CPU (zstd) based on file size and type
- **Watch daemon** (`hammer watch`) — drop-folder automation for compress/decompress pipelines
- **Web dashboard** — real-time telemetry, file browser, compress/decompress UI, CLI console
- **Right-click integration** — Nautilus, Thunar, Nemo file manager support
- **jtop telemetry** — Jetson-specific thermal, power, CPU, memory, EMC monitoring
- **10 CLI commands** — compress, decompress, batch, watch, benchmark, info, config, monitor, webui, version
- **TOML configuration** — persistent config with `hammer config --save`
- **Benchmark suite** — reproducible GPU vs CPU testing with real video files
- **325 tests** passing with comprehensive coverage
- **Apache 2.0** open source license with commercial licensing option
- Docker images for Jetson (L4T) and x86 (CUDA)
- GitHub Actions CI/CD
- Pre-commit hooks (black, ruff, mypy)

### Compression Focus
- All file types route through compression (zstd/lz4/nvCOMP)
- Large files (>500MB) automatically route to GPU nvCOMP LZ4
- Video/audio/image files compressed losslessly (no re-encoding, no bloat)
- ML datasets optimized with streaming decompression for PyTorch DataLoader

### Platform Support
- NVIDIA Jetson AGX Orin (primary target, fully tested)
- NVIDIA Jetson Xavier NX, Orin NX/Nano
- Desktop NVIDIA GPUs (RTX 3000/4000/5000)
- CPU-only fallback for non-NVIDIA systems

### Creator
- ResilientMind AI | ResilientMindai.com | Joseph C McGinty Jr
