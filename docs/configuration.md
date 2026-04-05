# Configuration Reference

Copyright 2026 ResilientMind AI | Joseph C McGinty Jr

## Config File Locations

HammerIO searches for configuration in this order:

1. Path specified via `HAMMERIO_CONFIG` environment variable
2. `./hammerio.toml` (project-local)
3. `~/.config/hammerio/config.toml` (user-level)
4. Built-in defaults

## Generate Default Config

```bash
hammer config --generate
# Creates hammerio.toml with all options documented
```

## View Current Config

```bash
hammer config --show
```

## All Options

```toml
[general]
quality = "balanced"     # fast, balanced, quality, lossless
workers = 4              # Parallel workers for batch processing
log_level = "INFO"       # DEBUG, INFO, WARNING, ERROR
output_format = "auto"   # auto, zst, gz, bz2

[video]
gpu_codec = "h264_nvenc"   # h264_nvenc, hevc_nvenc, av1_nvenc
cpu_codec = "libx264"      # libx264, libx265
preset = "p4"              # NVENC: p1 (fastest) - p7 (best quality)
crf = 23                   # Constant rate factor (0-51, lower=better)
container = "mp4"          # mp4, mkv, webm

[bulk]
algorithm = "zstd"         # zstd, lz4, snappy, deflate
chunk_size_mb = 64         # Chunk size for streaming compression
gpu_threshold_mb = 100     # Min file size to use GPU compression

[image]
output_format = "webp"     # webp, jpg, png
quality = 85               # Output quality (1-100)
max_dimension = 0          # Max width/height (0 = no resize)

[audio]
codec = "libmp3lame"       # libmp3lame, aac, libopus, flac
bitrate = "192k"           # Audio bitrate

[dataset]
algorithm = "zstd"         # zstd, lz4
compression_level = 3      # 1-22 for zstd
streaming = true           # Enable streaming decompression

[telemetry]
interval_seconds = 1.0     # Telemetry polling interval
history_size = 300         # Max history samples
thermal_warning_c = 80.0   # Warning threshold
thermal_critical_c = 90.0  # Critical threshold

[web]
host = "0.0.0.0"
port = 5000
debug = false

[jetson]
power_mode = "auto"        # auto, maxn, 15w, 30w
unified_memory_aware = true
nvme_io_threads = 4
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HAMMERIO_CONFIG` | Path to config file | Auto-discover |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | All |
| `HAMMERIO_LOG_LEVEL` | Override log level | INFO |
