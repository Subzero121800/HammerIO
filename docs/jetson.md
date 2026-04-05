# Jetson Deployment Guide

Copyright 2026 ResilientMind AI | Joseph C McGinty Jr

## Supported Platforms

| Platform | JetPack | L4T | CUDA | Compute | Status |
|----------|---------|-----|------|---------|--------|
| AGX Orin 64GB | 6.x | R36 | 12.6 | 8.7 | Fully tested |
| AGX Orin 32GB | 6.x | R36 | 12.6 | 8.7 | Supported |
| Orin NX 16GB | 6.x | R36 | 12.6 | 8.7 | Supported |
| Orin Nano 8GB | 6.x | R36 | 12.6 | 8.7 | Supported |
| Xavier NX | 5.x | R35 | 11.4 | 7.2 | Supported |
| Xavier AGX | 5.x | R35 | 11.4 | 7.2 | Supported |
| Jetson Nano | 4.x | R32 | 10.2 | 5.3 | Limited |

## Installation on Jetson

```bash
# JetPack 6.x (recommended)
sudo apt-get update
sudo apt-get install -y python3-pip ffmpeg

# Install HammerIO
pip3 install hammerio[all]

# Verify
hammer info --hardware
```

## Power Mode Configuration

HammerIO automatically detects the active power mode and adjusts behavior:

```bash
# Set MAXN for maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Check current mode
hammer info --hardware
```

| Mode | Power | Best For |
|------|-------|----------|
| MAXN | Full | Benchmarks, batch processing |
| 50W | 50W | Sustained workloads |
| 30W | 30W | Continuous operation |
| 15W | 15W | Power-constrained environments |

## Unified Memory

Jetson uses unified memory — CPU and GPU share the same physical RAM. HammerIO is aware of this:

- **No CPU→GPU copies needed** for many operations
- Memory reported as "unified" in hardware profile
- Large file processing can use full system RAM
- Be mindful of total memory pressure (GPU + CPU + system)

## nvCOMP on Jetson

nvCOMP is not available via pip on ARM64. Build from source:

```bash
# Clone nvCOMP
git clone https://github.com/NVIDIA/nvcomp.git
cd nvcomp
mkdir build && cd build

# Build for Jetson
cmake .. -DCMAKE_CUDA_ARCHITECTURES=87  # Orin
make -j$(nproc)
sudo make install

# Verify
python3 -c "from hammerio.core.hardware import detect_nvcomp; print(detect_nvcomp())"
```

## VPI on Jetson

VPI is included with JetPack. Verify:

```bash
python3 -c "import vpi; print(vpi.__version__)"
```

If not available:
```bash
sudo apt-get install python3-vpi3  # JetPack 6.x
# or
sudo apt-get install python3-vpi2  # JetPack 5.x
```

## Thermal Management

HammerIO monitors thermal sensors and warns about throttling:

```bash
# Check thermal status
hammer info --telemetry

# Start web dashboard for continuous monitoring
hammer webui
```

If you see throttling warnings during benchmarks:
1. Ensure adequate cooling (fan at max)
2. Run `sudo jetson_clocks` to lock clocks
3. Consider adding a heatsink or active cooling
4. Reduce worker count: `hammer batch --workers 2`

## NVMe Storage

For best I/O performance with large files:

1. Mount NVMe with optimized settings:
   ```bash
   sudo mount -o noatime,discard /dev/nvme0n1p1 /mnt/nvme
   ```

2. Use NVMe for both input and output paths:
   ```bash
   hammer compress /mnt/nvme/dataset/ --output /mnt/nvme/compressed/
   ```

## Docker on Jetson

```bash
# Build Jetson Docker image
docker build -f docker/Dockerfile.jetson -t hammerio:jetson .

# Run with GPU access
docker run --runtime nvidia --rm -it hammerio:jetson hammer info --hardware
```

## GStreamer NVENC Support

HammerIO now supports GStreamer as an alternative video encoding path using NVENC
hardware acceleration. This is particularly useful on Jetson platforms where
GStreamer is tightly integrated with the multimedia stack.

### Prerequisites

GStreamer with NVIDIA plugins is included in JetPack. Verify availability:

```bash
gst-inspect-1.0 nvv4l2h264enc
gst-inspect-1.0 nvv4l2h265enc
```

### Using GStreamer NVENC

HammerIO automatically detects GStreamer NVENC availability and can use it as a
video encoding backend alongside FFmpeg NVENC:

```bash
# Auto-detect will prefer GStreamer NVENC on Jetson when available
hammer compress video.mp4

# Force GStreamer backend
hammer compress video.mp4 --video-backend gstreamer
```

```python
from hammerio.core.router import JobRouter

router = JobRouter(quality="balanced")
job = router.route("input.mp4", video_backend="gstreamer")
result = router.execute(job)
```

### GStreamer Pipeline Details

HammerIO constructs optimized GStreamer pipelines for Jetson:

- **H.264 encoding**: `nvv4l2h264enc` with tunable bitrate and preset
- **H.265/HEVC encoding**: `nvv4l2h265enc` for higher compression efficiency
- **Zero-copy**: Uses `nvvidconv` for colorspace conversion without CPU copies
- **Hardware scaling**: Resolution changes happen on the GPU via `nvvidconv`

### GStreamer vs FFmpeg NVENC

| Feature | GStreamer NVENC | FFmpeg NVENC |
|---------|----------------|--------------|
| Jetson integration | Native (nvv4l2) | Via nvmpi/jetson-ffmpeg |
| Zero-copy pipeline | Yes | Partial |
| Container formats | Limited (mp4, mkv) | Broad |
| Subtitle handling | Basic | Full |
| HammerIO default | Jetson preferred | Desktop/server preferred |

HammerIO's smart router selects the best backend based on the platform and input
file characteristics. On Jetson, GStreamer is preferred for simple transcode
operations, while FFmpeg is used when advanced muxing or subtitle support is
needed.

## Troubleshooting

### "NVENC not available"
- Ensure JetPack is properly installed
- Check: `ffmpeg -encoders 2>&1 | grep nvenc`
- NVENC is available on all Orin and Xavier models

### "GPU Memory: 0 MB"
- This was fixed in HammerIO v0.1.0 — update to latest
- Jetson reports unified memory equal to system RAM

### High thermal warnings
- Normal under sustained load; Orin throttles at ~97°C
- Use `sudo jetson_clocks --fan` for max cooling
- Monitor with `hammer webui` or `jtop`
