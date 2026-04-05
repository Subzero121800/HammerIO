# Privacy Policy

**HammerIO — GPU-Accelerated Compression & Media Processing**

Copyright 2026 ResilientMind AI | ResilientMindai.com | Joseph C McGinty Jr

Last Updated: April 2026

## Overview

HammerIO is a local-first, privacy-respecting software tool. We are committed to protecting your privacy and being transparent about our data practices.

## Data Collection

**HammerIO does NOT collect, transmit, or store any personal data.**

Specifically:
- **No telemetry** is sent to external servers
- **No usage analytics** are collected
- **No file contents** are transmitted anywhere
- **No personal information** is gathered
- **No cookies** are used (the web dashboard runs locally)
- **No network requests** are made except those you explicitly initiate

## Local Data Processing

All data processing occurs entirely on your local hardware:
- Files are compressed/decompressed locally
- Hardware detection reads local system information
- Telemetry monitoring reads local sensors (temperature, power, GPU utilization)
- The web dashboard runs on localhost and is accessible only on your network
- Benchmark results are saved to local files only

## System Information Access

HammerIO accesses the following local system information for hardware detection and optimization:
- CUDA device properties (GPU model, memory, compute capability)
- CPU core count and frequency
- System RAM and swap usage
- Thermal sensor readings
- Power consumption readings (Jetson platforms)
- NVENC/NVDEC encoder/decoder availability
- Operating system and architecture information

This information is used solely for routing decisions and performance optimization. It is never transmitted externally.

## Web Dashboard

The HammerIO web dashboard:
- Runs on your local machine (default: localhost:5000)
- Uses WebSocket connections for real-time updates (local only)
- Loads Chart.js and Socket.IO from CDN for rendering (these CDN requests are the only external network activity)
- Does not set cookies or use browser storage for tracking

## Third-Party Services

HammerIO itself makes no calls to third-party services. However:
- **CDN resources**: The web dashboard loads JavaScript libraries (Chart.js, Socket.IO) from public CDNs. These requests are subject to the CDN provider's privacy policies.
- **pip install**: Installing via pip contacts the Python Package Index (PyPI), subject to PyPI's privacy policy.
- **Docker images**: Pulling Docker images contacts Docker Hub or NVIDIA NGC, subject to their privacy policies.

## Open Source Transparency

HammerIO is open source. You can audit the complete source code to verify these privacy claims. We encourage security researchers and privacy advocates to review our code.

## Children's Privacy

HammerIO is a developer tool and is not directed at children under 13. We do not knowingly collect information from children.

## Changes to This Policy

We may update this Privacy Policy from time to time. Changes will be committed to the repository with clear changelog entries.

## Contact

For privacy-related questions:
- Website: [resilientmindai.com](https://resilientmindai.com)
- GitHub: [github.com/hammerio/hammerio](https://github.com/hammerio/hammerio)

---

*ResilientMind AI | Joseph C McGinty Jr | 2026*
