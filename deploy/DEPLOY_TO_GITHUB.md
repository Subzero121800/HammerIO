# Deploy HammerIO to GitHub

## Prerequisites

1. A GitHub account
2. `git` and `gh` (GitHub CLI) installed on the deployment machine
3. Copy this entire `HammerIO` project folder to the deployment machine

## Step-by-Step Deployment

### 1. Authenticate with GitHub

```bash
gh auth login
```

### 2. Navigate to the project

```bash
cd /path/to/HammerIO
```

### 3. Create the GitHub repository

```bash
gh repo create Subzero121800/HammerIO \
    --public \
    --description "GPU where it matters. CPU where it doesn't. Zero configuration. CUDA-accelerated compression & media processing for NVIDIA Jetson." \
    --homepage "https://resilientmindai.com" \
    \
    --source . \
    --remote origin
```

Or if you prefer to create the repo first on GitHub.com:
1. Go to https://github.com/new
2. Name: `hammerio`
3. Description: `GPU where it matters. CPU where it doesn't. Zero configuration.`
4. Public, no README (we have one), no license template (we have our own)
5. Then add the remote:

```bash
git remote add origin https://github.com/Subzero121800/HammerIO.git
```

### 4. Push to GitHub

```bash
git branch -M main
git push -u origin main
```

### 5. Set repository topics (for discoverability)

```bash
gh repo edit Subzero121800/HammerIO \
    --add-topic cuda \
    --add-topic compression \
    --add-topic nvidia-jetson \
    --add-topic gpu \
    --add-topic nvenc \
    --add-topic edge-ai \
    --add-topic media-processing \
    --add-topic python \
    --add-topic zstd \
    --add-topic gstreamer
```

### 6. Create the first release

```bash
gh release create v0.1.0 \
    --title "HammerIO v0.1.0 — Initial Release" \
    --notes "$(cat <<'NOTES'
# HammerIO v0.1.0

**GPU where it matters. CPU where it doesn't. Zero configuration.**

## Highlights

- Smart job routing — auto-detects hardware and routes to optimal processor
- GStreamer NVENC hardware video encoding on Jetson (~19x faster than CPU)
- zstd/gzip/bzip2/lz4 compression (239x ratio on text, 198 MB/s)
- Batch image processing via OpenCV CUDA / PIL
- jtop telemetry integration (thermal, power, CPU, memory, EMC)
- Flask web dashboard with real-time WebSocket monitoring
- 9 CLI commands, 325 tests passing
- PyTorch StreamingDataset for on-the-fly decompression

## Install

```bash
pip install hammerio
```

## Quick Start

```bash
hammer compress video.mp4
hammer info --hardware
hammer webui
```

## Platform

Tested on NVIDIA Jetson AGX Orin (JetPack 6.2.2, CUDA 12.6).
Works on any NVIDIA CUDA-capable GPU.

Created by ResilientMind AI | Joseph C McGinty Jr
NOTES
)"
```

### 7. Enable GitHub Pages (optional, for docs)

```bash
gh repo edit Subzero121800/HammerIO --enable-wiki=false
```

### 8. Verify

```bash
gh repo view Subzero121800/HammerIO --web
```

## Post-Deployment Checklist

- [ ] Repository created and pushed
- [ ] Topics set for discoverability
- [ ] v0.1.0 release created
- [ ] README renders correctly on GitHub
- [ ] CI workflow runs on first push
- [ ] Issue templates visible under "New Issue"
- [ ] Star your own repo!

## Repository Settings (manual on GitHub.com)

1. **Settings → General**:
   - Social preview image (create a banner)
   - Check "Releases" in sidebar

2. **Settings → Branches**:
   - Add branch protection rule for `main`
   - Require PR reviews before merging

3. **Settings → Actions → General**:
   - Allow all actions

4. **Settings → Pages** (optional):
   - Source: Deploy from branch `main`, `/docs` folder

---

*Copyright 2026 ResilientMind AI | ResilientMindai.com | Joseph C McGinty Jr*
