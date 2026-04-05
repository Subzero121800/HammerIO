# HammerIO Desktop Applications

Copyright 2026 ResilientMind AI | Joseph C McGinty Jr

## Build Targets

| Platform | GPU Support | Build Command | Output |
|----------|-------------|---------------|--------|
| **Jetson/Ubuntu ARM64** | nvCOMP GPU LZ4 | `./apps/jetson/build.sh` | `.deb` package + tarball |
| **macOS** | CPU only | `./apps/macos/build.sh` | `.app` bundle + `.dmg` |
| **Windows** | NVIDIA CUDA | `.\apps\windows\build.ps1` | Portable `.zip` |

## Jetson / Ubuntu (Primary)

```bash
# Build .deb package
./apps/jetson/build.sh

# Install
sudo dpkg -i apps/jetson/build/hammerio_1.0.0_arm64.deb

# Launches:
#   hammer            — CLI
#   hammerio-dashboard — Web dashboard (in app menu)
```

The .deb package:
- Creates a venv at `/usr/lib/hammerio/venv` with `--system-site-packages` (jtop, VPI access)
- Installs `hammer` CLI to `/usr/local/bin/`
- Adds HammerIO Dashboard to the application menu
- Installs right-click file manager integration

## macOS

```bash
./apps/macos/build.sh

# Install
cp -r apps/macos/build/HammerIO.app /Applications/

# Run — opens dashboard in browser
open /Applications/HammerIO.app
```

Note: macOS runs CPU-only mode. GPU acceleration requires NVIDIA hardware.

## Windows

```powershell
.\apps\windows\build.ps1

# Extract zip, then:
install.bat                # First-time setup
hammer.bat compress file   # CLI
"HammerIO Dashboard.bat"   # Web dashboard
```

GPU acceleration requires NVIDIA CUDA drivers installed.
