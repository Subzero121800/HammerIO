#!/usr/bin/env bash
# HammerIO Desktop Application — Jetson/Ubuntu ARM64 Build
# Copyright 2026 ResilientMind AI | Joseph C McGinty Jr
#
# Builds a self-contained .deb package and AppImage for Jetson/Ubuntu ARM64.
#
# Prerequisites:
#   sudo apt install dpkg-dev fakeroot python3-venv
#
# Usage:
#   ./apps/jetson/build.sh           # Build both .deb and standalone
#   ./apps/jetson/build.sh deb       # Build .deb only
#   ./apps/jetson/build.sh appimage  # Build AppImage only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
VERSION=$(python3 -c "import tomllib; print(tomllib.load(open('$PROJECT_DIR/pyproject.toml','rb'))['project']['version'])" 2>/dev/null || echo "1.0.0")
ARCH=$(dpkg --print-architecture 2>/dev/null || echo "arm64")
BUILD_DIR="$SCRIPT_DIR/build"
PKG_NAME="hammerio"
PKG_DIR="$BUILD_DIR/${PKG_NAME}_${VERSION}_${ARCH}"

echo "╔══════════════════════════════════════════════════╗"
echo "║   HammerIO Desktop App Builder — Jetson/Ubuntu   ║"
echo "║   Version: $VERSION  Arch: $ARCH                      ║"
echo "╚══════════════════════════════════════════════════╝"

MODE="${1:-all}"

# ─── .deb Package ─────────────────────────────────────────────────────────────

build_deb() {
    echo ""
    echo "[1/4] Building .deb package..."

    rm -rf "$PKG_DIR"
    mkdir -p "$PKG_DIR/DEBIAN"
    mkdir -p "$PKG_DIR/usr/local/bin"
    mkdir -p "$PKG_DIR/usr/lib/hammerio"
    mkdir -p "$PKG_DIR/usr/share/applications"
    mkdir -p "$PKG_DIR/usr/share/doc/hammerio"

    # Control file
    cat > "$PKG_DIR/DEBIAN/control" << EOF
Package: hammerio
Version: $VERSION
Section: utils
Priority: optional
Architecture: $ARCH
Depends: python3 (>= 3.10), python3-pip, ffmpeg
Recommends: python3-jetson-stats, gstreamer1.0-plugins-bad
Maintainer: Joseph C McGinty Jr <contact@resilientmindai.com>
Homepage: https://resilientmindai.com
Description: GPU-accelerated compression for NVIDIA Jetson
 HammerIO automatically routes compression to GPU (nvCOMP LZ4) or CPU (zstd)
 based on file size and available hardware. Built for edge AI deployment.
 .
 Features: CLI, web dashboard, watch daemon, right-click integration.
 GPU decompression: 10+ GB/s via nvCOMP.
EOF

    # Post-install: create venv and install
    cat > "$PKG_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/sh
set -e
python3 -m venv --system-site-packages /usr/lib/hammerio/venv
/usr/lib/hammerio/venv/bin/pip install --upgrade pip wheel >/dev/null 2>&1
/usr/lib/hammerio/venv/bin/pip install /usr/lib/hammerio/src/.[all] 2>&1 | tail -3
# Install desktop integration
if [ -x /usr/lib/hammerio/src/desktop-integration/install.sh ]; then
    /usr/lib/hammerio/src/desktop-integration/install.sh || true
fi
echo "HammerIO installed. Run: hammer --help"
EOF
    chmod 755 "$PKG_DIR/DEBIAN/postinst"

    # Pre-remove
    cat > "$PKG_DIR/DEBIAN/prerm" << 'EOF'
#!/bin/sh
set -e
rm -rf /usr/lib/hammerio/venv
echo "HammerIO removed."
EOF
    chmod 755 "$PKG_DIR/DEBIAN/prerm"

    # Copy project source
    echo "[2/4] Copying source..."
    rsync -a --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
        --exclude='*.pyc' --exclude='.pytest_cache' --exclude='*.egg-info' \
        --exclude='apps/*/build' --exclude='Test Video.mp4' \
        "$PROJECT_DIR/" "$PKG_DIR/usr/lib/hammerio/src/"

    # Launcher script
    cat > "$PKG_DIR/usr/local/bin/hammer" << 'EOF'
#!/bin/sh
exec /usr/lib/hammerio/venv/bin/python3 -m hammerio "$@"
EOF
    chmod 755 "$PKG_DIR/usr/local/bin/hammer"

    # Dashboard launcher
    cat > "$PKG_DIR/usr/local/bin/hammerio-dashboard" << 'EOF'
#!/bin/sh
exec /usr/lib/hammerio/venv/bin/python3 /usr/lib/hammerio/src/start_webui.py "$@"
EOF
    chmod 755 "$PKG_DIR/usr/local/bin/hammerio-dashboard"

    # .desktop file for application menu
    cat > "$PKG_DIR/usr/share/applications/hammerio.desktop" << EOF
[Desktop Entry]
Type=Application
Name=HammerIO Dashboard
Comment=GPU-accelerated compression monitoring and control
Exec=hammerio-dashboard
Icon=utilities-file-archiver
Terminal=false
Categories=Utility;System;
Keywords=compression;gpu;nvidia;jetson;
EOF

    # Docs
    cp "$PROJECT_DIR/LICENSE" "$PKG_DIR/usr/share/doc/hammerio/"
    cp "$PROJECT_DIR/CHANGELOG.md" "$PKG_DIR/usr/share/doc/hammerio/"

    # Build
    echo "[3/4] Building package..."
    cd "$BUILD_DIR"
    dpkg-deb --build "${PKG_NAME}_${VERSION}_${ARCH}" 2>/dev/null

    DEB_FILE="$BUILD_DIR/${PKG_NAME}_${VERSION}_${ARCH}.deb"
    echo "[4/4] Package built: $DEB_FILE"
    echo "  Size: $(du -sh "$DEB_FILE" | cut -f1)"
    echo ""
    echo "  Install: sudo dpkg -i $DEB_FILE"
    echo "  Remove:  sudo dpkg -r hammerio"
}

# ─── Standalone Bundle ────────────────────────────────────────────────────────

build_standalone() {
    echo ""
    echo "Building standalone bundle..."

    BUNDLE_DIR="$BUILD_DIR/HammerIO-${VERSION}-${ARCH}"
    rm -rf "$BUNDLE_DIR"
    mkdir -p "$BUNDLE_DIR"

    # Copy project
    rsync -a --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
        --exclude='*.pyc' --exclude='.pytest_cache' --exclude='*.egg-info' \
        --exclude='apps/*/build' --exclude='Test Video.mp4' \
        "$PROJECT_DIR/" "$BUNDLE_DIR/"

    # Create install script
    cat > "$BUNDLE_DIR/install.sh" << 'INSTALL_EOF'
#!/bin/bash
set -e
echo "Installing HammerIO..."
cd "$(dirname "$0")"
./setup_venv.sh
echo ""
echo "Add to PATH: export PATH=\"$(pwd)/.venv/bin:\$PATH\""
echo "Or run: ./start.sh"
INSTALL_EOF
    chmod +x "$BUNDLE_DIR/install.sh"

    # Create tarball
    cd "$BUILD_DIR"
    tar czf "HammerIO-${VERSION}-linux-${ARCH}.tar.gz" "HammerIO-${VERSION}-${ARCH}/"
    echo "  Bundle: $BUILD_DIR/HammerIO-${VERSION}-linux-${ARCH}.tar.gz"
    echo "  Size: $(du -sh "HammerIO-${VERSION}-linux-${ARCH}.tar.gz" | cut -f1)"
}

# ─── Run ──────────────────────────────────────────────────────────────────────

case "$MODE" in
    deb) build_deb ;;
    standalone|appimage) build_standalone ;;
    all) build_deb; build_standalone ;;
    *) echo "Usage: $0 [deb|standalone|all]"; exit 1 ;;
esac

echo ""
echo "Build complete!"
