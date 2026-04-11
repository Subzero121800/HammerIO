#!/usr/bin/env bash
# HammerIO Desktop Application — macOS Build
# Copyright 2026 ResilientMind AI | Joseph C McGinty Jr
#
# Builds a macOS .app bundle with the HammerIO dashboard.
# Note: GPU compression requires NVIDIA CUDA (macOS runs CPU-only mode).
#
# Prerequisites:
#   brew install python@3.12
#
# Usage:
#   ./apps/macos/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
VERSION=$(python3 -c "
try:
    import tomllib
    print(tomllib.load(open('$PROJECT_DIR/pyproject.toml','rb'))['project']['version'])
except:
    print('1.0.0')
")
BUILD_DIR="$SCRIPT_DIR/build"
APP_NAME="HammerIO"
APP_DIR="$BUILD_DIR/$APP_NAME.app"

echo "╔══════════════════════════════════════════════════╗"
echo "║   HammerIO Desktop App Builder — macOS           ║"
echo "║   Version: $VERSION                                    ║"
echo "╚══════════════════════════════════════════════════╝"

rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# ─── Info.plist ───────────────────────────────────────────────────────────────

cat > "$APP_DIR/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>hammerio</string>
    <key>CFBundleIdentifier</key>
    <string>com.resilientmindai.hammerio</string>
    <key>CFBundleName</key>
    <string>HammerIO</string>
    <key>CFBundleDisplayName</key>
    <string>HammerIO Dashboard</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright 2026 ResilientMind AI - Joseph C McGinty Jr</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
</dict>
</plist>
EOF

# ─── Launcher script ──────────────────────────────────────────────────────────

cat > "$APP_DIR/Contents/MacOS/hammerio" << 'EOF'
#!/bin/bash
DIR="$(cd "$(dirname "$0")/../Resources" && pwd)"
cd "$DIR/src"

# Create venv if needed
if [ ! -d "$DIR/venv" ]; then
    python3 -m venv "$DIR/venv"
    "$DIR/venv/bin/pip" install -e ".[web]" > /dev/null 2>&1
fi

# Launch dashboard and open browser
"$DIR/venv/bin/python3" -c "
import webbrowser, threading, time
threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5000')).start()
from hammerio.web.app import create_app, socketio
app = create_app()
socketio.run(app, host='127.0.0.1', port=5000, allow_unsafe_werkzeug=True)
"
EOF
chmod +x "$APP_DIR/Contents/MacOS/hammerio"

# ─── Copy source ──────────────────────────────────────────────────────────────

rsync -a --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
    --exclude='*.pyc' --exclude='.pytest_cache' --exclude='*.egg-info' \
    --exclude='apps/*/build' --exclude='Test Video.mp4' \
    "$PROJECT_DIR/" "$APP_DIR/Contents/Resources/src/"

echo ""
echo "macOS app built: $APP_DIR"
echo "  To install: cp -r '$APP_DIR' /Applications/"
echo "  To run: open '$APP_DIR'"
echo ""
echo "Note: macOS runs CPU-only mode (no NVIDIA GPU)."
echo "GPU acceleration requires NVIDIA hardware (Jetson/RTX)."

# Create DMG if hdiutil available
if command -v hdiutil &>/dev/null; then
    DMG="$BUILD_DIR/HammerIO-${VERSION}-macOS.dmg"
    rm -f "$DMG"
    hdiutil create -volname "HammerIO" -srcfolder "$APP_DIR" -ov -format UDZO "$DMG" 2>/dev/null && \
        echo "DMG created: $DMG" || echo "DMG creation skipped"
fi
