#!/usr/bin/env bash
# HammerIO Full Installer
# Copyright 2026 ResilientMind AI | Joseph C McGinty Jr
#
# Installs everything:
#   1. Python venv with all dependencies
#   2. 'hammer' command available system-wide
#   3. Right-click file manager integration
#   4. systemd services (jtop hardening + dashboard + watch)
#   5. Verifies the installation
#
# Usage:
#   ./install.sh           # Full install (prompts for sudo)
#   ./install.sh --user    # User-only (no systemd, no sudo required)
#   ./install.sh --check   # Check installation status
#   ./install.sh --uninstall

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
USER_BIN="$HOME/.local/bin"
IS_JETSON=false
[ -f /etc/nv_tegra_release ] && IS_JETSON=true
OS_TYPE="$(uname -s)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
info() { echo -e "  ${CYAN}→${NC} $1"; }

MODE="${1:-full}"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║         HammerIO Installer v1.0.0                ║"
echo "║   ResilientMind AI | Joseph C McGinty Jr         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ─── Check installation ──────────────────────────────────────────────────────

if [ "$MODE" = "--check" ]; then
    echo "Installation Status:"
    echo ""

    # Venv
    if [ -d "$VENV_DIR" ]; then
        ok "Virtual environment: $VENV_DIR"
    else
        fail "Virtual environment: not found"
    fi

    # hammer CLI
    if command -v hammer &>/dev/null; then
        ok "hammer CLI: $(which hammer)"
    elif [ -f "$USER_BIN/hammer" ]; then
        ok "hammer CLI: $USER_BIN/hammer (add $USER_BIN to PATH)"
    else
        fail "hammer CLI: not in PATH"
    fi

    # System link
    if [ -f /usr/local/bin/hammer ]; then
        ok "System CLI: /usr/local/bin/hammer"
    else
        warn "System CLI: not installed (run with sudo for system-wide)"
    fi

    # Right-click
    if ls "$HOME/.local/share/nautilus/scripts/HammerIO"* &>/dev/null 2>&1; then
        ok "Right-click: Nautilus scripts installed"
    elif [ -f "$HOME/.local/share/applications/hammerio-compress.desktop" ]; then
        ok "Right-click: .desktop files installed"
    else
        warn "Right-click: not installed"
    fi

    # systemd
    if systemctl is-active --quiet jtop.service 2>/dev/null; then
        ok "jtop service: active"
    elif [ "$IS_JETSON" = true ]; then
        fail "jtop service: not running"
    else
        info "jtop service: N/A (not Jetson)"
    fi

    if systemctl is-enabled --quiet hammerio-dashboard.service 2>/dev/null; then
        svc_status=$(systemctl is-active hammerio-dashboard.service 2>/dev/null || echo "inactive")
        ok "Dashboard service: enabled ($svc_status)"
    else
        warn "Dashboard service: not installed"
    fi

    # Python version
    if [ -f "$VENV_DIR/bin/python3" ]; then
        pyver=$("$VENV_DIR/bin/python3" --version 2>&1 | awk '{print $2}')
        ok "Python: $pyver"
    fi

    # Test import
    if [ -f "$VENV_DIR/bin/python3" ]; then
        if "$VENV_DIR/bin/python3" -c "import hammerio; print(f'v{hammerio.__version__}')" 2>/dev/null; then
            ok "HammerIO import: working"
        else
            fail "HammerIO import: broken"
        fi
    fi

    echo ""
    exit 0
fi

# ─── Uninstall ────────────────────────────────────────────────────────────────

if [ "$MODE" = "--uninstall" ]; then
    echo "Uninstalling HammerIO..."
    echo ""

    # Stop services
    sudo systemctl stop hammerio-dashboard.service 2>/dev/null && ok "Stopped dashboard service" || true
    sudo systemctl stop hammerio-watch.service 2>/dev/null && ok "Stopped watch service" || true
    sudo systemctl disable hammerio-dashboard.service 2>/dev/null || true
    sudo systemctl disable hammerio-watch.service 2>/dev/null || true
    sudo rm -f /etc/systemd/system/hammerio-dashboard.service
    sudo rm -f /etc/systemd/system/hammerio-watch.service
    sudo rm -rf /etc/systemd/system/jtop.service.d/hammerio-harden.conf
    sudo systemctl daemon-reload 2>/dev/null || true
    ok "Removed systemd services"

    # Remove CLI links
    sudo rm -f /usr/local/bin/hammer /usr/local/bin/hammerio-dashboard 2>/dev/null && ok "Removed /usr/local/bin links" || true
    rm -f "$USER_BIN/hammer" "$USER_BIN/hammerio-dashboard" 2>/dev/null && ok "Removed ~/.local/bin links" || true

    # Remove right-click
    "$SCRIPT_DIR/desktop-integration/install.sh" --uninstall 2>/dev/null && ok "Removed right-click integration" || true

    # Remove venv
    rm -rf "$VENV_DIR" && ok "Removed virtual environment" || true
    rm -f "$SCRIPT_DIR/.hammerio.pid" "$SCRIPT_DIR/.hammerio.log"

    echo ""
    echo "HammerIO uninstalled. Source code preserved in: $SCRIPT_DIR"
    exit 0
fi

# ─── Step 1: Virtual environment + dependencies ──────────────────────────────

echo "[1/5] Setting up virtual environment..."
"$SCRIPT_DIR/setup_venv.sh" 2>&1 | grep -E "^\[|^  [A-Za-z]" | head -20
echo ""

# ─── Step 2: System-wide CLI ─────────────────────────────────────────────────

echo "[2/5] Installing 'hammer' command..."

HAMMER_BIN="$VENV_DIR/bin/hammer"
PYTHON_BIN="$VENV_DIR/bin/python3"

if [ ! -f "$HAMMER_BIN" ]; then
    # Create a wrapper if the entry point wasn't installed
    HAMMER_BIN="$VENV_DIR/bin/hammer"
    cat > "$HAMMER_BIN" << EOF
#!/bin/sh
exec "$PYTHON_BIN" -m hammerio "\$@"
EOF
    chmod +x "$HAMMER_BIN"
fi

# User-local bin (always)
mkdir -p "$USER_BIN"
ln -sf "$HAMMER_BIN" "$USER_BIN/hammer"
ln -sf "$SCRIPT_DIR/start_webui.py" "$USER_BIN/hammerio-dashboard"
chmod +x "$USER_BIN/hammerio-dashboard" 2>/dev/null || true
ok "Linked to $USER_BIN/hammer"

# System-wide (if full install with sudo)
if [ "$MODE" != "--user" ]; then
    if sudo ln -sf "$HAMMER_BIN" /usr/local/bin/hammer 2>/dev/null; then
        ok "Linked to /usr/local/bin/hammer"
    else
        warn "Could not link to /usr/local/bin (run with sudo for system-wide)"
    fi

    # Dashboard launcher
    cat > /tmp/hammerio-dashboard << EOF
#!/bin/sh
exec "$PYTHON_BIN" "$SCRIPT_DIR/start_webui.py" "\$@"
EOF
    chmod +x /tmp/hammerio-dashboard
    if sudo mv /tmp/hammerio-dashboard /usr/local/bin/hammerio-dashboard 2>/dev/null; then
        ok "Linked to /usr/local/bin/hammerio-dashboard"
    fi
fi

# Ensure ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$USER_BIN:"* ]]; then
    warn "$USER_BIN not in PATH — add to your shell profile:"
    info "echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
fi

echo ""

# ─── Step 3: Right-click integration ─────────────────────────────────────────

echo "[3/5] Installing right-click file manager integration..."
if [ -f "$SCRIPT_DIR/desktop-integration/install.sh" ]; then
    "$SCRIPT_DIR/desktop-integration/install.sh" 2>&1 | grep -E "^  Installed|Right-click" | head -5
    ok "Right-click integration installed"
else
    warn "desktop-integration/install.sh not found"
fi
echo ""

# ─── Step 4: systemd services ────────────────────────────────────────────────

echo "[4/5] Installing systemd services..."
if [ "$MODE" = "--user" ]; then
    warn "Skipped (--user mode, no sudo)"
elif [ "$OS_TYPE" != "Linux" ]; then
    warn "Skipped (not Linux)"
elif [ -f "$SCRIPT_DIR/systemd/install-services.sh" ]; then
    if sudo "$SCRIPT_DIR/systemd/install-services.sh" 2>&1 | grep -E "^\[|^  [a-z]|Created|Hardening|active|enabled" | head -10; then
        ok "systemd services installed"
    else
        warn "systemd install had issues (check manually)"
    fi
else
    warn "systemd/install-services.sh not found"
fi
echo ""

# ─── Step 5: Verify ──────────────────────────────────────────────────────────

echo "[5/5] Verifying installation..."
echo ""

ERRORS=0

# Check hammer command
if "$VENV_DIR/bin/python3" -m hammerio version 2>/dev/null | head -1 | grep -q "HammerIO"; then
    ok "hammer version: $("$VENV_DIR/bin/python3" -c "import hammerio; print(hammerio.__version__)" 2>/dev/null)"
else
    fail "hammer command not working"
    ERRORS=$((ERRORS + 1))
fi

# Check hardware detection
if "$VENV_DIR/bin/python3" -c "
from hammerio.core.hardware import detect_hardware
hw = detect_hardware()
print(hw.platform_name)
" 2>/dev/null | head -1 | grep -q .; then
    ok "Hardware detection: working"
else
    fail "Hardware detection: failed"
    ERRORS=$((ERRORS + 1))
fi

# Check web app
if "$VENV_DIR/bin/python3" -c "from hammerio.web.app import create_app; print('ok')" 2>/dev/null | grep -q ok; then
    ok "Web dashboard: importable"
else
    warn "Web dashboard: missing dependencies (pip install hammerio[web])"
fi

# Platform-specific
if [ "$IS_JETSON" = true ]; then
    if "$VENV_DIR/bin/python3" -c "import jtop; print('ok')" 2>/dev/null | grep -q ok; then
        ok "jtop: available"
    else
        warn "jtop: not importable (run: sudo pip3 install jetson-stats)"
    fi
fi

if "$VENV_DIR/bin/python3" -c "
try:
    from nvidia.nvcomp import Codec
    print('ok')
except:
    pass
" 2>/dev/null | grep -q ok; then
    ok "nvCOMP GPU: available"
else
    info "nvCOMP GPU: not available (CPU compression only)"
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║         Installation Complete!                   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Commands:"
echo "    hammer --help              All CLI commands"
echo "    hammer compress <file>     Compress a file"
echo "    hammer benchmark --quick   Run benchmark"
echo "    ./start.sh                 Start dashboard"
echo "    ./start.sh stop            Stop dashboard"
echo "    ./start.sh status          Check running instances"
echo ""
if [ $ERRORS -gt 0 ]; then
    echo -e "  ${RED}$ERRORS verification(s) failed — check above${NC}"
else
    echo -e "  ${GREEN}All verifications passed!${NC}"
fi
echo ""
