#!/usr/bin/env bash
# HammerIO systemd service installer
# Hardens jtop and installs HammerIO dashboard as a system service
#
# Copyright 2026 ResilientMind AI | Joseph C McGinty Jr
#
# Usage:
#   sudo ./systemd/install-services.sh          # Install all
#   sudo ./systemd/install-services.sh jtop      # Harden jtop only
#   sudo ./systemd/install-services.sh dashboard  # Install dashboard only
#   sudo ./systemd/install-services.sh --uninstall

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ "$(id -u)" -ne 0 ]; then
    echo "Error: Run with sudo"
    echo "  sudo $0 $*"
    exit 1
fi

MODE="${1:-all}"

echo "╔══════════════════════════════════════════════════╗"
echo "║   HammerIO System Services Installer             ║"
echo "║   ResilientMind AI | Joseph C McGinty Jr         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ─── Harden jtop service ─────────────────────────────────────────────────────

install_jtop_override() {
    echo "[1] Hardening jtop.service with auto-restart..."

    mkdir -p /etc/systemd/system/jtop.service.d

    cat > /etc/systemd/system/jtop.service.d/hammerio-harden.conf << 'EOF'
# HammerIO jtop hardening — auto-restart on any failure
# Installed by: systemd/install-services.sh

[Service]
# Restart on ANY exit, not just non-zero (catches SIGKILL, OOM, etc.)
Restart=always
RestartSec=5s

# Limit restart burst (max 5 restarts in 60 seconds, then stop trying)
StartLimitIntervalSec=60
StartLimitBurst=5

# Watchdog: if jtop doesn't respond within 30s, systemd kills and restarts it
WatchdogSec=30s

# OOM protection: don't let the kernel kill jtop
OOMScoreAdjust=-500

# Resource limits
MemoryMax=200M
CPUQuota=25%
EOF

    echo "  Created: /etc/systemd/system/jtop.service.d/hammerio-harden.conf"
    echo "  - Restart=always (survives crashes, SIGKILL, OOM)"
    echo "  - RestartSec=5s (fast recovery)"
    echo "  - WatchdogSec=30s (auto-kill if hung)"
    echo "  - OOMScoreAdjust=-500 (protected from OOM killer)"
    echo "  - StartLimitBurst=5/60s (prevents restart loops)"
}

# ─── HammerIO Dashboard service ──────────────────────────────────────────────

install_dashboard() {
    echo ""
    echo "[2] Installing hammerio-dashboard.service..."

    # Find the venv
    VENV_PYTHON=""
    if [ -f "$PROJECT_DIR/.venv/bin/python3" ]; then
        VENV_PYTHON="$PROJECT_DIR/.venv/bin/python3"
    elif [ -f "/usr/lib/hammerio/venv/bin/python3" ]; then
        VENV_PYTHON="/usr/lib/hammerio/venv/bin/python3"
    else
        echo "  Warning: No venv found. Run setup_venv.sh first."
        VENV_PYTHON="/usr/bin/python3"
    fi

    # Find the start script
    START_SCRIPT="$PROJECT_DIR/start_webui.py"

    # Detect the user who owns the project
    OWNER=$(stat -c '%U' "$PROJECT_DIR" 2>/dev/null || echo "root")

    cat > /etc/systemd/system/hammerio-dashboard.service << EOF
# HammerIO Web Dashboard — systemd service
# Copyright 2026 ResilientMind AI | Joseph C McGinty Jr
#
# Starts the web dashboard on boot with auto-restart.
# Access: http://localhost:5000

[Unit]
Description=HammerIO Web Dashboard
Documentation=https://github.com/Subzero121800/HammerIO
After=network.target jtop.service
Wants=jtop.service

[Service]
Type=simple
User=$OWNER
Group=$OWNER
WorkingDirectory=$PROJECT_DIR

# Environment
Environment=EGL_LOG_LEVEL=fatal
Environment=PYTHONUNBUFFERED=1

# Start the dashboard
ExecStart=$VENV_PYTHON $START_SCRIPT --host 0.0.0.0 --port 5000

# Auto-restart on any failure
Restart=always
RestartSec=5s
StartLimitIntervalSec=60
StartLimitBurst=5

# Watchdog: restart if unresponsive for 60s
WatchdogSec=60s

# Resource limits
MemoryMax=512M
CPUQuota=50%

# Security hardening
ProtectSystem=strict
ReadWritePaths=$PROJECT_DIR /tmp /home/$OWNER/.config/hammerio
ProtectHome=read-only
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF

    echo "  Created: /etc/systemd/system/hammerio-dashboard.service"
    echo "  User: $OWNER"
    echo "  Python: $VENV_PYTHON"
    echo "  Port: 5000"
}

# ─── HammerIO Watch Daemon service ───────────────────────────────────────────

install_watch() {
    echo ""
    echo "[3] Installing hammerio-watch.service..."

    VENV_PYTHON=""
    if [ -f "$PROJECT_DIR/.venv/bin/python3" ]; then
        VENV_PYTHON="$PROJECT_DIR/.venv/bin/python3"
    elif [ -f "/usr/lib/hammerio/venv/bin/python3" ]; then
        VENV_PYTHON="/usr/lib/hammerio/venv/bin/python3"
    else
        VENV_PYTHON="/usr/bin/python3"
    fi

    OWNER=$(stat -c '%U' "$PROJECT_DIR" 2>/dev/null || echo "root")
    WATCH_ROOT="/home/$OWNER/hammer-watch"

    cat > /etc/systemd/system/hammerio-watch.service << EOF
# HammerIO Watch Daemon — systemd service
# Copyright 2026 ResilientMind AI | Joseph C McGinty Jr
#
# Monitors drop folders for auto compress/decompress.

[Unit]
Description=HammerIO Watch Daemon
Documentation=https://github.com/Subzero121800/HammerIO
After=network.target jtop.service

[Service]
Type=simple
User=$OWNER
Group=$OWNER
WorkingDirectory=$PROJECT_DIR

Environment=EGL_LOG_LEVEL=fatal
Environment=PYTHONUNBUFFERED=1

ExecStart=$VENV_PYTHON -m hammerio watch --watch-root $WATCH_ROOT

Restart=always
RestartSec=5s
StartLimitIntervalSec=60
StartLimitBurst=5

MemoryMax=1G
CPUQuota=80%

[Install]
WantedBy=multi-user.target
EOF

    echo "  Created: /etc/systemd/system/hammerio-watch.service"
    echo "  Watch root: $WATCH_ROOT"
}

# ─── Uninstall ────────────────────────────────────────────────────────────────

uninstall() {
    echo "Uninstalling HammerIO services..."
    systemctl stop hammerio-dashboard.service 2>/dev/null || true
    systemctl stop hammerio-watch.service 2>/dev/null || true
    systemctl disable hammerio-dashboard.service 2>/dev/null || true
    systemctl disable hammerio-watch.service 2>/dev/null || true
    rm -f /etc/systemd/system/hammerio-dashboard.service
    rm -f /etc/systemd/system/hammerio-watch.service
    rm -rf /etc/systemd/system/jtop.service.d/hammerio-harden.conf
    systemctl daemon-reload
    echo "Done. jtop service restored to defaults."
    exit 0
}

# ─── Execute ──────────────────────────────────────────────────────────────────

case "$MODE" in
    --uninstall) uninstall ;;
    jtop) install_jtop_override ;;
    dashboard) install_dashboard ;;
    watch) install_watch ;;
    all) install_jtop_override; install_dashboard; install_watch ;;
    *) echo "Usage: sudo $0 [all|jtop|dashboard|watch|--uninstall]"; exit 1 ;;
esac

# Reload and apply
echo ""
echo "[4] Reloading systemd..."
systemctl daemon-reload

# Restart jtop with new config
systemctl restart jtop.service
echo "  jtop.service: $(systemctl is-active jtop.service)"

# Enable but don't start dashboard/watch (user can start manually)
if [ -f /etc/systemd/system/hammerio-dashboard.service ]; then
    systemctl enable hammerio-dashboard.service 2>/dev/null || true
    echo "  hammerio-dashboard.service: enabled (start with: sudo systemctl start hammerio-dashboard)"
fi
if [ -f /etc/systemd/system/hammerio-watch.service ]; then
    systemctl enable hammerio-watch.service 2>/dev/null || true
    echo "  hammerio-watch.service: enabled (start with: sudo systemctl start hammerio-watch)"
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Services Installed!                            ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  jtop:       Hardened with auto-restart (always)"
echo "  dashboard:  sudo systemctl start hammerio-dashboard"
echo "  watch:      sudo systemctl start hammerio-watch"
echo ""
echo "  Status:     sudo systemctl status jtop hammerio-dashboard hammerio-watch"
echo "  Logs:       sudo journalctl -u hammerio-dashboard -f"
echo "  Uninstall:  sudo $0 --uninstall"
echo ""
