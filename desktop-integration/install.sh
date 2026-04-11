#!/usr/bin/env bash
# HammerIO Desktop Integration Installer
# Adds right-click "Compress with HammerIO" and "Decompress with HammerIO"
# to your file manager (Nautilus, Thunar, Nemo, PCManFM, Dolphin).
#
# Copyright 2026 ResilientMind AI | Joseph C McGinty Jr
#
# Usage: ./desktop-integration/install.sh
#        ./desktop-integration/install.sh --uninstall

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
HAMMER_BIN=""

# Find hammer binary
if [ -f "$VENV_DIR/bin/hammer" ]; then
    HAMMER_BIN="$VENV_DIR/bin/hammer"
elif command -v hammer &>/dev/null; then
    HAMMER_BIN="$(which hammer)"
else
    HAMMER_BIN="$VENV_DIR/bin/python3 -m hammerio"
fi

echo "╔══════════════════════════════════════════════════╗"
echo "║   HammerIO Right-Click Integration               ║"
echo "║   ResilientMind AI | Joseph C McGinty Jr         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Hammer: $HAMMER_BIN"
echo ""

if [ "${1:-}" = "--uninstall" ]; then
    echo "Uninstalling..."
    rm -f ~/.local/share/nautilus/scripts/"HammerIO: Compress (GPU)"
    rm -f ~/.local/share/nautilus/scripts/"HammerIO: Decompress"
    rm -f ~/.local/share/nautilus/scripts/"HammerIO: Analyze Route"
    rm -f ~/.local/share/nautilus/scripts/"HammerIO: Open Terminal Here"
    rm -f ~/.local/share/nemo/actions/hammerio-compress.nemo_action
    rm -f ~/.local/share/nemo/actions/hammerio-decompress.nemo_action
    rm -f ~/.local/share/applications/hammerio-compress.desktop
    rm -f ~/.local/share/applications/hammerio-decompress.desktop
    echo "Done. Restart your file manager to apply."
    exit 0
fi

# ─── Create the helper script ─────────────────────────────────────────────────
# This script is called by all file managers. It handles venv activation,
# notification popups, and error handling.

HELPER="$PROJECT_DIR/desktop-integration/hammerio-action.sh"
cat > "$HELPER" << 'HELPEREOF'
#!/usr/bin/env bash
# HammerIO file manager action helper
# Called by Nautilus scripts, Thunar custom actions, Nemo actions, .desktop files
#
# Usage: hammerio-action.sh <compress|decompress|analyze> <file1> [file2] ...

ACTION="${1:-compress}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
LOG="/tmp/hammerio-action.log"

# Activate venv if available
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Notification helper (uses notify-send if available, else zenity)
notify() {
    local title="$1" body="$2" icon="${3:-dialog-information}"
    if command -v notify-send &>/dev/null; then
        notify-send -i "$icon" "$title" "$body" 2>/dev/null || true
    fi
}

# Progress dialog (zenity if available)
show_progress() {
    if command -v zenity &>/dev/null; then
        zenity --progress --title="HammerIO" --text="$1" --pulsate --auto-close --no-cancel 2>/dev/null &
        echo $!
    else
        echo ""
    fi
}

close_progress() {
    [ -n "$1" ] && kill "$1" 2>/dev/null || true
}

# Process files
TOTAL=$#
DONE=0
ERRORS=0

for FILE in "$@"; do
    [ ! -e "$FILE" ] && continue
    BASENAME="$(basename "$FILE")"

    case "$ACTION" in
        compress)
            notify "HammerIO" "Compressing: $BASENAME" "utilities-file-archiver"
            PID=$(show_progress "Compressing $BASENAME...")

            if hammer compress "$FILE" --quality balanced >> "$LOG" 2>&1; then
                DONE=$((DONE + 1))
                OUTPUT="${FILE}.zst"
                # Check various possible output extensions
                for ext in .zst .lz4 .gz .bz2 .hammer .hammer.mp4; do
                    if [ -f "${FILE}${ext}" ]; then
                        OUTPUT="${FILE}${ext}"
                        break
                    fi
                done
                OUTSIZE=$(du -sh "$OUTPUT" 2>/dev/null | cut -f1)
                INSIZE=$(du -sh "$FILE" 2>/dev/null | cut -f1)
                notify "HammerIO" "Done: $BASENAME\n$INSIZE → $OUTSIZE" "emblem-default"
            else
                ERRORS=$((ERRORS + 1))
                notify "HammerIO" "Failed: $BASENAME\nCheck $LOG" "dialog-error"
            fi

            close_progress "$PID"
            ;;

        decompress)
            notify "HammerIO" "Decompressing: $BASENAME" "utilities-file-archiver"
            PID=$(show_progress "Decompressing $BASENAME...")

            if hammer decompress "$FILE" >> "$LOG" 2>&1; then
                DONE=$((DONE + 1))
                notify "HammerIO" "Decompressed: $BASENAME" "emblem-default"
            else
                ERRORS=$((ERRORS + 1))
                notify "HammerIO" "Failed: $BASENAME\nCheck $LOG" "dialog-error"
            fi

            close_progress "$PID"
            ;;

        analyze)
            # Show routing analysis in a dialog
            RESULT=$(hammer info --routes "$FILE" 2>&1 | head -20)
            if command -v zenity &>/dev/null; then
                zenity --info --title="HammerIO: Route Analysis" \
                    --text="$RESULT" --width=500 --height=300 2>/dev/null
            else
                notify "HammerIO" "$RESULT"
            fi
            DONE=$((DONE + 1))
            ;;
    esac
done

if [ $TOTAL -gt 1 ]; then
    notify "HammerIO" "Batch complete: $DONE/$TOTAL succeeded, $ERRORS errors" \
        "$([ $ERRORS -eq 0 ] && echo 'emblem-default' || echo 'dialog-warning')"
fi
HELPEREOF
chmod +x "$HELPER"
echo "  Created: $HELPER"

# ─── Nautilus Scripts ──────────────────────────────────────────────────────────

if command -v nautilus &>/dev/null; then
    echo ""
    echo "Installing Nautilus right-click scripts..."
    NAUTILUS_DIR="$HOME/.local/share/nautilus/scripts"
    mkdir -p "$NAUTILUS_DIR"

    # Compress
    cat > "$NAUTILUS_DIR/HammerIO: Compress (GPU)" << EOF
#!/usr/bin/env bash
# HammerIO Compress — right-click action for Nautilus
"$HELPER" compress \$NAUTILUS_SCRIPT_SELECTED_FILE_PATHS
EOF
    chmod +x "$NAUTILUS_DIR/HammerIO: Compress (GPU)"

    # Decompress
    cat > "$NAUTILUS_DIR/HammerIO: Decompress" << EOF
#!/usr/bin/env bash
# HammerIO Decompress — right-click action for Nautilus
"$HELPER" decompress \$NAUTILUS_SCRIPT_SELECTED_FILE_PATHS
EOF
    chmod +x "$NAUTILUS_DIR/HammerIO: Decompress"

    # Analyze
    cat > "$NAUTILUS_DIR/HammerIO: Analyze Route" << EOF
#!/usr/bin/env bash
# HammerIO Analyze — right-click action for Nautilus
"$HELPER" analyze \$NAUTILUS_SCRIPT_SELECTED_FILE_PATHS
EOF
    chmod +x "$NAUTILUS_DIR/HammerIO: Analyze Route"

    # Open Terminal Here
    cat > "$NAUTILUS_DIR/HammerIO: Open Terminal Here" << EOF
#!/usr/bin/env bash
# HammerIO Terminal — right-click action for Nautilus
"$HELPER" terminal \$NAUTILUS_SCRIPT_SELECTED_FILE_PATHS
EOF
    chmod +x "$NAUTILUS_DIR/HammerIO: Open Terminal Here"

    echo "  Installed 4 Nautilus scripts"
    echo "  Right-click → Scripts → HammerIO: Compress/Decompress/Analyze/Terminal"
fi

# ─── Thunar Custom Actions ────────────────────────────────────────────────────

if command -v thunar &>/dev/null; then
    echo ""
    echo "Installing Thunar custom actions..."
    THUNAR_UCA="$HOME/.config/Thunar/uca.xml"

    # Check if our actions already exist
    if grep -q "HammerIO" "$THUNAR_UCA" 2>/dev/null; then
        echo "  Thunar actions already installed (skipping)"
    else
        # Insert before closing </actions> tag
        if [ -f "$THUNAR_UCA" ]; then
            # Remove closing tag, add our actions, re-add closing tag
            sed -i 's|</actions>||' "$THUNAR_UCA"
            cat >> "$THUNAR_UCA" << EOF
<action>
    <icon>utilities-file-archiver</icon>
    <name>HammerIO: Compress (GPU)</name>
    <unique-id>hammerio-compress</unique-id>
    <command>$HELPER compress %F</command>
    <description>Compress with GPU acceleration via HammerIO</description>
    <patterns>*</patterns>
    <startup-notify/>
    <directories/>
    <audio-files/>
    <image-files/>
    <other-files/>
    <text-files/>
    <video-files/>
</action>
<action>
    <icon>utilities-file-archiver</icon>
    <name>HammerIO: Decompress</name>
    <unique-id>hammerio-decompress</unique-id>
    <command>$HELPER decompress %F</command>
    <description>Decompress with HammerIO</description>
    <patterns>*.zst;*.lz4;*.hammer;*.gz;*.bz2</patterns>
    <other-files/>
</action>
<action>
    <icon>dialog-information</icon>
    <name>HammerIO: Analyze Route</name>
    <unique-id>hammerio-analyze</unique-id>
    <command>$HELPER analyze %F</command>
    <description>Show how HammerIO would process this file</description>
    <patterns>*</patterns>
    <directories/>
    <audio-files/>
    <image-files/>
    <other-files/>
    <text-files/>
    <video-files/>
</action>
</actions>
EOF
            echo "  Installed 3 Thunar custom actions"
            echo "  Right-click → HammerIO: Compress/Decompress/Analyze"
        fi
    fi
fi

# ─── Nemo Actions ─────────────────────────────────────────────────────────────

if command -v nemo &>/dev/null; then
    echo ""
    echo "Installing Nemo actions..."
    NEMO_DIR="$HOME/.local/share/nemo/actions"
    mkdir -p "$NEMO_DIR"

    cat > "$NEMO_DIR/hammerio-compress.nemo_action" << EOF
[Nemo Action]
Name=HammerIO: Compress (GPU)
Comment=Compress with GPU acceleration via HammerIO
Exec=$HELPER compress %F
Icon-Name=utilities-file-archiver
Selection=any
Extensions=any
EOF

    cat > "$NEMO_DIR/hammerio-decompress.nemo_action" << EOF
[Nemo Action]
Name=HammerIO: Decompress
Comment=Decompress with HammerIO
Exec=$HELPER decompress %F
Icon-Name=utilities-file-archiver
Selection=s
Extensions=zst;lz4;hammer;gz;bz2
EOF

    echo "  Installed 2 Nemo actions"
fi

# ─── .desktop files (for "Open With" menu) ────────────────────────────────────

echo ""
echo "Installing .desktop entries (Open With menu)..."
APPS_DIR="$HOME/.local/share/applications"
mkdir -p "$APPS_DIR"

cat > "$APPS_DIR/hammerio-compress.desktop" << EOF
[Desktop Entry]
Type=Application
Name=Compress with HammerIO (GPU)
Comment=GPU-accelerated compression via HammerIO
Exec=$HELPER compress %F
Icon=utilities-file-archiver
Terminal=false
Categories=Utility;Archiving;
MimeType=application/octet-stream;text/plain;text/csv;video/mp4;video/x-matroska;image/jpeg;image/png;audio/wav;audio/mpeg;
NoDisplay=true
EOF

cat > "$APPS_DIR/hammerio-decompress.desktop" << EOF
[Desktop Entry]
Type=Application
Name=Decompress with HammerIO
Comment=Decompress files with HammerIO
Exec=$HELPER decompress %F
Icon=utilities-file-archiver
Terminal=false
Categories=Utility;Archiving;
MimeType=application/zstd;application/gzip;application/x-bzip2;application/x-lz4;
NoDisplay=true
EOF

echo "  Installed 2 .desktop files"

# Update desktop database
update-desktop-database "$APPS_DIR" 2>/dev/null || true

# ─── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Installation Complete!                         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Right-click any file to see HammerIO options:"
echo ""
echo "  Nautilus: Right-click → Scripts → HammerIO: ..."
echo "  Thunar:   Right-click → HammerIO: ..."
echo "  Nemo:     Right-click → HammerIO: ..."
echo "  Any app:  Right-click → Open With → Compress with HammerIO"
echo ""
echo "  Supported actions:"
echo "    • Compress (GPU)     — Auto-routes to fastest encoder"
echo "    • Decompress         — Handles .zst, .lz4, .gz, .bz2, .hammer"
echo "    • Analyze Route      — Shows how HammerIO would process the file"
echo "    • Open Terminal Here  — Opens terminal with HammerIO on PATH"
echo ""
echo "  Uninstall: $0 --uninstall"
echo ""
