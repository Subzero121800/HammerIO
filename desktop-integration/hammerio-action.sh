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
