#!/usr/bin/env bash
# HammerIO file manager action helper
# Called by Nautilus scripts, Thunar custom actions, Nemo actions, .desktop files
#
# Single file:  compresses directly → file.zst
# Multiple files: creates tar archive → selection.tar.zst (single output)
#
# Usage: hammerio-action.sh <compress|decompress|analyze> <file1> [file2] ...

ACTION="${1:-compress}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
LOG="/tmp/hammerio-action.log"

# Activate venv
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

export EGL_LOG_LEVEL=fatal

notify() {
    local title="$1" body="$2" icon="${3:-dialog-information}"
    command -v notify-send &>/dev/null && notify-send -i "$icon" "$title" "$body" 2>/dev/null || true
}

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

TOTAL=$#

case "$ACTION" in
    compress)
        if [ "$TOTAL" -gt 1 ]; then
            # ── Multiple files → single tar.zst archive ──────────────────
            # Determine output name from the parent directory
            FIRST_FILE="$1"
            PARENT_DIR="$(cd "$(dirname "$FIRST_FILE")" && pwd)"
            DIR_NAME="$(basename "$PARENT_DIR")"
            TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
            ARCHIVE_NAME="${DIR_NAME}_${TOTAL}files_${TIMESTAMP}"
            TAR_FILE="${PARENT_DIR}/${ARCHIVE_NAME}.tar"
            OUTPUT="${TAR_FILE}.zst"

            notify "HammerIO" "Archiving ${TOTAL} files into single compressed archive..." "utilities-file-archiver"
            PID=$(show_progress "Creating archive: ${ARCHIVE_NAME}.tar.zst (${TOTAL} files)...")

            # Build file list (basenames relative to parent dir)
            FILE_LIST=()
            for FILE in "$@"; do
                [ ! -e "$FILE" ] && continue
                FILE_LIST+=("$(basename "$FILE")")
            done

            # Create tar archive
            if tar -cf "$TAR_FILE" -C "$PARENT_DIR" "${FILE_LIST[@]}" >> "$LOG" 2>&1; then
                TAR_SIZE=$(du -sh "$TAR_FILE" 2>/dev/null | cut -f1)

                # Compress the tar with HammerIO
                if hammer compress "$TAR_FILE" --quality balanced >> "$LOG" 2>&1; then
                    # Remove the intermediate .tar
                    rm -f "$TAR_FILE"

                    # Find the output (could be .tar.zst or .tar.lz4)
                    for ext in .zst .lz4 .gz; do
                        if [ -f "${TAR_FILE}${ext}" ]; then
                            OUTPUT="${TAR_FILE}${ext}"
                            break
                        fi
                    done

                    OUT_SIZE=$(du -sh "$OUTPUT" 2>/dev/null | cut -f1)
                    notify "HammerIO" "Archive created: $(basename "$OUTPUT")\n${TOTAL} files: ${TAR_SIZE} → ${OUT_SIZE}" "emblem-default"
                else
                    rm -f "$TAR_FILE"
                    notify "HammerIO" "Compression failed\nCheck $LOG" "dialog-error"
                fi
            else
                notify "HammerIO" "Failed to create tar archive\nCheck $LOG" "dialog-error"
            fi

            close_progress "$PID"

        else
            # ── Single file → direct compress ────────────────────────────
            FILE="$1"
            [ ! -e "$FILE" ] && exit 1
            BASENAME="$(basename "$FILE")"

            notify "HammerIO" "Compressing: $BASENAME" "utilities-file-archiver"
            PID=$(show_progress "Compressing $BASENAME...")

            if hammer compress "$FILE" --quality balanced >> "$LOG" 2>&1; then
                # Find output
                OUTPUT="$FILE.zst"
                for ext in .zst .lz4 .gz .bz2; do
                    if [ -f "${FILE}${ext}" ]; then
                        OUTPUT="${FILE}${ext}"
                        break
                    fi
                done
                OUTSIZE=$(du -sh "$OUTPUT" 2>/dev/null | cut -f1)
                INSIZE=$(du -sh "$FILE" 2>/dev/null | cut -f1)
                notify "HammerIO" "Done: $BASENAME\n$INSIZE → $OUTSIZE" "emblem-default"
            else
                notify "HammerIO" "Failed: $BASENAME\nCheck $LOG" "dialog-error"
            fi

            close_progress "$PID"
        fi
        ;;

    decompress)
        DONE=0
        ERRORS=0
        for FILE in "$@"; do
            [ ! -e "$FILE" ] && continue
            BASENAME="$(basename "$FILE")"

            notify "HammerIO" "Decompressing: $BASENAME" "utilities-file-archiver"
            PID=$(show_progress "Decompressing $BASENAME...")

            if hammer decompress "$FILE" >> "$LOG" 2>&1; then
                DONE=$((DONE + 1))

                # If it's a .tar.zst, auto-extract the tar
                DECOMPRESSED="${FILE%.zst}"
                DECOMPRESSED="${DECOMPRESSED%.lz4}"
                DECOMPRESSED="${DECOMPRESSED%.gz}"
                if [ -f "$DECOMPRESSED" ] && [[ "$DECOMPRESSED" == *.tar ]]; then
                    EXTRACT_DIR="$(dirname "$DECOMPRESSED")"
                    tar -xf "$DECOMPRESSED" -C "$EXTRACT_DIR" >> "$LOG" 2>&1 && rm -f "$DECOMPRESSED"
                    notify "HammerIO" "Extracted: $BASENAME → $EXTRACT_DIR" "emblem-default"
                else
                    notify "HammerIO" "Decompressed: $BASENAME" "emblem-default"
                fi
            else
                ERRORS=$((ERRORS + 1))
                notify "HammerIO" "Failed: $BASENAME\nCheck $LOG" "dialog-error"
            fi

            close_progress "$PID"
        done

        if [ $TOTAL -gt 1 ]; then
            notify "HammerIO" "Batch decompress: $DONE/$TOTAL succeeded" \
                "$([ $ERRORS -eq 0 ] && echo 'emblem-default' || echo 'dialog-warning')"
        fi
        ;;

    terminal)
        DIR="$1"
        [ -f "$DIR" ] && DIR="$(dirname "$DIR")"
        [ ! -d "$DIR" ] && DIR="$HOME"
        if command -v gnome-terminal &>/dev/null; then
            gnome-terminal --working-directory="$DIR" -- bash -c 'export PATH="$HOME/.local/bin:$PATH"; echo "HammerIO ready — type: hammer --help"; exec bash'
        elif command -v xfce4-terminal &>/dev/null; then
            xfce4-terminal --working-directory="$DIR" -e 'bash -c "export PATH=\"$HOME/.local/bin:\$PATH\"; echo \"HammerIO ready — type: hammer --help\"; exec bash"'
        elif command -v konsole &>/dev/null; then
            konsole --workdir "$DIR" -e bash -c 'export PATH="$HOME/.local/bin:$PATH"; echo "HammerIO ready — type: hammer --help"; exec bash'
        elif command -v x-terminal-emulator &>/dev/null; then
            cd "$DIR" && x-terminal-emulator
        else
            notify "HammerIO" "No terminal emulator found" "dialog-error"
        fi
        ;;

    analyze)
        if [ $TOTAL -gt 1 ]; then
            RESULT="Routing analysis for $TOTAL files:\n\n"
            for FILE in "$@"; do
                [ ! -e "$FILE" ] && continue
                BASENAME="$(basename "$FILE")"
                ROUTE=$(hammer info --routes "$FILE" 2>&1 | grep -E "Route:|Algorithm:|Reason:" | head -3)
                RESULT="${RESULT}${BASENAME}:\n${ROUTE}\n\n"
            done
        else
            RESULT=$(hammer info --routes "$1" 2>&1 | head -20)
        fi

        if command -v zenity &>/dev/null; then
            echo -e "$RESULT" | zenity --text-info --title="HammerIO: Route Analysis" \
                --width=600 --height=400 2>/dev/null
        else
            notify "HammerIO" "$RESULT"
        fi
        ;;
esac
