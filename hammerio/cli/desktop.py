"""Desktop integration — right-click context menu installer for HammerIO.

Installs 'Compress with HammerIO' and 'Decompress with HammerIO' into
the right-click menu of Nautilus, Nemo, Thunar, and the generic
'Open With' menu via .desktop files.

Usage:
    hammer install-desktop          # Install right-click integration
    hammer install-desktop --remove # Uninstall

Copyright 2026 ResilientMind AI | Joseph C McGinty Jr
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Where we place the action helper script and assets
_DATA_DIR = Path.home() / ".local" / "share" / "hammerio"
_HELPER = _DATA_DIR / "hammerio-action.sh"
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

_NAUTILUS_DIR = Path.home() / ".local" / "share" / "nautilus" / "scripts"
_NEMO_DIR = Path.home() / ".local" / "share" / "nemo" / "actions"
_APPS_DIR = Path.home() / ".local" / "share" / "applications"
_MIME_DIR = Path.home() / ".local" / "share" / "mime"
_ICONS_DIR = Path.home() / ".local" / "share" / "icons" / "hicolor"
_THUNAR_UCA = Path.home() / ".config" / "Thunar" / "uca.xml"

# ── Helper script (bash) ────────────────────────────────────────────────

_ACTION_SCRIPT = r'''#!/usr/bin/env bash
# HammerIO file manager action helper
# Called by Nautilus scripts, Nemo actions, Thunar custom actions, .desktop files
#
# Single file:  compresses directly -> file.zst
# Multiple files: creates tar archive -> selection.tar.zst
#
# Ensure hammer is on PATH (pip --user installs to ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"

# Usage: hammerio-action.sh <compress|decompress|analyze> <file1> [file2] ...

ACTION="${1:-compress}"
shift

LOG="/tmp/hammerio-action.log"
export EGL_LOG_LEVEL=fatal

notify() {
    local title="$1" body="$2" icon="${3:-dialog-information}"
    command -v notify-send &>/dev/null && notify-send -i "$icon" "$title" "$body" 2>/dev/null || true
}

show_progress() {
    if command -v zenity &>/dev/null; then
        zenity --progress --title="HammerIO" --text="$1" --percentage=0 --auto-close --no-cancel --width=350 2>/dev/null &
        echo $!
    else
        echo ""
    fi
}

update_progress() {
    # Feed percentage to zenity via /proc/$PID/fd/0
    local pid="$1" pct="$2" text="$3"
    if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
        echo "$pct" > /proc/$pid/fd/0 2>/dev/null || true
        [ -n "$text" ] && echo "# $text" > /proc/$pid/fd/0 2>/dev/null || true
    fi
}

close_progress() {
    [ -n "$1" ] && kill "$1" 2>/dev/null || true
}

TOTAL=$#

case "$ACTION" in
    compress)
        if [ "$TOTAL" -gt 1 ]; then
            FIRST_FILE="$1"
            PARENT_DIR="$(cd "$(dirname "$FIRST_FILE")" && pwd)"
            DIR_NAME="$(basename "$PARENT_DIR")"
            TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
            ARCHIVE_NAME="${DIR_NAME}_${TOTAL}files_${TIMESTAMP}"
            TAR_FILE="${PARENT_DIR}/${ARCHIVE_NAME}.tar"
            OUTPUT="${TAR_FILE}.zst"

            notify "HammerIO" "Archiving ${TOTAL} files into single compressed archive..." "utilities-file-archiver"
            PID=$(show_progress "Creating archive: ${ARCHIVE_NAME}.tar.zst (${TOTAL} files)...")

            FILE_LIST=()
            for FILE in "$@"; do
                [ ! -e "$FILE" ] && continue
                FILE_LIST+=("$(basename "$FILE")")
            done

            if tar -cf "$TAR_FILE" -C "$PARENT_DIR" "${FILE_LIST[@]}" >> "$LOG" 2>&1; then
                TAR_SIZE=$(du -sh "$TAR_FILE" 2>/dev/null | cut -f1)
                if hammer compress "$TAR_FILE" --quality balanced >> "$LOG" 2>&1; then
                    rm -f "$TAR_FILE"
                    for ext in .zst .lz4 .gz; do
                        if [ -f "${TAR_FILE}${ext}" ]; then OUTPUT="${TAR_FILE}${ext}"; break; fi
                    done
                    OUT_SIZE=$(du -sh "$OUTPUT" 2>/dev/null | cut -f1)
                    notify "HammerIO" "Archive created: $(basename "$OUTPUT")\n${TOTAL} files: ${TAR_SIZE} -> ${OUT_SIZE}" "emblem-default"
                else
                    rm -f "$TAR_FILE"
                    notify "HammerIO" "Compression failed\nCheck $LOG" "dialog-error"
                fi
            else
                notify "HammerIO" "Failed to create tar archive\nCheck $LOG" "dialog-error"
            fi
            close_progress "$PID"

        else
            FILE="$1"
            [ ! -e "$FILE" ] && exit 1
            BASENAME="$(basename "$FILE")"
            notify "HammerIO" "Compressing: $BASENAME" "utilities-file-archiver"
            PID=$(show_progress "Compressing $BASENAME...")
            if hammer compress "$FILE" --quality balanced >> "$LOG" 2>&1; then
                OUTPUT="$FILE.zst"
                for ext in .zst .lz4 .gz .bz2; do
                    if [ -f "${FILE}${ext}" ]; then OUTPUT="${FILE}${ext}"; break; fi
                done
                OUTSIZE=$(du -sh "$OUTPUT" 2>/dev/null | cut -f1)
                INSIZE=$(du -sh "$FILE" 2>/dev/null | cut -f1)
                notify "HammerIO" "Done: $BASENAME\n$INSIZE -> $OUTSIZE" "emblem-default"
            else
                notify "HammerIO" "Failed: $BASENAME\nCheck $LOG" "dialog-error"
            fi
            close_progress "$PID"
        fi
        ;;

    decompress)
        DONE=0; ERRORS=0
        for FILE in "$@"; do
            [ ! -e "$FILE" ] && continue
            BASENAME="$(basename "$FILE")"
            notify "HammerIO" "Decompressing: $BASENAME" "utilities-file-archiver"
            PID=$(show_progress "Decompressing $BASENAME...")
            if hammer decompress "$FILE" >> "$LOG" 2>&1; then
                DONE=$((DONE + 1))
                DECOMPRESSED="${FILE%.zst}"; DECOMPRESSED="${DECOMPRESSED%.lz4}"; DECOMPRESSED="${DECOMPRESSED%.gz}"
                if [ -f "$DECOMPRESSED" ] && [[ "$DECOMPRESSED" == *.tar ]]; then
                    EXTRACT_DIR="$(dirname "$DECOMPRESSED")"
                    tar -xf "$DECOMPRESSED" -C "$EXTRACT_DIR" >> "$LOG" 2>&1 && rm -f "$DECOMPRESSED"
                    notify "HammerIO" "Extracted: $BASENAME -> $EXTRACT_DIR" "emblem-default"
                else
                    notify "HammerIO" "Decompressed: $BASENAME" "emblem-default"
                fi
            else
                ERRORS=$((ERRORS + 1))
                notify "HammerIO" "Failed: $BASENAME\nCheck $LOG" "dialog-error"
            fi
            close_progress "$PID"
        done
        [ $TOTAL -gt 1 ] && notify "HammerIO" "Batch decompress: $DONE/$TOTAL succeeded" \
            "$([ $ERRORS -eq 0 ] && echo 'emblem-default' || echo 'dialog-warning')"
        ;;

    decompress_to)
        # Prompt user for destination folder
        if command -v zenity &>/dev/null; then
            DEST=$(zenity --file-selection --directory --title="HammerIO: Decompress to..." 2>/dev/null)
        elif command -v kdialog &>/dev/null; then
            DEST=$(kdialog --getexistingdirectory "$HOME" --title "HammerIO: Decompress to..." 2>/dev/null)
        else
            notify "HammerIO" "No file picker available (install zenity)" "dialog-error"
            exit 1
        fi
        [ -z "$DEST" ] && exit 0

        DONE=0; ERRORS=0
        for FILE in "$@"; do
            [ ! -e "$FILE" ] && continue
            BASENAME="$(basename "$FILE")"
            notify "HammerIO" "Decompressing: $BASENAME → $DEST" "utilities-file-archiver"
            PID=$(show_progress "Decompressing $BASENAME to $DEST...")
            if hammer decompress "$FILE" -o "$DEST" >> "$LOG" 2>&1; then
                DONE=$((DONE + 1))
                notify "HammerIO" "Decompressed: $BASENAME → $DEST" "emblem-default"
            else
                ERRORS=$((ERRORS + 1))
                notify "HammerIO" "Failed: $BASENAME\nCheck $LOG" "dialog-error"
            fi
            close_progress "$PID"
        done
        [ $TOTAL -gt 1 ] && notify "HammerIO" "Batch decompress: $DONE/$TOTAL to $DEST" \
            "$([ $ERRORS -eq 0 ] && echo 'emblem-default' || echo 'dialog-warning')"
        ;;

    terminal)
        # Open a terminal in the selected directory with hammer on PATH
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
            echo -e "$RESULT" | zenity --text-info --title="HammerIO: Route Analysis" --width=600 --height=400 2>/dev/null
        else
            notify "HammerIO" "$RESULT"
        fi
        ;;
esac
'''


# ── Install / Uninstall Logic ────────────────────────────────────────────

def _write_helper() -> Path:
    """Write the action helper script and return its path."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _HELPER.write_text(_ACTION_SCRIPT)
    _HELPER.chmod(_HELPER.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return _HELPER


def _has_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def _install_mime_and_icon() -> None:
    """Register the application/x-hammerio MIME type and .hammer file icon."""
    mime_src = _ASSETS_DIR / "hammerio-mime.xml"
    icon_src = _ASSETS_DIR / "hammer-icon.svg"

    if not mime_src.exists() or not icon_src.exists():
        console.print("[yellow]Warning: MIME/icon assets not found — skipping icon registration[/yellow]")
        return

    # Install MIME type via xdg-mime
    try:
        subprocess.run(
            ["xdg-mime", "install", "--novendor", str(mime_src)],
            capture_output=True, timeout=10,
        )
        console.print("  MIME type: [green]application/x-hammerio registered[/green]")
    except FileNotFoundError:
        console.print("  [yellow]xdg-mime not found — MIME type not registered[/yellow]")

    # Install icon at multiple sizes via xdg-icon-resource
    for size in (48, 64, 128, 256):
        try:
            subprocess.run(
                [
                    "xdg-icon-resource", "install", "--novendor",
                    "--context", "mimetypes",
                    "--size", str(size),
                    str(icon_src), "application-x-hammerio",
                ],
                capture_output=True, timeout=10,
            )
        except FileNotFoundError:
            break

    # Also copy the SVG directly for scalable icon support
    scalable_dir = _ICONS_DIR / "scalable" / "mimetypes"
    scalable_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(icon_src), str(scalable_dir / "application-x-hammerio.svg"))

    # Update icon cache
    try:
        subprocess.run(
            ["gtk-update-icon-cache", "-f", "-t", str(_ICONS_DIR.parent)],
            capture_output=True, timeout=30,
        )
    except FileNotFoundError:
        pass

    # Update MIME database
    try:
        subprocess.run(
            ["update-mime-database", str(_MIME_DIR)],
            capture_output=True, timeout=30,
        )
    except FileNotFoundError:
        pass

    console.print("  File icon: [green].hammer files now have the HammerIO icon[/green]")


def _uninstall_mime_and_icon() -> None:
    """Remove MIME type and icon registrations."""
    mime_src = _ASSETS_DIR / "hammerio-mime.xml"
    try:
        subprocess.run(
            ["xdg-mime", "uninstall", str(mime_src)],
            capture_output=True, timeout=10,
        )
    except (FileNotFoundError, Exception):
        pass

    for size in (48, 64, 128, 256):
        try:
            subprocess.run(
                [
                    "xdg-icon-resource", "uninstall",
                    "--context", "mimetypes",
                    "--size", str(size),
                    "application-x-hammerio",
                ],
                capture_output=True, timeout=10,
            )
        except (FileNotFoundError, Exception):
            pass

    scalable = _ICONS_DIR / "scalable" / "mimetypes" / "application-x-hammerio.svg"
    if scalable.exists():
        scalable.unlink()

    try:
        subprocess.run(["update-mime-database", str(_MIME_DIR)], capture_output=True, timeout=30)
    except (FileNotFoundError, Exception):
        pass
    try:
        subprocess.run(["gtk-update-icon-cache", "-f", "-t", str(_ICONS_DIR.parent)], capture_output=True, timeout=30)
    except (FileNotFoundError, Exception):
        pass


def install(remove: bool = False) -> None:
    """Install or remove desktop right-click integration."""

    if remove:
        _uninstall()
        return

    console.print(Panel(
        "[bold]HammerIO Desktop Integration[/bold]\n"
        "Right-click compress / decompress from your file manager",
        border_style="cyan",
    ))

    # Verify hammer is on PATH
    hammer_path = shutil.which("hammer")
    if not hammer_path:
        console.print("[yellow]Warning: 'hammer' not found on PATH.[/yellow]")
        console.print("  Make sure HammerIO is installed: pip install hammerio")
        console.print("  Or add ~/.local/bin to your PATH\n")

    helper = _write_helper()
    console.print(f"  Action script: [dim]{helper}[/dim]")

    # ── MIME type & icon ──
    _install_mime_and_icon()

    table = Table(show_header=True, header_style="bold")
    table.add_column("File Manager")
    table.add_column("Status", justify="center")
    table.add_column("Actions")

    # ── Nautilus ──
    if _has_cmd("nautilus"):
        _NAUTILUS_DIR.mkdir(parents=True, exist_ok=True)

        for name, action in [
            ("HammerIO: Compress (GPU)", "compress"),
            ("HammerIO: Decompress", "decompress"),
            ("HammerIO: Decompress to...", "decompress_to"),
            ("HammerIO: Analyze Route", "analyze"),
            ("HammerIO: Open Terminal Here", "terminal"),
        ]:
            script = _NAUTILUS_DIR / name
            script.write_text(
                f'#!/usr/bin/env bash\n'
                f'"{helper}" {action} $NAUTILUS_SCRIPT_SELECTED_FILE_PATHS\n'
            )
            script.chmod(script.stat().st_mode | stat.S_IEXEC)

        table.add_row("Nautilus (GNOME)", "[green]Installed[/green]",
                       "Right-click > Scripts > HammerIO")
    else:
        table.add_row("Nautilus (GNOME)", "[dim]Not found[/dim]", "Skipped")

    # ── Nemo ──
    if _has_cmd("nemo"):
        _NEMO_DIR.mkdir(parents=True, exist_ok=True)

        (_NEMO_DIR / "hammerio-compress.nemo_action").write_text(
            f"[Nemo Action]\n"
            f"Name=HammerIO: Compress (GPU)\n"
            f"Comment=Compress with GPU acceleration via HammerIO\n"
            f"Exec={helper} compress %F\n"
            f"Icon-Name=utilities-file-archiver\n"
            f"Selection=any\n"
            f"Extensions=any\n"
        )
        (_NEMO_DIR / "hammerio-decompress.nemo_action").write_text(
            f"[Nemo Action]\n"
            f"Name=HammerIO: Decompress\n"
            f"Comment=Decompress with HammerIO\n"
            f"Exec={helper} decompress %F\n"
            f"Icon-Name=utilities-file-archiver\n"
            f"Selection=s\n"
            f"Extensions=zst;lz4;hammer;gz;bz2\n"
        )
        (_NEMO_DIR / "hammerio-decompress-to.nemo_action").write_text(
            f"[Nemo Action]\n"
            f"Name=HammerIO: Decompress to...\n"
            f"Comment=Decompress to a chosen folder\n"
            f"Exec={helper} decompress_to %F\n"
            f"Icon-Name=utilities-file-archiver\n"
            f"Selection=s\n"
            f"Extensions=zst;lz4;hammer;gz;bz2\n"
        )

        (_NEMO_DIR / "hammerio-terminal.nemo_action").write_text(
            f"[Nemo Action]\n"
            f"Name=HammerIO: Open Terminal Here\n"
            f"Comment=Open terminal with HammerIO on PATH\n"
            f"Exec={helper} terminal %F\n"
            f"Icon-Name=utilities-terminal\n"
            f"Selection=any\n"
            f"Extensions=dir\n"
        )

        table.add_row("Nemo (Cinnamon)", "[green]Installed[/green]",
                       "Right-click > HammerIO")
    else:
        table.add_row("Nemo (Cinnamon)", "[dim]Not found[/dim]", "Skipped")

    # ── Thunar ──
    if _has_cmd("thunar"):
        if _THUNAR_UCA.exists():
            import re as _re
            uca = _THUNAR_UCA.read_text()
            # Remove existing HammerIO actions before re-adding
            uca = _re.sub(
                r'<action>\s*<[^>]*>.*?hammerio.*?</action>\s*',
                '', uca, flags=_re.DOTALL | _re.IGNORECASE,
            )
            actions_xml = (
                f'<action>\n'
                f'    <icon>application-x-hammerio</icon>\n'
                f'    <name>HammerIO: Compress (GPU)</name>\n'
                f'    <unique-id>hammerio-compress</unique-id>\n'
                f'    <command>{helper} compress %F</command>\n'
                f'    <description>Compress with GPU acceleration</description>\n'
                f'    <patterns>*</patterns>\n'
                f'    <directories/><audio-files/><image-files/><other-files/><text-files/><video-files/>\n'
                f'</action>\n'
                f'<action>\n'
                f'    <icon>application-x-hammerio</icon>\n'
                f'    <name>HammerIO: Decompress</name>\n'
                f'    <unique-id>hammerio-decompress</unique-id>\n'
                f'    <command>{helper} decompress %F</command>\n'
                f'    <description>Decompress with HammerIO</description>\n'
                f'    <patterns>*.zst;*.lz4;*.hammer;*.gz;*.bz2</patterns>\n'
                f'    <other-files/>\n'
                f'</action>\n'
                f'<action>\n'
                f'    <icon>application-x-hammerio</icon>\n'
                f'    <name>HammerIO: Decompress to...</name>\n'
                f'    <unique-id>hammerio-decompress-to</unique-id>\n'
                f'    <command>{helper} decompress_to %F</command>\n'
                f'    <description>Decompress to a chosen folder</description>\n'
                f'    <patterns>*.zst;*.lz4;*.hammer;*.gz;*.bz2</patterns>\n'
                f'    <other-files/>\n'
                f'</action>\n'
                f'<action>\n'
                f'    <icon>utilities-terminal</icon>\n'
                f'    <name>HammerIO: Open Terminal Here</name>\n'
                f'    <unique-id>hammerio-terminal</unique-id>\n'
                f'    <command>{helper} terminal %F</command>\n'
                f'    <description>Open terminal with HammerIO on PATH</description>\n'
                f'    <patterns>*</patterns>\n'
                f'    <directories/>\n'
                f'</action>\n'
            )
            uca = uca.replace("</actions>", actions_xml + "</actions>")
            _THUNAR_UCA.write_text(uca)
            table.add_row("Thunar (XFCE)", "[green]Installed[/green]",
                           "Right-click > HammerIO")
        else:
            table.add_row("Thunar (XFCE)", "[yellow]No uca.xml[/yellow]", "Skipped")
    else:
        table.add_row("Thunar (XFCE)", "[dim]Not found[/dim]", "Skipped")

    # ── .desktop files (Open With) ──
    _APPS_DIR.mkdir(parents=True, exist_ok=True)

    (_APPS_DIR / "hammerio-compress.desktop").write_text(
        f"[Desktop Entry]\n"
        f"Type=Application\n"
        f"Name=Compress with HammerIO (GPU)\n"
        f"Comment=GPU-accelerated compression via HammerIO\n"
        f"Exec={helper} compress %F\n"
        f"Icon=utilities-file-archiver\n"
        f"Terminal=false\n"
        f"Categories=Utility;Archiving;\n"
        f"MimeType=application/octet-stream;text/plain;text/csv;"
        f"video/mp4;image/jpeg;image/png;audio/wav;\n"
        f"NoDisplay=true\n"
    )
    (_APPS_DIR / "hammerio-decompress.desktop").write_text(
        f"[Desktop Entry]\n"
        f"Type=Application\n"
        f"Name=Decompress with HammerIO\n"
        f"Comment=Decompress files with HammerIO\n"
        f"Exec={helper} decompress %F\n"
        f"Icon=application-x-hammerio\n"
        f"Terminal=false\n"
        f"Categories=Utility;Archiving;\n"
        f"MimeType=application/x-hammerio;application/zstd;application/gzip;"
        f"application/x-bzip2;application/x-lz4;\n"
    )
    table.add_row("Open With (.desktop)", "[green]Installed[/green]",
                   "Right-click > Open With")

    # Update desktop database
    subprocess.run(
        ["update-desktop-database", str(_APPS_DIR)],
        capture_output=True, timeout=10,
    )

    # Set HammerIO as the default app for .hammer files
    try:
        subprocess.run(
            ["xdg-mime", "default", "hammerio-decompress.desktop", "application/x-hammerio"],
            capture_output=True, timeout=10,
        )
    except FileNotFoundError:
        pass

    console.print(table)

    console.print(f"\n[bold green]Desktop integration installed.[/bold green]")
    console.print("  Right-click any file to see:")
    console.print("    [cyan]HammerIO: Compress (GPU)[/cyan]")
    console.print("    [cyan]HammerIO: Decompress[/cyan]")
    console.print("    [cyan]HammerIO: Analyze Route[/cyan]")
    console.print("    [cyan]HammerIO: Open Terminal Here[/cyan]")
    console.print(f"\n  Uninstall: [dim]hammer install-desktop --remove[/dim]")
    console.print("  You may need to restart your file manager for changes to take effect.")


def _uninstall() -> None:
    """Remove all desktop integration files."""
    console.print("[bold]Removing HammerIO desktop integration...[/bold]\n")

    _uninstall_mime_and_icon()
    removed = 0

    # Nautilus scripts
    for name in ["HammerIO: Compress (GPU)", "HammerIO: Decompress", "HammerIO: Decompress to...", "HammerIO: Analyze Route", "HammerIO: Open Terminal Here"]:
        f = _NAUTILUS_DIR / name
        if f.exists():
            f.unlink()
            removed += 1

    # Nemo actions
    for name in ["hammerio-compress.nemo_action", "hammerio-decompress.nemo_action", "hammerio-decompress-to.nemo_action", "hammerio-terminal.nemo_action"]:
        f = _NEMO_DIR / name
        if f.exists():
            f.unlink()
            removed += 1

    # .desktop files
    for name in ["hammerio-compress.desktop", "hammerio-decompress.desktop"]:
        f = _APPS_DIR / name
        if f.exists():
            f.unlink()
            removed += 1

    # Thunar (remove our actions from uca.xml)
    if _THUNAR_UCA.exists():
        uca = _THUNAR_UCA.read_text()
        if "hammerio" in uca.lower():
            import re
            uca = re.sub(
                r'<action>\s*<[^>]*>.*?hammerio.*?</action>',
                '', uca, flags=re.DOTALL | re.IGNORECASE,
            )
            _THUNAR_UCA.write_text(uca)
            removed += 1

    # Helper script
    if _HELPER.exists():
        _HELPER.unlink()
        removed += 1

    if _DATA_DIR.exists() and not any(_DATA_DIR.iterdir()):
        _DATA_DIR.rmdir()

    subprocess.run(
        ["update-desktop-database", str(_APPS_DIR)],
        capture_output=True, timeout=10,
    )

    console.print(f"  Removed {removed} files.")
    console.print("[green]Desktop integration removed.[/green]")
    console.print("  Restart your file manager for changes to take effect.")
