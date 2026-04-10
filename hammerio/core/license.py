"""First-run license agreement for HammerIO.

On the very first CLI invocation, displays the license terms and requires
the user to type 'I AGREE' before any command will execute.  Acceptance is
persisted to ~/.config/hammerio/license_accepted so the prompt only appears
once.

Copyright 2026 ResilientMind AI | Joseph C McGinty Jr
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_CONFIG_DIR = Path.home() / ".config" / "hammerio"
_ACCEPTANCE_FILE = _CONFIG_DIR / "license_accepted"

LICENSE_TEXT = """\
HammerIO — License Agreement
Copyright 2026 ResilientMind AI | ResilientMindai.com | Joseph C McGinty Jr

By using HammerIO you agree to the following terms:

PROPRIETARY LICENSE
  HammerIO is proprietary software. All rights reserved.
  You may use HammerIO solely for personal, non-commercial evaluation.

  You may NOT without prior written permission:
    - Copy, reproduce, or distribute the Software
    - Use the Software for any commercial purpose
    - Modify, reverse engineer, or create derivative works
    - Redistribute or make the Software available to third parties

COMMERCIAL USE
  A commercial license is required if you:
    - Embed HammerIO in a commercial product or service
    - Distribute HammerIO as part of a paid offering
    - Use HammerIO in a SaaS platform
    - OEM HammerIO into hardware products

  Commercial licenses: Joseph@ResilientMindAI.com
  Details: https://hammerio.dev/license

DISCLAIMER
  THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.
  See the LICENSE file for the full terms and conditions.

Full license text: https://github.com/Subzero121800/HammerIO/blob/main/LICENSE
"""


def is_license_accepted() -> bool:
    """Return True if the user has previously accepted the license."""
    return _ACCEPTANCE_FILE.exists()


def record_acceptance() -> None:
    """Write the acceptance marker file."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    _ACCEPTANCE_FILE.write_text(
        f"License accepted: {timestamp}\n"
        f"HammerIO — Proprietary License\n"
        f"Copyright 2026 ResilientMind AI | Joseph C McGinty Jr\n"
    )


def require_license_acceptance() -> None:
    """Show the license and block until the user accepts.

    Called once on first CLI invocation.  If the terminal is not
    interactive (piped stdin), prints the license and exits with
    instructions to accept manually.
    """
    if is_license_accepted():
        return

    # Non-interactive — can't prompt
    if not sys.stdin.isatty():
        print(LICENSE_TEXT)
        print(
            "\nTo accept the license non-interactively, run:\n"
            "  hammer --accept-license\n"
            "\nOr create the marker file:\n"
            f"  mkdir -p {_CONFIG_DIR} && touch {_ACCEPTANCE_FILE}\n"
        )
        sys.exit(1)

    # Interactive — show license and prompt
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        console.print()
        console.print(Panel(
            LICENSE_TEXT,
            title="[bold]HammerIO License Agreement[/bold]",
            border_style="yellow",
            expand=False,
        ))
        console.print()
        console.print(
            '[bold]To accept these terms and use HammerIO, '
            'type [green]I AGREE[/green] below.[/bold]'
        )
        console.print('[dim]To decline, press Ctrl+C or type anything else.[/dim]\n')

        response = input("  > ").strip()

        if response == "I AGREE":
            record_acceptance()
            console.print("\n[bold green]License accepted.[/bold green] Welcome to HammerIO.\n")
        else:
            console.print("\n[yellow]License not accepted. Exiting.[/yellow]")
            sys.exit(1)

    except (ImportError, KeyboardInterrupt):
        # Fallback without Rich, or user hit Ctrl+C
        print(LICENSE_TEXT)
        print('\nType "I AGREE" to accept:')
        try:
            response = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(1)

        if response == "I AGREE":
            record_acceptance()
            print("\nLicense accepted. Welcome to HammerIO.\n")
        else:
            print("\nLicense not accepted. Exiting.")
            sys.exit(1)


def reset_license() -> None:
    """Remove the acceptance marker (for testing)."""
    if _ACCEPTANCE_FILE.exists():
        _ACCEPTANCE_FILE.unlink()
