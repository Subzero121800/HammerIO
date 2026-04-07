"""HammerIO CLI — Typer-based command-line interface.

GPU where it matters. CPU where it doesn't. Zero configuration.

Copyright 2026 ResilientMind AI | ResilientMindai.com | Joseph C McGinty Jr
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from enum import Enum
from pathlib import Path

# Suppress libEGL DRI2 authentication warning on Jetson
os.environ.setdefault("EGL_LOG_LEVEL", "fatal")
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

app = typer.Typer(
    name="hammer",
    help="HammerIO — GPU-accelerated compression & media processing.\n\n"
    "GPU where it matters. CPU where it doesn't. Zero configuration.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()


class QualityPreset(str, Enum):
    fast = "fast"
    balanced = "balanced"
    quality = "quality"
    lossless = "lossless"


class CompressMode(str, Enum):
    auto = "auto"
    gpu = "gpu"
    cpu = "cpu"
    bulk = "bulk"


@app.command()
def compress(
    input_path: str = typer.Argument(..., help="File or directory to compress"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path"),
    mode: CompressMode = typer.Option(CompressMode.auto, "--mode", "-m", help="Processing mode"),
    algo: Optional[str] = typer.Option(None, "--algo", "-a", help="Force algorithm (zstd, lz4, gzip, zip, bzip2)"),
    quality: QualityPreset = typer.Option(QualityPreset.balanced, "--quality", "-q", help="Quality preset"),
    workers: int = typer.Option(4, "--workers", "-w", help="Parallel workers for batch"),
) -> None:
    """Compress a file or directory using optimal GPU/CPU routing."""
    try:
        from hammerio.core.router import JobRouter

        with console.status("[bold green]Detecting hardware..."):
            router = JobRouter(quality=quality.value, force_mode=mode.value if mode != CompressMode.auto else None)

        input_p = Path(input_path)
        if not input_p.exists():
            console.print(f"[red]Error:[/red] Input not found: {input_path}")
            raise typer.Exit(1)

        # Show routing decision
        console.print(Panel(
            router.explain_route(input_p),
            title="[bold cyan]Routing Decision[/bold cyan]",
            border_style="cyan",
        ))

        # Execute with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Compressing...", total=100)

            def _progress_cb(job_id: str, pct: float) -> None:
                progress.update(task, completed=pct)

            router.set_progress_callback(_progress_cb)

            try:
                job = router.route(input_p, output, mode=mode.value if mode != CompressMode.auto else None, algorithm=algo)
                result = router.execute(job)
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1)

            progress.update(task, completed=100)

        # Show results
        _print_result(result)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Compress failed:[/red] {exc}")
        raise typer.Exit(1)


@app.command()
def decompress(
    input_path: str = typer.Argument(..., help="File to decompress"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path"),
) -> None:
    """Decompress a HammerIO-compressed file."""
    try:
        from hammerio.encoders.bulk import BulkEncoder
        from hammerio.encoders.general import GeneralEncoder
        from hammerio.core.hardware import detect_hardware

        input_p = Path(input_path)
        if not input_p.exists():
            console.print(f"[red]Error:[/red] File not found: {input_path}")
            raise typer.Exit(1)

        hw = detect_hardware()
        ext = input_p.suffix.lower()

        with console.status("[bold green]Decompressing..."):
            if ext == ".hammer":
                encoder = BulkEncoder(hw)
                out = encoder.decompress(input_p, Path(output) if output else None)
            elif ext in (".lzfse", ".hmac"):
                from hammerio.encoders.apple import AppleEncoder
                encoder = AppleEncoder(hw)
                out = encoder.decompress(input_p, Path(output) if output else None)
            elif ext in (".zst", ".gz", ".bz2", ".lz4", ".zip"):
                encoder = GeneralEncoder(hw)
                out = encoder.decompress(input_p, Path(output) if output else None)
            else:
                console.print(f"[red]Error:[/red] Unknown format: {ext}")
                raise typer.Exit(1)

            console.print(f"[green]Decompressed:[/green] {out}")
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Decompress failed:[/red] {exc}")
        raise typer.Exit(1)


@app.command()
def batch(
    input_dir: str = typer.Argument(..., help="Directory to process"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    mode: CompressMode = typer.Option(CompressMode.auto, "--mode", "-m", help="Processing mode"),
    workers: int = typer.Option(4, "--workers", "-w", help="Parallel workers"),
    quality: QualityPreset = typer.Option(QualityPreset.balanced, "--quality", "-q"),
) -> None:
    """Batch process all files in a directory."""
    try:
        from hammerio.core.router import JobRouter
        from hammerio.core.profiler import profile_directory

        input_p = Path(input_dir)
        if not input_p.is_dir():
            console.print(f"[red]Error:[/red] Not a directory: {input_dir}")
            raise typer.Exit(1)

        with console.status("[bold green]Analyzing directory..."):
            router = JobRouter(quality=quality.value)
            batch_profile = profile_directory(input_p, recursive=False)

        console.print(Panel(
            router.explain_route(input_p),
            title="[bold cyan]Batch Routing[/bold cyan]",
            border_style="cyan",
        ))

        file_count = batch_profile.file_count or 1

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing batch...", total=file_count)
            _completed = {"count": 0}

            # Wrap the original progress callback to track per-file completion
            _orig_cb = router._progress_callback

            def _batch_progress_cb(job_id: str, pct: float) -> None:
                if pct >= 100:
                    _completed["count"] += 1
                    progress.update(task, completed=_completed["count"])

            router.set_progress_callback(_batch_progress_cb)

            try:
                results = asyncio.run(router.execute_batch(
                    input_p, output_dir=output, mode=mode.value if mode != CompressMode.auto else None, workers=workers,
                ))
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1)

            progress.update(task, completed=file_count)

        # Summary table
        table = Table(title="Batch Results")
        table.add_column("File", style="cyan")
        table.add_column("Input", justify="right")
        table.add_column("Output", justify="right")
        table.add_column("Ratio", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("Processor", style="green")
        table.add_column("Status")

        for r in results:
            status = "[green]OK[/green]" if r.status.value == "completed" else f"[red]{r.status.value}[/red]"
            table.add_row(
                Path(r.input_path).name if r.input_path else "?",
                _human_size(r.input_size),
                _human_size(r.output_size),
                f"{r.compression_ratio:.2f}x",
                f"{r.elapsed_seconds:.1f}s",
                r.processor_used,
                status,
            )

        console.print(table)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Batch processing failed:[/red] {exc}")
        raise typer.Exit(1)




@app.command()
def benchmark(
    output: str = typer.Option("benchmarks/results/benchmark.json", "--output", "-o"),
    quick: bool = typer.Option(False, "--quick", help="Quick benchmark (100MB test data)"),
    large: bool = typer.Option(False, "--1gb", help="Large file benchmark (1GB download)"),
    huge: bool = typer.Option(False, "--10gb", help="Huge file benchmark (10GB generated)"),
    bench_type: str = typer.Option("all", "--type", "-t",
        help="Benchmark type: all, roundtrip, memory, random-io, scale"),
) -> None:
    """Run the HammerIO compression benchmark suite.

    Modes:
      --quick  100MB mixed data (fast, ~30 seconds)
      --1gb    1GB data from download (thorough, ~5 minutes)
      --10gb   10GB generated data (stress test, ~20 minutes)
      default  500MB mixed data (~2 minutes)

    Benchmark types (--type):
      all        Run all benchmark types (default)
      roundtrip  Sequential compress/decompress round-trip
      memory     In-memory only — pure algorithm speed, no disk I/O
      random-io  Random read/write/mixed patterns with IOPS & latency
      scale      Scalability sweep from 1MB to 1GB
    """
    console.print(Panel(
        "[bold]HammerIO Benchmark Suite[/bold]\n"
        "Compress → Decompress → Verify round-trip",
        border_style="yellow",
    ))

    try:
        import sys as _sys
        # Add project root to path so benchmarks module is always importable
        _project_root = str(Path(__file__).resolve().parent.parent.parent)
        if _project_root not in _sys.path:
            _sys.path.insert(0, _project_root)
        # Make output path absolute relative to project root if relative
        if not Path(output).is_absolute():
            output = str(Path(_project_root) / output)
        from benchmarks.run_benchmarks import run_all_benchmarks
        results = run_all_benchmarks(
            quick=quick, large=large, huge=huge,
            bench_type=bench_type, output_path=output,
        )
        console.print(f"\n[green]Results saved to:[/green] {output}")
    except ImportError as ie:
        console.print(f"[red]Benchmark module not found:[/red] {ie}")
        console.print(f"[dim]Searched: {_project_root}/benchmarks/[/dim]")
    except Exception as e:
        console.print(f"[red]Benchmark error:[/red] {e}")


@app.command()
def info(
    hardware: bool = typer.Option(False, "--hardware", "-hw", help="Show hardware profile"),
    routes: Optional[str] = typer.Option(None, "--routes", "-r", help="Show routes for a file/dir"),
    telemetry: bool = typer.Option(False, "--telemetry", "-t", help="Show live telemetry"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Display hardware info, routing decisions, or live telemetry."""
    try:
        if hardware or (not routes and not telemetry):
            _show_hardware(json_output)

        if routes:
            _show_routes(routes)

        if telemetry:
            _show_telemetry(json_output)
    except Exception as exc:
        console.print(f"[red]Info command failed:[/red] {exc}")
        raise typer.Exit(1)


@app.command()
def config(
    generate: bool = typer.Option(False, "--generate", "-g", help="Generate default config file"),
    show: bool = typer.Option(False, "--show", "-s", help="Show current config"),
    save: bool = typer.Option(False, "--save", help="Save current defaults to ~/.config/hammerio/config.toml"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="Config file path"),
) -> None:
    """Manage HammerIO configuration."""
    try:
        from hammerio.core.config import load_config, generate_default_config

        if generate:
            out = generate_default_config(path or "hammerio.toml")
            console.print(f"[green]Config generated:[/green] {out}")
            return

        if save:
            cfg = load_config(path)
            save_path = Path(path) if path else Path.home() / ".config" / "hammerio" / "config.toml"
            cfg.save(save_path)
            console.print(f"[green]Config saved to:[/green] {save_path}")
            return

        cfg = load_config(path)
        if cfg.config_path:
            console.print(f"[cyan]Config loaded from:[/cyan] {cfg.config_path}")
        else:
            console.print("[yellow]No config file found — using defaults[/yellow]")

        if show or not generate:
            import json as json_mod
            console.print_json(json_mod.dumps(cfg.to_dict(), indent=2))
    except Exception as exc:
        console.print(f"[red]Config command failed:[/red] {exc}")
        raise typer.Exit(1)


@app.command()
def monitor(
    interval: float = typer.Option(1.0, "--interval", "-i", help="Update interval in seconds"),
    count: int = typer.Option(0, "--count", "-n", help="Number of updates (0=infinite)"),
) -> None:
    """Live system telemetry monitor (jtop-style for HammerIO)."""
    try:
        from hammerio.core.telemetry import TelemetryCollector
        import time as time_mod

        collector = TelemetryCollector()
        iterations = 0

        try:
            while True:
                snap = collector.get_snapshot()
                # Clear screen and print
                console.clear()
                console.print(Panel(
                    collector.format_live_display(snap),
                    title="[bold yellow]HammerIO Monitor[/bold yellow] (Ctrl+C to exit)",
                    border_style="yellow",
                ))

                iterations += 1
                if count > 0 and iterations >= count:
                    break
                time_mod.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[dim]Monitor stopped.[/dim]")
    except Exception as exc:
        console.print(f"[red]Monitor failed:[/red] {exc}")
        raise typer.Exit(1)


@app.command()
def webui(
    host: str = typer.Option("0.0.0.0", "--host", "-H"),
    port: int = typer.Option(5000, "--port", "-p"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    """Launch the HammerIO web dashboard."""
    try:
        from hammerio.web.app import create_app, socketio
        app_instance = create_app()
        console.print(f"[bold green]HammerIO Dashboard[/bold green] → http://{host}:{port}")
        console.print(f"  Health:  http://{host}:{port}/health")
        console.print(f"  API:     http://{host}:{port}/api/hardware")
        console.print("[dim]Press Ctrl+C to stop.[/dim]")
        socketio.run(app_instance, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    except ImportError as e:
        console.print(f"[red]Web dependencies missing:[/red] pip install hammerio[web]\n{e}")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Web dashboard failed:[/red] {exc}")
        raise typer.Exit(1)


@app.command()
def watch(
    watch_root: str = typer.Option("./hammer-watch", "--watch-root", help="Root directory for watch folders"),
    compress_out: Optional[str] = typer.Option(None, "--compress-out", help="Output directory for compressed files"),
    decompress_out: Optional[str] = typer.Option(None, "--decompress-out", help="Output directory for decompressed files"),
    threshold_mb: int = typer.Option(100, "--threshold-mb", help="File size threshold (MB) for GPU vs CPU compression"),
    workers: int = typer.Option(0, "--workers", "-w", help="Parallel workers (0=from config)"),
    stable_wait: float = typer.Option(0.5, "--stable-wait", help="Seconds to wait for file stability check"),
) -> None:
    """Watch compress/ and decompress/ folders and auto-process files.

    Drop files into <watch-root>/compress/ to compress them, or into
    <watch-root>/decompress/ to decompress them. Results appear in
    compressed/ and decompressed/ respectively.
    """
    try:
        from hammerio.core.config import load_config
        from hammerio.watch import WatchDaemon

        cfg = load_config()

        effective_workers = workers if workers > 0 else cfg.workers

        daemon = WatchDaemon(
            watch_root=watch_root,
            compress_output=compress_out,
            decompress_output=decompress_out,
            gpu_threshold_mb=threshold_mb,
            workers=effective_workers,
            stable_wait=stable_wait,
            move_to_processed=cfg.get("watch", "move_to_processed", True),
        )

        console.print(Panel(
            f"[bold]Watch Root:[/bold] {Path(watch_root).resolve()}\n"
            f"[bold]Compress:[/bold]   drop files in compress/\n"
            f"[bold]Decompress:[/bold] drop files in decompress/\n"
            f"[bold]GPU threshold:[/bold] {threshold_mb} MB\n"
            f"[bold]Workers:[/bold] {effective_workers}\n"
            f"\n[dim]Press Ctrl+C to stop.[/dim]",
            title="[bold cyan]HammerIO Watch Daemon[/bold cyan]",
            border_style="cyan",
        ))

        daemon.start()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        console.print(f"[red]Watch failed:[/red] {exc}")
        raise typer.Exit(1)


def _show_hardware(as_json: bool) -> None:
    from hammerio.core.hardware import detect_hardware, format_hardware_report

    with console.status("[bold green]Detecting hardware..."):
        profile = detect_hardware()

    if as_json:
        data = {
            "platform": profile.platform_name,
            "architecture": profile.architecture,
            "cuda": {
                "available": profile.has_cuda,
                "version": profile.cuda_device.cuda_version if profile.cuda_device else None,
                "device": profile.cuda_device.name if profile.cuda_device else None,
                "memory_mb": profile.gpu_memory_mb,
            },
            "nvenc": {"available": profile.has_nvenc, "codecs": profile.nvenc.codecs},
            "nvdec": {"available": profile.nvdec.available, "codecs": profile.nvdec.codecs},
            "nvcomp": {"available": profile.has_nvcomp, "algorithms": profile.nvcomp.algorithms},
            "vpi": {"available": profile.has_vpi, "version": profile.vpi.version},
            "cpu_cores": profile.cpu_cores,
            "ram_mb": profile.total_ram_mb,
            "routes": profile.routing_summary(),
        }
        console.print_json(json.dumps(data, indent=2))
    else:
        console.print(Panel(
            format_hardware_report(profile),
            title="[bold cyan]HammerIO Hardware Profile[/bold cyan]",
            border_style="cyan",
        ))


def _show_routes(path: str) -> None:
    from hammerio.core.router import JobRouter

    router = JobRouter()
    console.print(Panel(
        router.explain_route(path),
        title="[bold cyan]Routing Analysis[/bold cyan]",
        border_style="cyan",
    ))


def _show_telemetry(as_json: bool) -> None:
    from hammerio.core.telemetry import TelemetryCollector

    collector = TelemetryCollector()
    snap = collector.get_snapshot()

    if as_json:
        console.print_json(json.dumps(snap.to_dict(), indent=2, default=str))
    else:
        console.print(Panel(
            collector.format_live_display(snap),
            title="[bold yellow]System Telemetry[/bold yellow]",
            border_style="yellow",
        ))


def _print_result(result: "JobResult") -> None:  # type: ignore[name-defined]
    table = Table(title="Compression Result")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Input", result.input_path)
    table.add_row("Output", result.output_path)
    table.add_row("Input Size", _human_size(result.input_size))
    table.add_row("Output Size", _human_size(result.output_size))
    table.add_row("Ratio", f"{result.compression_ratio:.2f}x")
    table.add_row("Savings", f"{result.savings_pct:.1f}%")
    table.add_row("Time", f"{result.elapsed_seconds:.2f}s")
    table.add_row("Throughput", f"{result.throughput_mbps:.1f} MB/s")
    table.add_row("Processor", result.processor_used)
    table.add_row("Algorithm", result.algorithm)
    table.add_row("Reason", result.routing_reason)

    if result.used_fallback:
        table.add_row("[yellow]Fallback[/yellow]", result.fallback_reason)

    console.print(table)


def _human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"


@app.command()
def version() -> None:
    """Show HammerIO version and system info."""
    try:
        from hammerio import __version__
        from hammerio.core.hardware import detect_hardware

        hw = detect_hardware()
        console.print(f"[bold cyan]HammerIO[/bold cyan] v{__version__}")
        console.print(f"  Platform: {hw.platform_name}")
        console.print(f"  CUDA: {hw.cuda_device.cuda_version if hw.cuda_device else 'N/A'}")
        console.print(f"  NVENC: {'Yes' if hw.has_nvenc else 'No'}")
        console.print(f"  Python: {sys.version.split()[0]}")
        console.print(f"  Copyright 2026 ResilientMind AI | Joseph C McGinty Jr")
    except Exception as exc:
        console.print(f"[red]Version check failed:[/red] {exc}")
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    one_gb: bool = typer.Option(False, "--1gb", help="Shortcut for 'benchmark --1gb'"),
    ten_gb: bool = typer.Option(False, "--10gb", help="Shortcut for 'benchmark --10gb'"),
    accept_license: bool = typer.Option(False, "--accept-license",
        help="Accept the license agreement non-interactively", hidden=True),
) -> None:
    """HammerIO — GPU where it matters. CPU where it doesn't."""
    from hammerio.core.license import (
        is_license_accepted, record_acceptance, require_license_acceptance,
    )

    # --accept-license flag for CI/scripted environments
    if accept_license:
        record_acceptance()
        console.print("[green]License accepted.[/green]")
        if ctx.invoked_subcommand is None and not one_gb and not ten_gb:
            raise typer.Exit()

    # First-run license check — every command requires acceptance
    if not is_license_accepted():
        require_license_acceptance()

    if one_gb or ten_gb:
        from hammerio.cli.main import _run_benchmark_sized
        _run_benchmark_sized(large=one_gb, huge=ten_gb)
        raise typer.Exit()


def _run_benchmark_sized(large: bool = False, huge: bool = False) -> None:
    """Run a sized benchmark directly (bypasses Typer option resolution)."""
    label = "10GB" if huge else "1GB"
    console.print(Panel(
        f"[bold]HammerIO Benchmark Suite ({label})[/bold]\nCompress \u2192 Decompress \u2192 Verify round-trip",
        border_style="yellow",
    ))
    try:
        import sys as _sys
        _project_root = str(Path(__file__).resolve().parent.parent.parent)
        if _project_root not in _sys.path:
            _sys.path.insert(0, _project_root)
        output = str(Path(_project_root) / "benchmarks/results/benchmark.json")
        from benchmarks.run_benchmarks import run_all_benchmarks
        run_all_benchmarks(quick=False, large=large, huge=huge, output_path=output)
        console.print(f"\n[green]Results saved to:[/green] {output}")
    except ImportError as ie:
        console.print(f"[red]Benchmark module not found:[/red] {ie}")
    except Exception as e:
        console.print(f"[red]Benchmark error:[/red] {e}")


if __name__ == "__main__":
    app()
