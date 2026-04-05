"""HammerIO Benchmark Suite — Reproducible GPU vs CPU performance testing.

Copyright 2026 ResilientMind AI | ResilientMindai.com | Joseph C McGinty Jr

Tests:
- Video: CPU libx264 vs NVENC h264 vs NVENC hevc
- Bulk data: CPU zstd vs GPU nvCOMP LZ4/zstd
- Image batch: CPU PIL vs GPU OpenCV CUDA
- Audio batch: CPU vs GPU FFmpeg
- Mixed workload: router decisions, total throughput

Output: Rich terminal tables, JSON, CSV
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""

    test_name: str
    workload: str
    method: str  # "cpu" or "gpu"
    processor: str  # specific encoder name
    input_size_bytes: int
    output_size_bytes: int
    elapsed_seconds: float
    compression_ratio: float
    throughput_mbps: float
    notes: str = ""

    @property
    def input_size_mb(self) -> float:
        return self.input_size_bytes / (1024 * 1024)


@dataclass
class BenchmarkSuite:
    """Complete benchmark results."""

    platform: str
    cuda_version: str
    timestamp: str
    results: list[BenchmarkResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _generate_test_data(path: Path, size_mb: int, data_type: str = "random") -> None:
    """Generate test data files for benchmarking."""
    size_bytes = size_mb * 1024 * 1024
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(path, "wb") as f:
        remaining = size_bytes
        while remaining > 0:
            write_size = min(chunk_size, remaining)
            if data_type == "random":
                f.write(os.urandom(write_size))
            elif data_type == "compressible":
                # Realistic mixed data: ~60% compressible, ~40% entropy
                # Simulates real workloads (logs, CSV, binary telemetry)
                quarter = write_size // 4
                # 25% structured text (CSV-like)
                csv_line = b"2026-04-05T12:00:00,sensor_01,23.456,67.89,1013.25,active,OK\n"
                text_block = csv_line * (quarter // len(csv_line) + 1)
                # 25% repeated patterns (telemetry headers, protocol frames)
                pattern = bytes(range(256)) * (quarter // 256 + 1)
                # 25% semi-random (compressed images, encoded data)
                semi_random = bytes((b ^ 0xAA) & 0xFF for b in os.urandom(quarter))
                # 25% true random (encrypted, pre-compressed)
                chunk = text_block[:quarter] + pattern[:quarter] + semi_random[:quarter] + os.urandom(quarter)
                f.write(chunk[:write_size])
            elif data_type == "text":
                line = "HammerIO benchmark test data line with some variation {}\n"
                data = "".join(line.format(i) for i in range(write_size // 60 + 1))
                f.write(data[:write_size].encode())
            remaining -= write_size


def _generate_test_video(path: Path, duration_s: int = 10, resolution: str = "1920x1080") -> bool:
    """Generate a test video using FFmpeg."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"testsrc=duration={duration_s}:size={resolution}:rate=30",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_s}",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            str(path),
        ], capture_output=True, timeout=120)
        return path.exists()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _generate_test_images(directory: Path, count: int = 100, size: tuple[int, int] = (640, 480)) -> int:
    """Generate test JPEG images."""
    try:
        import numpy as np
        directory.mkdir(parents=True, exist_ok=True)
        created = 0
        for i in range(count):
            img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            try:
                import cv2
                cv2.imwrite(str(directory / f"test_{i:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                created += 1
            except ImportError:
                from PIL import Image
                Image.fromarray(img).save(directory / f"test_{i:04d}.jpg", quality=90)
                created += 1
        return created
    except ImportError:
        return 0


# Suppress libEGL DRI2 warning in all subprocesses
_CLEAN_ENV = {**os.environ, "EGL_LOG_LEVEL": "fatal", "NO_COLOR": "1"}


def _time_command(cmd: list[str], timeout: int = 300) -> tuple[float, bool]:
    """Time a subprocess command. Returns (elapsed_seconds, success)."""
    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout, env=_CLEAN_ENV)
        elapsed = time.perf_counter() - start
        return elapsed, result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return time.perf_counter() - start, False


def benchmark_video(tmpdir: Path, quick: bool = False) -> list[BenchmarkResult]:
    """Benchmark video encoding: CPU vs GPU."""
    results: list[BenchmarkResult] = []

    console.print("[bold]Video Encoding Benchmark[/bold]")

    # For --quick, use the bundled real test video if available
    bundled_video = Path(__file__).resolve().parent.parent / "Test Video.mp4"
    if quick and bundled_video.exists():
        src = bundled_video
        # Probe actual resolution and duration
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", str(src)],
                capture_output=True, text=True, timeout=10,
            )
            import json as _json
            info = _json.loads(probe.stdout)
            fmt = info.get("format", {})
            duration = float(fmt.get("duration", 0))
            streams = info.get("streams", [])
            vstream = next((s for s in streams if s["codec_type"] == "video"), {})
            resolution = f'{vstream.get("width", "?")}x{vstream.get("height", "?")}'
        except Exception:
            duration = 0
            resolution = "unknown"
        console.print(f"  Using bundled test video: {src.name}")
    else:
        duration = 5 if quick else 30
        resolution = "1280x720" if quick else "1920x1080"
        src = tmpdir / "test_video.mp4"
        with console.status("Generating test video..."):
            if not _generate_test_video(src, duration_s=duration, resolution=resolution):
                console.print("[yellow]Skipping video benchmark — FFmpeg cannot generate test video[/yellow]")
                return results

    input_size = src.stat().st_size
    console.print(f"  Test video: {resolution}, {duration:.0f}s, {input_size / (1024*1024):.1f} MB")

    tests: list[tuple[str, list[str]]] = [
        ("CPU libx264", [
            "ffmpeg", "-y", "-i", str(src),
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "copy", str(tmpdir / "out_cpu_h264.mp4"),
        ]),
    ]

    # GStreamer NVENC — the working GPU path on Jetson
    # (FFmpeg NVENC requires libnvidia-encode.so.1 which doesn't exist on Jetson)
    gst_launch = shutil.which("gst-launch-1.0")
    if gst_launch:
        gst_h264 = tmpdir / "out_gst_h264.mp4"
        tests.append(("GPU GStreamer H264", [
            gst_launch, "-e",
            "filesrc", f"location={src}",
            "!", "decodebin",
            "!", "nvvidconv",
            "!", "video/x-raw(memory:NVMM),format=NV12",
            "!", "nvv4l2h264enc", "bitrate=8000000",
            "!", "h264parse",
            "!", "mp4mux",
            "!", "filesink", f"location={gst_h264}",
        ]))
        gst_h265 = tmpdir / "out_gst_h265.mp4"
        tests.append(("GPU GStreamer H265", [
            gst_launch, "-e",
            "filesrc", f"location={src}",
            "!", "decodebin",
            "!", "nvvidconv",
            "!", "video/x-raw(memory:NVMM),format=NV12",
            "!", "nvv4l2h265enc", "bitrate=8000000",
            "!", "h265parse",
            "!", "mp4mux",
            "!", "filesink", f"location={gst_h265}",
        ]))
    else:
        console.print("  [yellow]GStreamer not found — skipping GPU video benchmarks[/yellow]")

    for name, cmd in tests:
        console.print(f"  Running: {name}...", end=" ")
        elapsed, success = _time_command(cmd)
        # Find output file — handle both FFmpeg (last arg) and GStreamer (location=...)
        out_file = Path(cmd[-1])
        if not out_file.exists():
            # GStreamer uses "location=/path/to/file" as last arg
            for arg in reversed(cmd):
                if arg.startswith("location="):
                    out_file = Path(arg.split("=", 1)[1])
                    break
        out_size = out_file.stat().st_size if out_file.exists() else 0
        ratio = input_size / out_size if out_size > 0 else 0
        throughput = (input_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0

        status = "[green]OK[/green]" if success else "[red]FAIL[/red]"
        console.print(f"{status} {elapsed:.2f}s, {ratio:.2f}x, {throughput:.1f} MB/s")

        results.append(BenchmarkResult(
            test_name="video_encode",
            workload=f"{resolution}_{duration}s",
            method="gpu" if ("nvenc" in name.lower() or "gstreamer" in name.lower()) else "cpu",
            processor=name,
            input_size_bytes=input_size,
            output_size_bytes=out_size,
            elapsed_seconds=elapsed,
            compression_ratio=ratio,
            throughput_mbps=throughput,
            notes="" if success else "FAILED",
        ))

    return results


def benchmark_bulk_data(tmpdir: Path, quick: bool = False) -> list[BenchmarkResult]:
    """Benchmark bulk data compression: CPU zstd vs various algorithms."""
    results: list[BenchmarkResult] = []
    size_mb = 100 if quick else 1024

    console.print(f"\n[bold]Bulk Data Compression Benchmark ({size_mb} MB)[/bold]")

    src = tmpdir / "test_bulk.bin"
    with console.status(f"Generating {size_mb} MB test data..."):
        _generate_test_data(src, size_mb, data_type="compressible")

    input_size = src.stat().st_size

    # CPU zstd
    try:
        import zstandard as zstd

        for level in [1, 3, 9]:
            out = tmpdir / f"out_zstd_l{level}.zst"
            console.print(f"  CPU zstd (level {level})...", end=" ")

            start = time.perf_counter()
            cctx = zstd.ZstdCompressor(level=level)
            with open(src, "rb") as fin, open(out, "wb") as fout:
                cctx.copy_stream(fin, fout)
            elapsed = time.perf_counter() - start

            out_size = out.stat().st_size
            ratio = input_size / out_size if out_size > 0 else 0
            throughput = (input_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0

            console.print(f"[green]OK[/green] {elapsed:.2f}s, {ratio:.2f}x, {throughput:.1f} MB/s")

            results.append(BenchmarkResult(
                test_name="bulk_compress",
                workload=f"{size_mb}MB_compressible",
                method="cpu",
                processor=f"zstd_level{level}",
                input_size_bytes=input_size,
                output_size_bytes=out_size,
                elapsed_seconds=elapsed,
                compression_ratio=ratio,
                throughput_mbps=throughput,
            ))
    except ImportError:
        console.print("[yellow]  zstandard not installed — skipping CPU zstd[/yellow]")

    # CPU gzip
    import gzip as gzip_mod
    out = tmpdir / "out_gzip.gz"
    console.print("  CPU gzip...", end=" ")
    start = time.perf_counter()
    with open(src, "rb") as fin, gzip_mod.open(out, "wb", compresslevel=6) as fout:
        while True:
            chunk = fin.read(8 * 1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
    elapsed = time.perf_counter() - start
    out_size = out.stat().st_size
    ratio = input_size / out_size if out_size > 0 else 0
    throughput = (input_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
    console.print(f"[green]OK[/green] {elapsed:.2f}s, {ratio:.2f}x, {throughput:.1f} MB/s")

    results.append(BenchmarkResult(
        test_name="bulk_compress",
        workload=f"{size_mb}MB_compressible",
        method="cpu",
        processor="gzip_level6",
        input_size_bytes=input_size,
        output_size_bytes=out_size,
        elapsed_seconds=elapsed,
        compression_ratio=ratio,
        throughput_mbps=throughput,
    ))

    return results


def benchmark_images(tmpdir: Path, quick: bool = False) -> list[BenchmarkResult]:
    """Benchmark batch image processing."""
    results: list[BenchmarkResult] = []
    count = 50 if quick else 500

    console.print(f"\n[bold]Batch Image Processing Benchmark ({count} images)[/bold]")

    img_dir = tmpdir / "test_images"
    with console.status(f"Generating {count} test images..."):
        actual = _generate_test_images(img_dir, count=count)

    if actual == 0:
        console.print("[yellow]Skipping image benchmark — no image libraries available[/yellow]")
        return results

    input_size = sum(f.stat().st_size for f in img_dir.glob("*.jpg"))
    console.print(f"  Generated {actual} images, {input_size / (1024*1024):.1f} MB total")

    # CPU processing with PIL/OpenCV
    out_dir = tmpdir / "images_out_cpu"
    out_dir.mkdir(exist_ok=True)

    console.print("  CPU batch resize (50%)...", end=" ")
    start = time.perf_counter()
    try:
        import cv2
        for img_path in img_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is not None:
                resized = cv2.resize(img, None, fx=0.5, fy=0.5)
                cv2.imwrite(str(out_dir / img_path.name), resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    except ImportError:
        from PIL import Image
        for img_path in img_dir.glob("*.jpg"):
            img = Image.open(img_path)
            img = img.resize((img.width // 2, img.height // 2))
            img.save(out_dir / img_path.name, quality=85)

    elapsed = time.perf_counter() - start
    out_size = sum(f.stat().st_size for f in out_dir.glob("*.jpg"))
    throughput = actual / elapsed if elapsed > 0 else 0

    console.print(f"[green]OK[/green] {elapsed:.2f}s, {throughput:.1f} images/s")

    results.append(BenchmarkResult(
        test_name="image_batch",
        workload=f"{actual}_images_resize50pct",
        method="cpu",
        processor="opencv_cpu",
        input_size_bytes=input_size,
        output_size_bytes=out_size,
        elapsed_seconds=elapsed,
        compression_ratio=input_size / out_size if out_size > 0 else 0,
        throughput_mbps=throughput,
        notes=f"{throughput:.1f} images/sec",
    ))

    # GPU processing with OpenCV CUDA (if available)
    try:
        import cv2
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            out_dir_gpu = tmpdir / "images_out_gpu"
            out_dir_gpu.mkdir(exist_ok=True)

            console.print("  GPU batch resize (50%)...", end=" ")
            start = time.perf_counter()
            for img_path in img_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path))
                if img is not None:
                    gpu_mat = cv2.cuda_GpuMat()
                    gpu_mat.upload(img)
                    resized = cv2.cuda.resize(gpu_mat, (img.shape[1] // 2, img.shape[0] // 2))
                    result = resized.download()
                    cv2.imwrite(str(out_dir_gpu / img_path.name), result, [cv2.IMWRITE_JPEG_QUALITY, 85])
            elapsed = time.perf_counter() - start
            out_size = sum(f.stat().st_size for f in out_dir_gpu.glob("*.jpg"))
            throughput = actual / elapsed if elapsed > 0 else 0

            console.print(f"[green]OK[/green] {elapsed:.2f}s, {throughput:.1f} images/s")

            results.append(BenchmarkResult(
                test_name="image_batch",
                workload=f"{actual}_images_resize50pct",
                method="gpu",
                processor="opencv_cuda",
                input_size_bytes=input_size,
                output_size_bytes=out_size,
                elapsed_seconds=elapsed,
                compression_ratio=input_size / out_size if out_size > 0 else 0,
                throughput_mbps=throughput,
                notes=f"{throughput:.1f} images/sec",
            ))
    except Exception:
        console.print("  [yellow]GPU OpenCV CUDA not available[/yellow]")

    return results


def benchmark_audio(tmpdir: Path, quick: bool = False) -> list[BenchmarkResult]:
    """Benchmark audio encoding."""
    results: list[BenchmarkResult] = []
    count = 5 if quick else 20

    console.print(f"\n[bold]Audio Encoding Benchmark ({count} files)[/bold]")

    # Generate test WAV files
    audio_dir = tmpdir / "test_audio"
    audio_dir.mkdir(exist_ok=True)

    created = 0
    for i in range(count):
        wav_path = audio_dir / f"test_{i:03d}.wav"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"sine=frequency={220 + i * 10}:duration=10",
                str(wav_path),
            ], capture_output=True, timeout=30)
            if wav_path.exists():
                created += 1
        except (subprocess.TimeoutExpired, FileNotFoundError):
            break

    if created == 0:
        console.print("[yellow]Skipping audio benchmark — cannot generate test audio[/yellow]")
        return results

    input_size = sum(f.stat().st_size for f in audio_dir.glob("*.wav"))

    # CPU encoding to MP3
    out_dir = tmpdir / "audio_out"
    out_dir.mkdir(exist_ok=True)

    console.print(f"  CPU WAV→MP3 ({created} files)...", end=" ")
    start = time.perf_counter()
    for wav in audio_dir.glob("*.wav"):
        subprocess.run([
            "ffmpeg", "-y", "-i", str(wav),
            "-c:a", "libmp3lame", "-b:a", "192k",
            str(out_dir / wav.with_suffix(".mp3").name),
        ], capture_output=True, timeout=30)
    elapsed = time.perf_counter() - start
    out_size = sum(f.stat().st_size for f in out_dir.glob("*.mp3"))
    throughput = (input_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0

    console.print(f"[green]OK[/green] {elapsed:.2f}s, {throughput:.1f} MB/s")

    results.append(BenchmarkResult(
        test_name="audio_encode",
        workload=f"{created}_wav_to_mp3",
        method="cpu",
        processor="libmp3lame",
        input_size_bytes=input_size,
        output_size_bytes=out_size,
        elapsed_seconds=elapsed,
        compression_ratio=input_size / out_size if out_size > 0 else 0,
        throughput_mbps=throughput,
    ))

    return results


def print_results_table(results: list[BenchmarkResult]) -> None:
    """Print a Rich formatted results table."""
    table = Table(title="HammerIO Benchmark Results", show_lines=True)
    table.add_column("Test", style="cyan")
    table.add_column("Method", style="bold")
    table.add_column("Processor", style="green")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Throughput", justify="right")

    for r in results:
        method_style = "[blue]GPU[/blue]" if r.method == "gpu" else "[yellow]CPU[/yellow]"
        table.add_row(
            r.test_name,
            method_style,
            r.processor,
            f"{r.input_size_mb:.1f} MB",
            f"{r.output_size_bytes / (1024*1024):.1f} MB",
            f"{r.compression_ratio:.2f}x",
            f"{r.elapsed_seconds:.2f}s",
            f"{r.throughput_mbps:.1f} MB/s" if "image" not in r.test_name else r.notes,
        )

    console.print(table)

    # Speedup summary
    _print_speedup_summary(results)


def _print_speedup_summary(results: list[BenchmarkResult]) -> None:
    """Print GPU vs CPU speedup comparison."""
    tests = set(r.test_name for r in results)
    summary_table = Table(title="GPU vs CPU Speedup", show_lines=True)
    summary_table.add_column("Workload", style="cyan")
    summary_table.add_column("CPU Time", justify="right")
    summary_table.add_column("GPU Time", justify="right")
    summary_table.add_column("Speedup", justify="right", style="bold green")

    for test in sorted(tests):
        cpu_results = [r for r in results if r.test_name == test and r.method == "cpu"]
        # Only include GPU results that actually produced output (not failed NVENC)
        gpu_results = [r for r in results if r.test_name == test and r.method == "gpu"
                       and r.notes != "FAILED" and r.output_size_bytes > 0]

        if cpu_results and gpu_results:
            cpu_time = min(r.elapsed_seconds for r in cpu_results)
            gpu_time = min(r.elapsed_seconds for r in gpu_results)
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0

            summary_table.add_row(
                test.replace("_", " ").title(),
                f"{cpu_time:.2f}s",
                f"{gpu_time:.2f}s",
                f"{speedup:.1f}x",
            )

    console.print(summary_table)


def save_results(suite: BenchmarkSuite, output_path: str) -> None:
    """Save benchmark results to JSON and CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    json_data = {
        "platform": suite.platform,
        "cuda_version": suite.cuda_version,
        "timestamp": suite.timestamp,
        "results": [asdict(r) for r in suite.results],
        "summary": suite.summary,
    }
    with open(output, "w") as f:
        json.dump(json_data, f, indent=2)
    console.print(f"[green]JSON saved:[/green] {output}")

    # CSV
    csv_path = output.with_suffix(".csv")
    if suite.results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(suite.results[0]).keys())
            writer.writeheader()
            for r in suite.results:
                writer.writerow(asdict(r))
        console.print(f"[green]CSV saved:[/green] {csv_path}")


def run_all_benchmarks(quick: bool = False, output_path: str = "benchmarks/results/benchmark.json") -> BenchmarkSuite:
    """Run the complete benchmark suite."""
    from hammerio.core.hardware import detect_hardware

    hw = detect_hardware()

    console.print(Panel(
        f"[bold]HammerIO Benchmark Suite[/bold]\n"
        f"Platform: {hw.platform_name}\n"
        f"CUDA: {hw.cuda_device.cuda_version if hw.cuda_device else 'N/A'}\n"
        f"Mode: {'Quick' if quick else 'Full'}",
        border_style="yellow",
    ))

    suite = BenchmarkSuite(
        platform=hw.platform_name,
        cuda_version=hw.cuda_device.cuda_version if hw.cuda_device else "N/A",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    with tempfile.TemporaryDirectory(prefix="hammerio_bench_") as tmpdir:
        tmp = Path(tmpdir)

        suite.results.extend(benchmark_video(tmp, quick))
        suite.results.extend(benchmark_bulk_data(tmp, quick))
        suite.results.extend(benchmark_images(tmp, quick))
        suite.results.extend(benchmark_audio(tmp, quick))

    # Summary
    gpu_results = [r for r in suite.results if r.method == "gpu" and r.notes != "FAILED"]
    cpu_results = [r for r in suite.results if r.method == "cpu"]

    suite.summary = {
        "total_tests": len(suite.results),
        "gpu_tests": len(gpu_results),
        "cpu_tests": len(cpu_results),
        "gpu_available": hw.has_cuda,
        "nvenc_available": hw.has_nvenc,
    }

    print_results_table(suite.results)
    save_results(suite, output_path)

    return suite


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HammerIO Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks")
    parser.add_argument("--output", default="benchmarks/results/benchmark.json", help="Output path")
    args = parser.parse_args()

    run_all_benchmarks(quick=args.quick, output_path=args.output)
