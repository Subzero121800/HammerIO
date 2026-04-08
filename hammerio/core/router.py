"""Smart job routing engine for HammerIO.

Routes compression/processing jobs to the optimal processor based on
hardware capabilities, file characteristics, and current system state.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from hammerio.core.hardware import HardwareProfile, detect_hardware
from hammerio.core.profiler import (
    BatchProfile,
    CompressionMode,
    CompressionRecommendation,
    FileCategory,
    FileProfile,
    profile_directory,
    profile_file,
    recommend_batch,
    recommend_compression,
)

logger = logging.getLogger("hammerio.router")


class JobStatus(Enum):
    PENDING = "pending"
    ROUTING = "routing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    FALLBACK = "fallback"


@dataclass
class JobResult:
    """Result of a completed compression/processing job."""

    input_path: str
    output_path: str
    input_size: int
    output_size: int
    compression_ratio: float
    elapsed_seconds: float
    throughput_mbps: float
    processor_used: str
    mode: CompressionMode
    algorithm: str
    routing_reason: str
    status: JobStatus
    error: str = ""
    used_fallback: bool = False
    fallback_reason: str = ""
    profiling_overhead_ms: float = 0.0

    @property
    def savings_pct(self) -> float:
        if self.input_size == 0:
            return 0.0
        return (1.0 - self.output_size / self.input_size) * 100


@dataclass
class Job:
    """A single processing job tracked by the router."""

    job_id: str
    input_path: Path
    output_path: Optional[Path]
    profile: Optional[FileProfile] = None
    recommendation: Optional[CompressionRecommendation] = None
    status: JobStatus = JobStatus.PENDING
    result: Optional[JobResult] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class JobRouter:
    """Routes jobs to optimal processors based on hardware and file analysis.

    The router:
    1. Detects available hardware (CUDA, NVENC, nvCOMP, VPI)
    2. Profiles input files (type, size, entropy)
    3. Recommends optimal processor with reasoning
    4. Executes with automatic fallback on GPU failure
    5. Reports detailed metrics
    """

    def __init__(
        self,
        hardware: Optional[HardwareProfile] = None,
        quality: str = "balanced",
        force_mode: Optional[str] = None,
        max_workers: int = 4,
    ) -> None:
        self.hardware = hardware or detect_hardware()
        self.quality = quality
        self.force_mode = force_mode
        self.max_workers = max_workers
        self._jobs: dict[str, Job] = {}
        self._job_counter = 0
        self._encoders: dict[CompressionMode, Any] = {}
        self._progress_callback: Optional[Callable[[str, float], None]] = None

        self._register_encoders()

    def _register_encoders(self) -> None:
        """Register compression backends. Focus: compress/decompress only."""
        from hammerio.encoders.bulk import BulkEncoder
        from hammerio.encoders.general import GeneralEncoder

        general = GeneralEncoder(self.hardware)
        self._encoders[CompressionMode.GPU_NVCOMP] = BulkEncoder(self.hardware)
        self._encoders[CompressionMode.CPU_ZSTD] = general
        self._encoders[CompressionMode.CPU_GZIP] = general
        self._encoders[CompressionMode.CPU_BZIP2] = general

        # Apple Silicon Accelerate framework (macOS)
        if self.hardware.apple_compression:
            try:
                from hammerio.encoders.apple import AppleEncoder
                self._encoders[CompressionMode.APPLE_LZFSE] = AppleEncoder(self.hardware)
                logger.info("Apple LZFSE encoder registered")
            except Exception:
                pass

        # Media encoders registered only if explicitly needed (advanced use)
        try:
            from hammerio.encoders.video import VideoEncoder
            self._encoders[CompressionMode.GPU_NVENC] = VideoEncoder(self.hardware)
        except ImportError:
            pass
        try:
            from hammerio.encoders.image import ImageEncoder
            self._encoders[CompressionMode.GPU_VPI] = ImageEncoder(self.hardware)
        except ImportError:
            pass
        try:
            from hammerio.encoders.audio import AudioEncoder
            self._encoders[CompressionMode.GPU_FFMPEG] = AudioEncoder(self.hardware)
        except ImportError:
            pass

    def set_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set a callback for progress updates: callback(job_id, progress_pct)."""
        self._progress_callback = callback

    def _next_job_id(self) -> str:
        self._job_counter += 1
        return f"job_{self._job_counter:04d}"

    def route(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
        mode: Optional[str] = None,
        algorithm: Optional[str] = None,
    ) -> Job:
        """Analyze input and create a routed job (does not execute).

        Args:
            input_path: File or directory to process
            output_path: Where to write output (auto-generated if None)
            mode: Force a specific mode ("gpu", "cpu", "auto")
            algorithm: Force a specific algorithm
        """
        route_start = time.time()
        input_path = Path(input_path).resolve()
        job_id = self._next_job_id()

        # --- File profiling with timing ---
        profile_start = time.time()
        if input_path.is_dir():
            # Only profile the first file for routing — scanning the entire
            # directory tree is prohibitively slow for large directories and
            # the result is only used to pick the compression mode anyway.
            first_file = next(
                (f for f in input_path.rglob("*") if f.is_file()), None,
            )
            if first_file is None:
                raise ValueError(f"Empty directory: {input_path}")
            fp = profile_file(first_file)
        elif input_path.is_file():
            fp = profile_file(input_path)
        else:
            raise FileNotFoundError(f"Input not found: {input_path}")
        profile_elapsed_ms = (time.time() - profile_start) * 1000
        logger.info(
            "File profiling took %.3f ms for %s",
            profile_elapsed_ms,
            input_path.name,
        )

        # --- Hardware-aware recommendation with timing ---
        recommend_start = time.time()
        effective_mode = mode or self.force_mode
        rec = self._get_recommendation(fp, effective_mode, algorithm)

        # Directories are always tar'd then compressed as a single archive.
        # Override passthrough/none since the tar itself is compressible even
        # if individual files inside are already compressed.
        if input_path.is_dir() and (rec.algorithm in ("none", "") or rec.mode.value == "passthrough"):
            rec.algorithm = algorithm or "zstd"
            if self.hardware.has_nvcomp and effective_mode != "cpu":
                rec.mode = CompressionMode.GPU_NVCOMP
                rec.reason = f"Directory archive → nvCOMP {rec.algorithm}"
            else:
                rec.mode = CompressionMode.CPU_ZSTD
                rec.reason = f"Directory archive → CPU {rec.algorithm}"
        recommend_elapsed_ms = (time.time() - recommend_start) * 1000
        logger.info(
            "Hardware detection / recommendation took %.3f ms",
            recommend_elapsed_ms,
        )

        # Determine output path
        if output_path is None:
            output_path = self._auto_output_path(input_path, rec)
        else:
            output_path = Path(output_path).resolve()
            # If -o points to an existing directory, place the output
            # file inside it (e.g. -o /dest/ → /dest/MyFolder.hammer)
            if output_path.is_dir():
                out_name = input_path.name + ".hammer"
                output_path = output_path / out_name

        route_elapsed_ms = (time.time() - route_start) * 1000

        job = Job(
            job_id=job_id,
            input_path=input_path,
            output_path=output_path,
            profile=fp,
            recommendation=rec,
            status=JobStatus.ROUTING,
        )
        self._jobs[job_id] = job

        logger.info(
            "Routed %s → %s (%s) reason: %s "
            "[profiling=%.1fms, recommend=%.1fms, total=%.1fms]",
            input_path.name,
            rec.mode.value,
            rec.algorithm,
            rec.reason,
            profile_elapsed_ms,
            recommend_elapsed_ms,
            route_elapsed_ms,
        )
        return job

    def _get_recommendation(
        self,
        profile: FileProfile,
        mode: Optional[str],
        algorithm: Optional[str],
    ) -> CompressionRecommendation:
        """Get compression recommendation, respecting forced modes."""
        hw = self.hardware

        if mode == "gpu" and not hw.has_cuda:
            logger.warning("GPU mode forced but no CUDA device — will attempt with fallback")

        # Treat GStreamer NVENC as equivalent to FFmpeg NVENC for routing.
        nvenc_available = hw.has_nvenc or hw.has_gstreamer_nvenc

        rec = recommend_compression(
            profile,
            gpu_available=hw.has_cuda,
            nvenc_available=nvenc_available,
            nvcomp_available=hw.has_nvcomp,
            vpi_available=hw.has_vpi,
            apple_available=hw.apple_compression,
            target_quality=self.quality,
        )

        # Override if mode forced
        if mode == "cpu":
            rec.mode = CompressionMode.CPU_ZSTD
            rec.algorithm = algorithm or "zstd"
            rec.gpu_preferred = False
            rec.reason = f"CPU mode forced → {rec.algorithm}"
        elif mode == "gpu":
            if profile.category == FileCategory.VIDEO and nvenc_available:
                rec.mode = CompressionMode.GPU_NVENC
                rec.reason = "GPU forced → NVENC video encoding"
            elif profile.category == FileCategory.IMAGE and (hw.has_vpi or hw.has_cuda):
                rec.mode = CompressionMode.GPU_VPI
                rec.reason = "GPU forced → VPI/CUDA image processing"
            elif hw.has_nvcomp:
                rec.mode = CompressionMode.GPU_NVCOMP
                rec.reason = "GPU forced → nvCOMP bulk compression"
            else:
                rec.reason = (
                    f"GPU forced but no GPU encoder for {profile.category.value} "
                    f"(nvCOMP: {'yes' if hw.has_nvcomp else 'no'}, "
                    f"NVENC: {'yes' if nvenc_available else 'no'}) "
                    f"→ {rec.mode.value} (CPU fallback)"
                )

        if algorithm:
            rec.algorithm = algorithm

        return rec

    def _auto_output_path(self, input_path: Path, rec: CompressionRecommendation) -> Path:
        """Generate output path based on input and compression mode/algorithm."""
        if rec.mode == CompressionMode.GPU_NVENC:
            # Video: change extension based on codec
            ext = ".mp4"
            if "hevc" in rec.algorithm:
                ext = ".mp4"  # HEVC in MP4 container
            return input_path.with_suffix(f".hammer{ext}")
        elif rec.mode == CompressionMode.GPU_NVCOMP:
            return input_path.with_suffix(input_path.suffix + ".hammer")

        # For CPU modes, use the actual algorithm to pick the right extension
        algo = rec.algorithm.lower()
        algo_ext_map = {
            "zstd": ".zst",
            "gzip": ".gz",
            "bzip2": ".bz2",
            "lz4": ".lz4",
            "zip": ".zip",
            "lzfse": ".lzfse",
            "zlib": ".zz",
            "lzma": ".lzma",
        }
        ext = algo_ext_map.get(algo)
        if ext:
            if input_path.is_dir():
                return input_path.parent / (input_path.name + ".tar" + ext)
            return input_path.with_suffix(input_path.suffix + ext)

        # Fallback for non-standard algorithms
        if rec.mode == CompressionMode.CPU_GZIP:
            return input_path.with_suffix(input_path.suffix + ".gz")
        elif rec.mode == CompressionMode.CPU_BZIP2:
            return input_path.with_suffix(input_path.suffix + ".bz2")
        elif rec.mode == CompressionMode.CPU_ZSTD:
            return input_path.with_suffix(input_path.suffix + ".zst")
        else:
            return input_path.with_suffix(input_path.suffix + ".hammer")

    def execute(self, job: Job) -> JobResult:
        """Execute a routed job synchronously."""
        job.status = JobStatus.PROCESSING
        job.started_at = time.time()

        # Calculate profiling overhead (time between job creation and execution start)
        profiling_overhead_ms = (job.started_at - job.created_at) * 1000

        try:
            result = self._run_encoder(job)
            result.profiling_overhead_ms = profiling_overhead_ms
            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = time.time()
            return result
        except Exception as e:
            logger.warning("Primary encoder failed: %s — attempting fallback", e)
            result = self._try_fallback(job, str(e))
            result.profiling_overhead_ms = profiling_overhead_ms
            return result

    def _run_encoder(self, job: Job) -> JobResult:
        """Run the recommended encoder for a job."""
        rec = job.recommendation
        if rec is None:
            raise ValueError("Job has no recommendation — call route() first")

        encoder = self._encoders.get(rec.mode)
        if encoder is None:
            raise ValueError(f"No encoder registered for mode {rec.mode}")

        start = time.time()
        output_path = encoder.process(
            input_path=job.input_path,
            output_path=job.output_path,
            algorithm=rec.algorithm,
            quality=self.quality,
            progress_callback=self._progress_callback,
            job_id=job.job_id,
        )
        elapsed = time.time() - start

        input_size = job.input_path.stat().st_size if job.input_path.is_file() else self._dir_size(job.input_path)
        output_size = Path(output_path).stat().st_size if Path(output_path).exists() else 0
        ratio = input_size / output_size if output_size > 0 else 0.0
        throughput = (input_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0

        return JobResult(
            input_path=str(job.input_path),
            output_path=str(output_path),
            input_size=input_size,
            output_size=output_size,
            compression_ratio=ratio,
            elapsed_seconds=elapsed,
            throughput_mbps=throughput,
            processor_used=rec.mode.value,
            mode=rec.mode,
            algorithm=rec.algorithm,
            routing_reason=rec.reason,
            status=JobStatus.COMPLETED,
        )

    def _try_fallback(self, job: Job, primary_error: str) -> JobResult:
        """Attempt fallback when primary encoder fails."""
        rec = job.recommendation
        if rec is None or rec.fallback_mode is None:
            job.status = JobStatus.FAILED
            result = JobResult(
                input_path=str(job.input_path),
                output_path="",
                input_size=0,
                output_size=0,
                compression_ratio=0,
                elapsed_seconds=0,
                throughput_mbps=0,
                processor_used="none",
                mode=rec.mode if rec else CompressionMode.CPU_ZSTD,
                algorithm="",
                routing_reason="",
                status=JobStatus.FAILED,
                error=primary_error,
            )
            job.result = result
            return result

        logger.info(
            "Falling back: %s → %s (%s)",
            rec.mode.value,
            rec.fallback_mode.value,
            rec.fallback_algorithm,
        )

        # Swap to fallback
        original_mode = rec.mode
        rec.mode = rec.fallback_mode
        rec.algorithm = rec.fallback_algorithm

        try:
            result = self._run_encoder(job)
            result.used_fallback = True
            result.fallback_reason = f"Primary ({original_mode.value}) failed: {primary_error}"
            result.status = JobStatus.COMPLETED
            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = time.time()
            return result
        except Exception as e2:
            job.status = JobStatus.FAILED
            result = JobResult(
                input_path=str(job.input_path),
                output_path="",
                input_size=0,
                output_size=0,
                compression_ratio=0,
                elapsed_seconds=0,
                throughput_mbps=0,
                processor_used="none",
                mode=rec.fallback_mode,
                algorithm=rec.fallback_algorithm,
                routing_reason="",
                status=JobStatus.FAILED,
                error=f"Primary: {primary_error}; Fallback: {e2}",
            )
            job.result = result
            return result

    async def execute_async(self, job: Job) -> JobResult:
        """Execute a job asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, job)

    async def execute_batch(
        self,
        input_path: str | Path,
        output_dir: Optional[str | Path] = None,
        mode: Optional[str] = None,
        workers: int = 4,
    ) -> list[JobResult]:
        """Process a directory of files concurrently.

        Displays a Rich progress bar per file (when Rich is available),
        handles errors per-file without stopping the batch, and reports
        overall batch statistics after completion.
        """
        input_path = Path(input_path).resolve()
        if not input_path.is_dir():
            raise ValueError(f"Not a directory: {input_path}")

        output_dir_path = Path(output_dir).resolve() if output_dir else input_path / "compressed"
        output_dir_path.mkdir(parents=True, exist_ok=True)

        batch = profile_directory(input_path, recursive=False)
        jobs: list[Job] = []
        for fp in batch.files:
            out = output_dir_path / (fp.path.stem + ".hammer" + fp.path.suffix)
            job = self.route(fp.path, out, mode=mode)
            jobs.append(job)

        # Try to use Rich for per-file progress bars
        progress_ctx: Any = None
        task_ids: dict[str, Any] = {}
        try:
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn  # type: ignore[import-untyped]

            progress_ctx = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("{task.percentage:>5.1f}%"),
                TimeRemainingColumn(),
            )
        except ImportError:
            progress_ctx = None

        sem = asyncio.Semaphore(workers)

        async def _run(j: Job, prog: Any) -> JobResult:
            async with sem:
                # Register a per-file progress callback when Rich is available
                if prog is not None:
                    tid = prog.add_task(j.input_path.name, total=100.0)
                    task_ids[j.job_id] = tid
                    original_cb = self._progress_callback

                    def _file_progress(job_id: str, pct: float) -> None:
                        if job_id in task_ids:
                            prog.update(task_ids[job_id], completed=pct)
                        if original_cb:
                            original_cb(job_id, pct)

                    self._progress_callback = _file_progress
                try:
                    result = await self.execute_async(j)
                    return result
                except Exception as exc:
                    logger.error("Batch job failed for %s: %s", j.input_path.name, exc)
                    return JobResult(
                        input_path=str(j.input_path),
                        output_path="",
                        input_size=j.input_path.stat().st_size if j.input_path.is_file() else 0,
                        output_size=0,
                        compression_ratio=0,
                        elapsed_seconds=0,
                        throughput_mbps=0,
                        processor_used="error",
                        mode=CompressionMode.CPU_ZSTD,
                        algorithm="",
                        routing_reason="",
                        status=JobStatus.FAILED,
                        error=str(exc),
                    )

        if progress_ctx is not None:
            with progress_ctx as prog:
                overall_tid = prog.add_task("Overall", total=len(jobs))
                results_raw = []
                for coro in asyncio.as_completed([_run(j, prog) for j in jobs]):
                    r = await coro
                    results_raw.append(r)
                    prog.update(overall_tid, advance=1)
        else:
            gathered = await asyncio.gather(
                *[_run(j, None) for j in jobs], return_exceptions=True
            )
            results_raw = []
            for r in gathered:
                if isinstance(r, Exception):
                    logger.error("Batch job failed: %s", r)
                    results_raw.append(JobResult(
                        input_path="", output_path="", input_size=0, output_size=0,
                        compression_ratio=0, elapsed_seconds=0, throughput_mbps=0,
                        processor_used="error", mode=CompressionMode.CPU_ZSTD,
                        algorithm="", routing_reason="", status=JobStatus.FAILED,
                        error=str(r),
                    ))
                else:
                    results_raw.append(r)

        # Compile batch statistics
        final: list[JobResult] = list(results_raw)
        succeeded = [r for r in final if r.status == JobStatus.COMPLETED]
        failed = [r for r in final if r.status == JobStatus.FAILED]
        total_input = sum(r.input_size for r in final)
        total_output = sum(r.output_size for r in succeeded)
        total_elapsed = sum(r.elapsed_seconds for r in succeeded)
        avg_ratio = (
            sum(r.compression_ratio for r in succeeded) / len(succeeded)
            if succeeded else 0.0
        )
        overall_throughput = (
            (total_input / (1024 * 1024)) / total_elapsed
            if total_elapsed > 0 else 0.0
        )

        stats_msg = (
            f"Batch complete: {len(succeeded)}/{len(final)} succeeded, "
            f"{len(failed)} failed | "
            f"Input: {total_input / (1024 * 1024):.1f} MB, "
            f"Output: {total_output / (1024 * 1024):.1f} MB | "
            f"Avg ratio: {avg_ratio:.2f}x | "
            f"Throughput: {overall_throughput:.1f} MB/s"
        )
        logger.info(stats_msg)

        # Store stats as a module-level attribute for programmatic access
        self._last_batch_stats = {
            "total_files": len(final),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "total_input_bytes": total_input,
            "total_output_bytes": total_output,
            "avg_compression_ratio": avg_ratio,
            "total_elapsed_seconds": total_elapsed,
            "overall_throughput_mbps": overall_throughput,
        }

        return final

    def explain_route(self, input_path: str | Path) -> str:
        """Explain routing decision for an input without executing."""
        input_path = Path(input_path).resolve()

        nvenc_ok = self.hardware.has_nvenc or self.hardware.has_gstreamer_nvenc

        if input_path.is_dir():
            batch = profile_directory(input_path)
            lines = [f"Directory: {input_path} ({batch.file_count} files, {batch.total_size_mb:.1f} MB)"]
            lines.append(f"Categories: {dict(batch.category_counts)}")
            for fp in batch.files[:10]:
                rec = recommend_compression(
                    fp,
                    gpu_available=self.hardware.has_cuda,
                    nvenc_available=nvenc_ok,
                    nvcomp_available=self.hardware.has_nvcomp,
                    vpi_available=self.hardware.has_vpi,
                    target_quality=self.quality,
                )
                lines.append(f"  {fp.path.name} ({fp.size_human}) → {rec.mode.value} ({rec.algorithm}): {rec.reason}")
            if batch.file_count > 10:
                lines.append(f"  ... and {batch.file_count - 10} more files")
            return "\n".join(lines)
        else:
            fp = profile_file(input_path)
            rec = recommend_compression(
                fp,
                gpu_available=self.hardware.has_cuda,
                nvenc_available=nvenc_ok,
                nvcomp_available=self.hardware.has_nvcomp,
                vpi_available=self.hardware.has_vpi,
                target_quality=self.quality,
            )
            lines = [
                f"File: {fp.path.name}",
                f"Size: {fp.size_human}",
                f"Category: {fp.category.value}",
                f"Entropy: {fp.estimated_entropy:.2f} bits/byte",
                f"Already compressed: {fp.is_already_compressed}",
                f"",
                f"Route: {rec.mode.value}",
                f"Algorithm: {rec.algorithm}",
                f"Reason: {rec.reason}",
                f"GPU preferred: {rec.gpu_preferred}",
            ]
            if rec.fallback_mode:
                lines.append(f"Fallback: {rec.fallback_mode.value} ({rec.fallback_algorithm})")
            if rec.estimated_ratio:
                lines.append(f"Est. ratio: {rec.estimated_ratio:.1f}x")
            return "\n".join(lines)

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return list(self._jobs.values())

    @staticmethod
    def _dir_size(path: Path) -> int:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
