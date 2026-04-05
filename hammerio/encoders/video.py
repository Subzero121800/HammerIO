"""Video encoder for HammerIO with Jetson GStreamer NVENC support.

Encoding priority:
1. GStreamer V4L2 NVENC (Jetson preferred -- works with distro packages)
2. FFmpeg NVENC (desktop GPU with NVIDIA-compiled FFmpeg)
3. FFmpeg libx264/libx265 CPU fallback

The GStreamer path uses ``nvv4l2h264enc`` / ``nvv4l2h265enc`` which
leverage the Jetson hardware encoder through V4L2, bypassing the
FFmpeg NVENC limitation on distro builds.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from hammerio.core.hardware import HardwareProfile

logger = logging.getLogger("hammerio.encoders.video")

# Containers we accept as video input.
SUPPORTED_EXTENSIONS: set[str] = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

# ---------------------------------------------------------------------------
# Quality presets
# ---------------------------------------------------------------------------
# Each preset maps to parameters for *both* GPU (NVENC) and CPU (libx264/x265)
# paths.  The GPU path uses constant-quality rate-control (-cq / -qp) because
# NVENC does not support CRF in the traditional x264 sense.  The CPU path
# uses CRF.
#
# Structure:  preset_name -> {
#     "gpu_preset":  NVENC -preset value,
#     "gpu_rc":      rate-control mode,
#     "gpu_cq":      constant-quality value (lower = better),
#     "cpu_preset":  libx264/x265 -preset value,
#     "cpu_crf":     CRF value,
# }

_QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "gpu_preset": "p1",
        "gpu_rc": "constqp",
        "gpu_cq": 28,
        "cpu_preset": "ultrafast",
        "cpu_crf": 28,
    },
    "balanced": {
        "gpu_preset": "p4",
        "gpu_rc": "constqp",
        "gpu_cq": 23,
        "cpu_preset": "medium",
        "cpu_crf": 23,
    },
    "quality": {
        "gpu_preset": "p7",
        "gpu_rc": "constqp",
        "gpu_cq": 18,
        "cpu_preset": "slow",
        "cpu_crf": 18,
    },
    "lossless": {
        "gpu_preset": "p7",
        "gpu_rc": "lossless",
        "gpu_cq": 0,
        "cpu_preset": "veryslow",
        "cpu_crf": 0,
    },
}


class VideoEncoder:
    """GPU/CPU video encoder with GStreamer NVENC support for Jetson.

    Encoding path selection:
    - On Jetson with GStreamer NVENC: uses ``nvv4l2h264enc`` / ``nvv4l2h265enc``
    - On desktop with FFmpeg NVENC: uses ``h264_nvenc`` / ``hevc_nvenc``
    - Fallback: CPU-based ``libx264`` / ``libx265``

    Args:
        hardware: A ``HardwareProfile`` instance from hardware detection.
    """

    def __init__(self, hardware: HardwareProfile) -> None:
        self._hw = hardware
        self._ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        self._ffprobe = shutil.which("ffprobe") or "ffprobe"
        self._gst_launch = shutil.which("gst-launch-1.0") or ""
        self._use_gstreamer = (
            hasattr(hardware, "gstreamer_nvenc")
            and hardware.gstreamer_nvenc.available
            and bool(self._gst_launch)
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_quality_presets() -> dict[str, dict[str, Any]]:
        """Return a copy of the quality-preset to FFmpeg parameter mapping."""
        return {k: dict(v) for k, v in _QUALITY_PRESETS.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
        algorithm: str,
        quality: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Encode a video file.

        Args:
            input_path: Source video file.
            output_path: Destination file (container inferred from suffix).
            algorithm: Codec hint, e.g. ``"h264"``, ``"hevc"``, ``"h264_nvenc"``.
            quality: One of ``"fast"``, ``"balanced"``, ``"quality"``,
                ``"lossless"``.
            progress_callback: Optional ``callback(job_id, pct)`` called with
                a percentage ``0.0 .. 100.0`` as encoding progresses.
            job_id: Identifier forwarded to *progress_callback*.

        Returns:
            The resolved *output_path* as a string.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            ValueError: If the file extension is unsupported or the quality
                preset is unknown.
            RuntimeError: If FFmpeg exits with a non-zero return code.
        """
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        self._validate_input(input_path)

        if quality not in _QUALITY_PRESETS:
            raise ValueError(
                f"Unknown quality preset '{quality}'. "
                f"Choose from: {', '.join(_QUALITY_PRESETS)}"
            )

        preset = _QUALITY_PRESETS[quality]
        use_gpu = self._should_use_gpu(algorithm)

        # Try GStreamer NVENC on Jetson (preferred hardware path)
        if self._use_gstreamer and not use_gpu:
            # GStreamer available but FFmpeg NVENC isn't — try GStreamer
            try:
                return self._encode_gstreamer(input_path, output_path, algorithm, quality)
            except Exception as e:
                logger.warning("GStreamer encoding failed: %s — falling back to FFmpeg", e)

        codec = self._resolve_codec(algorithm, use_gpu)

        cmd = self._build_command(
            input_path, output_path, codec, preset, use_gpu,
        )

        logger.info(
            "Encoding %s -> %s  codec=%s  quality=%s  gpu=%s",
            input_path.name, output_path.name, codec, quality, use_gpu,
        )
        logger.debug("FFmpeg command: %s", " ".join(cmd))

        # Ensure the output directory exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        duration = self._probe_duration(input_path)
        try:
            self._run_ffmpeg(cmd, duration, progress_callback, job_id)
        except FileNotFoundError:
            raise FileNotFoundError(
                "ffmpeg not found on PATH — install FFmpeg to use VideoEncoder"
            )

        if not output_path.exists():
            raise RuntimeError(
                f"FFmpeg completed but output file was not created: {output_path}"
            )

        logger.info(
            "Encoding complete: %s (%.2f MB)",
            output_path.name,
            output_path.stat().st_size / (1024 * 1024),
        )
        return str(output_path)

    def probe(self, path: str | Path) -> dict[str, Any]:
        """Return metadata for a media file using ffprobe.

        Args:
            path: Path to the media file.

        Returns:
            A dict with keys from ffprobe's JSON output (``format``,
            ``streams``, etc.).

        Raises:
            RuntimeError: If ffprobe fails.
        """
        path = Path(path).resolve()
        cmd = [
            self._ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed (rc={result.returncode}): {result.stderr.strip()}"
            )
        try:
            return json.loads(result.stdout)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse ffprobe output: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_gstreamer(
        self,
        input_path: Path,
        output_path: Path,
        algorithm: str,
        quality: str,
    ) -> str:
        """Encode using GStreamer NVENC (Jetson V4L2 hardware encoder).

        This is the preferred hardware encoding path on Jetson devices where
        FFmpeg's NVENC support requires custom compilation.  Uses the
        ``nvv4l2h264enc`` / ``nvv4l2h265enc`` elements which go through
        the V4L2 interface to the Jetson hardware encoder.

        The pipeline: filesrc -> decodebin -> nvvidconv (to NVMM) ->
        nvv4l2h264enc -> h264parse -> mux -> filesink
        """
        algo = algorithm.lower()
        gst_cap = self._hw.gstreamer_nvenc
        is_hevc = "hevc" in algo or "h265" in algo or "x265" in algo

        if is_hevc and gst_cap.has_h265:
            encoder = "nvv4l2h265enc"
            parser = "h265parse"
        elif not is_hevc and gst_cap.has_h264:
            encoder = "nvv4l2h264enc"
            parser = "h264parse"
        else:
            raise RuntimeError(
                f"GStreamer encoder not available for {'h265' if is_hevc else 'h264'}"
            )

        # Map quality to bitrate (GStreamer V4L2 NVENC uses bitrate, not CRF).
        # Values in bits/sec.
        _GST_BITRATE: dict[str, int] = {
            "fast": 4_000_000,
            "balanced": 8_000_000,
            "quality": 15_000_000,
            "lossless": 30_000_000,
        }
        bitrate = _GST_BITRATE.get(quality, 8_000_000)

        # Determine output muxer from extension.
        suffix = output_path.suffix.lower()
        if suffix in (".mkv", ".webm"):
            muxer = "matroskamux"
        elif suffix == ".mov":
            muxer = "qtmux"
        else:
            muxer = "mp4mux"

        # Build the GStreamer pipeline as a list of arguments.
        # Video-only pipeline (audio passthrough is handled by a second
        # branch when the input has audio).
        #
        # For robustness we use a video-only pipeline first, then try
        # with audio if the input has it.  This avoids failures on
        # audio-less inputs.
        pipeline_str = self._build_gstreamer_pipeline(
            input_path, output_path, encoder, parser, muxer, bitrate,
        )

        logger.info(
            "GStreamer encode: %s -> %s (%s, %d bps)",
            input_path.name, output_path.name, encoder, bitrate,
        )
        logger.debug("GStreamer pipeline: %s", pipeline_str)

        # Ensure the output directory exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [self._gst_launch, "-e"] + pipeline_str.split(),
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            stderr_snippet = result.stderr[-500:] if result.stderr else "(no stderr)"
            raise RuntimeError(
                f"GStreamer failed (rc={result.returncode}): {stderr_snippet}"
            )

        if not output_path.exists():
            raise RuntimeError(
                "GStreamer completed but output file was not created"
            )

        logger.info(
            "GStreamer encode complete: %s (%.2f MB)",
            output_path.name,
            output_path.stat().st_size / (1024 * 1024),
        )
        return str(output_path)

    @staticmethod
    def _build_gstreamer_pipeline(
        input_path: Path,
        output_path: Path,
        encoder: str,
        parser: str,
        muxer: str,
        bitrate: int,
    ) -> str:
        """Build a GStreamer pipeline string for Jetson NVENC transcoding.

        Pipeline flow:
          filesrc -> decodebin -> nvvidconv -> NVMM NV12 -> encoder -> parser -> mux -> filesink

        Uses ``nvvidconv`` to convert decoded frames into NVMM memory which
        is required by the V4L2 hardware encoder on Jetson.
        """
        # Video-only pipeline (simpler, more reliable).
        # Audio could be added with a decodebin branch but that adds
        # complexity; for now we transcode video and drop audio.
        # A future enhancement can add audio passthrough.
        # Quote file paths to handle spaces and special characters
        in_loc = str(input_path).replace('"', '\\"')
        out_loc = str(output_path).replace('"', '\\"')
        pipeline = (
            f'filesrc location="{in_loc}" '
            f"! decodebin "
            f"! nvvidconv "
            f"! video/x-raw(memory:NVMM),format=NV12 "
            f"! {encoder} bitrate={bitrate} "
            f"! {parser} "
            f"! {muxer} "
            f'! filesink location="{out_loc}"'
        )
        return pipeline

    def _validate_input(self, path: Path) -> None:
        """Raise on missing file or unsupported extension."""
        if not path.is_file():
            raise FileNotFoundError(f"Input video not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    def _should_use_gpu(self, algorithm: str) -> bool:
        """Decide whether to use GPU encoding for the requested algorithm."""
        algo_lower = algorithm.lower()

        # Explicit NVENC request.
        if "nvenc" in algo_lower:
            if self._hw.has_nvenc:
                return True
            logger.warning(
                "NVENC explicitly requested but unavailable; falling back to CPU"
            )
            return False

        # Generic codec name — prefer GPU when available.
        if "h264" in algo_lower or "x264" in algo_lower:
            return self._hw.has_nvenc and "h264" in self._hw.nvenc.codecs
        if "hevc" in algo_lower or "h265" in algo_lower or "x265" in algo_lower:
            return self._hw.has_nvenc and "hevc" in self._hw.nvenc.codecs
        if "av1" in algo_lower:
            return self._hw.has_nvenc and "av1" in self._hw.nvenc.codecs

        # Default: use GPU if any NVENC codec is present.
        return self._hw.has_nvenc

    def _resolve_codec(self, algorithm: str, use_gpu: bool) -> str:
        """Map an algorithm hint to a concrete FFmpeg encoder name."""
        algo_lower = algorithm.lower()

        # If the caller already passed a full encoder name, honour it.
        if algo_lower in {
            "h264_nvenc", "hevc_nvenc", "av1_nvenc",
            "libx264", "libx265", "libsvtav1",
        }:
            return algo_lower

        is_hevc = "hevc" in algo_lower or "h265" in algo_lower or "x265" in algo_lower
        is_av1 = "av1" in algo_lower

        if is_av1:
            return "av1_nvenc" if use_gpu else "libsvtav1"
        if is_hevc:
            return "hevc_nvenc" if use_gpu else "libx265"
        # Default to H.264.
        return "h264_nvenc" if use_gpu else "libx264"

    def _build_command(
        self,
        input_path: Path,
        output_path: Path,
        codec: str,
        preset: dict[str, Any],
        use_gpu: bool,
    ) -> list[str]:
        """Assemble the full FFmpeg command line."""
        cmd: list[str] = [self._ffmpeg, "-y", "-hide_banner"]

        # Use NVDEC for decoding when available and encoding on GPU.
        if use_gpu and self._hw.nvdec.available:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

        cmd += ["-i", str(input_path)]

        # Video codec and quality params.
        cmd += ["-c:v", codec]

        if use_gpu:
            cmd += ["-preset", preset["gpu_preset"]]
            if preset["gpu_rc"] == "lossless":
                cmd += ["-tune", "lossless"]
            else:
                cmd += ["-rc", preset["gpu_rc"], "-qp", str(preset["gpu_cq"])]
        else:
            cmd += ["-preset", preset["cpu_preset"]]
            if preset["cpu_crf"] == 0:
                # libx264 lossless.
                cmd += ["-crf", "0"]
            else:
                cmd += ["-crf", str(preset["cpu_crf"])]

        # Copy audio stream without re-encoding.
        cmd += ["-c:a", "copy"]

        # Progress reporting — FFmpeg writes machine-readable updates when
        # given ``-progress pipe:1``.
        cmd += ["-progress", "pipe:1", "-stats_period", "0.5"]

        cmd.append(str(output_path))
        return cmd

    # ------------------------------------------------------------------
    # FFmpeg execution & progress parsing
    # ------------------------------------------------------------------

    def _probe_duration(self, path: Path) -> Optional[float]:
        """Return the duration in seconds via ffprobe, or None."""
        try:
            cmd = [
                self._ffprobe,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                dur = data.get("format", {}).get("duration")
                if dur is not None:
                    return float(dur)
        except Exception as exc:
            logger.debug("Could not probe duration for %s: %s", path.name, exc)
        return None

    def _run_ffmpeg(
        self,
        cmd: list[str],
        duration: Optional[float],
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: Optional[str],
    ) -> None:
        """Execute FFmpeg and stream progress updates.

        Uses ``subprocess.Popen`` so that we can read ``-progress pipe:1``
        output line-by-line and report percentage completion.

        Args:
            cmd: The full FFmpeg argument list.
            duration: Total input duration in seconds (used to compute %).
            progress_callback: Optional ``callback(job_id, pct)``.
            job_id: Passed through to *progress_callback*.

        Raises:
            RuntimeError: On non-zero exit code.
        """
        if progress_callback is None or duration is None or duration <= 0:
            # Simple path — no progress parsing needed.
            result = subprocess.run(
                cmd, capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg failed (rc={result.returncode}): "
                    f"{result.stderr[-2000:]}"
                )
            return

        # Streaming path with progress.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        _TIME_US_RE = re.compile(r"^out_time_us=(\d+)")
        _TIME_HMS_RE = re.compile(r"out_time=(\d{2}):(\d{2}):(\d{2})\.(\d+)")

        try:
            assert proc.stdout is not None  # for type checker
            for line in proc.stdout:
                line = line.strip()
                elapsed_s: float | None = None

                m = _TIME_US_RE.match(line)
                if m:
                    elapsed_s = int(m.group(1)) / 1_000_000
                else:
                    m2 = _TIME_HMS_RE.search(line)
                    if m2:
                        h, mn, s = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
                        frac = int(m2.group(4)) / (10 ** len(m2.group(4)))
                        elapsed_s = h * 3600 + mn * 60 + s + frac

                if elapsed_s is not None:
                    pct = min(elapsed_s / duration * 100.0, 100.0)
                    try:
                        progress_callback(job_id or "", pct)
                    except Exception:
                        logger.debug("Progress callback raised; ignoring")

                # ``progress=end`` is emitted once encoding finishes.
                if line.startswith("progress=end"):
                    try:
                        progress_callback(job_id or "", 100.0)
                    except Exception:
                        pass

            proc.wait()
        except Exception:
            proc.kill()
            proc.wait()
            raise

        if proc.returncode != 0:
            stderr_tail = ""
            if proc.stderr is not None:
                stderr_tail = proc.stderr.read()[-2000:]
            raise RuntimeError(
                f"FFmpeg failed (rc={proc.returncode}): {stderr_tail}"
            )
