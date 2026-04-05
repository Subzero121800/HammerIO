"""Batch audio processing encoder for HammerIO.

Uses FFmpeg subprocess calls for audio format conversion, quality
adjustment, and batch directory transcoding.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional, Union

from hammerio.core.hardware import HardwareProfile

logger = logging.getLogger("hammerio.encoders.audio")

# Supported audio extensions → canonical format name
_SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".mp3": "mp3",
    ".wav": "wav",
    ".flac": "flac",
    ".aac": "aac",
    ".m4a": "aac",
    ".ogg": "ogg",
    ".opus": "opus",
}

# Quality presets → bitrate strings used by FFmpeg
_BITRATE_PRESETS: dict[str, str] = {
    "fast": "128k",
    "balanced": "192k",
    "quality": "256k",
    "lossless": "320k",
}

# FFmpeg codec flags for each output format
_FORMAT_CODECS: dict[str, dict[str, str]] = {
    "mp3": {"codec": "libmp3lame", "ext": ".mp3"},
    "wav": {"codec": "pcm_s16le", "ext": ".wav"},
    "flac": {"codec": "flac", "ext": ".flac"},
    "aac": {"codec": "aac", "ext": ".m4a"},
    "ogg": {"codec": "libvorbis", "ext": ".ogg"},
    "opus": {"codec": "libopus", "ext": ".opus"},
}


def _is_supported_audio(path: Path) -> bool:
    """Check whether a file has a supported audio extension."""
    return path.suffix.lower() in _SUPPORTED_EXTENSIONS


def _find_ffmpeg() -> str:
    """Locate the ``ffmpeg`` binary or raise."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise FileNotFoundError(
            "ffmpeg not found on PATH — install FFmpeg to use AudioEncoder"
        )
    return path


def _probe_duration(ffmpeg: str, input_file: Path) -> Optional[float]:
    """Use ffprobe to get the duration in seconds, or None."""
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    if not shutil.which(ffprobe):
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(input_file),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return None


class AudioEncoder:
    """Batch audio transcoder backed by FFmpeg.

    Converts between MP3, WAV, FLAC, AAC, OGG, and OPUS formats with
    configurable quality presets that map to standard audio bitrates.

    Args:
        hardware: Detected hardware profile.  Currently used only for
            logging; all encoding is performed via FFmpeg subprocess.
    """

    def __init__(self, hardware: HardwareProfile) -> None:
        self.hardware = hardware
        self._ffmpeg: Optional[str] = None

        try:
            self._ffmpeg = _find_ffmpeg()
            logger.info("Audio backend: FFmpeg at %s", self._ffmpeg)
        except FileNotFoundError:
            logger.warning("FFmpeg not found — AudioEncoder will not be functional")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path, None],
        algorithm: str,
        quality: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Transcode audio file(s).

        For a single file the audio is converted and written to
        *output_path*.  For a directory every supported audio file is
        transcoded into the *output_path* directory, preserving base
        names (extension changes to match the target format).

        Args:
            input_path: Source file or directory.
            output_path: Destination file or directory.  Auto-generated
                when ``None``.
            algorithm: Target format — one of ``"mp3"``, ``"wav"``,
                ``"flac"``, ``"aac"``, ``"ogg"``, ``"opus"``.
            quality: Quality preset — ``"fast"`` (128 kbps),
                ``"balanced"`` (192 kbps), ``"quality"`` (256 kbps),
                or ``"lossless"`` (320 kbps).
            progress_callback: Optional ``(job_id, percent)`` callable.
            job_id: Identifier forwarded to the progress callback.

        Returns:
            Absolute path to the output file or directory.

        Raises:
            FileNotFoundError: If *input_path* or FFmpeg is missing.
            ValueError: If the target format is not recognised.
        """
        if self._ffmpeg is None:
            raise FileNotFoundError(
                "ffmpeg not found — install FFmpeg to use AudioEncoder"
            )

        input_path = Path(input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        fmt_info = _FORMAT_CODECS.get(algorithm.lower().strip())
        if fmt_info is None:
            raise ValueError(
                f"Unsupported audio format '{algorithm}'. "
                f"Choose from: {', '.join(_FORMAT_CODECS)}"
            )

        bitrate = _BITRATE_PRESETS.get(quality, _BITRATE_PRESETS["balanced"])
        jid = job_id or "audio"

        if output_path is None:
            if input_path.is_dir():
                output_path = input_path.parent / (input_path.name + "_transcoded")
            else:
                output_path = input_path.with_suffix(fmt_info["ext"])
        output_path = Path(output_path).resolve()

        if input_path.is_dir():
            return self._process_directory(
                input_path, output_path, algorithm, fmt_info, bitrate,
                progress_callback, jid,
            )
        else:
            return self._process_single(
                input_path, output_path, fmt_info, bitrate,
                progress_callback, jid,
            )

    # ------------------------------------------------------------------
    # Directory batch processing
    # ------------------------------------------------------------------

    def _process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        algorithm: str,
        fmt_info: dict[str, str],
        bitrate: str,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: str,
    ) -> str:
        """Transcode all supported audio files in a directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_files = [
            f for f in sorted(input_dir.iterdir())
            if f.is_file() and _is_supported_audio(f)
        ]
        total = len(audio_files)
        if total == 0:
            logger.warning("No supported audio files found in %s", input_dir)
            return str(output_dir)

        logger.info("Batch transcoding %d audio files from %s", total, input_dir)
        for idx, src in enumerate(audio_files):
            dst = output_dir / (src.stem + fmt_info["ext"])
            try:
                self._transcode(src, dst, fmt_info, bitrate, progress_callback, job_id, idx, total)
            except Exception:
                logger.exception("Failed to transcode %s", src.name)

            if progress_callback is not None:
                pct = (idx + 1) / total * 100.0
                progress_callback(job_id, pct)

        logger.info("Batch transcode complete — output in %s", output_dir)
        return str(output_dir)

    # ------------------------------------------------------------------
    # Single-file processing
    # ------------------------------------------------------------------

    def _process_single(
        self,
        input_file: Path,
        output_file: Path,
        fmt_info: dict[str, str],
        bitrate: str,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: str,
    ) -> str:
        """Transcode a single audio file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if progress_callback is not None:
            progress_callback(job_id, 0.0)

        self._transcode(input_file, output_file, fmt_info, bitrate, progress_callback, job_id)

        if progress_callback is not None:
            progress_callback(job_id, 100.0)

        in_size = input_file.stat().st_size
        out_size = output_file.stat().st_size
        logger.info(
            "Transcoded %s → %s (%.1f KB → %.1f KB)",
            input_file.name,
            output_file.name,
            in_size / 1024,
            out_size / 1024,
        )
        return str(output_file)

    # ------------------------------------------------------------------
    # FFmpeg invocation
    # ------------------------------------------------------------------

    def _transcode(
        self,
        src: Path,
        dst: Path,
        fmt_info: dict[str, str],
        bitrate: str,
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: str,
        batch_idx: int = 0,
        batch_total: int = 1,
    ) -> None:
        """Run FFmpeg to transcode a single file with progress parsing."""
        assert self._ffmpeg is not None

        codec = fmt_info["codec"]

        cmd: list[str] = [
            self._ffmpeg,
            "-y",              # overwrite output
            "-hide_banner",
            "-loglevel", "info",
            "-i", str(src),
            "-c:a", codec,
        ]

        # Bitrate is irrelevant for lossless PCM or FLAC
        if codec not in ("pcm_s16le", "flac"):
            cmd += ["-b:a", bitrate]

        # Opus requires 48 kHz sample rate
        if codec == "libopus":
            cmd += ["-ar", "48000"]

        cmd.append(str(dst))

        logger.debug("FFmpeg command: %s", " ".join(cmd))

        duration = _probe_duration(self._ffmpeg, src)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Parse FFmpeg stderr for progress (time=HH:MM:SS.xx)
        stderr_lines: list[str] = []
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)
            if progress_callback and duration and duration > 0:
                match = re.search(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})", line)
                if match:
                    h, m, s, cs = (int(g) for g in match.groups())
                    current = h * 3600 + m * 60 + s + cs / 100.0
                    file_pct = min(current / duration * 100.0, 99.9)
                    # Scale into batch range
                    overall_pct = (batch_idx / batch_total + file_pct / 100.0 / batch_total) * 100.0
                    progress_callback(job_id, overall_pct)

        proc.wait()
        if proc.returncode != 0:
            stderr_text = "".join(stderr_lines)
            raise RuntimeError(
                f"FFmpeg exited with code {proc.returncode} for {src.name}:\n{stderr_text}"
            )
