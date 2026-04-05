"""Input analysis and algorithm selection for HammerIO.

Profiles input files to determine optimal compression strategy:
file type, size, estimated entropy, batch characteristics.
"""

from __future__ import annotations

import mimetypes
import os
import random
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class FileCategory(Enum):
    """High-level file category for routing decisions."""

    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    ARCHIVE = "archive"
    DATASET = "dataset"
    DOCUMENT = "document"
    BINARY = "binary"
    TEXT = "text"
    DIRECTORY = "directory"
    UNKNOWN = "unknown"


class CompressionMode(Enum):
    """Recommended compression mode."""

    GPU_NVENC = "gpu_nvenc"       # Video via NVENC
    GPU_NVCOMP = "gpu_nvcomp"    # Bulk data via nvCOMP
    GPU_VPI = "gpu_vpi"          # Image batch via VPI
    GPU_FFMPEG = "gpu_ffmpeg"    # Audio/video via FFmpeg CUDA
    CPU_ZSTD = "cpu_zstd"        # General CPU compression
    CPU_GZIP = "cpu_gzip"        # Compatibility mode
    CPU_BZIP2 = "cpu_bzip2"      # Max ratio CPU
    PASSTHROUGH = "passthrough"  # Already compressed, skip


# File extension mappings
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".ts", ".mpg", ".mpeg"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif", ".raw", ".cr2", ".nef"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus", ".aiff"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".zst", ".lz4", ".hammer"}
DATASET_EXTENSIONS = {".csv", ".parquet", ".arrow", ".hdf5", ".h5", ".npy", ".npz", ".tfrecord", ".pt", ".safetensors"}
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt"}


@dataclass
class FileProfile:
    """Profile of a single file for routing."""

    path: Path
    size_bytes: int
    category: FileCategory
    mime_type: str
    extension: str
    estimated_entropy: float = 0.0  # 0-8 bits per byte
    is_already_compressed: bool = False

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def size_human(self) -> str:
        if self.size_bytes < 1024:
            return f"{self.size_bytes} B"
        elif self.size_bytes < 1024 * 1024:
            return f"{self.size_bytes / 1024:.1f} KB"
        elif self.size_bytes < 1024 * 1024 * 1024:
            return f"{self.size_mb:.1f} MB"
        else:
            return f"{self.size_bytes / (1024**3):.2f} GB"


@dataclass
class BatchProfile:
    """Profile of a batch of files."""

    files: list[FileProfile] = field(default_factory=list)
    total_size_bytes: int = 0
    category_counts: dict[FileCategory, int] = field(default_factory=dict)
    dominant_category: FileCategory = FileCategory.UNKNOWN

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)


@dataclass
class CompressionRecommendation:
    """Algorithm and mode recommendation from the profiler."""

    mode: CompressionMode
    algorithm: str  # e.g. "lz4", "zstd", "h264_nvenc"
    reason: str
    estimated_ratio: Optional[float] = None  # estimated compression ratio
    gpu_preferred: bool = False
    fallback_mode: Optional[CompressionMode] = None
    fallback_algorithm: str = ""


def categorize_file(path: Path) -> FileCategory:
    """Determine file category from extension and mime type."""
    ext = path.suffix.lower()

    if ext in VIDEO_EXTENSIONS:
        return FileCategory.VIDEO
    if ext in IMAGE_EXTENSIONS:
        return FileCategory.IMAGE
    if ext in AUDIO_EXTENSIONS:
        return FileCategory.AUDIO
    if ext in ARCHIVE_EXTENSIONS:
        return FileCategory.ARCHIVE
    if ext in DATASET_EXTENSIONS:
        return FileCategory.DATASET
    if ext in DOCUMENT_EXTENSIONS:
        return FileCategory.DOCUMENT

    # Try mime type
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        if mime.startswith("video/"):
            return FileCategory.VIDEO
        if mime.startswith("image/"):
            return FileCategory.IMAGE
        if mime.startswith("audio/"):
            return FileCategory.AUDIO
        if mime.startswith("text/"):
            return FileCategory.TEXT

    # Check if text by sampling
    try:
        with open(path, "rb") as f:
            sample = f.read(8192)
        if not sample:
            return FileCategory.UNKNOWN
        # Heuristic: if most bytes are printable ASCII, it's text
        text_chars = sum(1 for b in sample if 32 <= b <= 126 or b in (9, 10, 13))
        if text_chars / len(sample) > 0.85:
            return FileCategory.TEXT
    except OSError:
        pass

    return FileCategory.BINARY


def estimate_entropy(path: Path, sample_size: int = 65536) -> float:
    """Estimate Shannon entropy of file content (bits per byte).

    Higher entropy (close to 8) means data is already compressed or random.
    Lower entropy means better compression potential.
    """
    import math

    try:
        file_size = path.stat().st_size
        if file_size == 0:
            return 0.0

        # Sample from multiple positions for large files
        with open(path, "rb") as f:
            if file_size <= sample_size:
                data = f.read()
            else:
                # Sample beginning, middle, and end
                chunk = sample_size // 3
                data = f.read(chunk)
                f.seek(file_size // 2)
                data += f.read(chunk)
                f.seek(max(0, file_size - chunk))
                data += f.read(chunk)

        if not data:
            return 0.0

        # Calculate byte frequency distribution
        freq = [0] * 256
        for byte in data:
            freq[byte] += 1

        length = len(data)
        entropy = 0.0
        for count in freq:
            if count > 0:
                p = count / length
                entropy -= p * math.log2(p)

        return entropy
    except OSError:
        return 0.0


def is_already_compressed(path: Path, category: FileCategory) -> bool:
    """Check if file is already in a compressed format."""
    if category == FileCategory.ARCHIVE:
        return True
    # Most video/audio formats use lossy compression
    ext = path.suffix.lower()
    compressed_exts = {
        ".mp4", ".mkv", ".webm", ".mp3", ".aac", ".ogg", ".opus",
        ".jpg", ".jpeg", ".webp", ".gif", ".zip", ".gz", ".bz2",
        ".xz", ".zst", ".7z", ".rar", ".lz4",
    }
    return ext in compressed_exts


def profile_file(path: Path) -> FileProfile:
    """Create a complete profile for a single file."""
    path = Path(path).resolve()
    stat = path.stat()
    category = categorize_file(path)
    mime, _ = mimetypes.guess_type(str(path))

    entropy = 0.0
    if stat.st_size > 0 and category not in (FileCategory.VIDEO, FileCategory.AUDIO):
        entropy = estimate_entropy(path)

    return FileProfile(
        path=path,
        size_bytes=stat.st_size,
        category=category,
        mime_type=mime or "application/octet-stream",
        extension=path.suffix.lower(),
        estimated_entropy=entropy,
        is_already_compressed=is_already_compressed(path, category),
    )


def profile_directory(directory: Path, recursive: bool = True) -> BatchProfile:
    """Profile all files in a directory."""
    directory = Path(directory).resolve()
    files: list[FileProfile] = []
    total_size = 0
    counts: dict[FileCategory, int] = {}

    glob_pattern = "**/*" if recursive else "*"
    for entry in directory.glob(glob_pattern):
        if entry.is_file():
            fp = profile_file(entry)
            files.append(fp)
            total_size += fp.size_bytes
            counts[fp.category] = counts.get(fp.category, 0) + 1

    # Determine dominant category
    dominant = FileCategory.UNKNOWN
    if counts:
        dominant = max(counts, key=lambda k: counts[k])

    return BatchProfile(
        files=files,
        total_size_bytes=total_size,
        category_counts=counts,
        dominant_category=dominant,
    )


def recommend_compression(
    profile: FileProfile,
    gpu_available: bool = True,
    nvenc_available: bool = True,
    nvcomp_available: bool = False,
    vpi_available: bool = False,
    target_quality: str = "balanced",
) -> CompressionRecommendation:
    """Recommend optimal compression strategy for a file profile.

    Args:
        profile: File profile from profile_file()
        gpu_available: Whether CUDA GPU is available
        nvenc_available: Whether NVENC encoder is available
        nvcomp_available: Whether nvCOMP library is available
        vpi_available: Whether VPI is available
        target_quality: One of "fast", "balanced", "quality", "lossless"
    """
    cat = profile.category

    # Already compressed — skip
    if profile.is_already_compressed and cat == FileCategory.ARCHIVE:
        return CompressionRecommendation(
            mode=CompressionMode.PASSTHROUGH,
            algorithm="none",
            reason=f"Already compressed archive ({profile.extension})",
        )

    # VIDEO — compress with zstd (lossless, fast, preserves original)
    # Transcoding (re-encoding) is a separate operation via 'hammer transcode'
    if cat == FileCategory.VIDEO:
        if profile.is_already_compressed:
            # Already compressed video — zstd won't help much but it's safe
            algo = "zstd" if target_quality != "fast" else "lz4"
            return CompressionRecommendation(
                mode=CompressionMode.CPU_ZSTD,
                algorithm=algo,
                reason=f"Video file (already compressed {profile.extension}) → {algo} archive",
                estimated_ratio=1.1,  # Minimal gain on compressed video
            )
        algo = "zstd" if target_quality != "fast" else "lz4"
        return CompressionRecommendation(
            mode=CompressionMode.CPU_ZSTD,
            algorithm=algo,
            reason=f"Video file → {algo} compression (lossless, preserves original)",
            estimated_ratio=1.1,
        )

    # IMAGE — compress with zstd (images are already compressed, minimal gain)
    if cat == FileCategory.IMAGE:
        algo = "zstd" if target_quality != "fast" else "lz4"
        return CompressionRecommendation(
            mode=CompressionMode.CPU_ZSTD,
            algorithm=algo,
            reason=f"Image file → {algo} compression",
            estimated_ratio=1.05 if profile.is_already_compressed else 2.0,
        )

    # AUDIO — compress with zstd
    if cat == FileCategory.AUDIO:
        algo = "zstd" if target_quality != "fast" else "lz4"
        return CompressionRecommendation(
            mode=CompressionMode.CPU_ZSTD,
            algorithm=algo,
            reason=f"Audio file → {algo} compression",
            estimated_ratio=1.05 if profile.is_already_compressed else 1.5,
        )

    # DATASET (large data files)
    if cat == FileCategory.DATASET:
        if gpu_available and nvcomp_available and profile.size_mb > 100:
            algo = "lz4" if target_quality == "fast" else "zstd"
            return CompressionRecommendation(
                mode=CompressionMode.GPU_NVCOMP,
                algorithm=f"nvcomp_{algo}",
                reason=f"Large dataset ({profile.size_human}) → nvCOMP GPU compression",
                estimated_ratio=2.5 if algo == "lz4" else 3.5,
                gpu_preferred=True,
                fallback_mode=CompressionMode.CPU_ZSTD,
                fallback_algorithm="zstd",
            )
        return CompressionRecommendation(
            mode=CompressionMode.CPU_ZSTD,
            algorithm="zstd",
            reason=f"Dataset ({profile.size_human}) → CPU zstd compression",
            estimated_ratio=3.0,
        )

    # BULK BINARY — large files benefit from GPU compression
    if cat == FileCategory.BINARY and profile.size_mb > 500:
        if gpu_available and nvcomp_available:
            return CompressionRecommendation(
                mode=CompressionMode.GPU_NVCOMP,
                algorithm="nvcomp_lz4",
                reason=f"Large binary ({profile.size_human}) → nvCOMP GPU for throughput",
                gpu_preferred=True,
                fallback_mode=CompressionMode.CPU_ZSTD,
                fallback_algorithm="zstd",
            )

    # TEXT — high compressibility
    if cat == FileCategory.TEXT:
        algo = "zstd" if target_quality != "fast" else "lz4"
        return CompressionRecommendation(
            mode=CompressionMode.CPU_ZSTD,
            algorithm=algo,
            reason=f"Text file (entropy {profile.estimated_entropy:.1f}) → CPU {algo}",
            estimated_ratio=5.0 if profile.estimated_entropy < 5 else 2.5,
        )

    # DEFAULT — CPU zstd
    algo = "zstd"
    if target_quality == "fast":
        algo = "lz4"
    elif profile.extension in (".gz", ".tgz"):
        algo = "gzip"

    return CompressionRecommendation(
        mode=CompressionMode.CPU_ZSTD,
        algorithm=algo,
        reason=f"General file ({profile.size_human}) → CPU {algo}",
        estimated_ratio=2.0,
    )


def recommend_batch(
    batch: BatchProfile,
    gpu_available: bool = True,
    nvenc_available: bool = True,
    nvcomp_available: bool = False,
    vpi_available: bool = False,
    target_quality: str = "balanced",
) -> list[CompressionRecommendation]:
    """Recommend compression strategy for a batch of files."""
    return [
        recommend_compression(
            fp,
            gpu_available=gpu_available,
            nvenc_available=nvenc_available,
            nvcomp_available=nvcomp_available,
            vpi_available=vpi_available,
            target_quality=target_quality,
        )
        for fp in batch.files
    ]
