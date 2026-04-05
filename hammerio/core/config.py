"""Configuration management for HammerIO.

Loads settings from TOML config file with sensible defaults.
Config locations (in priority order):
  1. Path specified via HAMMERIO_CONFIG env var
  2. ./hammerio.toml (project-local)
  3. ~/.config/hammerio/config.toml (user-level)

Copyright 2026 ResilientMind AI | ResilientMindai.com | Joseph C McGinty Jr
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import toml


DEFAULTS: dict[str, Any] = {
    "general": {
        "quality": "balanced",
        "workers": 8,
        "log_level": "INFO",
        "output_format": "auto",  # auto, zst, gz, bz2
    },
    "video": {
        "gpu_codec": "h264_nvenc",
        "cpu_codec": "libx264",
        "preset": "p2",          # NVENC preset (p1=fastest, p7=best)
        "crf": 23,
        "container": "mp4",
    },
    "bulk": {
        "algorithm": "zstd",
        "chunk_size_mb": 128,
        "gpu_threshold_mb": 500,  # Min file size for GPU compression
    },
    "image": {
        "output_format": "webp",
        "quality": 85,
        "max_dimension": 0,       # 0 = no resize
    },
    "audio": {
        "codec": "aac",
        "bitrate": "128k",
    },
    "dataset": {
        "algorithm": "zstd",
        "compression_level": 3,
        "streaming": True,
    },
    "telemetry": {
        "interval_seconds": 1.0,
        "history_size": 300,
        "thermal_warning_c": 75.0,
        "thermal_critical_c": 90.0,
    },
    "web": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": False,
    },
    "jetson": {
        "power_mode": "auto",     # auto, maxn, 15w, 30w
        "unified_memory_aware": True,
        "nvme_io_threads": 4,
    },
    "watch": {
        "watch_root": "./hammer-watch",
        "compress_output": None,
        "decompress_output": None,
        "gpu_threshold_mb": 100,
        "stable_wait_seconds": 0.5,
        "move_to_processed": True,
    },
}


@dataclass
class HammerConfig:
    """HammerIO configuration container."""

    data: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULTS))
    config_path: Optional[Path] = None

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a config value with dotted section.key lookup."""
        return self.data.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a config value."""
        if section not in self.data:
            self.data[section] = {}
        self.data[section][key] = value

    @property
    def quality(self) -> str:
        return self.get("general", "quality", "balanced")

    @property
    def workers(self) -> int:
        return self.get("general", "workers", 4)

    @property
    def log_level(self) -> str:
        return self.get("general", "log_level", "INFO")

    def save(self, path: Optional[Path] = None) -> None:
        """Save config to TOML file."""
        out = path or self.config_path
        if out is None:
            out = Path.home() / ".config" / "hammerio" / "config.toml"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            toml.dump(self.data, f)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base, returning new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Optional[str | Path] = None) -> HammerConfig:
    """Load configuration from TOML file with defaults.

    Search order:
    1. Explicit path argument
    2. HAMMERIO_CONFIG environment variable
    3. ./hammerio.toml
    4. ~/.config/hammerio/config.toml
    5. Built-in defaults
    """
    search_paths: list[Path] = []

    if path:
        search_paths.append(Path(path))

    env_path = os.environ.get("HAMMERIO_CONFIG")
    if env_path:
        search_paths.append(Path(env_path))

    search_paths.extend([
        Path("hammerio.toml"),
        Path.home() / ".config" / "hammerio" / "config.toml",
    ])

    for config_path in search_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = toml.load(f)
                merged = _deep_merge(copy.deepcopy(DEFAULTS), user_config)
                return HammerConfig(data=merged, config_path=config_path)
            except (toml.TomlDecodeError, OSError):
                continue

    return HammerConfig(data=copy.deepcopy(DEFAULTS))


def generate_default_config(path: Optional[str | Path] = None) -> Path:
    """Generate a default config file with all options documented."""
    out = Path(path) if path else Path("hammerio.toml")

    content = """# HammerIO Configuration
# Copyright 2026 ResilientMind AI | Joseph C McGinty Jr
# See: https://github.com/resilientmindai/hammerio

[general]
quality = "balanced"     # fast, balanced, quality, lossless
workers = 8              # Parallel workers for batch processing
log_level = "INFO"       # DEBUG, INFO, WARNING, ERROR
output_format = "auto"   # auto, zst, gz, bz2

[video]
gpu_codec = "h264_nvenc"   # h264_nvenc, hevc_nvenc, av1_nvenc
cpu_codec = "libx264"      # libx264, libx265
preset = "p2"              # NVENC: p1 (fastest) - p7 (best quality)
crf = 23                   # Constant rate factor (0-51, lower=better)
container = "mp4"          # mp4, mkv, webm

[bulk]
algorithm = "zstd"         # zstd, lz4, snappy, deflate
chunk_size_mb = 128        # Chunk size for streaming compression
gpu_threshold_mb = 500     # Min file size to use GPU compression

[image]
output_format = "webp"     # webp, jpg, png
quality = 85               # Output quality (1-100)
max_dimension = 0          # Max width/height (0 = no resize)

[audio]
codec = "aac"              # aac, libmp3lame, libopus, flac
bitrate = "128k"           # Audio bitrate

[dataset]
algorithm = "zstd"         # zstd, lz4
compression_level = 3      # 1-22 for zstd
streaming = true           # Enable streaming decompression

[telemetry]
interval_seconds = 1.0     # Telemetry polling interval
history_size = 300         # Max telemetry history samples
thermal_warning_c = 75.0   # Temperature warning threshold
thermal_critical_c = 90.0  # Temperature critical threshold

[web]
host = "0.0.0.0"
port = 5000
debug = false

[jetson]
power_mode = "auto"        # auto, maxn, 15w, 30w
unified_memory_aware = true
nvme_io_threads = 4
"""
    out.write_text(content)
    return out
