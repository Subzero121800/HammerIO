"""Tests for configuration management."""

from __future__ import annotations

from pathlib import Path

import pytest

from hammerio.core.config import HammerConfig, load_config, generate_default_config


class TestConfig:
    def test_load_defaults(self) -> None:
        config = load_config()
        assert config.quality == "balanced"
        assert config.workers == 8
        assert config.log_level == "INFO"

    def test_get_nested(self) -> None:
        config = load_config()
        assert config.get("video", "gpu_codec") == "h264_nvenc"
        assert config.get("bulk", "algorithm") == "zstd"
        assert config.get("nonexistent", "key", "fallback") == "fallback"

    def test_set_value(self) -> None:
        config = load_config()
        config.set("general", "quality", "fast")
        assert config.quality == "fast"

    def test_generate_default_config(self, tmp_dir: Path) -> None:
        path = generate_default_config(tmp_dir / "test_config.toml")
        assert path.exists()
        content = path.read_text()
        assert "[general]" in content
        assert "[video]" in content
        assert "[jetson]" in content

    def test_load_from_file(self, tmp_dir: Path) -> None:
        config_file = tmp_dir / "hammerio.toml"
        config_file.write_text('[general]\nquality = "fast"\nworkers = 8\n')
        config = load_config(config_file)
        assert config.quality == "fast"
        assert config.workers == 8
        # Defaults should still be present for unspecified keys
        assert config.get("video", "gpu_codec") == "h264_nvenc"

    def test_save_config(self, tmp_dir: Path) -> None:
        config = load_config()
        config.set("general", "quality", "quality")
        save_path = tmp_dir / "saved.toml"
        config.save(save_path)
        assert save_path.exists()
        reloaded = load_config(save_path)
        assert reloaded.quality == "quality"

    def test_to_dict(self) -> None:
        config = load_config()
        d = config.to_dict()
        assert "general" in d
        assert "video" in d
        assert "jetson" in d

    def test_env_var_override(self, tmp_dir: Path) -> None:
        """Verify HAMMERIO_CONFIG env var overrides default config search."""
        import os

        config_file = tmp_dir / "env_override.toml"
        config_file.write_text('[general]\nquality = "lossless"\nworkers = 16\n')

        old_val = os.environ.get("HAMMERIO_CONFIG")
        try:
            os.environ["HAMMERIO_CONFIG"] = str(config_file)
            config = load_config()
            assert config.quality == "lossless"
            assert config.workers == 16
            assert config.config_path == config_file
        finally:
            if old_val is None:
                os.environ.pop("HAMMERIO_CONFIG", None)
            else:
                os.environ["HAMMERIO_CONFIG"] = old_val

    def test_env_var_nonexistent_path_falls_through(self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When HAMMERIO_CONFIG points to a non-existent file, defaults are used."""
        # Run from a clean tmp_dir so no local hammerio.toml is found
        monkeypatch.chdir(tmp_dir)
        monkeypatch.setenv("HAMMERIO_CONFIG", str(tmp_dir / "does_not_exist.toml"))
        config = load_config()
        # Should fall through to defaults
        assert config.quality == "balanced"
        assert config.config_path is None
