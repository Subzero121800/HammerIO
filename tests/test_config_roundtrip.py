"""Tests for config save/load round-trip and edge cases."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import toml

from hammerio.core.config import (
    DEFAULTS,
    HammerConfig,
    _deep_merge,
    generate_default_config,
    load_config,
)


class TestConfigRoundTrip:
    """Verify save -> load produces identical configuration."""

    def test_full_round_trip(self, tmp_dir: Path) -> None:
        config = load_config()
        config.set("general", "quality", "fast")
        config.set("video", "crf", 18)
        config.set("bulk", "chunk_size_mb", 128)
        save_path = tmp_dir / "rt.toml"
        config.save(save_path)

        reloaded = load_config(save_path)
        assert reloaded.quality == "fast"
        assert reloaded.get("video", "crf") == 18
        assert reloaded.get("bulk", "chunk_size_mb") == 128

    def test_round_trip_preserves_all_sections(self, tmp_dir: Path) -> None:
        config = load_config()
        save_path = tmp_dir / "allsections.toml"
        config.save(save_path)
        reloaded = load_config(save_path)
        for section in DEFAULTS:
            for key in DEFAULTS[section]:
                assert reloaded.get(section, key) == config.get(section, key), (
                    f"Mismatch in [{section}].{key}"
                )

    def test_new_section_survives_round_trip(self, tmp_dir: Path) -> None:
        config = load_config()
        config.set("custom", "my_key", "my_value")
        save_path = tmp_dir / "custom.toml"
        config.save(save_path)
        reloaded = load_config(save_path)
        assert reloaded.get("custom", "my_key") == "my_value"


class TestConfigEdgeCases:
    """Edge cases for config loading."""

    def test_corrupt_toml_falls_through(self, tmp_dir: Path) -> None:
        corrupt = tmp_dir / "corrupt.toml"
        corrupt.write_text("this is [not valid toml {{{")
        config = load_config(corrupt)
        # Should fall through to defaults
        assert config.quality == "balanced"

    def test_empty_toml_uses_defaults(self, tmp_dir: Path) -> None:
        empty = tmp_dir / "empty.toml"
        empty.write_text("")
        config = load_config(empty)
        assert config.quality == "balanced"

    def test_partial_override_preserves_defaults(self, tmp_dir: Path) -> None:
        partial = tmp_dir / "partial.toml"
        partial.write_text('[general]\nworkers = 16\n')
        config = load_config(partial)
        assert config.workers == 16
        assert config.quality == "balanced"  # default preserved
        assert config.get("video", "gpu_codec") == "h264_nvenc"

    def test_env_var_config_path(self, tmp_dir: Path, monkeypatch) -> None:
        cfg = tmp_dir / "env_config.toml"
        cfg.write_text('[general]\nquality = "quality"\n')
        monkeypatch.setenv("HAMMERIO_CONFIG", str(cfg))
        config = load_config()
        assert config.quality == "quality"

    def test_save_creates_parent_dirs(self, tmp_dir: Path) -> None:
        config = load_config()
        deep_path = tmp_dir / "a" / "b" / "c" / "config.toml"
        config.save(deep_path)
        assert deep_path.exists()

    def test_save_default_path_when_none(self, tmp_dir: Path, monkeypatch) -> None:
        """When config_path is None and no explicit path given, save to ~/.config."""
        config = HammerConfig()
        # Monkeypatch home to tmp_dir to avoid polluting real home
        monkeypatch.setattr(Path, "home", lambda: tmp_dir)
        config.save()
        expected = tmp_dir / ".config" / "hammerio" / "config.toml"
        assert expected.exists()

    def test_to_dict_returns_copy(self) -> None:
        config = load_config()
        d = config.to_dict()
        d["general"]["quality"] = "MODIFIED"
        # Original should be unchanged
        assert config.quality == "balanced"

    def test_get_missing_section(self) -> None:
        config = load_config()
        assert config.get("nonexistent_section", "key", "default_val") == "default_val"

    def test_get_missing_key(self) -> None:
        config = load_config()
        assert config.get("general", "nonexistent_key", 42) == 42


class TestDeepMerge:
    """Test _deep_merge utility."""

    def test_simple_override(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3, "c": 4}}
        result = _deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_does_not_mutate_base(self) -> None:
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert "y" not in base["a"]

    def test_replace_dict_with_scalar(self) -> None:
        base = {"a": {"x": 1}}
        override = {"a": 42}
        result = _deep_merge(base, override)
        assert result["a"] == 42


class TestGenerateDefaultConfig:
    """Test default config file generation."""

    def test_generates_valid_toml(self, tmp_dir: Path) -> None:
        path = generate_default_config(tmp_dir / "default.toml")
        # Should be parseable TOML
        data = toml.load(path)
        assert "general" in data
        assert data["general"]["quality"] == "balanced"

    def test_default_filename(self, tmp_dir: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_dir)
        path = generate_default_config()
        assert path.name == "hammerio.toml"
        assert path.exists()
