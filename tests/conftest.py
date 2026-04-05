"""Shared test fixtures for HammerIO."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory(prefix="hammerio_test_") as d:
        yield Path(d)


@pytest.fixture
def sample_text_file(tmp_dir: Path) -> Path:
    """Create a sample text file."""
    p = tmp_dir / "sample.txt"
    p.write_text("Hello, HammerIO!\n" * 1000)
    return p


@pytest.fixture
def sample_binary_file(tmp_dir: Path) -> Path:
    """Create a sample binary file."""
    p = tmp_dir / "sample.bin"
    p.write_bytes(os.urandom(1024 * 100))  # 100KB
    return p


@pytest.fixture
def sample_compressible_file(tmp_dir: Path) -> Path:
    """Create a highly compressible file."""
    p = tmp_dir / "compressible.dat"
    data = b"AAAA" * 256000  # ~1MB, single repeated byte = very low entropy
    p.write_bytes(data)
    return p


@pytest.fixture
def sample_csv_file(tmp_dir: Path) -> Path:
    """Create a sample CSV dataset file."""
    p = tmp_dir / "dataset.csv"
    lines = ["id,value,label\n"]
    for i in range(10000):
        lines.append(f"{i},{i * 3.14:.6f},class_{i % 5}\n")
    p.write_text("".join(lines))
    return p


@pytest.fixture
def sample_directory(tmp_dir: Path) -> Path:
    """Create a directory with mixed files."""
    d = tmp_dir / "mixed"
    d.mkdir()
    (d / "file1.txt").write_text("Text content\n" * 100)
    (d / "file2.txt").write_text("More text\n" * 200)
    (d / "data.bin").write_bytes(os.urandom(10000))
    (d / "config.json").write_text('{"key": "value", "count": 42}\n')
    return d
