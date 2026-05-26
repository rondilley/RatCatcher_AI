"""Shared test fixtures for RatCatcher AI tests."""

import os
import tempfile
from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory for tests that write files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "clips").mkdir()
    (data_dir / "thumbnails").mkdir()
    return data_dir


@pytest.fixture
def config_dir() -> Path:
    """Path to the project config directory."""
    return Path(__file__).parent.parent / "config"
