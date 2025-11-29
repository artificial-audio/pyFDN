"""Pytest configuration for path setup."""

import sys
from pathlib import Path

import pytest
from .matloader import load_mat_workspace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

@pytest.fixture(scope="session")
def loadmat():
    return load_mat_workspace

