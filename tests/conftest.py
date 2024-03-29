import os
from pathlib import Path

import pytest


@pytest.fixture
def cache_path(tmp_path):
    return Path(os.getenv("BIOIMAGEIO_CACHE_PATH", tmp_path))


@pytest.fixture
def manifest_path():
    return Path(__file__).parent.parent / "manifest.yaml"
