import requests
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional
from zipfile import ZipFile

import pytest

from pybio.spec.utils import get_instance

MODEL_EXTENSIONS = (".model.yaml", ".model.yml")
UNET_2D_NUCLEI_BROAD_PACKAGE_URL = (
    "https://github.com/bioimage-io/pytorch-bioimage-io/releases/download/v0.1.1/UNet2DNucleiBroad.model.zip"
)


def guess_model_path(file_names: List[str]) -> Optional[str]:
    for file_name in file_names:
        if file_name.endswith(MODEL_EXTENSIONS):
            return file_name

    return None


def eval_model_zip(model_zip: ZipFile, cache_path: Path):
    with TemporaryDirectory() as tempdir:
        temp_path = Path(tempdir)
        if cache_path is None:
            cache_path = temp_path / "cache"

        model_zip.extractall(temp_path)
        spec_file_str = guess_model_path([str(file_name) for file_name in temp_path.glob("*")])
        pybio_model = load_model_config(spec_file_str, root_path=temp_path, cache_path=cache_path)

        return get_instance(pybio_model)
