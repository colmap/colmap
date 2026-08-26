"""Install PEP 517 build requirements into the active Python environment."""

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "tomli"], check=True
    )

tomllib = importlib.import_module(
    "tomllib" if sys.version_info >= (3, 11) else "tomli"
)
project_dir = Path(__file__).resolve().parents[2]
with (project_dir / "pyproject.toml").open("rb") as fid:
    requirements = tomllib.load(fid)["build-system"]["requires"]

subprocess.run(
    [sys.executable, "-m", "pip", "install", *requirements], check=True
)

# cibuildwheel creates its virtual environment below a randomized temporary
# directory on macOS and Windows. Keep pybind11's headers at a stable path so
# that the absolute include directory does not invalidate compiler cache keys.
ccache_dir = os.environ.get("CCACHE_DIR")
if ccache_dir:
    pybind11 = importlib.import_module("pybind11")
    pybind11_source = Path(pybind11.__file__).parent
    pybind11_destination = (
        Path(ccache_dir).parent / "build-requirements" / "pybind11"
    )
    shutil.rmtree(pybind11_destination, ignore_errors=True)
    shutil.copytree(pybind11_source, pybind11_destination)
