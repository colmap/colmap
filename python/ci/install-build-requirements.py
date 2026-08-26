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
if "CCACHE_DIR" in os.environ:
    pybind11 = importlib.import_module("pybind11")
    if pybind11.__file__ is None:
        raise RuntimeError("Cannot locate the installed pybind11 package")
    pybind11_source = Path(pybind11.__file__).parent
    pybind11_destination = (
        Path(os.environ["CCACHE_DIR"]).parent
        / "build-requirements"
        / "pybind11"
    )
    shutil.rmtree(pybind11_destination, ignore_errors=True)
    shutil.copytree(pybind11_source, pybind11_destination)
