"""Install PEP 517 build requirements into the active Python environment."""

import importlib
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
