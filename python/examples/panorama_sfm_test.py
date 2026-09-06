import subprocess
import sys
from pathlib import Path


def test_help() -> None:
    script_path = Path(__file__).with_name("panorama_sfm.py")
    subprocess.run(
        [sys.executable, script_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
