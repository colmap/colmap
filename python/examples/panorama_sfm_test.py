# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
from pathlib import Path

import panorama_sfm
import pytest
from pycolmap.panorama import PanoramaReconstructionOptions


def test_help() -> None:
    script_path = Path(__file__).with_name("panorama_sfm.py")
    subprocess.run(
        [sys.executable, script_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )


def test_input_mask_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []

    def reconstruct(
        input_path: Path,
        output_path: Path,
        options: PanoramaReconstructionOptions,
    ) -> dict:
        captured.append((input_path, output_path, options))
        return {}

    monkeypatch.setattr(panorama_sfm, "reconstruct", reconstruct)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "panorama_sfm.py",
            "--input_image_path",
            "panoramas",
            "--output_path",
            "output",
            "--input_mask_path",
            "masks",
            "--input_camera_mask_path",
            "camera.png",
            "--use_cpu",
        ],
    )
    panorama_sfm.main()
    assert len(captured) == 1
    input_path, output_path, options = captured[0]
    assert input_path == Path("panoramas")
    assert output_path == Path("output")
    assert options.input_mask_path == Path("masks")
    assert options.input_camera_mask_path == Path("camera.png")
    assert not options.use_gpu
