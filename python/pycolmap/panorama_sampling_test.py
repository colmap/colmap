# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest

from .panorama import PanoProcessor, PanoRenderOptions


@pytest.mark.parametrize("pitch", [-90.0, 90.0])
def test_process_does_not_wrap_between_panorama_poles(
    tmp_path: Path, pitch: float
) -> None:
    pytest.importorskip("cv2")
    image_module = pytest.importorskip("PIL.Image")
    pixels = np.zeros((50, 100, 3), dtype=np.uint8)
    pixels[0] = [255, 0, 0]
    pixels[-1] = [0, 0, 255]
    image_module.fromarray(pixels).save(tmp_path / "pano.png")
    processor = PanoProcessor(
        tmp_path,
        tmp_path / "images",
        tmp_path / "masks",
        PanoRenderOptions(
            num_steps_yaw=1,
            pitches_deg=[pitch],
            hfov_deg=90.0,
            vfov_deg=90.0,
        ),
    )
    processor.process("pano.png")
    with image_module.open(
        tmp_path / "images/pano_camera0/pano.png"
    ) as rendered:
        output = np.asarray(rendered)
    # An odd-sized virtual camera has a pixel pointing exactly at the pole.
    centre = output[output.shape[0] // 2, output.shape[1] // 2]
    expected = pixels[0, 0] if pitch > 0 else pixels[-1, 0]
    np.testing.assert_array_equal(centre, expected)


def test_process_wraps_panorama_longitude(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    image_module = pytest.importorskip("PIL.Image")
    pixels = np.zeros((50, 100, 3), dtype=np.uint8)
    pixels[:, 0] = [255, 0, 0]
    pixels[:, -1] = [0, 0, 255]
    image_module.fromarray(pixels).save(tmp_path / "pano.png")
    processor = PanoProcessor(
        tmp_path,
        tmp_path / "images",
        tmp_path / "masks",
        PanoRenderOptions(
            num_steps_yaw=2,
            pitches_deg=[0.0],
            hfov_deg=90.0,
            vfov_deg=90.0,
        ),
    )
    processor.process("pano.png")
    with image_module.open(
        tmp_path / "images/pano_camera1/pano.png"
    ) as rendered:
        output = np.asarray(rendered)
    centre = output[output.shape[0] // 2, output.shape[1] // 2]
    np.testing.assert_array_equal(centre, [128, 0, 128])
