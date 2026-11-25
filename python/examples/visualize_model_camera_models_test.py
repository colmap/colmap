# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import pycolmap


@pytest.mark.parametrize(
    ("model_name", "params", "fx", "fy"),
    [
        ("SIMPLE_PINHOLE", [400, 320, 240], 400, 400),
        ("SIMPLE_RADIAL_FISHEYE", [400, 320, 240, 0.05], 400, 400),
        (
            "THIN_PRISM_FISHEYE",
            [400, 500, 320, 240, 0, 0, 0, 0, 0, 0, 0, 0],
            400,
            500,
        ),
    ],
)
def test_draws_perspective_camera_models(
    tmp_path: Path, model_name: str, params: list[float], fx: float, fy: float
) -> None:
    open3d = pytest.importorskip("open3d")
    from visualize_model import Model

    width, height = 640, 480
    options = pycolmap.SyntheticDatasetOptions(
        num_rigs=1,
        num_cameras_per_rig=1,
        num_frames_per_rig=1,
        num_points3D=0,
        camera_width=width,
        camera_height=height,
        camera_model_id=pycolmap.CameraModelId(model_name),
        camera_params=params,
    )
    reconstruction = pycolmap.synthesize_dataset(options)
    reconstruction.write(tmp_path)
    model = Model()
    model.read_model(str(tmp_path))
    model.visualizer = Mock(spec=open3d.visualization.Visualizer)

    model.add_cameras()

    calls = model.visualizer.add_geometry.call_args_list
    assert len(calls) == 3
    frustum = calls[2].args[0]
    assert isinstance(frustum, open3d.geometry.LineSet)
    img = next(iter(model.reconstruction.images.values()))
    frustum_in_camera = img.cam_from_world() * np.asarray(frustum.points)
    x, y = width / (2 * fx), height / (2 * fy)
    np.testing.assert_allclose(
        frustum_in_camera,
        [[0, 0, 0], [-x, -y, 1], [x, -y, 1], [-x, y, 1], [x, y, 1]],
        atol=1e-10,
    )
