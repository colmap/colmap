# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np

import pycolmap

from .package_eth3d_centered import (
    SOURCE_CALIBRATION,
    TARGET_CALIBRATION,
    undistort_scene,
)


def test_undistort_scene(tmp_path: Path) -> None:
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 1
    options.num_frames_per_rig = 3
    options.num_points3D = 50
    source = pycolmap.synthesize_dataset(options)
    input_path = tmp_path / "input"
    (input_path / SOURCE_CALIBRATION).mkdir(parents=True)
    source.write(input_path / SOURCE_CALIBRATION)
    (input_path / "images").mkdir()
    pycolmap.synthesize_images(
        pycolmap.SyntheticImageOptions(), source, input_path / "images"
    )
    target_camera = pycolmap.Camera(
        camera_id=1,
        model="PINHOLE",
        width=700,
        height=500,
        params=[1000.0, 1000.0, 350.0, 250.0],
    )

    output_path = tmp_path / "output"
    undistort_scene(input_path, output_path, target_camera, num_threads=1)

    target = pycolmap.Reconstruction(output_path / TARGET_CALIBRATION)
    assert list(target.cameras.values()) == [target_camera]
    assert target.num_points3D() == source.num_points3D()
    for image_id, image in target.images.items():
        np.testing.assert_allclose(
            image.cam_from_world().matrix(),
            source.images[image_id].cam_from_world().matrix(),
        )
        bitmap = pycolmap.Bitmap.read(output_path / "images" / image.name, True)
        assert bitmap is not None
        assert (bitmap.width, bitmap.height) == (700, 500)
    # The synthetic observations are noise-free.
    assert target.compute_mean_reprojection_error() < 1e-6
