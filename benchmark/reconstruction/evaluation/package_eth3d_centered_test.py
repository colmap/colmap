# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest

import pycolmap

from .package_eth3d_centered import (
    TARGET_IMAGE_DIR,
    check_no_blank_pixels,
    create_target_camera,
    map_pixels,
    undistort_image,
    undistort_reconstruction,
)


def _pinhole(width: int, height: int, focal: float) -> pycolmap.Camera:
    return pycolmap.Camera(
        camera_id=1,
        model="PINHOLE",
        width=width,
        height=height,
        params=[focal, focal, width / 2, height / 2],
    )


def test_create_target_camera_has_centered_principal_point() -> None:
    camera = create_target_camera()
    assert camera.model == pycolmap.CameraModelId.PINHOLE
    assert camera.principal_point_x == camera.width / 2
    assert camera.principal_point_y == camera.height / 2


def test_map_pixels_between_scaled_cameras() -> None:
    source_camera = _pinhole(100, 80, 50.0)
    target_camera = _pinhole(200, 160, 100.0)
    points2D = np.array([[0.5, 0.5], [50.0, 40.0], [99.0, 79.0]])
    mapped, valid = map_pixels(source_camera, target_camera, points2D)
    np.testing.assert_allclose(mapped, 2 * points2D)
    assert valid.all()


def test_map_pixels_marks_points_outside_target() -> None:
    source_camera = _pinhole(100, 80, 50.0)
    target_camera = _pinhole(50, 40, 50.0)
    _, valid = map_pixels(
        source_camera, target_camera, np.array([[50.0, 40.0], [1.0, 1.0]])
    )
    np.testing.assert_array_equal(valid, [True, False])


def test_check_no_blank_pixels() -> None:
    source_camera = _pinhole(100, 80, 50.0)
    check_no_blank_pixels(source_camera, _pinhole(90, 70, 50.0))
    with pytest.raises(ValueError):
        check_no_blank_pixels(source_camera, _pinhole(100, 80, 50.0))


def test_undistort_reconstruction_preserves_geometry() -> None:
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 1
    options.num_frames_per_rig = 4
    options.num_points3D = 50
    options.num_points2D_without_point3D = 0
    source = pycolmap.synthesize_dataset(options)
    target_camera = _pinhole(800, 600, 1000.0)

    target = undistort_reconstruction(source, target_camera)

    assert list(target.cameras) == [target_camera.camera_id]
    assert set(target.images) == set(source.images)
    assert target.num_points3D() > 0
    for image_id, image in target.images.items():
        source_image = source.images[image_id]
        assert image.name == (
            f"{TARGET_IMAGE_DIR}/{Path(source_image.name).name}"
        )
        np.testing.assert_allclose(
            image.cam_from_world().matrix(),
            source_image.cam_from_world().matrix(),
        )
        for point2D in image.points2D:
            assert point2D.has_point3D()
            assert 0 <= point2D.xy[0] < target_camera.width
            assert 0 <= point2D.xy[1] < target_camera.height
    # The synthetic observations are noise-free.
    assert target.compute_mean_reprojection_error() < 1e-6


def test_undistort_image_keeps_only_orientation(tmp_path: Path) -> None:
    source_camera = _pinhole(64, 48, 50.0)
    target_camera = _pinhole(32, 24, 25.0)
    source_image = pycolmap.Bitmap.from_array(
        np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
    )
    source_image.set_metadata("Orientation", 1)
    source_image.set_metadata("Make", "Maker")
    source_image.set_metadata("Model", "Camera")
    source_image.set_metadata("Exif:FocalLength", 20.0)
    source_path = tmp_path / "source.jpg"
    assert source_image.write(source_path)

    target_path = tmp_path / "target.jpg"
    undistort_image(source_path, target_path, source_camera, target_camera)

    target_image = pycolmap.Bitmap.read(target_path, as_rgb=True)
    assert target_image is not None
    assert (target_image.width, target_image.height) == (32, 24)
    assert target_image.exif_orientation() == 1
    assert target_image.exif_camera_model() is None
    assert target_image.exif_focal_length() is None
