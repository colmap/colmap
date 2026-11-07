# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import pycolmap


def test_undistort_camera_options_default_init() -> None:
    options = pycolmap.UndistortCameraOptions()
    assert options is not None


def test_warp_image_interpolation_enum() -> None:
    assert {
        k: int(v)
        for k, v in pycolmap.WarpImageOptions.Interpolation.__members__.items()
    } == {
        "NEAREST_NEIGHBOR": 0,
        "BILINEAR": 1,
    }


def test_warp_image_options_interpolation_readwrite() -> None:
    options = pycolmap.WarpImageOptions()
    assert (
        options.interpolation
        == pycolmap.WarpImageOptions.Interpolation.BILINEAR
    )
    options.interpolation = (
        pycolmap.WarpImageOptions.Interpolation.NEAREST_NEIGHBOR
    )
    assert (
        options.interpolation
        == pycolmap.WarpImageOptions.Interpolation.NEAREST_NEIGHBOR
    )


def test_warp_image_options_direct_warp_min_scale_readwrite() -> None:
    options = pycolmap.WarpImageOptions()
    assert options.direct_warp_min_scale == 0.5
    options.direct_warp_min_scale = 0.4
    assert options.direct_warp_min_scale == 0.4


def test_undistort_camera_options_warp_options_readwrite() -> None:
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.warp_options, pycolmap.WarpImageOptions)
    options.warp_options.direct_warp_min_scale = 0.5
    assert options.warp_options.direct_warp_min_scale == 0.5


def test_undistort_camera_options_blank_pixels_readwrite() -> None:
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.blank_pixels, float)
    options.blank_pixels = 1.0
    assert options.blank_pixels == 1.0


def test_undistort_camera_options_min_scale_readwrite() -> None:
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.min_scale, float)
    options.min_scale = 0.5
    assert options.min_scale == 0.5


def test_undistort_camera_options_max_scale_readwrite() -> None:
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.max_scale, float)
    options.max_scale = 1.5
    assert options.max_scale == 1.5


def test_undistort_camera_options_max_image_size_readwrite() -> None:
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.max_image_size, int)
    options.max_image_size = 2048
    assert options.max_image_size == 2048


def test_undistort_camera_options_roi_min_x_readwrite() -> None:
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.roi_min_x, float)
    options.roi_min_x = 0.1
    assert options.roi_min_x == 0.1


def test_undistort_camera_options_roi_min_y_readwrite() -> None:
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.roi_min_y, float)
    options.roi_min_y = 0.2
    assert options.roi_min_y == 0.2


def test_undistort_camera_options_roi_max_x_readwrite() -> None:
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.roi_max_x, float)
    options.roi_max_x = 0.9
    assert options.roi_max_x == 0.9


def test_undistort_camera_options_roi_max_y_readwrite() -> None:
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.roi_max_y, float)
    options.roi_max_y = 0.8
    assert options.roi_max_y == 0.8


def test_undistort_camera() -> None:
    options = pycolmap.UndistortCameraOptions()
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.PINHOLE, 500.0, 1024, 768
    )
    undistorted_camera = pycolmap.undistort_camera(options, camera)
    assert isinstance(undistorted_camera, pycolmap.Camera)


def test_undistort_image() -> None:
    options = pycolmap.UndistortCameraOptions()
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.PINHOLE, 500.0, 1024, 768
    )
    array = np.zeros((768, 1024, 3), dtype=np.uint8)
    bitmap = pycolmap.Bitmap.from_array(array)
    result = pycolmap.undistort_image(options, bitmap, camera)
    assert isinstance(result, tuple)
    assert len(result) == 2
    undistorted_bitmap, undistorted_camera = result
    assert isinstance(undistorted_bitmap, pycolmap.Bitmap)
    assert isinstance(undistorted_camera, pycolmap.Camera)


def test_warp_image_between_cameras() -> None:
    source_camera = pycolmap.Camera(
        model="PINHOLE", width=64, height=48, params=[50.0, 50.0, 32.0, 24.0]
    )
    # Shifting the principal point by an integer number of pixels shifts the
    # image by the same amount.
    target_camera = pycolmap.Camera(
        model="PINHOLE", width=64, height=48, params=[50.0, 50.0, 34.0, 24.0]
    )
    array = np.random.randint(1, 256, (48, 64, 3), dtype=np.uint8)
    bitmap = pycolmap.Bitmap.from_array(array)
    options = pycolmap.WarpImageOptions()
    options.interpolation = (
        pycolmap.WarpImageOptions.Interpolation.NEAREST_NEIGHBOR
    )
    warped = pycolmap.warp_image_between_cameras(
        options, source_camera, target_camera, bitmap
    )
    assert warped.width == 64
    assert warped.height == 48
    warped_array = warped.to_array()
    np.testing.assert_array_equal(warped_array[:, 2:], array[:, :-2])
    # Target pixels outside of the source image are black.
    np.testing.assert_array_equal(warped_array[:, :2], 0)
