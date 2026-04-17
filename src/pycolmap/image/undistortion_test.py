import numpy as np

import pycolmap


def test_undistort_camera_options_default_init():
    options = pycolmap.UndistortCameraOptions()
    assert options is not None


def test_undistort_camera_options_blank_pixels_readwrite():
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.blank_pixels, float)
    options.blank_pixels = 1.0
    assert options.blank_pixels == 1.0


def test_undistort_camera_options_min_scale_readwrite():
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.min_scale, float)
    options.min_scale = 0.5
    assert options.min_scale == 0.5


def test_undistort_camera_options_max_scale_readwrite():
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.max_scale, float)
    options.max_scale = 1.5
    assert options.max_scale == 1.5


def test_undistort_camera_options_max_image_size_readwrite():
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.max_image_size, int)
    options.max_image_size = 2048
    assert options.max_image_size == 2048


def test_undistort_camera_options_roi_min_x_readwrite():
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.roi_min_x, float)
    options.roi_min_x = 0.1
    assert options.roi_min_x == 0.1


def test_undistort_camera_options_roi_min_y_readwrite():
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.roi_min_y, float)
    options.roi_min_y = 0.2
    assert options.roi_min_y == 0.2


def test_undistort_camera_options_roi_max_x_readwrite():
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.roi_max_x, float)
    options.roi_max_x = 0.9
    assert options.roi_max_x == 0.9


def test_undistort_camera_options_roi_max_y_readwrite():
    options = pycolmap.UndistortCameraOptions()
    assert isinstance(options.roi_max_y, float)
    options.roi_max_y = 0.8
    assert options.roi_max_y == 0.8


def test_undistort_camera():
    options = pycolmap.UndistortCameraOptions()
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.PINHOLE, 500.0, 1024, 768
    )
    undistorted_camera = pycolmap.undistort_camera(options, camera)
    assert isinstance(undistorted_camera, pycolmap.Camera)


def test_undistort_image():
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
