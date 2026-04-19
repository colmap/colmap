import numpy as np

import pycolmap


def test_camera_model_id_invalid():
    assert pycolmap.CameraModelId.INVALID is not None


def test_camera_model_id_pinhole():
    assert pycolmap.CameraModelId.PINHOLE is not None


def test_camera_model_id_simple_pinhole():
    assert pycolmap.CameraModelId.SIMPLE_PINHOLE is not None


def test_camera_model_id_simple_radial():
    assert pycolmap.CameraModelId.SIMPLE_RADIAL is not None


def test_camera_model_id_radial():
    assert pycolmap.CameraModelId.RADIAL is not None


def test_camera_model_id_opencv():
    assert pycolmap.CameraModelId.OPENCV is not None


def test_camera_model_id_string_construction():
    model = pycolmap.CameraModelId("PINHOLE")
    assert model == pycolmap.CameraModelId.PINHOLE


def test_camera_default_init():
    camera = pycolmap.Camera()
    assert camera is not None


def test_camera_create_from_model_id():
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.PINHOLE, 500.0, 1024, 768
    )
    assert camera is not None
    assert camera.camera_id == 1


def test_camera_create_from_model_name():
    camera = pycolmap.Camera.create_from_model_name(
        2, "PINHOLE", 500.0, 1024, 768
    )
    assert camera is not None
    assert camera.camera_id == 2


def test_camera_camera_id_readwrite(simple_camera):
    simple_camera.camera_id = 10
    assert simple_camera.camera_id == 10


def test_camera_model_readwrite(simple_camera):
    assert simple_camera.model == pycolmap.CameraModelId.PINHOLE
    simple_camera.model = pycolmap.CameraModelId.SIMPLE_PINHOLE
    assert simple_camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE


def test_camera_width_height_readwrite(simple_camera):
    assert simple_camera.width == 1024
    assert simple_camera.height == 768
    simple_camera.width = 2048
    simple_camera.height = 1536
    assert simple_camera.width == 2048
    assert simple_camera.height == 1536


def test_camera_params_readwrite(simple_camera):
    params = simple_camera.params
    assert len(params) > 0
    new_params = list(params)
    new_params[0] = 600.0
    simple_camera.params = new_params
    assert simple_camera.params[0] == 600.0


def test_camera_has_prior_focal_length_readwrite(simple_camera):
    simple_camera.has_prior_focal_length = True
    assert simple_camera.has_prior_focal_length is True
    simple_camera.has_prior_focal_length = False
    assert simple_camera.has_prior_focal_length is False


def test_camera_focal_length_readwrite():
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 500.0, 1024, 768
    )
    camera.focal_length = 600.0
    assert camera.focal_length == 600.0


def test_camera_focal_length_x_y(simple_camera):
    assert isinstance(simple_camera.focal_length_x, float)
    assert isinstance(simple_camera.focal_length_y, float)


def test_camera_principal_point_x_y(simple_camera):
    assert isinstance(simple_camera.principal_point_x, float)
    assert isinstance(simple_camera.principal_point_y, float)


def test_camera_sensor_id_readonly(simple_camera):
    sensor_id = simple_camera.sensor_id
    assert sensor_id is not None


def test_camera_model_name_readonly(simple_camera):
    assert simple_camera.model_name == "PINHOLE"


def test_camera_params_info_readonly(simple_camera):
    info = simple_camera.params_info
    assert isinstance(info, str)


def test_camera_mean_focal_length(simple_camera):
    mean_focal = simple_camera.mean_focal_length()
    assert isinstance(mean_focal, float)


def test_camera_focal_length_idxs(simple_camera):
    idxs = simple_camera.focal_length_idxs()
    assert len(idxs) > 0


def test_camera_principal_point_idxs(simple_camera):
    idxs = simple_camera.principal_point_idxs()
    assert len(idxs) > 0


def test_camera_extra_params_idxs(simple_camera):
    idxs = simple_camera.extra_params_idxs()
    assert isinstance(idxs, list)


def test_camera_calibration_matrix(simple_camera):
    matrix = simple_camera.calibration_matrix()
    assert matrix.shape == (3, 3)


def test_camera_cam_from_img_point2d(simple_camera):
    point2d = np.array([512.0, 384.0])
    result = simple_camera.cam_from_img(point2d)
    assert result is not None


def test_camera_cam_from_img_matrix(simple_camera):
    points = np.array([[512.0, 384.0], [100.0, 200.0]])
    result = simple_camera.cam_from_img(points)
    assert len(result) == 2


def test_camera_img_from_cam_point3d(simple_camera):
    point3d = np.array([0.0, 0.0, 1.0])
    result = simple_camera.img_from_cam(point3d)
    assert result is not None


def test_camera_img_from_cam_matrix(simple_camera):
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.2, 1.0]])
    result = simple_camera.img_from_cam(points)
    assert len(result) == 2


def test_camera_cam_from_img_threshold(simple_camera):
    threshold = simple_camera.cam_from_img_threshold(1.0)
    assert isinstance(threshold, float)


def test_camera_verify_params(simple_camera):
    assert simple_camera.verify_params()


def test_camera_is_undistorted(simple_camera):
    result = simple_camera.is_undistorted()
    assert isinstance(result, bool)


def test_camera_has_bogus_params(simple_camera):
    result = simple_camera.has_bogus_params(0.1, 100.0, 1.0)
    assert isinstance(result, bool)


def test_camera_rescale_factor(simple_camera):
    simple_camera.rescale(2.0)
    assert simple_camera.width == 2048
    assert simple_camera.height == 1536


def test_camera_rescale_dimensions(simple_camera):
    simple_camera.rescale(512, 384)
    assert simple_camera.width == 512
    assert simple_camera.height == 384


def test_camera_params_to_string(simple_camera):
    params_string = simple_camera.params_to_string()
    assert isinstance(params_string, str)
    assert len(params_string) > 0


def test_camera_set_params_from_string(simple_camera):
    params_string = simple_camera.params_to_string()
    simple_camera.set_params_from_string(params_string)
    assert simple_camera.verify_params()


def test_camera_map_empty():
    camera_map = pycolmap.CameraMap()
    assert len(camera_map) == 0


def test_camera_map_insert_and_access(simple_camera):
    camera_map = pycolmap.CameraMap()
    camera_map[1] = simple_camera
    assert len(camera_map) == 1
    assert camera_map[1].camera_id == 1
