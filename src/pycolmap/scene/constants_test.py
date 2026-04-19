import pycolmap


def test_invalid_camera_id():
    assert pycolmap.INVALID_CAMERA_ID is not None


def test_invalid_image_id():
    assert pycolmap.INVALID_IMAGE_ID is not None


def test_invalid_image_pair_id():
    assert pycolmap.INVALID_IMAGE_PAIR_ID is not None


def test_invalid_point2d_idx():
    assert pycolmap.INVALID_POINT2D_IDX is not None


def test_invalid_point3d_id():
    assert pycolmap.INVALID_POINT3D_ID is not None


def test_invalid_pose_prior_id():
    assert pycolmap.INVALID_POSE_PRIOR_ID is not None


def test_invalid_sensor_id():
    assert pycolmap.INVALID_SENSOR_ID is not None


def test_invalid_data_id():
    assert pycolmap.INVALID_DATA_ID is not None
