import pycolmap


def test_ba_covariance_options_params_enum():
    assert {m.name: int(m) for m in pycolmap.BACovarianceOptionsParams} == {
        "POSES": 0,
        "POINTS": 1,
        "POSES_AND_POINTS": 2,
        "ALL": 3,
    }


def test_ba_covariance_options_default_init():
    options = pycolmap.BACovarianceOptions()
    assert options is not None


def test_ba_covariance_options_params_readwrite():
    options = pycolmap.BACovarianceOptions()
    options.params = pycolmap.BACovarianceOptionsParams.POINTS
    assert options.params == pycolmap.BACovarianceOptionsParams.POINTS


def test_ba_covariance_options_damping_readwrite():
    options = pycolmap.BACovarianceOptions()
    assert isinstance(options.damping, float)
    options.damping = 1e-6
    assert options.damping == 1e-6


def test_experimental_pose_param_default_init():
    param = pycolmap.ExperimentalPoseParam()
    assert param is not None


def test_experimental_pose_param_image_id_readwrite():
    param = pycolmap.ExperimentalPoseParam()
    param.image_id = 42
    assert param.image_id == 42


def test_estimate_ba_covariance_is_callable():
    assert callable(pycolmap.estimate_ba_covariance)
