import pycolmap


def test_ba_covariance_options_params_poses():
    assert pycolmap.BACovarianceOptionsParams.POSES is not None


def test_ba_covariance_options_params_points():
    assert pycolmap.BACovarianceOptionsParams.POINTS is not None


def test_ba_covariance_options_params_poses_and_points():
    assert pycolmap.BACovarianceOptionsParams.POSES_AND_POINTS is not None


def test_ba_covariance_options_params_all():
    assert pycolmap.BACovarianceOptionsParams.ALL is not None


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
