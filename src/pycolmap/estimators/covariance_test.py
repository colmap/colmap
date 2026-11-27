# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import pycolmap


def test_ba_covariance_options_params_enum() -> None:
    assert {
        k: int(v)
        for k, v in pycolmap.BACovarianceOptionsParams.__members__.items()
    } == {
        "POSES": 0,
        "POINTS": 1,
        "POSES_AND_POINTS": 2,
        "ALL": 3,
    }


def test_ba_covariance_options_default_init() -> None:
    options = pycolmap.BACovarianceOptions()
    assert options is not None


def test_ba_covariance_options_params_readwrite() -> None:
    options = pycolmap.BACovarianceOptions()
    options.params = pycolmap.BACovarianceOptionsParams.POINTS
    assert options.params == pycolmap.BACovarianceOptionsParams.POINTS


def test_ba_covariance_options_damping_readwrite() -> None:
    options = pycolmap.BACovarianceOptions()
    assert isinstance(options.damping, float)
    options.damping = 1e-6
    assert options.damping == 1e-6


def test_experimental_pose_param_default_init() -> None:
    param = pycolmap.ExperimentalPoseParam()
    assert param is not None


def test_experimental_pose_param_image_id_readwrite() -> None:
    param = pycolmap.ExperimentalPoseParam()
    param.image_id = 42
    assert param.image_id == 42


def test_experimental_pose_param_from_rigid3d() -> None:
    pose = pycolmap.Rigid3d()
    pose.translation = np.array([1.0, 2.0, 3.0])
    param = pycolmap.ExperimentalPoseParam()
    param.cam_from_world = pose.params
    np.testing.assert_array_equal(param.cam_from_world, pose.params)

    options = pycolmap.BACovarianceOptions(experimental_custom_poses=[param])
    pose.translation[:] = [4.0, 5.0, 6.0]
    np.testing.assert_array_equal(
        options.experimental_custom_poses[0].cam_from_world, pose.params
    )
    with pytest.raises(ValueError):
        param.cam_from_world = np.zeros(4)
    with pytest.raises(TypeError):
        param.cam_from_world = np.zeros(7, dtype=np.float32)


def test_estimate_ba_covariance_is_callable() -> None:
    assert callable(pycolmap.estimate_ba_covariance)
