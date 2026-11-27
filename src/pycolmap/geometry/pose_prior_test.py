# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import pycolmap


def test_pose_prior_coordinate_system_enum() -> None:
    assert pycolmap.PosePriorCoordinateSystem.UNDEFINED is not None
    assert pycolmap.PosePriorCoordinateSystem.WGS84 is not None
    assert pycolmap.PosePriorCoordinateSystem.CARTESIAN is not None


def test_pose_prior_default_init() -> None:
    pose_prior = pycolmap.PosePrior()
    assert pose_prior is not None


def test_pose_prior_position_readwrite() -> None:
    pose_prior = pycolmap.PosePrior()
    pose_prior.position = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(pose_prior.position, [1.0, 2.0, 3.0])


def test_pose_prior_position_covariance_readwrite() -> None:
    pose_prior = pycolmap.PosePrior()
    covariance = np.eye(3) * 0.5
    pose_prior.position_covariance = covariance
    np.testing.assert_array_almost_equal(
        pose_prior.position_covariance, covariance
    )


def test_pose_prior_coordinate_system_readwrite() -> None:
    pose_prior = pycolmap.PosePrior()
    pose_prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.WGS84
    assert (
        pose_prior.coordinate_system == pycolmap.PosePriorCoordinateSystem.WGS84
    )


def test_pose_prior_gravity_readwrite() -> None:
    pose_prior = pycolmap.PosePrior()
    gravity = np.array([0.0, 0.0, -9.81])
    pose_prior.gravity = gravity
    np.testing.assert_array_almost_equal(pose_prior.gravity, gravity)


def test_pose_prior_has_position() -> None:
    pose_prior = pycolmap.PosePrior()
    assert isinstance(pose_prior.has_position(), bool)


def test_pose_prior_has_position_cov() -> None:
    pose_prior = pycolmap.PosePrior()
    assert isinstance(pose_prior.has_position_cov(), bool)


def test_pose_prior_has_gravity() -> None:
    pose_prior = pycolmap.PosePrior()
    assert isinstance(pose_prior.has_gravity(), bool)


def test_pose_prior_heading() -> None:
    prior = pycolmap.PosePrior()
    assert np.isnan(prior.heading)
    assert np.isnan(prior.heading_stddev)
    assert not prior.has_heading()
    assert not prior.has_heading_stddev()

    prior = pycolmap.PosePrior(heading=0.0, heading_stddev=0.1)
    assert prior.has_heading()
    assert not prior.has_position()
    assert prior.has_heading_stddev()
    assert prior.todict()["heading"] == 0.0
    assert prior.todict()["heading_stddev"] == 0.1

    prior.heading = float("inf")
    assert not prior.has_heading()
    for invalid_stddev in (0.0, -0.1, float("inf")):
        prior.heading_stddev = invalid_stddev
        assert not prior.has_heading_stddev()


def test_pose_prior_pose_prior_id_readwrite() -> None:
    pose_prior = pycolmap.PosePrior()
    pose_prior.pose_prior_id = 42
    assert pose_prior.pose_prior_id == 42


def test_pose_prior_corr_data_id_readwrite() -> None:
    pose_prior = pycolmap.PosePrior()
    data_id = pycolmap.data_t(
        pycolmap.sensor_t(pycolmap.SensorType.CAMERA, 0), 5
    )
    pose_prior.corr_data_id = data_id
    assert pose_prior.corr_data_id == data_id
