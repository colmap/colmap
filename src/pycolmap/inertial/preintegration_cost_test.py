# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import pycolmap
import pycolmap.inertial


def _make_dummy_preintegrated_data() -> pycolmap.inertial.PreintegratedImuData:
    calib = pycolmap.ImuCalibration()
    opt = pycolmap.inertial.ImuPreintegrationOptions()
    integrator = pycolmap.inertial.ImuPreintegrator(opt, calib, 0, 10000000)
    integrator.integrate(
        pycolmap.ImuMeasurement(0, np.zeros(3), np.array([0, 0, 9.81]))
    )
    integrator.integrate(
        pycolmap.ImuMeasurement(10000000, np.zeros(3), np.array([0, 0, 9.81]))
    )
    data = integrator.extract()
    data.finalize()
    return data


def test_visual_centric_imu_preintegration_cost_constructs() -> None:
    assert hasattr(pycolmap.inertial, "VisualCentricImuPreintegrationCost")
    pytest.importorskip("pyceres")
    data = _make_dummy_preintegrated_data()
    c = pycolmap.inertial.VisualCentricImuPreintegrationCost(data)
    assert c is not None


def test_analytical_visual_centric_imu_preintegration_cost_constructs() -> None:
    assert hasattr(
        pycolmap.inertial, "AnalyticalVisualCentricImuPreintegrationCost"
    )
    pytest.importorskip("pyceres")
    data = _make_dummy_preintegrated_data()
    c = pycolmap.inertial.AnalyticalVisualCentricImuPreintegrationCost(data)
    assert c is not None


def test_inertial_rotation_cost_constructs() -> None:
    assert hasattr(pycolmap.inertial, "InertialRotationCost")
    pytest.importorskip("pyceres")
    data = _make_dummy_preintegrated_data()
    rig = pycolmap.Rigid3d()
    q = pycolmap.Rotation3d()

    c_rig = pycolmap.inertial.InertialRotationCost(data, rig)
    assert c_rig is not None

    c_q = pycolmap.inertial.InertialRotationCost(data, q)
    assert c_q is not None


def test_inertial_global_positioning_cost_constructs() -> None:
    assert hasattr(pycolmap.inertial, "InertialGlobalPositioningCost")
    pytest.importorskip("pyceres")
    data = _make_dummy_preintegrated_data()
    rig = pycolmap.Rigid3d()
    q_cw = pycolmap.Rotation3d()

    c = pycolmap.inertial.InertialGlobalPositioningCost(data, rig, q_cw, q_cw)
    assert c is not None


def test_bias_prior_cost_constructs() -> None:
    cf = pycolmap.inertial
    assert hasattr(cf, "BiasPriorCost")
    pytest.importorskip("pyceres")
    prior = np.array([0.01, -0.02, 0.03])
    cov = np.eye(3) * 0.001
    stddev_vec = np.array([0.05, 0.05, 0.05])

    c_raw = cf.BiasPriorCost(prior, 3)
    assert c_raw is not None

    c_stddev = cf.BiasPriorCost(0.05, prior, 3)
    assert c_stddev is not None

    c_stddev2 = cf.BiasPriorCost(prior, 0.05, 3)
    assert c_stddev2 is not None

    c_cov = cf.BiasPriorCost(cov, prior, 3)
    assert c_cov is not None

    c_cov2 = cf.BiasPriorCost(prior, cov, 3)
    assert c_cov2 is not None

    c_stddev_vec = cf.BiasPriorCost(stddev_vec, prior, 3)
    assert c_stddev_vec is not None


def test_gyro_bias_prior_cost_constructs() -> None:
    cf = pycolmap.inertial
    assert hasattr(cf, "GyroBiasPriorCost")
    pytest.importorskip("pyceres")
    prior = np.array([0.01, -0.02, 0.03])
    cov = np.eye(3) * 0.001
    stddev_vec = np.array([0.05, 0.05, 0.05])

    c_raw = cf.GyroBiasPriorCost(prior)
    assert c_raw is not None

    c_stddev = cf.GyroBiasPriorCost(0.05, prior)
    assert c_stddev is not None

    c_stddev2 = cf.GyroBiasPriorCost(prior, 0.05)
    assert c_stddev2 is not None

    c_cov = cf.GyroBiasPriorCost(cov, prior)
    assert c_cov is not None

    c_cov2 = cf.GyroBiasPriorCost(prior, cov)
    assert c_cov2 is not None

    c_stddev_vec = cf.GyroBiasPriorCost(stddev_vec, prior)
    assert c_stddev_vec is not None


def test_accel_bias_prior_cost_constructs() -> None:
    cf = pycolmap.inertial
    assert hasattr(cf, "AccelBiasPriorCost")
    pytest.importorskip("pyceres")
    prior = np.array([0.01, -0.02, 0.03])
    cov = np.eye(3) * 0.001
    stddev_vec = np.array([0.1, 0.1, 0.1])

    c_raw = cf.AccelBiasPriorCost(prior)
    assert c_raw is not None

    c_stddev = cf.AccelBiasPriorCost(0.1, prior)
    assert c_stddev is not None

    c_stddev2 = cf.AccelBiasPriorCost(prior, 0.1)
    assert c_stddev2 is not None

    c_cov = cf.AccelBiasPriorCost(cov, prior)
    assert c_cov is not None

    c_cov2 = cf.AccelBiasPriorCost(prior, cov)
    assert c_cov2 is not None

    c_stddev_vec = cf.AccelBiasPriorCost(stddev_vec, prior)
    assert c_stddev_vec is not None
