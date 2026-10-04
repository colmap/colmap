# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pycolmap


def test_cost_functions_submodule_exists() -> None:
    assert hasattr(pycolmap._core, "cost_functions")


def test_reproj_error_cost_exists() -> None:
    assert hasattr(pycolmap._core.cost_functions, "ReprojErrorCost")


def test_rig_reproj_error_cost_exists() -> None:
    assert hasattr(pycolmap._core.cost_functions, "RigReprojErrorCost")


def test_scaled_rig_reproj_error_cost_exists() -> None:
    assert hasattr(pycolmap._core.cost_functions, "ScaledRigReprojErrorCost")


def test_sampson_error_cost_exists() -> None:
    assert hasattr(pycolmap._core.cost_functions, "SampsonErrorCost")


def test_absolute_pose_prior_cost_exists() -> None:
    assert hasattr(pycolmap._core.cost_functions, "AbsolutePosePriorCost")


def test_absolute_pose_position_prior_cost_exists() -> None:
    assert hasattr(
        pycolmap._core.cost_functions, "AbsolutePosePositionPriorCost"
    )


def test_relative_pose_prior_cost_exists() -> None:
    assert hasattr(pycolmap._core.cost_functions, "RelativePosePriorCost")


def test_point3d_alignment_cost_exists() -> None:
    assert hasattr(pycolmap._core.cost_functions, "Point3DAlignmentCost")


def _make_dummy_preintegrated_data() -> pycolmap.PreintegratedImuData:
    import numpy as np

    calib = pycolmap.ImuCalibration()
    opt = pycolmap.ImuPreintegrationOptions()
    integrator = pycolmap.ImuPreintegrator(opt, calib, 0, 10000000)
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
    cf = pycolmap._core.cost_functions
    assert hasattr(cf, "VisualCentricImuPreintegrationCost")
    pytest.importorskip("pyceres")
    data = _make_dummy_preintegrated_data()
    c = cf.VisualCentricImuPreintegrationCost(data)
    assert c is not None

    q_iori = pycolmap.Rotation3d()
    c_stab = cf.VisualCentricImuPreintegrationCost(data, q_iori, q_iori)
    assert c_stab is not None


def test_analytical_visual_centric_imu_preintegration_cost_constructs() -> None:
    cf = pycolmap._core.cost_functions
    assert hasattr(cf, "AnalyticalVisualCentricImuPreintegrationCost")
    pytest.importorskip("pyceres")
    data = _make_dummy_preintegrated_data()
    c = cf.AnalyticalVisualCentricImuPreintegrationCost(data)
    assert c is not None

    q_iori = pycolmap.Rotation3d()
    c_stab = cf.AnalyticalVisualCentricImuPreintegrationCost(
        data, q_iori, q_iori
    )
    assert c_stab is not None


def test_inertial_rotation_cost_constructs() -> None:
    import numpy as np

    cf = pycolmap._core.cost_functions
    assert hasattr(cf, "InertialRotationCost")
    pytest.importorskip("pyceres")
    data = _make_dummy_preintegrated_data()
    rig = pycolmap.Rigid3d()
    q = pycolmap.Rotation3d()
    q_iori = pycolmap.Rotation3d()
    sqrt_info = np.eye(6)

    c_rig = cf.InertialRotationCost(data, rig)
    assert c_rig is not None

    c_q = cf.InertialRotationCost(data, q)
    assert c_q is not None

    c_stab_rig = cf.InertialRotationCost(data, rig, q_iori, q_iori)
    assert c_stab_rig is not None

    c_stab_q = cf.InertialRotationCost(data, q, q_iori, q_iori)
    assert c_stab_q is not None

    c_sqrt_stab = cf.InertialRotationCost(data, rig, sqrt_info, q_iori, q_iori)
    assert c_sqrt_stab is not None


def test_inertial_global_positioning_cost_constructs() -> None:
    cf = pycolmap._core.cost_functions
    assert hasattr(cf, "InertialGlobalPositioningCost")
    pytest.importorskip("pyceres")
    data = _make_dummy_preintegrated_data()
    rig = pycolmap.Rigid3d()
    q_cw = pycolmap.Rotation3d()

    c = cf.InertialGlobalPositioningCost(data, rig, q_cw, q_cw)
    assert c is not None


def test_bias_prior_cost_constructs() -> None:
    import numpy as np

    cf = pycolmap._core.cost_functions
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
    import numpy as np

    cf = pycolmap._core.cost_functions
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
    import numpy as np

    cf = pycolmap._core.cost_functions
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
