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


def test_analytical_visual_centric_imu_preintegration_cost_constructs() -> None:
    cf = pycolmap._core.cost_functions
    assert hasattr(cf, "AnalyticalVisualCentricImuPreintegrationCost")
    pytest.importorskip("pyceres")
    data = _make_dummy_preintegrated_data()
    c = cf.AnalyticalVisualCentricImuPreintegrationCost(data)
    assert c is not None


def test_inertial_rotation_cost_constructs() -> None:
    cf = pycolmap._core.cost_functions
    assert hasattr(cf, "InertialRotationCost")
    pytest.importorskip("pyceres")
    data = _make_dummy_preintegrated_data()
    rig = pycolmap.Rigid3d()
    q = pycolmap.Rotation3d()

    c_rig = cf.InertialRotationCost(data, rig)
    assert c_rig is not None

    c_q = cf.InertialRotationCost(data, q)
    assert c_q is not None


def test_inertial_global_positioning_cost_constructs() -> None:
    cf = pycolmap._core.cost_functions
    assert hasattr(cf, "InertialGlobalPositioningCost")
    pytest.importorskip("pyceres")
    data = _make_dummy_preintegrated_data()
    rig = pycolmap.Rigid3d()
    q_cw = pycolmap.Rotation3d()

    c = cf.InertialGlobalPositioningCost(data, rig, q_cw, q_cw)
    assert c is not None
