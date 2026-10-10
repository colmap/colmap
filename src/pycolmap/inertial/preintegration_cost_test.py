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
