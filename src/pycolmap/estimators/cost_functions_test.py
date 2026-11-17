# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import pycolmap
from pycolmap import cost_functions


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


def test_relative_rotation_cost() -> None:
    pytest.importorskip("pyceres")
    rotation1 = pycolmap.Rotation3d(np.array([0.2, -0.1, 0.3]))
    rotation2 = pycolmap.Rotation3d(np.array([-0.1, 0.4, 0.2]))
    relative_rotation = rotation2 * rotation1.inverse()
    cost = cost_functions.RelativeRotationCost(relative_rotation)
    residuals, _ = cost.evaluate(rotation1.quat, rotation2.quat)
    np.testing.assert_allclose(residuals, 0.0, atol=1e-12)
    residuals, _ = cost.evaluate(rotation1.quat, rotation1.quat)
    assert np.linalg.norm(residuals) == pytest.approx(relative_rotation.angle())
