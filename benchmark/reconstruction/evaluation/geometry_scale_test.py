# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import pycolmap

from .geometry import vec_angular_dist_deg
from .utils import compute_rel_pose_error


@pytest.mark.parametrize("scale1", [1e-200, 1e-12, 1.0, 1e200])
@pytest.mark.parametrize("scale2", [1e-200, 1e-12, 1.0, 1e200])
@pytest.mark.parametrize(
    "target,expected",
    [
        ([1.0, 0.0, 0.0], 0.0),
        ([1.0, 1.0, 0.0], 45.0),
        ([0.0, 1.0, 0.0], 90.0),
        ([-1.0, 0.0, 0.0], 180.0),
    ],
)
def test_angular_distance_is_scale_invariant(
    scale1: float, scale2: float, target: list[float], expected: float
) -> None:
    source = scale1 * np.array([1.0, 0.0, 0.0])
    target_vec = scale2 * np.array(target)
    source_before, target_before = source.copy(), target_vec.copy()
    assert vec_angular_dist_deg(source, target_vec) == pytest.approx(
        expected, abs=1e-8
    )
    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(target_vec, target_before)


@pytest.mark.parametrize("scale", [1e-200, 1e-12, 1.0, 1e200])
def test_relative_pose_direction_is_independent_of_reconstruction_scale(
    scale: float,
) -> None:
    estimated = pycolmap.Rigid3d(pycolmap.Rotation3d(), [scale, 0.0, 0.0])
    ground_truth = pycolmap.Rigid3d(pycolmap.Rotation3d(), [2.0, 0.0, 0.0])
    dt, dr = compute_rel_pose_error(estimated, ground_truth, 0.1)
    assert dt == pytest.approx(0.0, abs=1e-8)
    assert dr == pytest.approx(0.0, abs=1e-8)


def test_zero_vector_keeps_existing_undefined_direction_result() -> None:
    assert vec_angular_dist_deg(np.zeros(3), np.array([1.0, 0.0, 0.0])) == 90.0
