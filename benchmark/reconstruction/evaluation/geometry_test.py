# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from .geometry import normalize_vec, vec_angular_dist_deg


class TestNormalizeVec:
    @pytest.mark.parametrize("scale", [1e-200, 1e-12, 1.0, 1e200])
    def test_nonzero_vector(self, scale: float) -> None:
        vec = scale * np.array([3.0, 4.0, 0.0])
        vec_before = vec.copy()
        normalized = normalize_vec(vec)
        np.testing.assert_allclose(normalized, [0.6, 0.8, 0.0])
        np.testing.assert_almost_equal(np.linalg.norm(normalized), 1.0)
        np.testing.assert_array_equal(vec, vec_before)

    def test_zero_vector(self) -> None:
        vec = np.zeros(3)
        normalized = normalize_vec(vec)
        assert normalized is not vec
        np.testing.assert_array_equal(normalized, np.zeros(3))


class TestVecAngularDistDeg:
    @pytest.mark.parametrize("scale1", [1e-200, 1e-12, 1.0, 2.0, 1e200])
    @pytest.mark.parametrize("scale2", [1e-200, 1e-12, 1.0, 3.0, 1e200])
    @pytest.mark.parametrize(
        ("target_dir", "expected"),
        [
            ([1.0, 0.0, 0.0], 0.0),
            ([1.0, 1.0, 0.0], 45.0),
            ([0.0, 1.0, 0.0], 90.0),
            ([-1.0, 0.0, 0.0], 180.0),
        ],
    )
    def test_scale_invariant_angles(
        self,
        scale1: float,
        scale2: float,
        target_dir: list[float],
        expected: float,
    ) -> None:
        vec1 = scale1 * np.array([1.0, 0.0, 0.0])
        vec2 = scale2 * np.array(target_dir)
        vec1_before, vec2_before = vec1.copy(), vec2.copy()
        assert vec_angular_dist_deg(vec1, vec2) == pytest.approx(
            expected, abs=1e-8
        )
        np.testing.assert_array_equal(vec1, vec1_before)
        np.testing.assert_array_equal(vec2, vec2_before)

    def test_zero_vector(self) -> None:
        dist = vec_angular_dist_deg(np.zeros(3), np.array([1.0, 0.0, 0.0]))
        assert dist == 90.0
