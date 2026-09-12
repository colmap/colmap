# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .geometry import normalize_vec, vec_angular_dist_deg


class TestNormalizeVec:
    def test_unit_vector(self) -> None:
        vec = np.array([1.0, 0.0, 0.0])
        normalized = normalize_vec(vec)
        np.testing.assert_allclose(normalized, vec)
        np.testing.assert_almost_equal(np.linalg.norm(normalized), 1.0)

    def test_non_unit_vector(self) -> None:
        vec = np.array([3.0, 4.0, 0.0])
        normalized = normalize_vec(vec)
        expected = np.array([0.6, 0.8, 0.0])
        np.testing.assert_allclose(normalized, expected)
        np.testing.assert_almost_equal(np.linalg.norm(normalized), 1.0)

    def test_zero_vector(self) -> None:
        vec = np.array([0.0, 0.0, 0.0])
        normalized = normalize_vec(vec)
        assert np.linalg.norm(normalized) < 1e-8


class TestVecAngularDistDeg:
    def test_identical_vectors(self) -> None:
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        dist = vec_angular_dist_deg(vec1, vec2)
        np.testing.assert_almost_equal(dist, 0.0)

    def test_opposite_vectors(self) -> None:
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        dist = vec_angular_dist_deg(vec1, vec2)
        np.testing.assert_almost_equal(dist, 180.0)

    def test_perpendicular_vectors(self) -> None:
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        dist = vec_angular_dist_deg(vec1, vec2)
        np.testing.assert_almost_equal(dist, 90.0)

    def test_45_degree_vectors(self) -> None:
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 1.0, 0.0])
        dist = vec_angular_dist_deg(vec1, vec2)
        np.testing.assert_almost_equal(dist, 45.0)

    def test_non_unit_vectors(self) -> None:
        vec1 = np.array([2.0, 0.0, 0.0])
        vec2 = np.array([0.0, 3.0, 0.0])
        dist = vec_angular_dist_deg(vec1, vec2)
        # Should normalize internally
        np.testing.assert_almost_equal(dist, 90.0)
