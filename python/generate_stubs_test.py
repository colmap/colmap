"""Tests for generated pycolmap stubs.

These tests verify that the generated type stubs correctly accept valid usage
patterns. Each test contains code that should pass mypy type checking.
If mypy reports errors, it indicates the stubs need fixing.

Run with: mypy python/generate_stubs_test.py
"""

import numpy as np
from numpy.typing import NDArray

import pycolmap


def test_rotation3d_accepts_1d_array() -> None:
    """Rotation3d should accept a 1D array of shape (4,) for quaternion."""
    quat_1d: NDArray[np.float64] = np.array([0.0, 0.0, 0.0, 1.0])
    rotation = pycolmap.Rotation3d(quat_1d)
    assert rotation is not None


def test_rotation3d_accepts_2d_column_vector() -> None:
    """Rotation3d should accept a 2D column vector of shape (4, 1)."""
    quat_2d: NDArray[np.float64] = np.array([[0.0], [0.0], [0.0], [1.0]])
    rotation = pycolmap.Rotation3d(quat_2d)
    assert rotation is not None


def test_rigid3d_accepts_1d_translation() -> None:
    """Rigid3d should accept a 1D array of shape (3,) for translation."""
    rotation = pycolmap.Rotation3d()
    translation_1d: NDArray[np.float64] = np.array([1.0, 2.0, 3.0])
    rigid = pycolmap.Rigid3d(rotation, translation_1d)
    assert rigid is not None


def test_rigid3d_accepts_2d_column_translation() -> None:
    """Rigid3d should accept a 2D column vector of shape (3, 1) for translation."""
    rotation = pycolmap.Rotation3d()
    translation_2d: NDArray[np.float64] = np.array([[1.0], [2.0], [3.0]])
    rigid = pycolmap.Rigid3d(rotation, translation_2d)
    assert rigid is not None


def test_two_view_geometry_inlier_matches() -> None:
    """TwoViewGeometry.inlier_matches should accept uint32 array with 2 columns."""
    matches: NDArray[np.uint32] = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.uint32)
    two_view_geom = pycolmap.TwoViewGeometry()
    two_view_geom.inlier_matches = matches
    assert two_view_geom.inlier_matches is not None


if __name__ == "__main__":
    # Run basic sanity checks (runtime, not type checking)
    test_rotation3d_accepts_1d_array()
    test_rotation3d_accepts_2d_column_vector()
    test_rigid3d_accepts_1d_translation()
    test_rigid3d_accepts_2d_column_translation()
    test_two_view_geometry_inlier_matches()
    print("All runtime tests passed!")
