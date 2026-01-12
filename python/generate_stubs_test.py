"""Tests for generated pycolmap stubs.

These tests verify that the generated type stubs correctly accept valid usage
patterns. Each test contains code that should pass mypy type checking.
If mypy reports errors, it indicates the stubs need fixing.

Run with: mypy python/generate_stubs_test.py
"""

import numpy as np
from numpy.typing import NDArray

import pycolmap


# Issue: Eigen column vectors (Nx1) generate stubs with shape
# tuple[Literal[N], Literal[1]] but should also accept 1D arrays
# with shape tuple[Literal[N]]
class TestEigenColumnVectorShouldAccept1DArray:
    def test_rotation3d_quaternion(self) -> None:
        quat: NDArray[np.float64] = np.array([0.0, 0.0, 0.0, 1.0])
        rotation = pycolmap.Rotation3d(quat)
        assert rotation is not None

    def test_rigid3d_translation(self) -> None:
        rotation = pycolmap.Rotation3d()
        translation: NDArray[np.float64] = np.array([1.0, 2.0, 3.0])
        rigid = pycolmap.Rigid3d(rotation, translation)
        assert rigid is not None


# Issue: Eigen dynamic dimensions generate stubs with Never type
# but should use int to represent variable-length dimensions
class TestEigenDynamicDimensionShouldNotBeNever:
    def test_two_view_geometry_inlier_matches(self) -> None:
        matches: NDArray[np.uint32] = np.array(
            [[0, 1], [2, 3], [4, 5]], dtype=np.uint32
        )
        two_view_geom = pycolmap.TwoViewGeometry()
        two_view_geom.inlier_matches = matches
        assert two_view_geom.inlier_matches is not None


if __name__ == "__main__":
    # Run basic sanity checks (runtime, not type checking)
    test1 = TestEigenColumnVectorShouldAccept1DArray()
    test1.test_rotation3d_quaternion()
    test1.test_rigid3d_translation()

    test2 = TestEigenDynamicDimensionShouldNotBeNever()
    test2.test_two_view_geometry_inlier_matches()

    print("All runtime tests passed!")
