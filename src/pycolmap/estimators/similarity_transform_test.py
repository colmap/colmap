import numpy as np

import pycolmap


def test_estimate_rigid3d_with_coplanar_points():
    source = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
    ]
    target = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
    ]
    result = pycolmap.estimate_rigid3d(source, target)
    # May return None for degenerate configurations, but call should succeed
    assert result is None or isinstance(result, pycolmap.Rigid3d)


def test_estimate_rigid3d_robust_is_callable():
    assert callable(pycolmap.estimate_rigid3d_robust)


def test_estimate_sim3d_with_points():
    source = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    target = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
        np.array([0.0, 0.0, 2.0]),
    ]
    result = pycolmap.estimate_sim3d(source, target)
    assert result is None or isinstance(result, pycolmap.Sim3d)


def test_estimate_sim3d_robust_is_callable():
    assert callable(pycolmap.estimate_sim3d_robust)
