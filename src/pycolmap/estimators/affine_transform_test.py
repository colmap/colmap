import numpy as np

import pycolmap


def test_estimate_affine2d_with_points():
    source = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]
    target = [
        np.array([1.0, 1.0]),
        np.array([2.0, 1.0]),
        np.array([1.0, 2.0]),
    ]
    result = pycolmap.estimate_affine2d(source, target)
    assert result is not None
    result_array = np.array(result)
    assert result_array.shape == (2, 3)


def test_estimate_affine2d_robust_is_callable():
    assert callable(pycolmap.estimate_affine2d_robust)
