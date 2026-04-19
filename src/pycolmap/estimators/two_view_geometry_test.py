import numpy as np
import pytest

import pycolmap


def test_two_view_geometry_options_default_init():
    options = pycolmap.TwoViewGeometryOptions()
    assert options is not None


def test_two_view_geometry_options_min_num_inliers_readwrite():
    options = pycolmap.TwoViewGeometryOptions()
    assert isinstance(options.min_num_inliers, int)
    options.min_num_inliers = 30
    assert options.min_num_inliers == 30


@pytest.mark.parametrize(
    "name",
    [
        "estimate_calibrated_two_view_geometry",
        "estimate_two_view_geometry",
        "estimate_two_view_geometry_pose",
        "compute_squared_sampson_error",
    ],
)
def test_public_api_callable(name):
    assert callable(getattr(pycolmap, name))


def test_compute_squared_sampson_error_with_identity():
    points1 = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    points2 = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    matrix = np.eye(3)
    residuals = pycolmap.compute_squared_sampson_error(points1, points2, matrix)
    assert isinstance(residuals, list)
    assert len(residuals) == 2
