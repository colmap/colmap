import numpy as np

import pycolmap


def test_triangulate_point():
    cam1_from_world = np.eye(3, 4)
    cam2_from_world = np.eye(3, 4)
    cam2_from_world[0, 3] = -1.0
    point1 = np.array([0.0, 0.0])
    point2 = np.array([0.1, 0.0])
    result = pycolmap.triangulate_point(
        cam1_from_world, cam2_from_world, point1, point2
    )
    if result is not None:
        assert result.shape == (3,)


def test_calculate_triangulation_angle():
    center1 = np.array([0.0, 0.0, 0.0])
    center2 = np.array([1.0, 0.0, 0.0])
    point3d = np.array([0.5, 0.0, 1.0])
    angle = pycolmap.calculate_triangulation_angle(center1, center2, point3d)
    assert isinstance(angle, float)
    assert angle >= 0.0


def test_triangulate_mid_point_callable():
    assert callable(pycolmap.triangulate_mid_point)
