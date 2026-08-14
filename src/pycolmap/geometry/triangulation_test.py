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


def test_triangulate_point_from_rays():
    cam1_from_world = np.eye(3, 4)
    cam2_from_world = np.eye(3, 4)
    cam2_from_world[0, 3] = -1.0
    point3d = np.array([0.0, 0.0, 5.0])
    ray1 = cam1_from_world @ np.append(point3d, 1.0)
    ray1 /= np.linalg.norm(ray1)
    ray2 = cam2_from_world @ np.append(point3d, 1.0)
    ray2 /= np.linalg.norm(ray2)
    result = pycolmap.triangulate_point(
        cam1_from_world, cam2_from_world, ray1, ray2
    )
    assert result is not None
    np.testing.assert_allclose(result, point3d, atol=1e-6)


def test_triangulate_multi_view_point():
    cams_from_world = [np.eye(3, 4), np.eye(3, 4), np.eye(3, 4)]
    cams_from_world[1][0, 3] = -1.0
    cams_from_world[2][1, 3] = -1.0
    point3d = np.array([0.1, -0.2, 4.0])
    cam_rays = []
    for cam_from_world in cams_from_world:
        ray = cam_from_world @ np.append(point3d, 1.0)
        cam_rays.append(ray / np.linalg.norm(ray))
    result = pycolmap.triangulate_multi_view_point(cams_from_world, cam_rays)
    assert result is not None
    np.testing.assert_allclose(result, point3d, atol=1e-6)
