import numpy as np

import pycolmap


def test_point3d_default_init():
    point = pycolmap.Point3D()
    assert point is not None


def test_point3d_xyz_readwrite():
    point = pycolmap.Point3D()
    point.xyz = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(point.xyz, [1.0, 2.0, 3.0])


def test_point3d_color_readwrite():
    point = pycolmap.Point3D()
    point.color = np.array([255, 128, 0], dtype=np.uint8)
    np.testing.assert_array_equal(point.color, [255, 128, 0])


def test_point3d_error_readwrite():
    point = pycolmap.Point3D()
    point.error = 0.5
    assert point.error == 0.5


def test_point3d_has_error():
    point = pycolmap.Point3D()
    result = point.has_error()
    assert isinstance(result, bool)


def test_point3d_track_readwrite():
    point = pycolmap.Point3D()
    track = pycolmap.Track()
    track.add_element(image_id=1, point2D_idx=0)
    point.track = track
    assert point.track.length() == 1


def test_point3d_map_empty():
    point_map = pycolmap.Point3DMap()
    assert len(point_map) == 0
