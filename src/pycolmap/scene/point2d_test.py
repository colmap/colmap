import numpy as np

import pycolmap


def test_point2d_default_init() -> None:
    point = pycolmap.Point2D()
    assert point is not None


def test_point2d_init_with_xy() -> None:
    point = pycolmap.Point2D(xy=np.array([1.0, 2.0]))
    assert point is not None


def test_point2d_init_with_xy_and_point3d_id() -> None:
    point = pycolmap.Point2D(xy=np.array([1.0, 2.0]), point3D_id=5)
    assert point.point3D_id == 5


def test_point2d_xy_readwrite() -> None:
    point = pycolmap.Point2D()
    point.xy = np.array([3.0, 4.0])
    np.testing.assert_array_almost_equal(point.xy, [3.0, 4.0])


def test_point2d_x() -> None:
    point = pycolmap.Point2D(xy=np.array([10.0, 20.0]))
    assert point.x() == 10.0


def test_point2d_y() -> None:
    point = pycolmap.Point2D(xy=np.array([10.0, 20.0]))
    assert point.y() == 20.0


def test_point2d_point3d_id_readwrite() -> None:
    point = pycolmap.Point2D()
    point.point3D_id = 42
    assert point.point3D_id == 42


def test_point2d_has_point3d_false() -> None:
    point = pycolmap.Point2D()
    assert not point.has_point3D()


def test_point2d_has_point3d_true() -> None:
    point = pycolmap.Point2D()
    point.point3D_id = 1
    assert point.has_point3D()


def test_point2d_list_append_and_len() -> None:
    point_list = pycolmap.Point2DList()
    point_list.append(pycolmap.Point2D(xy=np.array([1.0, 2.0])))
    point_list.append(pycolmap.Point2D(xy=np.array([3.0, 4.0])))
    assert len(point_list) == 2
