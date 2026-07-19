import numpy as np

import pycolmap


def test_image_default_init():
    image = pycolmap.Image()
    assert image is not None


def test_image_image_id_readwrite():
    image = pycolmap.Image()
    image.image_id = 10
    assert image.image_id == 10


def test_image_camera_id_readwrite():
    image = pycolmap.Image()
    image.camera_id = 5
    assert image.camera_id == 5


def test_image_frame_id_readwrite():
    image = pycolmap.Image()
    image.frame_id = 3
    assert image.frame_id == 3


def test_image_name_readwrite():
    image = pycolmap.Image()
    image.name = "test_image.jpg"
    assert image.name == "test_image.jpg"


def test_image_points2d_readwrite():
    image = pycolmap.Image()
    point_list = pycolmap.Point2DList()
    point_list.append(pycolmap.Point2D(xy=np.array([1.0, 2.0])))
    point_list.append(pycolmap.Point2D(xy=np.array([3.0, 4.0])))
    image.points2D = point_list
    assert len(image.points2D) == 2


def test_image_data_id_readonly():
    image = pycolmap.Image()
    data_id = image.data_id
    assert data_id is not None


def test_image_has_camera_id():
    image = pycolmap.Image()
    assert not image.has_camera_id()
    image.camera_id = 1
    assert image.has_camera_id()


def test_image_has_frame_id():
    image = pycolmap.Image()
    assert not image.has_frame_id()
    image.frame_id = 1
    assert image.has_frame_id()


def test_image_num_points2d():
    image = pycolmap.Image()
    assert image.num_points2D() == 0
    point_list = pycolmap.Point2DList()
    point_list.append(pycolmap.Point2D(xy=np.array([1.0, 2.0])))
    image.points2D = point_list
    assert image.num_points2D() == 1


def test_image_num_points3d_readonly():
    image = pycolmap.Image()
    assert image.num_points3D == 0


def test_image_has_pose_readonly():
    image = pycolmap.Image()
    assert isinstance(image.has_pose, bool)


def test_image_map_empty():
    image_map = pycolmap.ImageMap()
    assert len(image_map) == 0


def test_image_map_insert_and_access():
    image_map = pycolmap.ImageMap()
    image = pycolmap.Image()
    image.image_id = 1
    image.name = "test.jpg"
    image_map[1] = image
    assert len(image_map) == 1
    assert image_map[1].name == "test.jpg"
