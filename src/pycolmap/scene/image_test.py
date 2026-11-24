# SPDX-License-Identifier: BSD-3-Clause

import gc
import weakref

import numpy as np
import pytest

import pycolmap


def test_image_default_init() -> None:
    image = pycolmap.Image()
    assert image is not None


def test_image_image_id_readwrite() -> None:
    image = pycolmap.Image()
    image.image_id = 10
    assert image.image_id == 10


def test_image_camera_id_readwrite() -> None:
    image = pycolmap.Image()
    image.camera_id = 5
    assert image.camera_id == 5


def test_image_frame_id_readwrite() -> None:
    image = pycolmap.Image()
    image.frame_id = 3
    assert image.frame_id == 3


def test_image_name_readwrite() -> None:
    image = pycolmap.Image()
    image.name = "test_image.jpg"
    assert image.name == "test_image.jpg"


def test_image_points2d_readwrite() -> None:
    image = pycolmap.Image()
    point_list = pycolmap.Point2DList()
    point_list.append(pycolmap.Point2D(xy=np.array([1.0, 2.0])))
    point_list.append(pycolmap.Point2D(xy=np.array([3.0, 4.0])))
    image.points2D = point_list
    assert len(image.points2D) == 2


def test_image_data_id_readonly() -> None:
    image = pycolmap.Image()
    data_id = image.data_id
    assert data_id is not None


def test_image_has_camera_id() -> None:
    image = pycolmap.Image()
    assert not image.has_camera_id()
    image.camera_id = 1
    assert image.has_camera_id()


def test_image_has_frame_id() -> None:
    image = pycolmap.Image()
    assert not image.has_frame_id()
    image.frame_id = 1
    assert image.has_frame_id()


def test_image_num_points2d() -> None:
    image = pycolmap.Image()
    assert image.num_points2D() == 0
    point_list = pycolmap.Point2DList()
    point_list.append(pycolmap.Point2D(xy=np.array([1.0, 2.0])))
    image.points2D = point_list
    assert image.num_points2D() == 1


def test_image_num_points3d_readonly() -> None:
    image = pycolmap.Image()
    assert image.num_points3D == 0


def test_image_has_pose_readonly() -> None:
    image = pycolmap.Image()
    assert isinstance(image.has_pose, bool)


def test_image_map_empty() -> None:
    image_map = pycolmap.ImageMap()
    assert len(image_map) == 0


def test_image_map_insert_and_access() -> None:
    image_map = pycolmap.ImageMap()
    image = pycolmap.Image()
    image.image_id = 1
    image.name = "test.jpg"
    image_map[1] = image
    assert len(image_map) == 1
    assert image_map[1].name == "test.jpg"


@pytest.mark.parametrize(
    ("owner_type", "property_name", "id_name"),
    [
        ("image", "camera", "camera_id"),
        ("image", "frame", "frame_id"),
        ("frame", "rig", "rig_id"),
    ],
)
def test_scene_pointer_property_keeps_reconstruction_alive(
    owner_type: str, property_name: str, id_name: str
) -> None:
    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera_with_trivial_rig(
        pycolmap.Camera(
            camera_id=1,
            model="SIMPLE_PINHOLE",
            width=10,
            height=10,
            params=[10.0, 5.0, 5.0],
        )
    )
    reconstruction.add_image_with_trivial_frame(
        pycolmap.Image(name="image.jpg", image_id=1, camera_id=1)
    )
    owner = getattr(reconstruction, owner_type)(1)
    borrowed = getattr(owner, property_name)
    reconstruction_ref = weakref.ref(reconstruction)
    del owner, reconstruction
    gc.collect()

    assert reconstruction_ref() is not None
    assert getattr(borrowed, id_name) == 1
    assert borrowed is getattr(
        getattr(reconstruction_ref(), owner_type)(1), property_name
    )

    del borrowed
    gc.collect()
    assert reconstruction_ref() is None


def test_unset_scene_pointer_properties() -> None:
    assert pycolmap.Image().camera is None
    assert pycolmap.Image().frame is None
    assert pycolmap.Frame().rig is None
