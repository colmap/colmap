# SPDX-License-Identifier: BSD-3-Clause

import copy
import gc
import weakref
from collections.abc import Callable

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


@pytest.mark.parametrize("kind", ["image", "frame"])
@pytest.mark.parametrize("copy_fn", [copy.copy, copy.deepcopy])
def test_scene_copy_keeps_reconstruction_alive(
    kind: str,
    copy_fn: Callable[
        [pycolmap.Image | pycolmap.Frame], pycolmap.Image | pycolmap.Frame
    ],
    simple_camera: pycolmap.Camera,
) -> None:
    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera_with_trivial_rig(simple_camera)
    reconstruction.add_image_with_trivial_frame(
        pycolmap.Image(name="frame.jpg", camera_id=1, image_id=1)
    )
    pose = pycolmap.Rigid3d(translation=np.array([1.0, 2.0, 3.0]))
    reconstruction.frame(1).rig_from_world = pose
    original = (
        reconstruction.image(1) if kind == "image" else reconstruction.frame(1)
    )
    duplicate = copy_fn(original)
    assert duplicate is not original
    owner_ref = weakref.ref(reconstruction)
    del original, reconstruction
    gc.collect()
    assert owner_ref() is not None

    if isinstance(duplicate, pycolmap.Image):
        assert duplicate.has_pose
        np.testing.assert_allclose(
            duplicate.cam_from_world().matrix(), pose.matrix()
        )
        projection = duplicate.project_point(np.array([0.0, 0.0, 1.0]))
        assert projection is not None
        np.testing.assert_allclose(projection, [637.0, 634.0])
    else:
        assert duplicate.has_pose()
        np.testing.assert_allclose(
            duplicate.sensor_from_world(
                pycolmap.sensor_t(pycolmap.SensorType.CAMERA, 1)
            ).matrix(),
            pose.matrix(),
        )
    del duplicate
    gc.collect()
    assert owner_ref() is None
