# SPDX-License-Identifier: BSD-3-Clause

import copy
import gc
import weakref
from collections.abc import Callable

import pytest

import pycolmap


def test_incremental_triangulator_options_init() -> None:
    options = pycolmap.IncrementalTriangulatorOptions()
    assert options is not None


def test_incremental_triangulator_options_max_transitivity() -> None:
    options = pycolmap.IncrementalTriangulatorOptions()
    options.max_transitivity = 2
    assert options.max_transitivity == 2


def test_incremental_triangulator_options_create_max_angle_error() -> None:
    options = pycolmap.IncrementalTriangulatorOptions()
    options.create_max_angle_error = 3.0
    assert options.create_max_angle_error == 3.0


def test_incremental_triangulator_options_check() -> None:
    options = pycolmap.IncrementalTriangulatorOptions()
    assert options.check()


def test_incremental_triangulator_class_exists() -> None:
    assert hasattr(pycolmap, "IncrementalTriangulator")


@pytest.mark.parametrize("copy_fn", [copy.copy, copy.deepcopy])
def test_incremental_triangulator_copy_keeps_reconstruction_alive(
    simple_camera: pycolmap.Camera,
    copy_fn: Callable[
        [pycolmap.IncrementalTriangulator], pycolmap.IncrementalTriangulator
    ],
) -> None:
    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera_with_trivial_rig(simple_camera)
    reconstruction.add_image_with_trivial_frame(
        pycolmap.Image(name="image.jpg", image_id=1, camera_id=1)
    )
    graph = pycolmap.CorrespondenceGraph()
    graph.add_image(1, 0)
    triangulator = pycolmap.IncrementalTriangulator(graph, reconstruction)
    copied = copy_fn(triangulator)
    reconstruction_ref = weakref.ref(reconstruction)
    del triangulator, reconstruction
    gc.collect()

    assert reconstruction_ref() is not None
    # The image has no pose, so triangulation returns without creating points.
    assert (
        copied.triangulate_image(pycolmap.IncrementalTriangulatorOptions(), 1)
        == 0
    )

    del copied
    gc.collect()
    assert reconstruction_ref() is None
