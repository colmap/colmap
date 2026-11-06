# SPDX-License-Identifier: BSD-3-Clause

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pycolmap

from .eth3d_slam import SCENES, DatasetETH3DSLAM, associate_poses
from .utils import compute_scene_errors


def test_association_preserves_missing_ground_truth() -> None:
    gt = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1.01, 1, 0, 0, 0, 0, 0, 1],
            [2, 2, 0, 0, 0, 0, 0, 1],
        ]
    )
    poses = associate_poses([0.9, 1, 1.005, 1.01, 1.5, 2, 2.1], gt, 1 / 75)
    assert set(poses) == {1, 2}
    np.testing.assert_allclose(poses[2][1], [0.5, 0, 0])


def test_sfm_interpolation_timespan() -> None:
    gt = np.array([[1, 0, 0, 0, 0, 0, 0, 1], [1.08, 1, 0, 0, 0, 0, 0, 1]])
    poses = associate_poses([1.000049, 1.04, 1.079951], gt, 1 / 11.25)
    assert len(poses) == 3
    np.testing.assert_allclose(poses[1][1], [0.5, 0, 0])
    assert not associate_poses([1.04], gt, 1 / 75)
    assert len(SCENES) == len(set(SCENES)) == 55


@pytest.mark.parametrize("error_type", ["relative", "absolute"])
def test_dataset_prepares_and_evaluates_ground_truth(
    tmp_path: Path, error_type: str
) -> None:
    source = tmp_path / "data" / "eth3d-slam" / "plant_1"
    (source / "rgb").mkdir(parents=True)
    timestamps = [1.0, 1.005, 1.5, 2.0]
    for timestamp in timestamps:
        Image.new("L", (100, 80)).save(source / "rgb" / f"{timestamp}.png")
    (source / "rgb.txt").write_text(
        "".join(f"{t} rgb/{t}.png\n" for t in timestamps)
    )
    np.savetxt(source / "calibration.txt", [90, 95, 49.5, 39.5])
    rotation = pycolmap.Rotation3d(np.array([0.1, 0.2, 0.3]))
    np.savetxt(
        source / "groundtruth.txt",
        [
            [t, x, 0, 0, *rotation.quat]
            for t, x in [(1, 0), (1.01, 1), (2, 2), (2.01, 3)]
        ],
    )
    dataset = DatasetETH3DSLAM(
        data_path=tmp_path / "data",
        categories=[],
        scenes=["plant_1"],
        run_path=tmp_path / "runs",
        run_name="test",
    )
    scene = dataset.list_scenes()[0]
    dataset.prepare_scene(scene)
    gt = pycolmap.Reconstruction(scene.sparse_gt_path)
    assert scene.num_images == 4
    assert gt.num_images() == 3
    assert gt.num_cameras() == 1
    assert scene.images_without_gt == {"1.5.png"}
    np.testing.assert_allclose(gt.cameras[1].params, [90, 95, 50, 40])
    for image_id, x in [(1, 0), (2, 0.5), (4, 2)]:
        image = gt.images[image_id]
        assert image.name == f"{timestamps[image_id - 1]}.png"
        np.testing.assert_allclose(
            image.projection_center(), [x, 0, 0], atol=1e-14
        )
        np.testing.assert_allclose(
            image.cam_from_world().rotation.matrix(), rotation.matrix().T
        )

    estimated = pycolmap.Reconstruction(scene.sparse_gt_path)
    estimated.add_image_with_trivial_frame(
        pycolmap.Image(image_id=3, camera_id=1, name="1.5.png"),
        pycolmap.Rigid3d(),
    )
    args = argparse.Namespace(error_type=error_type)
    errors = compute_scene_errors(args, scene, [estimated], gt, 0.001)
    assert len(errors) == (6 if error_type == "relative" else 3)
    np.testing.assert_allclose(errors, 0, atol=1e-5)

    if error_type == "relative":
        unfiltered_scene = dataclasses.replace(scene, images_without_gt=set())
        errors = compute_scene_errors(
            args, unfiltered_scene, [estimated], gt, 0.001
        )
        assert np.count_nonzero(errors == 180) == 6

    estimated.deregister_frame(estimated.images[2].frame_id)
    estimated.tear_down()
    errors = compute_scene_errors(args, scene, [estimated], gt, 0.001)
    assert np.count_nonzero(errors > 1) == (
        4 if error_type == "relative" else 1
    )

    cached_scene = dataset.list_scenes()[0]
    dataset.prepare_scene(cached_scene)
    assert cached_scene.images_without_gt == {"1.5.png"}
