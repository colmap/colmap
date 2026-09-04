# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

import pytest

from .eth3d import DatasetETH3DDistorted, DatasetETH3DUndistorted


def _make_scene(
    data_path: Path,
    dataset: str,
    category: str,
    scene: str,
    calibration_dir: str,
    image_dirs: dict[str, int],
) -> None:
    """Creates one ETH3D scene as published, with empty image files."""
    scene_path = data_path / dataset / category / scene
    (scene_path / calibration_dir).mkdir(parents=True)
    for image_dir, num_images in image_dirs.items():
        image_path = scene_path / "images" / image_dir
        image_path.mkdir(parents=True)
        for index in range(num_images):
            (image_path / f"{index}.JPG").touch()


def _make_dslr_scene(data_path: Path, scene: str = "courtyard") -> None:
    _make_scene(
        data_path,
        dataset="eth3d",
        category="dslr",
        scene=scene,
        calibration_dir="dslr_calibration_undistorted",
        image_dirs={"dslr_images_undistorted": 3},
    )


def _make_rig_scene(data_path: Path, scene: str = "delivery_area") -> None:
    # The rig publishes one folder per camera, named images_rig_cam<i>_*.
    _make_scene(
        data_path,
        dataset="eth3d",
        category="rig",
        scene=scene,
        calibration_dir="rig_calibration_undistorted",
        image_dirs={
            f"images_rig_cam{camera}_undistorted": 2 for camera in range(4, 8)
        },
    )


def _make_distorted_scene(data_path: Path, scene: str = "courtyard") -> None:
    _make_scene(
        data_path,
        dataset="eth3d-distorted",
        category="dslr",
        scene=scene,
        calibration_dir="dslr_calibration_jpg",
        image_dirs={"dslr_images": 3},
    )


def _list_scenes(dataset_cls, data_path: Path, categories=(), scenes=()):
    dataset = dataset_cls(
        data_path=data_path,
        categories=list(categories),
        scenes=list(scenes),
        run_path=data_path / "runs",
        run_name="test",
    )
    return dataset.list_scenes()


def test_lists_dslr_scene(tmp_path):
    _make_dslr_scene(tmp_path)

    scene_infos = _list_scenes(DatasetETH3DUndistorted, tmp_path)

    assert len(scene_infos) == 1
    scene_info = scene_infos[0]
    assert scene_info.dataset == "eth3d"
    assert scene_info.category == "dslr"
    assert scene_info.scene == "courtyard"
    assert scene_info.num_images == 3
    assert scene_info.image_path.name == "images"
    assert scene_info.sparse_gt_path.name == "dslr_calibration_undistorted"
    assert "--data_type" in scene_info.colmap_extra_args


def test_lists_rig_scene(tmp_path):
    _make_rig_scene(tmp_path)

    scene_infos = _list_scenes(DatasetETH3DUndistorted, tmp_path)

    assert len(scene_infos) == 1
    scene_info = scene_infos[0]
    assert scene_info.category == "rig"
    # All four per-camera folders contribute, none is matched by name.
    assert scene_info.num_images == 8
    assert scene_info.sparse_gt_path.name == "rig_calibration_undistorted"
    assert scene_info.colmap_extra_args == ["--data_type", "video"]


def test_lists_both_categories(tmp_path):
    _make_dslr_scene(tmp_path)
    _make_rig_scene(tmp_path)

    scene_infos = _list_scenes(DatasetETH3DUndistorted, tmp_path)

    assert {info.category for info in scene_infos} == {"dslr", "rig"}


def test_lists_distorted_scene(tmp_path):
    _make_distorted_scene(tmp_path)

    scene_infos = _list_scenes(DatasetETH3DDistorted, tmp_path)

    assert len(scene_infos) == 1
    assert scene_infos[0].dataset == "eth3d-distorted"
    assert scene_infos[0].sparse_gt_path.name == "dslr_calibration_jpg"


def test_skips_scene_without_calibration(tmp_path):
    _make_dslr_scene(tmp_path)
    calibration_path = (
        tmp_path / "eth3d/dslr/courtyard/dslr_calibration_undistorted"
    )
    calibration_path.rmdir()

    assert _list_scenes(DatasetETH3DUndistorted, tmp_path) == []


def test_filters_by_category_and_scene(tmp_path):
    _make_dslr_scene(tmp_path)
    _make_rig_scene(tmp_path)

    by_category = _list_scenes(
        DatasetETH3DUndistorted, tmp_path, categories=["rig"]
    )
    assert [info.scene for info in by_category] == ["delivery_area"]

    by_scene = _list_scenes(
        DatasetETH3DUndistorted, tmp_path, scenes=["courtyard"]
    )
    assert [info.scene for info in by_scene] == ["courtyard"]

    # An empty result is not an error once the dataset is present on disk.
    assert (
        _list_scenes(DatasetETH3DUndistorted, tmp_path, scenes=["typo"]) == []
    )
    assert (
        _list_scenes(DatasetETH3DUndistorted, tmp_path, categories=["typo"])
        == []
    )


def test_raises_when_not_downloaded(tmp_path):
    with pytest.raises(RuntimeError, match="download.py --datasets eth3d"):
        _list_scenes(DatasetETH3DUndistorted, tmp_path)
