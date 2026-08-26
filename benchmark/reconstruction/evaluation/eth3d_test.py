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
from types import SimpleNamespace

import pytest

from .eth3d import DatasetETH3D


def _make_dataset(tmp_path: Path, variant: str = "undistorted") -> DatasetETH3D:
    return DatasetETH3D(
        data_path=tmp_path / "data",
        categories=[],
        scenes=[],
        run_path=tmp_path / "runs",
        run_name="test",
        variant=variant,
    )


def _add_variant(
    tmp_path: Path,
    category: str,
    scene: str,
    image_dir: str,
    calibration_dir: str,
    image_names: list[str],
) -> None:
    scene_path = tmp_path / "data" / "eth3d" / category / scene
    images_path = scene_path / "images" / image_dir
    images_path.mkdir(parents=True)
    for image_name in image_names:
        (images_path / image_name).touch()
    (scene_path / calibration_dir).mkdir()


def test_list_scenes_keeps_coexisting_variants_separate(tmp_path):
    _add_variant(
        tmp_path,
        "dslr",
        "scene",
        "dslr_images_undistorted",
        "dslr_calibration_undistorted",
        ["one.JPG", "two.JPG"],
    )
    _add_variant(
        tmp_path,
        "dslr",
        "scene",
        "dslr_images",
        "dslr_calibration_jpg",
        ["one.JPG", "two.JPG", "three.JPG"],
    )

    undistorted = _make_dataset(tmp_path).list_scenes()[0]
    distorted = _make_dataset(tmp_path, "distorted").list_scenes()[0]

    assert undistorted.num_images == 2
    assert distorted.num_images == 3
    assert undistorted.image_path == distorted.image_path
    assert undistorted.image_path.name == "images"
    assert undistorted.sparse_gt_path.name == "dslr_calibration_undistorted"
    assert distorted.sparse_gt_path.name == "dslr_calibration_jpg"
    assert undistorted.colmap_extra_args[-2:] == [
        "--image_list_path",
        str(undistorted.workspace_path / "image_list.txt"),
    ]


def test_list_scenes_skips_missing_variant_and_raises_when_empty(tmp_path):
    _add_variant(
        tmp_path,
        "rig",
        "scene",
        "rig_images_undistorted",
        "rig_calibration_undistorted",
        ["one.JPG"],
    )

    with pytest.raises(
        RuntimeError, match="download.py --datasets eth3d:distorted"
    ):
        _make_dataset(tmp_path, "distorted").list_scenes()


def test_prepare_scene_writes_marker_and_sorted_gt_image_list(
    tmp_path, monkeypatch
):
    dataset = _make_dataset(tmp_path)
    _add_variant(
        tmp_path,
        "dslr",
        "scene",
        "dslr_images_undistorted",
        "dslr_calibration_undistorted",
        ["one.JPG", "two.JPG"],
    )
    scene_info = dataset.list_scenes()[0]
    images = {
        1: SimpleNamespace(name="dslr_images_undistorted/two.JPG"),
        2: SimpleNamespace(name="dslr_images_undistorted/one.JPG"),
    }
    monkeypatch.setattr(
        "evaluation.eth3d.pycolmap.Reconstruction",
        lambda unused_path: SimpleNamespace(images=images),
    )

    dataset.prepare_scene(scene_info)

    assert (scene_info.workspace_path / "variant.txt").read_text() == (
        "undistorted\n"
    )
    assert (scene_info.workspace_path / "image_list.txt").read_text() == (
        "dslr_images_undistorted/one.JPG\ndslr_images_undistorted/two.JPG\n"
    )


def test_prepare_scene_rejects_cross_variant_workspace_before_loading_gt(
    tmp_path, monkeypatch
):
    dataset = _make_dataset(tmp_path, "distorted")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "variant.txt").write_text("undistorted\n")
    scene_info = SimpleNamespace(workspace_path=workspace)
    monkeypatch.setattr(
        "evaluation.eth3d.pycolmap.Reconstruction",
        lambda unused_path: pytest.fail("GT must not load before marker check"),
    )

    with pytest.raises(RuntimeError, match="fresh --run_name"):
        dataset.prepare_scene(scene_info)


def test_prepare_scene_rejects_unmarked_legacy_state_for_distorted(tmp_path):
    dataset = _make_dataset(tmp_path, "distorted")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "database.db").touch()
    scene_info = SimpleNamespace(workspace_path=workspace)

    with pytest.raises(RuntimeError, match="assumed to be undistorted"):
        dataset.prepare_scene(scene_info)
