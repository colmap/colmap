# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest

from .eth3d import DatasetETH3DUndistorted


def create_dataset(data_path: Path) -> DatasetETH3DUndistorted:
    return DatasetETH3DUndistorted(
        data_path=data_path,
        categories=[],
        scenes=[],
        run_path=data_path.parent / "runs",
        run_name="run",
    )


def test_list_scenes_rig_image_prefixes(tmp_path: Path) -> None:
    scene_path = tmp_path / "data" / "eth3d-undistorted" / "rig" / "s"
    (scene_path / "rig_calibration_undistorted").mkdir(parents=True)
    for camera in ["images_rig_cam5", "images_rig_cam4"]:
        (scene_path / "images" / camera).mkdir(parents=True)
        (scene_path / "images" / camera / "0.png").touch()

    (scene_info,) = create_dataset(tmp_path / "data").list_scenes()

    assert scene_info.rig_image_prefixes == [
        "images_rig_cam4/",
        "images_rig_cam5/",
    ]


def test_list_scenes_rejects_old_eth3d_folder(tmp_path: Path) -> None:
    (tmp_path / "data" / "eth3d" / "rig").mkdir(parents=True)
    (tmp_path / "data" / "eth3d-undistorted" / "dslr").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="rmdir eth3d"):
        create_dataset(tmp_path / "data").list_scenes()
