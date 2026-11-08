# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from .eth3d import DatasetETH3DUndistortedLegacy


def test_list_scenes_rig_image_prefixes(tmp_path: Path) -> None:
    scene_path = tmp_path / "data" / "eth3d" / "rig" / "s"
    (scene_path / "rig_calibration_undistorted").mkdir(parents=True)
    for camera in ["images_rig_cam5", "images_rig_cam4"]:
        (scene_path / "images" / camera).mkdir(parents=True)
        (scene_path / "images" / camera / "0.png").touch()
    dataset = DatasetETH3DUndistortedLegacy(
        data_path=tmp_path / "data",
        categories=[],
        scenes=[],
        run_path=tmp_path / "runs",
        run_name="run",
    )

    (scene_info,) = dataset.list_scenes()

    assert scene_info.rig_image_prefixes == [
        "images_rig_cam4/",
        "images_rig_cam5/",
    ]
