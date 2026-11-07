# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest

from .eth3d import (
    DatasetETH3DCentered,
    DatasetETH3DDistorted,
    DatasetETH3DUndistorted,
    _DatasetETH3D,
)


@pytest.mark.parametrize(
    ("dataset_class", "calibration", "single_camera"),
    [
        (DatasetETH3DUndistorted, "dslr_calibration_undistorted", False),
        (DatasetETH3DDistorted, "dslr_calibration_jpg", False),
        (DatasetETH3DCentered, "dslr_calibration_undistorted", True),
    ],
)
def test_list_scenes_single_camera(
    tmp_path: Path,
    dataset_class: type[_DatasetETH3D],
    calibration: str,
    single_camera: bool,
) -> None:
    scene_path = tmp_path / "data" / dataset_class.dataset_name / "dslr" / "s"
    (scene_path / calibration).mkdir(parents=True)
    (scene_path / "images" / "dslr_images").mkdir(parents=True)
    (scene_path / "images" / "dslr_images" / "a.JPG").touch()
    dataset = dataset_class(
        data_path=tmp_path / "data",
        categories=[],
        scenes=[],
        run_path=tmp_path / "runs",
        run_name="run",
    )

    (scene_info,) = dataset.list_scenes()

    assert scene_info.num_images == 1
    assert scene_info.colmap_extra_args is not None
    assert ("--single_camera" in scene_info.colmap_extra_args) == single_camera
