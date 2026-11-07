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
    ("dataset_class", "calibration", "expected_args"),
    [
        (
            DatasetETH3DUndistorted,
            "dslr_calibration_undistorted",
            ["--data_type", "individual"],
        ),
        (
            DatasetETH3DDistorted,
            "dslr_calibration_jpg",
            ["--data_type", "individual"],
        ),
        (
            DatasetETH3DCentered,
            "dslr_calibration_undistorted",
            ["--data_type", "individual", "--single_camera", "1"],
        ),
    ],
)
def test_list_scenes_colmap_extra_args(
    tmp_path: Path,
    dataset_class: type[_DatasetETH3D],
    calibration: str,
    expected_args: list[str],
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

    assert scene_info.colmap_extra_args == expected_args
