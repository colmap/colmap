# SPDX-License-Identifier: BSD-3-Clause

# Equivalent tests to src/colmap/controllers/incremental_pipeline_test.cc

from pathlib import Path
from typing import Any
from unittest.mock import patch

import custom_incremental_pipeline
import pytest

import pycolmap


def expect_equal_reconstructions(
    gt: pycolmap.Reconstruction,
    computed: pycolmap.Reconstruction,
    max_rotation_error_deg: float,
    max_proj_center_error: float,
    num_obs_tolerance: float,
) -> None:
    assert computed.num_cameras() == gt.num_cameras()
    assert computed.num_images() == gt.num_images()
    assert computed.num_reg_images() == gt.num_reg_images()
    assert (
        computed.compute_num_observations()
        >= (1 - num_obs_tolerance) * gt.compute_num_observations()
    )

    result = pycolmap.compare_reconstructions(
        computed,
        gt,
        alignment_error="proj_center",
        max_proj_center_error=max_proj_center_error,
    )
    assert result is not None
    for error in result["errors"]:
        assert error.rotation_error_deg < max_rotation_error_deg
        assert error.proj_center_error < max_proj_center_error


def create_test_options() -> pycolmap.IncrementalPipelineOptions:
    options = pycolmap.IncrementalPipelineOptions()
    # Use single thread for deterministic behavior.
    options.num_threads = 1
    return options


def test_without_noise(tmp_path: Path) -> None:
    pycolmap.set_random_seed(0)

    database_path = tmp_path / "database.db"
    image_path = tmp_path / "images"
    image_path.mkdir()
    output_path = tmp_path / "sparse"
    output_path.mkdir()

    with pycolmap.Database.open(database_path) as database:
        synthetic_dataset_options = pycolmap.SyntheticDatasetOptions()
        synthetic_dataset_options.num_cameras_per_rig = 2
        synthetic_dataset_options.num_frames_per_rig = 7
        synthetic_dataset_options.num_points3D = 50
        gt_reconstruction = pycolmap.synthesize_dataset(
            synthetic_dataset_options, database
        )

    custom_incremental_pipeline.main(
        database_path=database_path,
        image_path=image_path,
        output_path=output_path,
        options=create_test_options(),
    )

    expect_equal_reconstructions(
        gt_reconstruction,
        pycolmap.Reconstruction(output_path / "0"),
        max_rotation_error_deg=1e-2,
        max_proj_center_error=1e-4,
        num_obs_tolerance=0,
    )


def test_with_noise(tmp_path: Path) -> None:
    pycolmap.set_random_seed(0)

    database_path = tmp_path / "database.db"
    image_path = tmp_path / "images"
    image_path.mkdir()
    output_path = tmp_path / "sparse"
    output_path.mkdir()

    with pycolmap.Database.open(database_path) as database:
        synthetic_dataset_options = pycolmap.SyntheticDatasetOptions()
        synthetic_dataset_options.num_cameras_per_rig = 2
        synthetic_dataset_options.num_frames_per_rig = 7
        synthetic_dataset_options.num_points3D = 100
        gt_reconstruction = pycolmap.synthesize_dataset(
            synthetic_dataset_options, database
        )
        synthetic_noise_options = pycolmap.SyntheticNoiseOptions()
        synthetic_noise_options.point2D_stddev = 0.5
        pycolmap.synthesize_noise(synthetic_noise_options, gt_reconstruction)

    custom_incremental_pipeline.main(
        database_path=database_path,
        image_path=image_path,
        output_path=output_path,
        options=create_test_options(),
    )

    expect_equal_reconstructions(
        gt_reconstruction,
        pycolmap.Reconstruction(output_path / "0"),
        max_rotation_error_deg=1e-1,
        max_proj_center_error=1e-1,
        num_obs_tolerance=0.02,
    )


def test_multi_reconstruction(tmp_path: Path) -> None:
    pycolmap.set_random_seed(0)

    database_path = tmp_path / "database.db"
    image_path = tmp_path / "images"
    image_path.mkdir()
    output_path = tmp_path / "sparse"
    output_path.mkdir()

    with pycolmap.Database.open(database_path) as database:
        synthetic_dataset_options = pycolmap.SyntheticDatasetOptions()
        synthetic_dataset_options.num_cameras_per_rig = 1
        synthetic_dataset_options.num_frames_per_rig = 5
        synthetic_dataset_options.num_points3D = 50
        gt_reconstruction1 = pycolmap.synthesize_dataset(
            synthetic_dataset_options, database
        )
        synthetic_dataset_options.num_frames_per_rig = 4
        gt_reconstruction2 = pycolmap.synthesize_dataset(
            synthetic_dataset_options, database
        )

    options = create_test_options()
    options.min_model_size = 4
    custom_incremental_pipeline.main(
        database_path=database_path,
        image_path=image_path,
        output_path=output_path,
        options=options,
    )

    assert len(list(output_path.iterdir())) == 2
    reconstruction1 = pycolmap.Reconstruction(output_path / "0")
    reconstruction2 = pycolmap.Reconstruction(output_path / "1")
    if reconstruction1 == gt_reconstruction2.num_reg_images():
        reconstruction1, reconstruction2 = reconstruction2, reconstruction1

    expect_equal_reconstructions(
        gt_reconstruction1,
        reconstruction1,
        max_rotation_error_deg=1e-2,
        max_proj_center_error=1e-4,
        num_obs_tolerance=0,
    )
    expect_equal_reconstructions(
        gt_reconstruction2,
        reconstruction2,
        max_rotation_error_deg=1e-2,
        max_proj_center_error=1e-4,
        num_obs_tolerance=0,
    )


def test_progress_bar_counts_registered_images(tmp_path: Path) -> None:
    pycolmap.set_random_seed(0)

    database_path = tmp_path / "database.db"
    image_path = tmp_path / "images"
    image_path.mkdir()
    output_path = tmp_path / "sparse"
    output_path.mkdir()

    with pycolmap.Database.open(database_path) as database:
        synthetic_dataset_options = pycolmap.SyntheticDatasetOptions()
        synthetic_dataset_options.num_rigs = 1
        synthetic_dataset_options.num_cameras_per_rig = 1
        synthetic_dataset_options.num_frames_per_rig = 5
        synthetic_dataset_options.num_points3D = 50
        pycolmap.synthesize_dataset(synthetic_dataset_options, database)
        # Second scene with fewer images than min_model_size.
        synthetic_dataset_options.num_frames_per_rig = 3
        pycolmap.synthesize_dataset(synthetic_dataset_options, database)

    pbars: list[Any] = []
    manager = custom_incremental_pipeline.enlighten.Manager
    create_counter = manager.counter

    def record_counter(*args: Any, **kwargs: Any) -> Any:
        pbars.append(create_counter(*args, **kwargs))
        return pbars[-1]

    options = create_test_options()
    options.min_model_size = 4
    with patch.object(manager, "counter", record_counter):
        reconstructions = custom_incremental_pipeline.main(
            database_path=database_path,
            image_path=image_path,
            output_path=output_path,
            options=options,
        )

    assert len(reconstructions) == 1
    assert reconstructions[0].num_reg_images() == 5
    assert len(pbars) == 1
    assert pbars[0].count == 5


def test_chained_matches(tmp_path: Path) -> None:
    pycolmap.set_random_seed(0)

    database_path = tmp_path / "database.db"
    image_path = tmp_path / "images"
    image_path.mkdir()
    output_path = tmp_path / "sparse"
    output_path.mkdir()

    with pycolmap.Database.open(database_path) as database:
        synthetic_dataset_options = pycolmap.SyntheticDatasetOptions()
        synthetic_dataset_options.num_cameras_per_rig = 1
        synthetic_dataset_options.num_frames_per_rig = 4
        synthetic_dataset_options.num_points3D = 100
        synthetic_dataset_options.match_config = (
            pycolmap.SyntheticDatasetMatchConfig.CHAINED
        )
        gt_reconstruction = pycolmap.synthesize_dataset(
            synthetic_dataset_options, database
        )

    custom_incremental_pipeline.main(
        database_path=database_path,
        image_path=image_path,
        output_path=output_path,
        options=create_test_options(),
    )

    expect_equal_reconstructions(
        gt_reconstruction,
        pycolmap.Reconstruction(output_path / "0"),
        max_rotation_error_deg=1e-2,
        max_proj_center_error=1e-4,
        num_obs_tolerance=0,
    )


def test_write_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        custom_incremental_pipeline.time, "time", lambda: 123.456
    )
    snapshot_root = tmp_path / "snapshots"
    reconstruction = pycolmap.Reconstruction()

    custom_incremental_pipeline.write_snapshot(reconstruction, snapshot_root)

    snapshot = pycolmap.Reconstruction(snapshot_root / "0000123456")
    assert snapshot.num_images() == 0
