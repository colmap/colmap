# SPDX-License-Identifier: BSD-3-Clause

# Equivalent tests to src/colmap/controllers/incremental_pipeline_test.cc

from pathlib import Path

import custom_incremental_pipeline
import numpy as np

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


def _set_pose_priors_with_shifted_anchor(
    database: pycolmap.Database,
    shifted_image_id: int,
    shift: np.ndarray,
    metric_from_synth: pycolmap.Sim3d | None = None,
) -> np.ndarray:
    shifted_prior_position = np.zeros(3)
    for prior in database.read_all_pose_priors():
        if metric_from_synth is not None:
            prior.position = metric_from_synth * prior.position
        if prior.corr_data_id.id == shifted_image_id:
            prior.position = prior.position + shift
            prior.position_covariance = 1e-12 * np.eye(3)
            shifted_prior_position = prior.position.copy()
        else:
            prior.position_covariance = 1e-6 * np.eye(3)
        database.update_pose_prior(prior)
    return shifted_prior_position


def test_with_pose_priors(tmp_path: Path) -> None:
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
        synthetic_dataset_options.num_points3D = 60
        synthetic_dataset_options.prior_position = True
        gt_reconstruction = pycolmap.synthesize_dataset(
            synthetic_dataset_options, database
        )
        metric_from_synth = pycolmap.Sim3d(
            3.0, pycolmap.Rotation3d(), [10.0, -5.0, 2.0]
        )
        gt_reconstruction.transform(metric_from_synth)
        shifted_image_id = next(
            iter(
                gt_reconstruction.frame(
                    gt_reconstruction.reg_frame_ids()[-1]
                ).image_ids
            )
        ).id
        shifted_prior_position = _set_pose_priors_with_shifted_anchor(
            database,
            shifted_image_id,
            np.array([0.01, 0.0, 0.0]),
            metric_from_synth=metric_from_synth,
        )

    options = create_test_options()
    options.use_prior_position = True
    custom_incremental_pipeline.main(
        database_path=database_path,
        image_path=image_path,
        output_path=output_path,
        options=options,
    )

    computed = pycolmap.Reconstruction(output_path / "0")
    assert computed.num_reg_images() == gt_reconstruction.num_reg_images()
    shifted_err = float(
        np.linalg.norm(
            computed.image(shifted_image_id).projection_center()
            - shifted_prior_position
        )
    )
    assert shifted_err < 1e-3
    for image_id, gt_image in gt_reconstruction.images.items():
        err = float(
            np.linalg.norm(
                computed.image(image_id).projection_center()
                - gt_image.projection_center()
            )
        )
        assert err < 2e-2


def test_continue_unaligned_with_fixed_frames_and_priors(
    tmp_path: Path,
) -> None:
    pycolmap.set_random_seed(0)

    database_path = tmp_path / "database.db"
    image_path = tmp_path / "images"
    image_path.mkdir()
    input_path = tmp_path / "input"
    input_path.mkdir()
    output_path = tmp_path / "sparse"
    output_path.mkdir()

    with pycolmap.Database.open(database_path) as database:
        synthetic_dataset_options = pycolmap.SyntheticDatasetOptions()
        synthetic_dataset_options.num_rigs = 1
        synthetic_dataset_options.num_cameras_per_rig = 1
        synthetic_dataset_options.num_frames_per_rig = 6
        synthetic_dataset_options.num_points3D = 80
        synthetic_dataset_options.prior_position = True
        gt_reconstruction = pycolmap.synthesize_dataset(
            synthetic_dataset_options, database
        )

        frame_ids = gt_reconstruction.reg_frame_ids()
        assert len(frame_ids) == 6
        shifted_image_id = next(
            iter(gt_reconstruction.frame(frame_ids[-1]).image_ids)
        ).id
        shifted_prior_position = _set_pose_priors_with_shifted_anchor(
            database, shifted_image_id, np.array([0.01, 0.0, 0.0])
        )

    initial_reconstruction = pycolmap.Reconstruction(gt_reconstruction)
    initial_reconstruction.deregister_frame(frame_ids[4])
    initial_reconstruction.deregister_frame(frame_ids[5])
    initial_reconstruction.transform(
        pycolmap.Sim3d(
            2.5,
            pycolmap.Rotation3d(
                np.array([0.0, 0.0, np.sin(0.25), np.cos(0.25)])
            ),
            [100.0, -50.0, 20.0],
        )
    )
    initial_reconstruction.write(input_path)

    options = create_test_options()
    options.fix_existing_frames = True
    options.use_prior_position = True
    options.ba_global_frames_freq = 1
    options.random_seed = 0

    custom_incremental_pipeline.main(
        database_path=database_path,
        image_path=image_path,
        input_path=input_path,
        output_path=output_path,
        options=options,
    )

    computed = pycolmap.Reconstruction(output_path / "0")
    assert computed.num_reg_frames() == 6
    shifted_err = float(
        np.linalg.norm(
            computed.image(shifted_image_id).projection_center()
            - shifted_prior_position
        )
    )
    assert shifted_err < 1e-3
    for image_id, gt_image in gt_reconstruction.images.items():
        err = float(
            np.linalg.norm(
                computed.image(image_id).projection_center()
                - gt_image.projection_center()
            )
        )
        assert err < 2e-2
