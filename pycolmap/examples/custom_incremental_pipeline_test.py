# Equivalent tests to src/colmap/controllers/incremental_pipeline_test.cc

import custom_incremental_pipeline

import pycolmap


def expect_equal_reconstructions(
    gt: pycolmap.Reconstruction,
    computed: pycolmap.Reconstruction,
    max_rotation_error_deg: float,
    max_proj_center_error: float,
    num_obs_tolerance: float,
):
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
    for error in result["errors"]:
        assert error.rotation_error_deg < max_rotation_error_deg
        assert error.proj_center_error < max_proj_center_error


def test_without_noise(tmp_path):
    database_path = tmp_path / "database.db"
    image_path = tmp_path / "images"
    image_path.mkdir()
    output_path = tmp_path / "sparse"
    output_path.mkdir()

    with pycolmap.Database(database_path) as database:
        synthetic_dataset_options = pycolmap.SyntheticDatasetOptions()
        synthetic_dataset_options.num_cameras = 2
        synthetic_dataset_options.num_images = 7
        synthetic_dataset_options.num_points3D = 50
        synthetic_dataset_options.point2D_stddev = 0
        gt_reconstruction = pycolmap.synthesize_dataset(
            synthetic_dataset_options, database
        )

    custom_incremental_pipeline.main(
        database_path=database_path,
        image_path=image_path,
        output_path=output_path,
    )

    expect_equal_reconstructions(
        gt_reconstruction,
        pycolmap.Reconstruction(output_path / "0"),
        max_rotation_error_deg=1e-2,
        max_proj_center_error=1e-4,
        num_obs_tolerance=0,
    )


def test_with_noise(tmp_path):
    database_path = tmp_path / "database.db"
    image_path = tmp_path / "images"
    image_path.mkdir()
    output_path = tmp_path / "sparse"
    output_path.mkdir()

    with pycolmap.Database(database_path) as database:
        synthetic_dataset_options = pycolmap.SyntheticDatasetOptions()
        synthetic_dataset_options.num_cameras = 2
        synthetic_dataset_options.num_images = 7
        synthetic_dataset_options.num_points3D = 100
        synthetic_dataset_options.point2D_stddev = 0.5
        gt_reconstruction = pycolmap.synthesize_dataset(
            synthetic_dataset_options, database
        )

    custom_incremental_pipeline.main(
        database_path=database_path,
        image_path=image_path,
        output_path=output_path,
    )

    expect_equal_reconstructions(
        gt_reconstruction,
        pycolmap.Reconstruction(output_path / "0"),
        max_rotation_error_deg=1e-1,
        max_proj_center_error=1e-1,
        num_obs_tolerance=0.02,
    )


def test_multi_reconstruction(tmp_path):
    database_path = tmp_path / "database.db"
    image_path = tmp_path / "images"
    image_path.mkdir()
    output_path = tmp_path / "sparse"
    output_path.mkdir()

    with pycolmap.Database(database_path) as database:
        synthetic_dataset_options = pycolmap.SyntheticDatasetOptions()
        synthetic_dataset_options.num_cameras = 1
        synthetic_dataset_options.num_images = 5
        synthetic_dataset_options.num_points3D = 50
        synthetic_dataset_options.point2D_stddev = 0
        gt_reconstruction1 = pycolmap.synthesize_dataset(
            synthetic_dataset_options, database
        )
        synthetic_dataset_options.num_images = 4
        gt_reconstruction2 = pycolmap.synthesize_dataset(
            synthetic_dataset_options, database
        )

    options = pycolmap.IncrementalPipelineOptions()
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


def test_chained_matches(tmp_path):
    database_path = tmp_path / "database.db"
    image_path = tmp_path / "images"
    image_path.mkdir()
    output_path = tmp_path / "sparse"
    output_path.mkdir()

    with pycolmap.Database(database_path) as database:
        synthetic_dataset_options = pycolmap.SyntheticDatasetOptions()
        synthetic_dataset_options.num_cameras = 1
        synthetic_dataset_options.num_images = 4
        synthetic_dataset_options.num_points3D = 100
        synthetic_dataset_options.point2D_stddev = 0
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
    )

    expect_equal_reconstructions(
        gt_reconstruction,
        pycolmap.Reconstruction(output_path / "0"),
        max_rotation_error_deg=1e-2,
        max_proj_center_error=1e-4,
        num_obs_tolerance=0,
    )
