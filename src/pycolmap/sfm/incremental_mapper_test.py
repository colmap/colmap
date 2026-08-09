from pathlib import Path

import pycolmap


def test_image_selection_method_max_visible_points_num():
    assert pycolmap.ImageSelectionMethod.MAX_VISIBLE_POINTS_NUM is not None


def test_image_selection_method_max_visible_points_ratio():
    assert pycolmap.ImageSelectionMethod.MAX_VISIBLE_POINTS_RATIO is not None


def test_image_selection_method_min_uncertainty():
    assert pycolmap.ImageSelectionMethod.MIN_UNCERTAINTY is not None


def test_incremental_mapper_options_init():
    options = pycolmap.IncrementalMapperOptions()
    assert options is not None


def test_incremental_mapper_options_check():
    options = pycolmap.IncrementalMapperOptions()
    assert options.check()


def test_local_bundle_adjustment_report_init():
    report = pycolmap.LocalBundleAdjustmentReport()
    assert report is not None


def test_local_bundle_adjustment_report_num_merged_observations():
    report = pycolmap.LocalBundleAdjustmentReport()
    report.num_merged_observations = 5
    assert report.num_merged_observations == 5


def test_local_bundle_adjustment_report_num_completed_observations():
    report = pycolmap.LocalBundleAdjustmentReport()
    report.num_completed_observations = 10
    assert report.num_completed_observations == 10


def test_local_bundle_adjustment_report_num_filtered_observations():
    report = pycolmap.LocalBundleAdjustmentReport()
    report.num_filtered_observations = 3
    assert report.num_filtered_observations == 3


def test_local_bundle_adjustment_report_num_adjusted_observations():
    report = pycolmap.LocalBundleAdjustmentReport()
    report.num_adjusted_observations = 7
    assert report.num_adjusted_observations == 7


def test_incremental_pipeline_callback_initial_image_pair():
    assert (
        pycolmap.IncrementalPipelineCallback.INITIAL_IMAGE_PAIR_REG_CALLBACK
        is not None
    )


def test_incremental_pipeline_callback_next_image():
    assert (
        pycolmap.IncrementalPipelineCallback.NEXT_IMAGE_REG_CALLBACK is not None
    )


def test_incremental_pipeline_callback_last_image():
    assert (
        pycolmap.IncrementalPipelineCallback.LAST_IMAGE_REG_CALLBACK is not None
    )


def test_incremental_pipeline_status_success():
    assert pycolmap.IncrementalPipelineStatus.SUCCESS is not None


def test_incremental_pipeline_status_interrupted():
    assert pycolmap.IncrementalPipelineStatus.INTERRUPTED is not None


def test_incremental_pipeline_status_continue():
    assert pycolmap.IncrementalPipelineStatus.CONTINUE is not None


def test_incremental_pipeline_status_stop():
    assert pycolmap.IncrementalPipelineStatus.STOP is not None


def test_incremental_pipeline_status_no_initial_pair():
    assert pycolmap.IncrementalPipelineStatus.NO_INITIAL_PAIR is not None


def test_incremental_pipeline_status_bad_initial_pair():
    assert pycolmap.IncrementalPipelineStatus.BAD_INITIAL_PAIR is not None


def test_incremental_pipeline_options_init():
    options = pycolmap.IncrementalPipelineOptions()
    assert options is not None


def test_incremental_pipeline_options_min_num_matches():
    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = 20
    assert options.min_num_matches == 20


def test_incremental_pipeline_options_check():
    options = pycolmap.IncrementalPipelineOptions()
    assert options.check()


def test_incremental_pipeline_options_is_initial_pair_provided():
    options = pycolmap.IncrementalPipelineOptions()
    result = options.is_initial_pair_provided()
    assert isinstance(result, bool)


def test_incremental_pipeline_options_get_mapper():
    options = pycolmap.IncrementalPipelineOptions()
    mapper_options = options.get_mapper()
    assert mapper_options is not None


def test_incremental_pipeline_options_get_triangulation():
    options = pycolmap.IncrementalPipelineOptions()
    triangulation_options = options.get_triangulation()
    assert triangulation_options is not None


def test_incremental_pipeline_options_get_local_bundle_adjustment():
    options = pycolmap.IncrementalPipelineOptions()
    bundle_adjustment_options = options.get_local_bundle_adjustment()
    assert bundle_adjustment_options is not None


def test_incremental_pipeline_options_get_global_bundle_adjustment():
    options = pycolmap.IncrementalPipelineOptions()
    bundle_adjustment_options = options.get_global_bundle_adjustment()
    assert bundle_adjustment_options is not None


def test_incremental_pipeline_options_ba_local_backend_readwrite():
    options = pycolmap.IncrementalPipelineOptions()
    options.ba_local_backend = pycolmap.BundleAdjustmentBackend.CERES
    assert options.ba_local_backend == pycolmap.BundleAdjustmentBackend.CERES
    options.ba_local_backend = pycolmap.BundleAdjustmentBackend.CASPAR
    assert options.ba_local_backend == pycolmap.BundleAdjustmentBackend.CASPAR


def test_incremental_pipeline_options_ba_global_backend_readwrite():
    options = pycolmap.IncrementalPipelineOptions()
    options.ba_global_backend = pycolmap.BundleAdjustmentBackend.CERES
    assert options.ba_global_backend == pycolmap.BundleAdjustmentBackend.CERES
    options.ba_global_backend = pycolmap.BundleAdjustmentBackend.CASPAR
    assert options.ba_global_backend == pycolmap.BundleAdjustmentBackend.CASPAR


def test_incremental_pipeline_run(tmp_path: Path):
    pycolmap.set_random_seed(0)

    database_path = tmp_path / "database.db"
    with pycolmap.Database.open(database_path) as database:
        synthetic_dataset_options = pycolmap.SyntheticDatasetOptions()
        synthetic_dataset_options.num_rigs = 2
        synthetic_dataset_options.num_cameras_per_rig = 1
        synthetic_dataset_options.num_frames_per_rig = 7
        synthetic_dataset_options.num_points3D = 50
        synthetic_dataset_options.camera_has_prior_focal_length = False
        gt_reconstruction = pycolmap.synthesize_dataset(
            synthetic_dataset_options, database
        )

        reconstruction_manager = pycolmap.ReconstructionManager()
        pipeline = pycolmap.IncrementalPipeline(
            pycolmap.IncrementalPipelineOptions(),
            database,
            reconstruction_manager,
        )
        pipeline.run()

    assert reconstruction_manager.size() == 1
    reconstruction = reconstruction_manager.get(0)
    assert reconstruction.num_reg_images() == gt_reconstruction.num_reg_images()
    result = pycolmap.compare_reconstructions(
        reconstruction,
        gt_reconstruction,
        alignment_error="proj_center",
        max_proj_center_error=1e-4,
    )
    assert result is not None
    for error in result["errors"]:
        assert error.rotation_error_deg < 1e-2
        assert error.proj_center_error < 1e-4
