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
