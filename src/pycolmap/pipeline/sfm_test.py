import pycolmap


def test_view_graph_calibration_options_init():
    options = pycolmap.ViewGraphCalibrationOptions()
    assert options is not None


def test_global_mapper_options_init():
    options = pycolmap.GlobalMapperOptions()
    assert options is not None


def test_global_pipeline_options_init():
    options = pycolmap.GlobalPipelineOptions()
    assert options is not None


def test_global_pipeline_options_min_num_matches():
    options = pycolmap.GlobalPipelineOptions()
    options.min_num_matches = 20
    assert options.min_num_matches == 20


def test_incremental_mapping_callable():
    assert callable(pycolmap.incremental_mapping)


def test_global_mapping_callable():
    assert callable(pycolmap.global_mapping)


def test_triangulate_points_callable():
    assert callable(pycolmap.triangulate_points)


def test_calibrate_view_graph_callable():
    assert callable(pycolmap.calibrate_view_graph)


def test_bundle_adjustment_callable():
    assert callable(pycolmap.bundle_adjustment)
