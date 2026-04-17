import pycolmap


def test_rotation_weight_type_geman_mcclure():
    assert pycolmap.RotationWeightType.GEMAN_MCCLURE is not None


def test_rotation_weight_type_half_norm():
    assert pycolmap.RotationWeightType.HALF_NORM is not None


def test_rotation_estimator_options_default_init():
    options = pycolmap.RotationEstimatorOptions()
    assert options is not None


def test_gravity_refiner_options_default_init():
    options = pycolmap.GravityRefinerOptions()
    assert options is not None


def test_gravity_refiner_options_max_outlier_ratio_readwrite():
    options = pycolmap.GravityRefinerOptions()
    assert isinstance(options.max_outlier_ratio, float)
    options.max_outlier_ratio = 0.5
    assert options.max_outlier_ratio == 0.5


def test_gravity_refiner_options_max_gravity_error_readwrite():
    options = pycolmap.GravityRefinerOptions()
    assert isinstance(options.max_gravity_error, float)
    options.max_gravity_error = 10.0
    assert options.max_gravity_error == 10.0


def test_gravity_refiner_options_min_num_neighbors_readwrite():
    options = pycolmap.GravityRefinerOptions()
    assert isinstance(options.min_num_neighbors, int)
    options.min_num_neighbors = 5
    assert options.min_num_neighbors == 5


def test_global_positioner_options_default_init():
    options = pycolmap.GlobalPositionerOptions()
    assert options is not None


def test_run_rotation_averaging_is_callable():
    assert callable(pycolmap.run_rotation_averaging)


def test_run_gravity_refinement_is_callable():
    assert callable(pycolmap.run_gravity_refinement)


def test_run_global_positioning_is_callable():
    assert callable(pycolmap.run_global_positioning)
