import pytest

import pycolmap


def test_rotation_weight_type_enum():
    assert {m.name: int(m) for m in pycolmap.RotationWeightType} == {
        "GEMAN_MCCLURE": 0,
        "HALF_NORM": 1,
    }


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


@pytest.mark.parametrize(
    "name",
    [
        "run_rotation_averaging",
        "run_gravity_refinement",
        "run_global_positioning",
    ],
)
def test_public_api_callable(name):
    assert callable(getattr(pycolmap, name))
