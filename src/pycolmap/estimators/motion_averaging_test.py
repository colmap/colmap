import gc

import pyceres
import pytest

import pycolmap


def test_rotation_weight_type_enum() -> None:
    assert {
        k: int(v) for k, v in pycolmap.RotationWeightType.__members__.items()
    } == {
        "GEMAN_MCCLURE": 0,
        "HALF_NORM": 1,
    }


def test_rotation_averaging_reweighting_enum() -> None:
    assert {
        k: int(v)
        for k, v in pycolmap.RotationAveragingReweighting.__members__.items()
    } == {
        "UNIFORM": 0,
        "INLIER_MATCH_COUNT": 1,
    }


def test_rotation_estimator_options_default_init() -> None:
    options = pycolmap.RotationEstimatorOptions()
    assert options is not None
    assert options.reweighting == pycolmap.RotationAveragingReweighting.UNIFORM


def test_rotation_estimator_options_reweighting_readwrite() -> None:
    options = pycolmap.RotationEstimatorOptions()
    options.reweighting = (
        pycolmap.RotationAveragingReweighting.INLIER_MATCH_COUNT
    )
    assert (
        options.reweighting
        == pycolmap.RotationAveragingReweighting.INLIER_MATCH_COUNT
    )
    options.reweighting = "UNIFORM"  # type: ignore[assignment]
    assert options.reweighting == pycolmap.RotationAveragingReweighting.UNIFORM


def test_gravity_refiner_options_default_init() -> None:
    options = pycolmap.GravityRefinerOptions()
    assert options is not None


def test_gravity_refiner_options_max_outlier_ratio_readwrite() -> None:
    options = pycolmap.GravityRefinerOptions()
    assert isinstance(options.max_outlier_ratio, float)
    options.max_outlier_ratio = 0.5
    assert options.max_outlier_ratio == 0.5


def test_gravity_refiner_options_max_gravity_error_readwrite() -> None:
    options = pycolmap.GravityRefinerOptions()
    assert isinstance(options.max_gravity_error, float)
    options.max_gravity_error = 10.0
    assert options.max_gravity_error == 10.0


def test_gravity_refiner_options_min_num_neighbors_readwrite() -> None:
    options = pycolmap.GravityRefinerOptions()
    assert isinstance(options.min_num_neighbors, int)
    options.min_num_neighbors = 5
    assert options.min_num_neighbors == 5


def test_global_positioner_options_default_init() -> None:
    options = pycolmap.GlobalPositionerOptions()
    assert options is not None


def test_global_positioner_retains_custom_loss():
    dataset_options = pycolmap.SyntheticDatasetOptions()
    dataset_options.num_rigs = 1
    dataset_options.num_cameras_per_rig = 1
    dataset_options.num_frames_per_rig = 4
    dataset_options.num_points3D = 30
    reconstruction = pycolmap.synthesize_dataset(dataset_options)

    options = pycolmap.GlobalPositionerOptions()
    options.use_gpu = False
    options.random_seed = 42
    loss = pyceres.CauchyLoss(0.1)
    owner = pycolmap.create_default_global_positioner(
        options,
        pycolmap.PoseGraph(),
        reconstruction,
        loss_function=loss,
    )
    del loss
    gc.collect()
    assert owner.problem.num_residual_blocks() > 0
    assert owner.solve().IsSolutionUsable()


@pytest.mark.parametrize(
    "name",
    [
        "run_rotation_averaging",
        "run_gravity_refinement",
        "run_global_positioning",
    ],
)
def test_public_api_callable(name: str) -> None:
    assert callable(getattr(pycolmap, name))
