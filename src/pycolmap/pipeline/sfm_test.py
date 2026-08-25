import os
from pathlib import Path

import pytest

import pycolmap

skip_in_cuda_wheel = pytest.mark.skipif(
    os.environ.get("BUILD_CUDA_ENABLED") == "true",
    reason="Covered by the CPU wheel; unsupported by the CUDA builder host",
)


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


@pytest.mark.parametrize(
    "name",
    [
        "incremental_mapping",
        "global_mapping",
        "hierarchical_mapping",
        "triangulate_points",
        "calibrate_view_graph",
        "bundle_adjustment",
    ],
)
def test_public_api_callable(name):
    assert callable(getattr(pycolmap, name))


def test_hierarchical_pipeline_options_init():
    options = pycolmap.HierarchicalPipelineOptions()
    assert options is not None


def test_incremental_mapping(tmp_path: Path):
    pycolmap.set_random_seed(0)

    database_path = tmp_path / "database.db"
    image_path = tmp_path / "images"
    image_path.mkdir()
    output_path = tmp_path / "sparse"
    output_path.mkdir()

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

    reconstructions = pycolmap.incremental_mapping(
        database_path, image_path, output_path
    )

    assert len(reconstructions) == 1
    assert (
        reconstructions[0].num_reg_images()
        == gt_reconstruction.num_reg_images()
    )
    assert (output_path / "0").exists()


@skip_in_cuda_wheel
def test_global_mapping(tmp_path: Path):
    pycolmap.set_random_seed(0)

    database_path = tmp_path / "database.db"
    image_path = tmp_path / "images"
    image_path.mkdir()
    output_path = tmp_path / "sparse"
    output_path.mkdir()

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

    # Global mapping requires calibrated two-view geometries.
    assert pycolmap.calibrate_view_graph(database_path)

    reconstructions = pycolmap.global_mapping(
        database_path, image_path, output_path
    )

    assert len(reconstructions) == 1
    assert (
        reconstructions[0].num_reg_images()
        == gt_reconstruction.num_reg_images()
    )
    assert (output_path / "0").exists()
    result = pycolmap.compare_reconstructions(
        reconstructions[0],
        gt_reconstruction,
        alignment_error="proj_center",
        max_proj_center_error=1e-4,
    )
    assert result is not None
    for error in result["errors"]:
        assert error.rotation_error_deg < 1e-2
        assert error.proj_center_error < 1e-4


@skip_in_cuda_wheel
def test_hierarchical_mapping(tmp_path: Path):
    pycolmap.set_random_seed(0)

    database_path = tmp_path / "database.db"
    image_path = tmp_path / "images"
    image_path.mkdir()
    output_path = tmp_path / "sparse"
    output_path.mkdir()

    with pycolmap.Database.open(database_path) as database:
        synthetic_dataset_options = pycolmap.SyntheticDatasetOptions()
        synthetic_dataset_options.num_rigs = 2
        synthetic_dataset_options.num_cameras_per_rig = 1
        synthetic_dataset_options.num_frames_per_rig = 10
        synthetic_dataset_options.num_points3D = 100
        gt_reconstruction = pycolmap.synthesize_dataset(
            synthetic_dataset_options, database
        )

    options = pycolmap.HierarchicalPipelineOptions()
    options.clustering_options.leaf_max_num_images = 5
    options.clustering_options.image_overlap = 3
    reconstructions = pycolmap.hierarchical_mapping(
        database_path, image_path, output_path, options
    )

    assert len(reconstructions) == 1
    assert (
        reconstructions[0].num_reg_images()
        == gt_reconstruction.num_reg_images()
    )
    assert (output_path / "0").exists()
