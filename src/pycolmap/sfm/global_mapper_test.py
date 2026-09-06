from pathlib import Path

import pycolmap


def test_global_mapper_options_init() -> None:
    options = pycolmap.GlobalMapperOptions()
    assert options is not None


def test_global_mapper_options_image_path() -> None:
    options = pycolmap.GlobalMapperOptions()
    options.image_path = "/tmp/images"
    assert options.image_path == Path("/tmp/images")


def test_global_mapper_options_ba_gpu_index() -> None:
    options = pycolmap.GlobalMapperOptions()
    options.ba_gpu_index = "0"
    assert options.ba_gpu_index == "0"


def test_global_mapper_options_get_rotation_averaging() -> None:
    options = pycolmap.GlobalMapperOptions()
    assert options.get_rotation_averaging() is not None


def test_global_mapper_options_get_global_positioning() -> None:
    options = pycolmap.GlobalMapperOptions()
    assert options.get_global_positioning() is not None


def test_global_mapper_options_get_bundle_adjustment() -> None:
    options = pycolmap.GlobalMapperOptions()
    assert options.get_bundle_adjustment() is not None


def test_global_mapper_options_get_retriangulation() -> None:
    options = pycolmap.GlobalMapperOptions()
    assert options.get_retriangulation() is not None


def test_global_pipeline_options_init() -> None:
    options = pycolmap.GlobalPipelineOptions()
    assert options is not None


def test_global_pipeline_options_nested_options() -> None:
    options = pycolmap.GlobalPipelineOptions()
    options.mapper.min_tri_angle_deg = 2.0
    assert options.mapper.min_tri_angle_deg == 2.0


def test_global_pipeline_options_from_dict() -> None:
    options = pycolmap.GlobalPipelineOptions(
        min_num_matches=20, mapper={"ba_num_iterations": 5}
    )
    assert options.min_num_matches == 20
    assert options.mapper.ba_num_iterations == 5


def test_global_pipeline_callback_model_update() -> None:
    assert pycolmap.GlobalPipelineCallback.MODEL_UPDATE_CALLBACK is not None


def test_global_pipeline_run(tmp_path: Path) -> None:
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

    # Global mapping requires calibrated two-view geometries.
    assert pycolmap.calibrate_view_graph(database_path)

    with pycolmap.Database.open(database_path) as database:
        num_model_updates = 0

        def on_model_update() -> None:
            nonlocal num_model_updates
            num_model_updates += 1

        reconstruction_manager = pycolmap.ReconstructionManager()
        pipeline = pycolmap.GlobalPipeline(
            pycolmap.GlobalPipelineOptions(),
            database,
            reconstruction_manager,
        )
        pipeline.add_callback(
            pycolmap.GlobalPipelineCallback.MODEL_UPDATE_CALLBACK,
            on_model_update,
        )
        pipeline.run()

    assert num_model_updates > 0
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
