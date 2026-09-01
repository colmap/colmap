from pathlib import Path

import pycolmap


def test_scene_clustering_options_init() -> None:
    options = pycolmap.SceneClusteringOptions()
    assert options is not None


def test_scene_clustering_options_check() -> None:
    options = pycolmap.SceneClusteringOptions()
    assert options.check()


def test_scene_clustering_options_leaf_max_num_images() -> None:
    options = pycolmap.SceneClusteringOptions()
    options.leaf_max_num_images = 100
    assert options.leaf_max_num_images == 100


def test_hierarchical_pipeline_options_init() -> None:
    options = pycolmap.HierarchicalPipelineOptions()
    assert options is not None


def test_hierarchical_pipeline_options_check() -> None:
    options = pycolmap.HierarchicalPipelineOptions()
    assert options.check()


def test_hierarchical_pipeline_options_num_workers() -> None:
    options = pycolmap.HierarchicalPipelineOptions()
    options.num_workers = 2
    assert options.num_workers == 2


def test_hierarchical_pipeline_options_nested_options() -> None:
    options = pycolmap.HierarchicalPipelineOptions()
    options.clustering_options.image_overlap = 10
    assert options.clustering_options.image_overlap == 10
    options.incremental_options.min_num_matches = 20
    assert options.incremental_options.min_num_matches == 20


def test_hierarchical_pipeline_options_from_dict() -> None:
    options = pycolmap.HierarchicalPipelineOptions(
        num_workers=3, clustering_options={"leaf_max_num_images": 50}
    )
    assert options.num_workers == 3
    assert options.clustering_options.leaf_max_num_images == 50


def test_hierarchical_pipeline_run(tmp_path: Path) -> None:
    pycolmap.set_random_seed(0)

    database_path = tmp_path / "database.db"
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
        reconstruction_manager = pycolmap.ReconstructionManager()
        pipeline = pycolmap.HierarchicalPipeline(
            options, database, reconstruction_manager
        )
        pipeline.run()

    assert reconstruction_manager.size() == 1
    reconstruction = reconstruction_manager.get(0)
    assert reconstruction.num_reg_images() == gt_reconstruction.num_reg_images()
    assert (
        reconstruction.compute_num_observations()
        >= gt_reconstruction.compute_num_observations()
    )
    result = pycolmap.compare_reconstructions(
        reconstruction,
        gt_reconstruction,
        alignment_error="proj_center",
        max_proj_center_error=1e-4,
    )
    assert result is not None
