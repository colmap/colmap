import gc

import pytest

import pycolmap


def test_ceres_rotation_averager_adds_residual_and_solves():
    pyceres = pytest.importorskip("pyceres")
    dataset_options = pycolmap.SyntheticDatasetOptions()
    dataset_options.num_rigs = 1
    dataset_options.num_cameras_per_rig = 1
    dataset_options.num_frames_per_rig = 2
    dataset_options.num_points3D = 10
    reconstruction = pycolmap.synthesize_dataset(dataset_options)
    image_ids = sorted(reconstruction.images)

    pose_graph = pycolmap.PoseGraph()
    pose_graph.add_edge(
        *image_ids,
        pycolmap.PoseGraphEdge(
            cam2_from_cam1=pycolmap.Rigid3d(), num_matches=10
        ),
    )
    options = pycolmap.CeresRotationAveragerOptions()
    averager = pycolmap.create_default_ceres_rotation_averager(
        options, pose_graph, reconstruction
    )
    assert averager.problem.num_residual_blocks() == 1
    loss = pyceres.CauchyLoss(0.05)
    averager.add_relative_rotation_residual(
        *image_ids, pycolmap.Rigid3d().rotation, loss
    )
    assert averager.problem.num_residual_blocks() == 2
    del loss
    del reconstruction
    gc.collect()
    assert averager.solve().IsSolutionUsable()
