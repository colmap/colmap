# SPDX-License-Identifier: BSD-3-Clause

import gc

import numpy as np
import pytest

import pycolmap


def test_ceres_rotation_averager_adds_residual_and_solves() -> None:
    pyceres = pytest.importorskip("pyceres")

    class PythonLoss(pyceres.LossFunction):
        def Evaluate(self, squared_norm: float, rho: np.ndarray) -> None:
            rho[:] = [squared_norm, 1.0, 0.0]

    dataset_options = pycolmap.SyntheticDatasetOptions()
    dataset_options.num_rigs = 1
    dataset_options.num_cameras_per_rig = 1
    dataset_options.num_frames_per_rig = 2
    dataset_options.num_points3D = 10
    reconstruction = pycolmap.synthesize_dataset(dataset_options)
    image_id1, image_id2 = sorted(reconstruction.images)

    pose_graph = pycolmap.PoseGraph()
    pose_graph.add_edge(
        image_id1,
        image_id2,
        pycolmap.PoseGraphEdge(
            cam2_from_cam1=pycolmap.Rigid3d(), num_matches=10
        ),
    )
    options = pycolmap.RotationEstimatorOptions()
    assert options.reweighting == pycolmap.RotationAveragingReweighting.UNIFORM
    options.reweighting = (
        pycolmap.RotationAveragingReweighting.INLIER_MATCH_COUNT
    )
    options.ceres.solver_options.num_threads = 1
    averager = pycolmap.create_default_ceres_rotation_averager(
        options, pose_graph, reconstruction
    )
    averager.solver_options.num_threads = 2
    assert averager.solver_options.num_threads == 2
    assert averager.problem.num_residual_blocks() == 1
    loss = PythonLoss()
    for _ in range(100):
        averager.add_relative_rotation_residual(
            image_id1, image_id2, pycolmap.Rigid3d().rotation, loss
        )
    assert averager.problem.num_residual_blocks() == 101
    del loss
    del reconstruction
    gc.collect()
    summary = averager.solve()
    assert summary.IsSolutionUsable()
    assert summary.num_threads_given == 2


def test_ceres_rotation_averager_covariance_reweighting() -> None:
    pyceres = pytest.importorskip("pyceres")
    dataset_options = pycolmap.SyntheticDatasetOptions()
    dataset_options.num_rigs = 1
    dataset_options.num_cameras_per_rig = 1
    dataset_options.num_frames_per_rig = 2
    dataset_options.num_points3D = 10
    reconstruction = pycolmap.synthesize_dataset(dataset_options)
    image_id1, image_id2 = sorted(reconstruction.images)
    rotation1 = reconstruction.images[image_id1].cam_from_world().rotation
    rotation2 = reconstruction.images[image_id2].cam_from_world().rotation
    # Residual convention: estimate = prior * Exp(residual).
    residual = np.array([0.01, -0.02, 0.03])
    prior = rotation2 * rotation1.inverse() * pycolmap.Rotation3d(-residual)
    cov = np.diag([1e-4, 4e-4, 9e-4])

    edge = pycolmap.PoseGraphEdge(
        cam2_from_cam1=pycolmap.Rigid3d(prior, np.zeros(3))
    )
    edge.cam2_from_cam1_rotation_cov = cov
    pose_graph = pycolmap.PoseGraph()
    pose_graph.add_edge(image_id1, image_id2, edge)
    options = pycolmap.RotationEstimatorOptions()
    options.reweighting = pycolmap.RotationAveragingReweighting.COVARIANCE
    options.skip_initialization = True
    options.ceres.loss_function_type = pycolmap.LossFunctionType.TRIVIAL
    averager = pycolmap.create_default_ceres_rotation_averager(
        options, pose_graph, reconstruction
    )
    averager.add_relative_rotation_residual(
        image_id1,
        image_id2,
        prior,
        pyceres.TrivialLoss(),
        cam2_from_cam1_cov=cov,
    )
    evaluation = pyceres.EvaluateOptions()
    residuals = np.asarray(averager.problem.evaluate_residuals(evaluation))
    expected = residual @ np.linalg.solve(cov, residual)
    np.testing.assert_allclose(residuals @ residuals, 2 * expected, rtol=1e-6)

    edge.cam2_from_cam1_rotation_cov = None
    pose_graph = pycolmap.PoseGraph()
    pose_graph.add_edge(image_id1, image_id2, edge)
    with pytest.raises(ValueError):
        pycolmap.create_default_ceres_rotation_averager(
            options, pose_graph, reconstruction
        )
