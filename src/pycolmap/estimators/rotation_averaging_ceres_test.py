# SPDX-License-Identifier: BSD-3-Clause

import gc
import subprocess
import sys

import pytest

import pycolmap


def test_ceres_rotation_averager_adds_residual_and_solves() -> None:
    pytest.importorskip("pyceres")
    subprocess.run([sys.executable, __file__], check=True, timeout=30)


def _run_averager() -> None:
    import pyceres

    class PythonLoss(pyceres.LossFunction):
        def Evaluate(self, squared_norm, rho):
            rho[:] = [squared_norm, 1.0, 0.0]

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
    options.solver_options.num_threads = 2
    averager = pycolmap.create_default_ceres_rotation_averager(
        options, pose_graph, reconstruction
    )
    assert averager.problem.num_residual_blocks() == 1
    loss = PythonLoss()
    for _ in range(100):
        averager.add_relative_rotation_residual(
            *image_ids, pycolmap.Rigid3d().rotation, loss
        )
    assert averager.problem.num_residual_blocks() == 101
    del loss
    del reconstruction
    gc.collect()
    assert averager.solve().IsSolutionUsable()


if __name__ == "__main__":
    _run_averager()
