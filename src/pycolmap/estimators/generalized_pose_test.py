# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import pycolmap


@pytest.mark.parametrize(
    "name",
    [
        "estimate_generalized_absolute_pose",
        "refine_generalized_absolute_pose",
        "estimate_and_refine_generalized_absolute_pose",
        "estimate_scaled_generalized_absolute_pose",
        "refine_scaled_generalized_absolute_pose",
        "estimate_and_refine_scaled_generalized_absolute_pose",
        "estimate_generalized_relative_pose",
    ],
)
def test_public_api_callable(name: str) -> None:
    assert callable(getattr(pycolmap, name))


def build_tiny_scaled_problem() -> dict:
    """Two-camera rig with exact correspondences for smoke testing."""
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 500, 640, 480
    )
    gt_rig_from_world = pycolmap.Sim3d(
        scale=1.5,
        rotation=pycolmap.Rotation3d(),
        translation=np.zeros(3),
    )
    cams_from_rig = [
        pycolmap.Rigid3d(),
        pycolmap.Rigid3d(
            rotation=pycolmap.Rotation3d(),
            translation=np.array([1.0, 0.0, 0.0]),
        ),
    ]
    points2D = []
    points3D = []
    camera_idxs = []
    for i in range(8):
        point3D = np.array([0.1 * i - 0.35, 0.05 * (i % 3) - 0.05, 5.0])
        point3D_in_cam = cams_from_rig[i % 2] * (gt_rig_from_world * point3D)
        points2D.append(camera.img_from_cam(point3D_in_cam))
        points3D.append(point3D)
        camera_idxs.append(i % 2)
    return dict(
        gt_rig_from_world=gt_rig_from_world,
        points2D=np.array(points2D),
        points3D=np.array(points3D),
        camera_idxs=np.array(camera_idxs),
        cams_from_rig=cams_from_rig,
        cameras=[camera, camera],
    )


def test_scaled_generalized_absolute_pose_smoke() -> None:
    problem = build_tiny_scaled_problem()
    ransac_options = pycolmap.RANSACOptions()
    ransac_options.max_error = 4.0
    ransac_options.random_seed = 0
    estimation = pycolmap.estimate_scaled_generalized_absolute_pose(
        points2D=problem["points2D"],
        points3D=problem["points3D"],
        camera_idxs=problem["camera_idxs"],
        cams_from_rig=problem["cams_from_rig"],
        cameras=problem["cameras"],
        estimation_options=ransac_options,
    )
    assert estimation is not None
    assert {"rig_from_world", "num_inliers", "inlier_mask"} <= set(
        estimation.keys()
    )

    refinement_options = pycolmap.AbsolutePoseRefinementOptions()
    refinement = pycolmap.refine_scaled_generalized_absolute_pose(
        rig_from_world=problem["gt_rig_from_world"],
        points2D=problem["points2D"],
        points3D=problem["points3D"],
        inlier_mask=np.ones(len(problem["points2D"]), dtype=bool),
        camera_idxs=problem["camera_idxs"],
        cams_from_rig=problem["cams_from_rig"],
        cameras=problem["cameras"],
        refinement_options=refinement_options,
    )
    assert refinement is not None
    assert {"rig_from_world", "cameras"} <= set(refinement.keys())

    result = pycolmap.estimate_and_refine_scaled_generalized_absolute_pose(
        points2D=problem["points2D"],
        points3D=problem["points3D"],
        camera_idxs=problem["camera_idxs"],
        cams_from_rig=problem["cams_from_rig"],
        cameras=problem["cameras"],
        estimation_options=ransac_options,
        refinement_options=refinement_options,
        return_covariance=True,
    )
    assert result is not None
    assert {
        "rig_from_world",
        "num_inliers",
        "inlier_mask",
        "cameras",
        "covariance",
    } <= set(result.keys())
