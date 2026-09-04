import copy

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


def build_scaled_generalized_absolute_pose_problem() -> dict:
    """Synthetic rig whose cams_from_rig are only known up to scale."""
    pycolmap.set_random_seed(0)
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 2
    options.num_cameras_per_rig = 2
    options.num_frames_per_rig = 1
    options.num_points3D = 50
    reconstruction = pycolmap.synthesize_dataset(options)

    rng = np.random.default_rng(0)
    xyzw = rng.normal(size=4)
    gt_rig_from_world = pycolmap.Sim3d(
        scale=1.5,
        rotation=pycolmap.Rotation3d(xyzw=xyzw / np.linalg.norm(xyzw)),
        translation=rng.uniform(-1.0, 1.0, size=3),
    )

    points2D = []
    points3D = []
    camera_idxs = []
    cams_from_rig = []
    cameras = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        for point2D in image.points2D:
            if point2D.has_point3D():
                points2D.append(point2D.xy)
                points3D.append(reconstruction.point3D(point2D.point3D_id).xyz)
                camera_idxs.append(len(cameras))
        # Copy: the camera references the reconstruction, which does not
        # outlive this function.
        cameras.append(copy.deepcopy(image.camera))
        # Rigid camera pose in the scaled rig frame. The uniform scaling of
        # the camera frame leaves the image projections unchanged.
        cams_from_rig.append(
            gt_rig_from_world.transform_camera_world(image.cam_from_world())
        )

    return dict(
        gt_rig_from_world=gt_rig_from_world,
        points2D=np.array(points2D),
        points3D=np.array(points3D),
        camera_idxs=np.array(camera_idxs),
        cams_from_rig=cams_from_rig,
        cameras=cameras,
    )


def assert_sim3d_near(
    actual: pycolmap.Sim3d, expected: pycolmap.Sim3d, tol: float = 1e-5
) -> None:
    assert actual.scale == pytest.approx(expected.scale, abs=tol)
    assert actual.rotation.angle_to(expected.rotation) < tol
    np.testing.assert_allclose(
        actual.translation, expected.translation, atol=tol
    )


def test_estimate_scaled_generalized_absolute_pose() -> None:
    problem = build_scaled_generalized_absolute_pose_problem()
    options = pycolmap.RANSACOptions()
    options.max_error = 2.0
    options.random_seed = 0
    estimation = pycolmap.estimate_scaled_generalized_absolute_pose(
        points2D=problem["points2D"],
        points3D=problem["points3D"],
        camera_idxs=problem["camera_idxs"],
        cams_from_rig=problem["cams_from_rig"],
        cameras=problem["cameras"],
        estimation_options=options,
    )
    assert estimation is not None
    assert np.all(estimation["inlier_mask"])
    assert_sim3d_near(
        estimation["rig_from_world"], problem["gt_rig_from_world"]
    )


def test_refine_scaled_generalized_absolute_pose() -> None:
    problem = build_scaled_generalized_absolute_pose_problem()
    init_rig_from_world = (
        pycolmap.Sim3d(
            scale=1.05,
            rotation=pycolmap.Rotation3d(np.array([0.0, 0.02, 0.0])),
            translation=np.array([0.05, -0.05, 0.05]),
        )
        * problem["gt_rig_from_world"]
    )
    options = pycolmap.AbsolutePoseRefinementOptions()
    options.refine_focal_length = False
    options.refine_extra_params = False
    refinement = pycolmap.refine_scaled_generalized_absolute_pose(
        rig_from_world=init_rig_from_world,
        points2D=problem["points2D"],
        points3D=problem["points3D"],
        inlier_mask=np.ones(len(problem["points2D"]), dtype=bool),
        camera_idxs=problem["camera_idxs"],
        cams_from_rig=problem["cams_from_rig"],
        cameras=problem["cameras"],
        refinement_options=options,
        return_covariance=True,
    )
    assert refinement is not None
    assert_sim3d_near(
        refinement["rig_from_world"], problem["gt_rig_from_world"]
    )
    assert len(refinement["cameras"]) == len(problem["cameras"])
    assert refinement["covariance"].shape == (7, 7)


def test_estimate_and_refine_scaled_generalized_absolute_pose() -> None:
    problem = build_scaled_generalized_absolute_pose_problem()
    ransac_options = pycolmap.RANSACOptions()
    ransac_options.max_error = 2.0
    ransac_options.random_seed = 0
    refinement_options = pycolmap.AbsolutePoseRefinementOptions()
    refinement_options.refine_focal_length = False
    refinement_options.refine_extra_params = False
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
    assert np.all(result["inlier_mask"])
    assert_sim3d_near(result["rig_from_world"], problem["gt_rig_from_world"])
    assert len(result["cameras"]) == len(problem["cameras"])
    assert result["covariance"].shape == (7, 7)
