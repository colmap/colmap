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


class ScaledGeneralizedAbsolutePoseProblem:
    """Synthetic rig whose cams_from_rig are only known up to scale."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        pycolmap.set_random_seed(seed)
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs = 2
        options.num_cameras_per_rig = 2
        options.num_frames_per_rig = 1
        options.num_points3D = 50
        reconstruction = pycolmap.synthesize_dataset(options)

        self.gt_rig_from_world = pycolmap.Sim3d(
            scale=rng.uniform(0.5, 2.0),
            rotation=random_rotation(rng),
            translation=rng.uniform(-1.0, 1.0, size=3),
        )
        points2D = []
        points3D = []
        point3D_ids = []
        camera_idxs = []
        self.cams_from_rig: list[pycolmap.Rigid3d] = []
        self.cameras: list[pycolmap.Camera] = []
        for image_id in reconstruction.reg_image_ids():
            image = reconstruction.image(image_id)
            for point2D in image.points2D:
                if point2D.has_point3D():
                    points2D.append(point2D.xy)
                    points3D.append(
                        reconstruction.point3D(point2D.point3D_id).xyz
                    )
                    point3D_ids.append(point2D.point3D_id)
                    camera_idxs.append(len(self.cameras))
            # Copy: the camera references the reconstruction, which does not
            # outlive this constructor.
            self.cameras.append(copy.deepcopy(image.camera))
            # Rigid camera pose in the scaled rig frame. The uniform scaling
            # of the camera frame leaves the image projections unchanged.
            self.cams_from_rig.append(
                self.gt_rig_from_world.transform_camera_world(
                    image.cam_from_world()
                )
            )
        self.points2D = np.array(points2D)
        self.points3D = np.array(points3D)
        self.point3D_ids = np.array(point3D_ids)
        self.camera_idxs = np.array(camera_idxs)

    def move_point_behind_camera(self, i: int) -> None:
        cam_from_rig = self.cams_from_rig[self.camera_idxs[i]]
        point3D_in_cam = cam_from_rig * (
            self.gt_rig_from_world * self.points3D[i]
        )
        point3D_in_cam[2] = -abs(point3D_in_cam[2])
        self.points3D[i] = self.gt_rig_from_world.inverse() * (
            cam_from_rig.inverse() * point3D_in_cam
        )


def random_rotation(rng: np.random.Generator) -> pycolmap.Rotation3d:
    xyzw = rng.normal(size=4)
    return pycolmap.Rotation3d(xyzw=xyzw / np.linalg.norm(xyzw))


def perturb_sim3d(
    rng: np.random.Generator, tform: pycolmap.Sim3d
) -> pycolmap.Sim3d:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    perturbation = pycolmap.Sim3d(
        scale=1.05,
        rotation=pycolmap.Rotation3d(np.deg2rad(1.0) * axis),
        translation=rng.uniform(-0.1, 0.1, size=3),
    )
    return perturbation * tform


def assert_sim3d_near(
    actual: pycolmap.Sim3d, expected: pycolmap.Sim3d, tol: float
) -> None:
    assert actual.scale == pytest.approx(expected.scale, abs=tol)
    assert actual.rotation.angle_to(expected.rotation) < tol
    np.testing.assert_allclose(
        actual.translation, expected.translation, atol=tol
    )


def test_estimate_scaled_generalized_absolute_pose() -> None:
    rng = np.random.default_rng(1)
    problem = ScaledGeneralizedAbsolutePoseProblem(seed=1)
    num_points = len(problem.points2D)

    gt_inlier_ratio = 0.8
    outlier_distance = 50.0
    num_gt_inliers = int(gt_inlier_ratio * num_points)
    shuffled_idxs = rng.permutation(num_points)
    outlier_idxs = shuffled_idxs[num_gt_inliers:]
    gt_inlier_mask = np.ones(num_points, dtype=bool)
    gt_inlier_mask[outlier_idxs] = False
    directions = rng.normal(size=(len(outlier_idxs), 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    problem.points2D[outlier_idxs] += outlier_distance * directions
    num_unique_inliers = len(np.unique(problem.point3D_ids[gt_inlier_mask]))

    options = pycolmap.RANSACOptions()
    options.max_error = 2.0
    options.min_inlier_ratio = gt_inlier_ratio / 2
    options.confidence = 0.99999
    options.random_seed = 0
    estimation = pycolmap.estimate_scaled_generalized_absolute_pose(
        points2D=problem.points2D,
        points3D=problem.points3D,
        camera_idxs=problem.camera_idxs,
        cams_from_rig=problem.cams_from_rig,
        cameras=problem.cameras,
        estimation_options=options,
    )
    assert estimation is not None
    assert estimation["num_inliers"] == num_unique_inliers
    np.testing.assert_array_equal(estimation["inlier_mask"], gt_inlier_mask)
    assert_sim3d_near(
        estimation["rig_from_world"], problem.gt_rig_from_world, tol=1e-5
    )


def test_refine_scaled_generalized_absolute_pose() -> None:
    rng = np.random.default_rng(2)
    problem = ScaledGeneralizedAbsolutePoseProblem(seed=2)
    init_rig_from_world = perturb_sim3d(rng, problem.gt_rig_from_world)
    init_params = np.array(init_rig_from_world.params)
    gt_focal_lengths = [camera.focal_length for camera in problem.cameras]

    options = pycolmap.AbsolutePoseRefinementOptions()
    options.refine_focal_length = False
    options.refine_extra_params = False
    refinement = pycolmap.refine_scaled_generalized_absolute_pose(
        rig_from_world=init_rig_from_world,
        points2D=problem.points2D,
        points3D=problem.points3D,
        inlier_mask=np.ones(len(problem.points2D), dtype=bool),
        camera_idxs=problem.camera_idxs,
        cams_from_rig=problem.cams_from_rig,
        cameras=problem.cameras,
        refinement_options=options,
        return_covariance=True,
    )
    assert refinement is not None
    rig_from_world = refinement["rig_from_world"]
    assert_sim3d_near(rig_from_world, problem.gt_rig_from_world, tol=1e-6)
    assert rig_from_world.scale > 0
    # The input transform is not modified in place.
    np.testing.assert_array_equal(init_rig_from_world.params, init_params)

    covariance = refinement["covariance"]
    assert covariance.shape == (7, 7)
    assert np.all(np.isfinite(covariance))
    assert np.any(covariance != 0)
    np.testing.assert_allclose(covariance, covariance.T)

    # Cameras are returned and unchanged when intrinsics are not refined.
    assert len(refinement["cameras"]) == len(problem.cameras)
    for camera, gt_focal_length in zip(
        refinement["cameras"], gt_focal_lengths, strict=True
    ):
        assert camera.focal_length == gt_focal_length


def test_estimate_and_refine_scaled_generalized_absolute_pose() -> None:
    problem = ScaledGeneralizedAbsolutePoseProblem(seed=4)

    ransac_options = pycolmap.RANSACOptions()
    ransac_options.max_error = 2.0
    ransac_options.random_seed = 0
    refinement_options = pycolmap.AbsolutePoseRefinementOptions()
    refinement_options.refine_focal_length = False
    refinement_options.refine_extra_params = False
    result = pycolmap.estimate_and_refine_scaled_generalized_absolute_pose(
        points2D=problem.points2D,
        points3D=problem.points3D,
        camera_idxs=problem.camera_idxs,
        cams_from_rig=problem.cams_from_rig,
        cameras=problem.cameras,
        estimation_options=ransac_options,
        refinement_options=refinement_options,
        return_covariance=True,
    )
    assert result is not None
    assert result["num_inliers"] == len(np.unique(problem.point3D_ids))
    assert np.all(result["inlier_mask"])
    assert_sim3d_near(
        result["rig_from_world"], problem.gt_rig_from_world, tol=1e-5
    )
    assert result["covariance"].shape == (7, 7)
    assert len(result["cameras"]) == len(problem.cameras)


def test_estimate_scaled_generalized_absolute_pose_empty_inputs() -> None:
    problem = ScaledGeneralizedAbsolutePoseProblem(seed=5)
    options = pycolmap.RANSACOptions()
    options.max_error = 2.0
    estimation = pycolmap.estimate_scaled_generalized_absolute_pose(
        points2D=np.zeros((0, 2)),
        points3D=np.zeros((0, 3)),
        camera_idxs=np.zeros(0, dtype=int),
        cams_from_rig=problem.cams_from_rig,
        cameras=problem.cameras,
        estimation_options=options,
    )
    assert estimation is None


def test_estimate_scaled_generalized_absolute_pose_panoramic_rig() -> None:
    problem = ScaledGeneralizedAbsolutePoseProblem(seed=6)
    # Move all cameras to a shared projection center, making the rig geometry
    # scale unobservable.
    center = np.array([0.1, -0.2, 0.3])
    cams_from_rig = [
        pycolmap.Rigid3d(
            rotation=cam_from_rig.rotation,
            translation=cam_from_rig.rotation * -center,
        )
        for cam_from_rig in problem.cams_from_rig
    ]
    options = pycolmap.RANSACOptions()
    options.max_error = 2.0
    options.random_seed = 0
    estimation = pycolmap.estimate_scaled_generalized_absolute_pose(
        points2D=problem.points2D,
        points3D=problem.points3D,
        camera_idxs=problem.camera_idxs,
        cams_from_rig=cams_from_rig,
        cameras=problem.cameras,
        estimation_options=options,
    )
    assert estimation is None


def test_refine_scaled_generalized_absolute_pose_single_center() -> None:
    problem = ScaledGeneralizedAbsolutePoseProblem(seed=7)
    options = pycolmap.AbsolutePoseRefinementOptions()
    options.refine_focal_length = False
    options.refine_extra_params = False

    # The scale is unobservable if the inlier mask only selects observations
    # from a single projection center, so the arbitrary initial scale must not
    # be reported as successfully refined.
    rig_from_world = pycolmap.Sim3d(
        scale=2.0,
        rotation=problem.gt_rig_from_world.rotation,
        translation=problem.gt_rig_from_world.translation,
    )
    refinement = pycolmap.refine_scaled_generalized_absolute_pose(
        rig_from_world=rig_from_world,
        points2D=problem.points2D,
        points3D=problem.points3D,
        inlier_mask=problem.camera_idxs == 0,
        camera_idxs=problem.camera_idxs,
        cams_from_rig=problem.cams_from_rig,
        cameras=problem.cameras,
        refinement_options=options,
    )
    assert refinement is None

    # An empty inlier set leaves the problem unconstrained.
    refinement = pycolmap.refine_scaled_generalized_absolute_pose(
        rig_from_world=rig_from_world,
        points2D=problem.points2D,
        points3D=problem.points3D,
        inlier_mask=np.zeros(len(problem.points2D), dtype=bool),
        camera_idxs=problem.camera_idxs,
        cams_from_rig=problem.cams_from_rig,
        cameras=problem.cameras,
        refinement_options=options,
    )
    assert refinement is None


def test_refine_scaled_generalized_absolute_pose_stale_inliers() -> None:
    rng = np.random.default_rng(8)
    problem = ScaledGeneralizedAbsolutePoseProblem(seed=8)
    # Inlier observations that do not project at the initial estimate are
    # excluded from the refinement, which must still converge to the ground
    # truth from the remaining exact observations.
    for i in (0, 7, 20):
        problem.move_point_behind_camera(i)

    options = pycolmap.AbsolutePoseRefinementOptions()
    options.refine_focal_length = False
    options.refine_extra_params = False
    refinement = pycolmap.refine_scaled_generalized_absolute_pose(
        rig_from_world=perturb_sim3d(rng, problem.gt_rig_from_world),
        points2D=problem.points2D,
        points3D=problem.points3D,
        inlier_mask=np.ones(len(problem.points2D), dtype=bool),
        camera_idxs=problem.camera_idxs,
        cams_from_rig=problem.cams_from_rig,
        cameras=problem.cameras,
        refinement_options=options,
    )
    assert refinement is not None
    assert_sim3d_near(
        refinement["rig_from_world"], problem.gt_rig_from_world, tol=1e-6
    )


def test_refine_scaled_generalized_absolute_pose_points_behind() -> None:
    problem = ScaledGeneralizedAbsolutePoseProblem(seed=9)
    # No observation constrains the reprojection cost at the initial estimate,
    # so refinement must fail instead of accepting the invalid pose as a
    # perfect fit.
    for i in range(len(problem.points3D)):
        problem.move_point_behind_camera(i)

    options = pycolmap.AbsolutePoseRefinementOptions()
    options.refine_focal_length = False
    options.refine_extra_params = False
    refinement = pycolmap.refine_scaled_generalized_absolute_pose(
        rig_from_world=problem.gt_rig_from_world,
        points2D=problem.points2D,
        points3D=problem.points3D,
        inlier_mask=np.ones(len(problem.points2D), dtype=bool),
        camera_idxs=problem.camera_idxs,
        cams_from_rig=problem.cams_from_rig,
        cameras=problem.cameras,
        refinement_options=options,
    )
    assert refinement is None


def test_refine_scaled_generalized_absolute_pose_nonpositive_scale() -> None:
    problem = ScaledGeneralizedAbsolutePoseProblem(seed=10)
    rig_from_world = pycolmap.Sim3d(
        scale=-1.0,
        rotation=problem.gt_rig_from_world.rotation,
        translation=problem.gt_rig_from_world.translation,
    )
    with pytest.raises(ValueError):
        pycolmap.refine_scaled_generalized_absolute_pose(
            rig_from_world=rig_from_world,
            points2D=problem.points2D,
            points3D=problem.points3D,
            inlier_mask=np.ones(len(problem.points2D), dtype=bool),
            camera_idxs=problem.camera_idxs,
            cams_from_rig=problem.cams_from_rig,
            cameras=problem.cameras,
        )
