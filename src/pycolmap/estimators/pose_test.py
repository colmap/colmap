import numpy as np
import pytest

import pycolmap


def test_absolute_pose_estimation_options_default_init() -> None:
    options = pycolmap.AbsolutePoseEstimationOptions()
    assert options is not None


def test_absolute_pose_estimation_options_estimate_focal_length_readwrite() -> (
    None
):
    options = pycolmap.AbsolutePoseEstimationOptions()
    original = options.estimate_focal_length
    assert isinstance(original, bool)
    options.estimate_focal_length = not original
    assert options.estimate_focal_length == (not original)


def test_absolute_pose_estimation_options_ransac_property() -> None:
    options = pycolmap.AbsolutePoseEstimationOptions()
    ransac = options.ransac
    assert isinstance(ransac, pycolmap.RANSACOptions)


def test_absolute_pose_refinement_options_default_init() -> None:
    options = pycolmap.AbsolutePoseRefinementOptions()
    assert options is not None


def test_absolute_pose_refinement_options_gradient_tolerance_readwrite() -> (
    None
):
    options = pycolmap.AbsolutePoseRefinementOptions()
    assert isinstance(options.gradient_tolerance, float)
    options.gradient_tolerance = 0.5
    assert options.gradient_tolerance == 0.5


def test_absolute_pose_refinement_options_max_num_iterations_readwrite() -> (
    None
):
    options = pycolmap.AbsolutePoseRefinementOptions()
    assert isinstance(options.max_num_iterations, int)
    options.max_num_iterations = 200
    assert options.max_num_iterations == 200


def test_absolute_pose_refinement_options_loss_function_scale_readwrite() -> (
    None
):
    options = pycolmap.AbsolutePoseRefinementOptions()
    assert isinstance(options.loss_function_scale, float)
    options.loss_function_scale = 2.0
    assert options.loss_function_scale == 2.0


def test_absolute_pose_refinement_options_refine_focal_length_readwrite() -> (
    None
):
    options = pycolmap.AbsolutePoseRefinementOptions()
    original = options.refine_focal_length
    assert isinstance(original, bool)
    options.refine_focal_length = not original
    assert options.refine_focal_length == (not original)


def test_absolute_pose_refinement_options_refine_extra_params_readwrite() -> (
    None
):
    options = pycolmap.AbsolutePoseRefinementOptions()
    original = options.refine_extra_params
    assert isinstance(original, bool)
    options.refine_extra_params = not original
    assert options.refine_extra_params == (not original)


def test_absolute_pose_refinement_options_print_summary_readwrite() -> None:
    options = pycolmap.AbsolutePoseRefinementOptions()
    original = options.print_summary
    assert isinstance(original, bool)
    options.print_summary = not original
    assert options.print_summary == (not original)


def test_absolute_pose_refinement_options_use_position_prior_readwrite() -> (
    None
):
    options = pycolmap.AbsolutePoseRefinementOptions()
    original = options.use_position_prior
    assert isinstance(original, bool)
    options.use_position_prior = not original
    assert options.use_position_prior == (not original)


def test_absolute_pose_refinement_position_prior_covariance_readwrite() -> None:
    options = pycolmap.AbsolutePoseRefinementOptions()
    covariance = options.position_prior_covariance
    assert covariance is not None
    new_covariance = np.eye(3) * 2.0
    options.position_prior_covariance = new_covariance
    result = options.position_prior_covariance
    np.testing.assert_array_almost_equal(result, new_covariance)


@pytest.mark.parametrize(
    "name",
    [
        "estimate_absolute_pose",
        "refine_absolute_pose",
        "estimate_and_refine_absolute_pose",
        "estimate_relative_pose",
        "refine_relative_pose",
    ],
)
def test_public_api_callable(name: str) -> None:
    assert callable(getattr(pycolmap, name))


def test_relative_pose_ray_overloads() -> None:
    rng = np.random.default_rng(0)
    points3D = rng.uniform(-1.0, 1.0, size=(20, 3))
    points3D[:, 2] += 4.0
    translation = np.array([0.5, 0.1, 0.0])
    cam_rays1 = points3D / np.linalg.norm(points3D, axis=1, keepdims=True)
    points3D_in_cam2 = points3D + translation
    cam_rays2 = points3D_in_cam2 / np.linalg.norm(
        points3D_in_cam2, axis=1, keepdims=True
    )

    options = pycolmap.RANSACOptions()
    options.max_error = 1e-3
    options.random_seed = 0
    estimation = pycolmap.estimate_relative_pose(
        cam_rays1=cam_rays1, cam_rays2=cam_rays2, options=options
    )
    assert estimation is not None
    assert estimation["num_inliers"] == len(points3D)

    refinement = pycolmap.refine_relative_pose(
        cam2_from_cam1=estimation["cam2_from_cam1"],
        cam_rays1=cam_rays1,
        cam_rays2=cam_rays2,
        inlier_mask=estimation["inlier_mask"],
    )
    assert refinement is not None
