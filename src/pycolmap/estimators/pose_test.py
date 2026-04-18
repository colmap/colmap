import numpy as np

import pycolmap


def test_absolute_pose_estimation_options_default_init():
    options = pycolmap.AbsolutePoseEstimationOptions()
    assert options is not None


def test_absolute_pose_estimation_options_estimate_focal_length_readwrite():
    options = pycolmap.AbsolutePoseEstimationOptions()
    original = options.estimate_focal_length
    assert isinstance(original, bool)
    options.estimate_focal_length = not original
    assert options.estimate_focal_length == (not original)


def test_absolute_pose_estimation_options_ransac_property():
    options = pycolmap.AbsolutePoseEstimationOptions()
    ransac = options.ransac
    assert isinstance(ransac, pycolmap.RANSACOptions)


def test_absolute_pose_refinement_options_default_init():
    options = pycolmap.AbsolutePoseRefinementOptions()
    assert options is not None


def test_absolute_pose_refinement_options_gradient_tolerance_readwrite():
    options = pycolmap.AbsolutePoseRefinementOptions()
    assert isinstance(options.gradient_tolerance, float)
    options.gradient_tolerance = 0.5
    assert options.gradient_tolerance == 0.5


def test_absolute_pose_refinement_options_max_num_iterations_readwrite():
    options = pycolmap.AbsolutePoseRefinementOptions()
    assert isinstance(options.max_num_iterations, int)
    options.max_num_iterations = 200
    assert options.max_num_iterations == 200


def test_absolute_pose_refinement_options_loss_function_scale_readwrite():
    options = pycolmap.AbsolutePoseRefinementOptions()
    assert isinstance(options.loss_function_scale, float)
    options.loss_function_scale = 2.0
    assert options.loss_function_scale == 2.0


def test_absolute_pose_refinement_options_refine_focal_length_readwrite():
    options = pycolmap.AbsolutePoseRefinementOptions()
    original = options.refine_focal_length
    assert isinstance(original, bool)
    options.refine_focal_length = not original
    assert options.refine_focal_length == (not original)


def test_absolute_pose_refinement_options_refine_extra_params_readwrite():
    options = pycolmap.AbsolutePoseRefinementOptions()
    original = options.refine_extra_params
    assert isinstance(original, bool)
    options.refine_extra_params = not original
    assert options.refine_extra_params == (not original)


def test_absolute_pose_refinement_options_print_summary_readwrite():
    options = pycolmap.AbsolutePoseRefinementOptions()
    original = options.print_summary
    assert isinstance(original, bool)
    options.print_summary = not original
    assert options.print_summary == (not original)


def test_absolute_pose_refinement_options_use_position_prior_readwrite():
    options = pycolmap.AbsolutePoseRefinementOptions()
    original = options.use_position_prior
    assert isinstance(original, bool)
    options.use_position_prior = not original
    assert options.use_position_prior == (not original)


def test_absolute_pose_refinement_options_position_prior_covariance_readwrite():
    options = pycolmap.AbsolutePoseRefinementOptions()
    covariance = options.position_prior_covariance
    assert covariance is not None
    new_covariance = np.eye(3) * 2.0
    options.position_prior_covariance = new_covariance
    result = options.position_prior_covariance
    np.testing.assert_array_almost_equal(result, new_covariance)


def test_estimate_absolute_pose_is_callable():
    assert callable(pycolmap.estimate_absolute_pose)


def test_refine_absolute_pose_is_callable():
    assert callable(pycolmap.refine_absolute_pose)


def test_estimate_and_refine_absolute_pose_is_callable():
    assert callable(pycolmap.estimate_and_refine_absolute_pose)


def test_estimate_relative_pose_is_callable():
    assert callable(pycolmap.estimate_relative_pose)


def test_refine_relative_pose_is_callable():
    assert callable(pycolmap.refine_relative_pose)
