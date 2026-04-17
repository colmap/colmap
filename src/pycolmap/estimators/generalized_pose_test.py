import pycolmap


def test_estimate_generalized_absolute_pose_is_callable():
    assert callable(pycolmap.estimate_generalized_absolute_pose)


def test_refine_generalized_absolute_pose_is_callable():
    assert callable(pycolmap.refine_generalized_absolute_pose)


def test_estimate_and_refine_generalized_absolute_pose_is_callable():
    assert callable(pycolmap.estimate_and_refine_generalized_absolute_pose)


def test_estimate_generalized_relative_pose_is_callable():
    assert callable(pycolmap.estimate_generalized_relative_pose)
