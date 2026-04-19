import pycolmap


def test_cost_functions_submodule_exists():
    assert hasattr(pycolmap._core, "cost_functions")


def test_reproj_error_cost_exists():
    assert hasattr(pycolmap._core.cost_functions, "ReprojErrorCost")


def test_sampson_error_cost_exists():
    assert hasattr(pycolmap._core.cost_functions, "SampsonErrorCost")


def test_absolute_pose_prior_cost_exists():
    assert hasattr(pycolmap._core.cost_functions, "AbsolutePosePriorCost")


def test_absolute_pose_position_prior_cost_exists():
    assert hasattr(
        pycolmap._core.cost_functions, "AbsolutePosePositionPriorCost"
    )


def test_relative_pose_prior_cost_exists():
    assert hasattr(pycolmap._core.cost_functions, "RelativePosePriorCost")


def test_point3d_alignment_cost_exists():
    assert hasattr(pycolmap._core.cost_functions, "Point3DAlignmentCost")
