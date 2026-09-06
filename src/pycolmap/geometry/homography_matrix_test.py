import pycolmap


def test_pose_from_homography_matrix_exists() -> None:
    assert hasattr(pycolmap, "pose_from_homography_matrix")
    assert callable(pycolmap.pose_from_homography_matrix)
