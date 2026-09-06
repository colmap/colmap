import pycolmap


def test_estimate_homography_matrix_is_callable() -> None:
    assert callable(pycolmap.estimate_homography_matrix)
