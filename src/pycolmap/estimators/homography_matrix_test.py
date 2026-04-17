import pycolmap


def test_estimate_homography_matrix_is_callable():
    assert callable(pycolmap.estimate_homography_matrix)
