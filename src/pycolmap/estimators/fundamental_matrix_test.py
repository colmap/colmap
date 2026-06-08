import pycolmap


def test_estimate_fundamental_matrix_is_callable():
    assert callable(pycolmap.estimate_fundamental_matrix)
