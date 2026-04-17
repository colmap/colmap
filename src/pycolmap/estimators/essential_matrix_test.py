import pycolmap


def test_estimate_essential_matrix_is_callable():
    assert callable(pycolmap.estimate_essential_matrix)
