import pycolmap


def test_extract_features_callable():
    assert callable(pycolmap.extract_features)
