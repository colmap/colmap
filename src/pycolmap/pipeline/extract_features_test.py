import pycolmap


def test_extract_features_callable() -> None:
    assert callable(pycolmap.extract_features)
