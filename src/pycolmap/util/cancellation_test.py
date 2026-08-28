import pytest

import pycolmap


def test_cancellation_token():
    token = pycolmap.CancellationToken()
    assert not token.is_cancelled
    token.cancel()
    assert token.is_cancelled


def test_cancelled_feature_extraction_raises(tmp_path):
    image_path = tmp_path / "images"
    image_path.mkdir()
    token = pycolmap.CancellationToken()
    token.cancel()

    with pytest.raises(InterruptedError):
        pycolmap.extract_features(
            tmp_path / "database.db",
            image_path,
            cancellation_token=token,
        )
