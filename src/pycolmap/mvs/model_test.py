import pytest

import pycolmap

pytestmark = pytest.mark.skipif(
    not hasattr(pycolmap, "MVSModel"),
    reason="MVSModel not available",
)


def test_mvs_model_init():
    model = pycolmap.MVSModel()
    assert model is not None
