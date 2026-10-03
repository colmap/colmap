# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pycolmap

pytestmark = pytest.mark.skipif(
    not hasattr(pycolmap, "MVSModel"),
    reason="MVSModel not available",
)


def test_mvs_model_init() -> None:
    model = pycolmap.MVSModel()
    assert model is not None
