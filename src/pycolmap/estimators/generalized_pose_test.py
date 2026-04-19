import pytest

import pycolmap


@pytest.mark.parametrize(
    "name",
    [
        "estimate_generalized_absolute_pose",
        "refine_generalized_absolute_pose",
        "estimate_and_refine_generalized_absolute_pose",
        "estimate_generalized_relative_pose",
    ],
)
def test_public_api_callable(name):
    assert callable(getattr(pycolmap, name))
