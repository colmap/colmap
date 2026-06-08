import pytest

import pycolmap

pytestmark = pytest.mark.skipif(
    not hasattr(pycolmap, "PatchMatchOptions"),
    reason="PatchMatchOptions not available (requires CUDA)",
)


def test_patch_match_options_init():
    options = pycolmap.PatchMatchOptions()
    assert options is not None


def test_stereo_fusion_options_init():
    options = pycolmap.StereoFusionOptions()
    assert options is not None


@pytest.mark.parametrize(
    "name",
    [
        "patch_match_stereo",
        "stereo_fusion",
    ],
)
def test_public_api_callable(name):
    assert callable(getattr(pycolmap, name))
