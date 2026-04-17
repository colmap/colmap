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


def test_patch_match_stereo_callable():
    assert callable(pycolmap.patch_match_stereo)


def test_stereo_fusion_callable():
    assert callable(pycolmap.stereo_fusion)
