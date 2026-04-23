import pytest

import pycolmap


def test_image_alignment_error_default_init():
    error = pycolmap.ImageAlignmentError()
    assert error is not None


def test_image_alignment_error_image_name_readwrite():
    error = pycolmap.ImageAlignmentError()
    error.image_name = "test_image.jpg"
    assert error.image_name == "test_image.jpg"


def test_image_alignment_error_rotation_error_deg_readwrite():
    error = pycolmap.ImageAlignmentError()
    error.rotation_error_deg = 1.5
    assert error.rotation_error_deg == 1.5


def test_image_alignment_error_proj_center_error_readwrite():
    error = pycolmap.ImageAlignmentError()
    error.proj_center_error = 0.01
    assert error.proj_center_error == 0.01


@pytest.mark.parametrize(
    "name",
    [
        "align_reconstructions_via_reprojections",
        "align_reconstructions_via_proj_centers",
        "align_reconstructions_via_points",
        "compare_reconstructions",
    ],
)
def test_public_api_callable(name):
    assert callable(getattr(pycolmap, name))


def test_compare_reconstructions_with_synthetic(synthetic_reconstruction):
    result = pycolmap.compare_reconstructions(
        synthetic_reconstruction, synthetic_reconstruction
    )
    assert result is not None
    assert "rec2_from_rec1" in result
    assert "errors" in result
