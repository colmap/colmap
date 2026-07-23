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
        "align_reconstruction_to_pose_priors",
        "align_reconstruction_to_pose_priors_robust",
        "refine_pose_prior_alignment_with_orientations",
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


def test_pose_prior_alignment_result_default_init():
    result = pycolmap.PosePriorAlignmentResult()
    assert result.success is False
    assert len(result.correspondence_image_ids) == 0
    assert result.inlier_mask.shape == (0,)
    assert result.orientation_requested is False
    assert result.orientation_engaged is False
    assert result.orientation_inlier_mask.shape == (0,)


def test_align_reconstruction_to_pose_priors_robust_no_priors(
    synthetic_reconstruction,
):
    result = pycolmap.align_reconstruction_to_pose_priors_robust(
        synthetic_reconstruction, [], pycolmap.RANSACOptions()
    )
    assert result.success is False


def test_anisotropic_position_gate_default_init():
    gate = pycolmap.AnisotropicPositionGate()
    assert gate.max_horizontal_error == -1.0
    assert gate.max_vertical_error == -1.0
    assert gate.is_set() is False


def test_anisotropic_position_gate_readwrite_and_is_set():
    gate = pycolmap.AnisotropicPositionGate()
    gate.max_horizontal_error = 10.0
    assert gate.is_set() is False
    gate.max_vertical_error = 20.0
    assert gate.max_horizontal_error == 10.0
    assert gate.max_vertical_error == 20.0
    assert gate.is_set() is True


def test_align_reconstruction_to_pose_priors_robust_with_anisotropic_gate(
    synthetic_reconstruction,
):
    # No priors resolve regardless of the gate, but this exercises the
    # anisotropic_gate keyword argument round-trip through the binding.
    gate = pycolmap.AnisotropicPositionGate()
    gate.max_horizontal_error = 10.0
    gate.max_vertical_error = 20.0
    result = pycolmap.align_reconstruction_to_pose_priors_robust(
        synthetic_reconstruction,
        [],
        pycolmap.RANSACOptions(),
        anisotropic_gate=gate,
    )
    assert result.success is False
