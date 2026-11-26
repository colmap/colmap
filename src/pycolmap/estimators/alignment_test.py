# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import pycolmap


def test_image_alignment_error_default_init() -> None:
    error = pycolmap.ImageAlignmentError()
    assert error is not None


def test_image_alignment_error_image_name_readwrite() -> None:
    error = pycolmap.ImageAlignmentError()
    error.image_name = "test_image.jpg"
    assert error.image_name == "test_image.jpg"


def test_image_alignment_error_rotation_error_deg_readwrite() -> None:
    error = pycolmap.ImageAlignmentError()
    error.rotation_error_deg = 1.5
    assert error.rotation_error_deg == 1.5


def test_image_alignment_error_proj_center_error_readwrite() -> None:
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
        "compare_reconstructions",
    ],
)
def test_public_api_callable(name: str) -> None:
    assert callable(getattr(pycolmap, name))


def test_compare_reconstructions_with_synthetic(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    result = pycolmap.compare_reconstructions(
        synthetic_reconstruction, synthetic_reconstruction
    )
    assert result is not None
    assert "rec2_from_rec1" in result
    assert "errors" in result


def test_align_reconstruction_to_pose_priors(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    gt_tgt_from_src = pycolmap.Sim3d(
        2.5,
        pycolmap.Rotation3d(np.array([0.0, 0.0, np.sin(0.2), np.cos(0.2)])),
        [10.0, -5.0, 3.0],
    )
    tgt_reconstruction = pycolmap.Reconstruction(synthetic_reconstruction)
    tgt_reconstruction.transform(gt_tgt_from_src)

    cartesian_priors = [
        pycolmap.PosePrior(
            corr_data_id=image.data_id,
            position=image.projection_center(),
            position_covariance=1e-4 * np.eye(3),
            coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
        )
        for image_id in tgt_reconstruction.reg_image_ids()
        for image in (tgt_reconstruction.image(image_id),)
    ]
    assert len(cartesian_priors) >= 3

    assert (
        pycolmap.align_reconstruction_to_pose_priors(
            synthetic_reconstruction, cartesian_priors[:2]
        )
        is None
    )

    tgt_from_src = pycolmap.align_reconstruction_to_pose_priors(
        synthetic_reconstruction, cartesian_priors
    )
    assert tgt_from_src is not None
    assert np.allclose(
        tgt_from_src.matrix(), gt_tgt_from_src.matrix(), atol=1e-5
    )

    wgs84_priors = [
        pycolmap.PosePrior(
            corr_data_id=p.corr_data_id,
            position=p.position,
            coordinate_system=pycolmap.PosePriorCoordinateSystem.WGS84,
        )
        for p in cartesian_priors
    ]
    with pytest.raises(ValueError, match="Cartesian coordinate system"):
        pycolmap.align_reconstruction_to_pose_priors(
            synthetic_reconstruction, wgs84_priors, pycolmap.RANSACOptions()
        )
