# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pycolmap


def test_bundle_adjustment_termination_type_enum() -> None:
    assert {
        k: int(v)
        for k, v in pycolmap.BundleAdjustmentTerminationType.__members__.items()
    } == {
        "CONVERGENCE": 0,
        "NO_CONVERGENCE": 1,
        "FAILURE": 2,
        "USER_SUCCESS": 3,
        "USER_FAILURE": 4,
    }


def test_bundle_adjustment_backend_enum() -> None:
    assert {
        k: int(v)
        for k, v in pycolmap.BundleAdjustmentBackend.__members__.items()
    } == {
        "CERES": 0,
        "CASPAR": 1,
    }
    assert (
        pycolmap.BundleAdjustmentBackend("CASPAR")
        == pycolmap.BundleAdjustmentBackend.CASPAR
    )


def test_loss_function_type_enum() -> None:
    assert {
        k: int(v) for k, v in pycolmap.LossFunctionType.__members__.items()
    } == {
        "TRIVIAL": 0,
        "SOFT_L1": 1,
        "CAUCHY": 2,
        "HUBER": 3,
        "GEMAN_MCCLURE": 4,
    }


def test_bundle_adjustment_summary_default_init() -> None:
    summary = pycolmap.BundleAdjustmentSummary()
    assert summary is not None


def test_bundle_adjustment_summary_num_residuals_readwrite() -> None:
    summary = pycolmap.BundleAdjustmentSummary()
    summary.num_residuals = 100
    assert summary.num_residuals == 100


def test_bundle_adjustment_summary_termination_type_readwrite() -> None:
    summary = pycolmap.BundleAdjustmentSummary()
    summary.termination_type = (
        pycolmap.BundleAdjustmentTerminationType.CONVERGENCE
    )
    assert (
        summary.termination_type
        == pycolmap.BundleAdjustmentTerminationType.CONVERGENCE
    )


def test_bundle_adjustment_summary_is_solution_usable() -> None:
    summary = pycolmap.BundleAdjustmentSummary()
    result = summary.is_solution_usable()
    assert isinstance(result, bool)


def test_bundle_adjustment_summary_brief_report() -> None:
    summary = pycolmap.BundleAdjustmentSummary()
    report = summary.brief_report()
    assert isinstance(report, str)


def test_ceres_bundle_adjustment_summary_default_init() -> None:
    summary = pycolmap.CeresBundleAdjustmentSummary()
    assert summary is not None


def test_ceres_bundle_adjustment_summary_inherits() -> None:
    summary = pycolmap.CeresBundleAdjustmentSummary()
    assert isinstance(summary, pycolmap.BundleAdjustmentSummary)


def test_bundle_adjustment_config_default_init() -> None:
    config = pycolmap.BundleAdjustmentConfig()
    assert config is not None


def test_bundle_adjustment_config_images_property() -> None:
    config = pycolmap.BundleAdjustmentConfig()
    images = config.images
    assert len(images) == 0


def test_ceres_ba_options_default_init() -> None:
    options = pycolmap.CeresBundleAdjustmentOptions()
    assert options is not None


def test_ceres_ba_options_check() -> None:
    options = pycolmap.CeresBundleAdjustmentOptions()
    result = options.check()
    assert isinstance(result, bool)


def test_caspar_ba_options_default_init() -> None:
    options = pycolmap.CasparBundleAdjustmentOptions()
    assert options is not None


def test_caspar_ba_options_readwrite() -> None:
    options = pycolmap.CasparBundleAdjustmentOptions()
    options.solver_iter_max = 17
    options.pcg_iter_max = 11
    options.diag_init = 2.0
    options.gpu_index = "0"
    assert options.solver_iter_max == 17
    assert options.pcg_iter_max == 11
    assert options.diag_init == 2.0
    assert options.gpu_index == "0"


def test_ba_options_default_init() -> None:
    options = pycolmap.BundleAdjustmentOptions()
    assert options is not None


def test_ba_options_refine_focal_length_readwrite() -> None:
    options = pycolmap.BundleAdjustmentOptions()
    original = options.refine_focal_length
    assert isinstance(original, bool)
    options.refine_focal_length = not original
    assert options.refine_focal_length == (not original)


def test_ba_options_refine_principal_point_readwrite() -> None:
    options = pycolmap.BundleAdjustmentOptions()
    original = options.refine_principal_point
    assert isinstance(original, bool)
    options.refine_principal_point = not original
    assert options.refine_principal_point == (not original)


def test_ba_options_refine_extra_params_readwrite() -> None:
    options = pycolmap.BundleAdjustmentOptions()
    original = options.refine_extra_params
    assert isinstance(original, bool)
    options.refine_extra_params = not original
    assert options.refine_extra_params == (not original)


def test_ba_options_refine_points3d_readwrite() -> None:
    options = pycolmap.BundleAdjustmentOptions()
    original = options.refine_points3D
    assert isinstance(original, bool)
    options.refine_points3D = not original
    assert options.refine_points3D == (not original)


def test_ba_options_min_track_length_readwrite() -> None:
    options = pycolmap.BundleAdjustmentOptions()
    assert isinstance(options.min_track_length, int)
    options.min_track_length = 5
    assert options.min_track_length == 5


def test_ba_options_print_summary_readwrite() -> None:
    options = pycolmap.BundleAdjustmentOptions()
    original = options.print_summary
    assert isinstance(original, bool)
    options.print_summary = not original
    assert options.print_summary == (not original)


def test_ba_options_backend_readwrite() -> None:
    options = pycolmap.BundleAdjustmentOptions()
    options.backend = pycolmap.BundleAdjustmentBackend.CERES
    assert options.backend == pycolmap.BundleAdjustmentBackend.CERES
    options.backend = pycolmap.BundleAdjustmentBackend.CASPAR
    assert options.backend == pycolmap.BundleAdjustmentBackend.CASPAR


def test_ba_options_ceres_property() -> None:
    options = pycolmap.BundleAdjustmentOptions()
    ceres = options.ceres
    assert isinstance(ceres, pycolmap.CeresBundleAdjustmentOptions)


def test_ba_options_caspar_property() -> None:
    options = pycolmap.BundleAdjustmentOptions()
    caspar = options.caspar
    assert isinstance(caspar, pycolmap.CasparBundleAdjustmentOptions)


def test_ba_options_check() -> None:
    options = pycolmap.BundleAdjustmentOptions()
    result = options.check()
    assert isinstance(result, bool)


def test_pose_prior_ba_options_default_init() -> None:
    options = pycolmap.PosePriorBundleAdjustmentOptions()
    assert options is not None


def test_pose_prior_ba_options_prior_position_fallback_stddev_readwrite() -> (
    None
):
    options = pycolmap.PosePriorBundleAdjustmentOptions()
    assert isinstance(options.prior_position_fallback_stddev, float)
    options.prior_position_fallback_stddev = 5.0
    assert options.prior_position_fallback_stddev == 5.0


def test_ceres_pose_prior_ba_options_default_init() -> None:
    options = pycolmap.CeresPosePriorBundleAdjustmentOptions()
    assert options is not None


def test_ceres_pose_prior_ba_options_check() -> None:
    options = pycolmap.CeresPosePriorBundleAdjustmentOptions()
    result = options.check()
    assert isinstance(result, bool)


def test_create_default_bundle_adjuster(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    options = pycolmap.BundleAdjustmentOptions()
    config = pycolmap.BundleAdjustmentConfig()
    adjuster = pycolmap.create_default_bundle_adjuster(
        options, config, synthetic_reconstruction
    )
    assert adjuster is not None


def test_create_default_ceres_bundle_adjuster(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    options = pycolmap.BundleAdjustmentOptions()
    config = pycolmap.BundleAdjustmentConfig()
    adjuster = pycolmap.create_default_ceres_bundle_adjuster(
        options, config, synthetic_reconstruction
    )
    assert adjuster is not None


def test_bundle_adjustment_pipeline(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    reconstruction = pycolmap.Reconstruction(synthetic_reconstruction)
    options = pycolmap.BundleAdjustmentOptions()
    options.print_summary = False
    pycolmap.bundle_adjustment(reconstruction, options)


def test_photometric_ba_options_and_summary() -> None:
    options = pycolmap.PhotometricBundleAdjustmentOptions()
    assert options.check()
    assert options.patch_size == 4
    assert options.geometric_reprojection_weight == 1.0
    assert options.loss_function_type == pycolmap.LossFunctionType.GEMAN_MCCLURE
    summary = pycolmap.PhotometricBundleAdjustmentSummary()
    assert isinstance(summary, pycolmap.BundleAdjustmentSummary)
    assert summary.num_tracks_total == 0


def _write_synthetic_textured_images(
    root_dir, reconstruction: pycolmap.Reconstruction
) -> None:
    import numpy as np
    from PIL import Image as PILImage

    for image in reconstruction.images.values():
        cam = reconstruction.cameras[image.camera_id]
        yy, xx = np.mgrid[0 : cam.height, 0 : cam.width]
        arr = np.clip(
            128 + 50 * np.sin(0.1 * xx) + 40 * np.cos(0.13 * yy), 0, 255
        ).astype(np.uint8)
        img_path = root_dir / image.name
        img_path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(arr).save(img_path)


def test_photometric_bundle_adjustment_pipeline(
    tmp_path,
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    import numpy as np

    reconstruction = pycolmap.Reconstruction(synthetic_reconstruction)
    _write_synthetic_textured_images(tmp_path, reconstruction)
    num_images_before = reconstruction.num_reg_images()
    num_points_before = reconstruction.num_points3D()

    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.print_summary = False
    ba_options.ceres.solver_options.max_num_iterations = 5
    photo_options = pycolmap.PhotometricBundleAdjustmentOptions()
    photo_options.pyramid_levels = "0"
    photo_options.refine_structure_first = False
    pycolmap.photometric_bundle_adjustment(
        reconstruction, tmp_path, ba_options, photo_options
    )

    assert reconstruction.num_reg_images() == num_images_before
    assert reconstruction.num_points3D() == num_points_before
    for point in reconstruction.points3D.values():
        assert np.all(np.isfinite(point.xyz))
    for cam in reconstruction.cameras.values():
        assert np.all(np.isfinite(cam.params))


def test_create_photometric_bundle_adjuster_and_lifetime(
    tmp_path,
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    import gc

    reconstruction = pycolmap.Reconstruction(synthetic_reconstruction)
    _write_synthetic_textured_images(tmp_path, reconstruction)

    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.print_summary = False
    ba_options.ceres.solver_options.max_num_iterations = 5
    photo_options = pycolmap.PhotometricBundleAdjustmentOptions()
    photo_options.pyramid_levels = "0"
    photo_options.refine_structure_first = False
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in reconstruction.reg_image_ids():
        config.add_image(image_id)

    adjuster = pycolmap.create_photometric_bundle_adjuster(
        ba_options, photo_options, config, tmp_path, reconstruction
    )
    # Drop the caller's Python reference and force GC to verify keep_alive.
    del reconstruction
    gc.collect()

    summary = adjuster.solve()
    assert summary.is_solution_usable()
    assert summary.num_tracks_total > 0
    assert summary.num_tracks_kept > 0
    assert summary.num_residuals > 0


def test_photometric_bundle_adjustment_invalid_image_path(
    tmp_path,
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    reconstruction = pycolmap.Reconstruction(synthetic_reconstruction)
    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.print_summary = False
    photo_options = pycolmap.PhotometricBundleAdjustmentOptions()
    photo_options.pyramid_levels = "0"
    with pytest.raises(ValueError):
        pycolmap.photometric_bundle_adjustment(
            reconstruction,
            tmp_path / "does_not_exist",
            ba_options,
            photo_options,
        )
