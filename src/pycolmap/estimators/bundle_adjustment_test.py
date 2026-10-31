# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
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


def test_bundle_adjustment_gauge_enum() -> None:
    assert {
        k: int(v) for k, v in pycolmap.BundleAdjustmentGauge.__members__.items()
    } == {
        "UNSPECIFIED": -1,
        "TWO_CAMS_FROM_WORLD": 0,
        "THREE_POINTS": 1,
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


def test_bundle_adjustment_priors_default_init() -> None:
    priors = pycolmap.BundleAdjustmentPriors()
    assert priors.pose_priors == []


def test_ba_options_prior_fields_and_helpers() -> None:
    options = pycolmap.BundleAdjustmentOptions()
    assert isinstance(
        options.pose_prior, pycolmap.BundleAdjustmentPosePriorOptions
    )
    assert not options.pose_prior.use_position
    assert not options.constrain_metric_scale()
    assert not options.pose_prior.overwrite_position_covariance
    np.testing.assert_allclose(
        options.pose_prior.position_stddev, [1.0, 1.0, 1.0]
    )

    options.pose_prior.use_position = True
    options.pose_prior.overwrite_position_covariance = True
    options.pose_prior.position_stddev = np.array([2.0, 3.0, 4.0])
    assert options.constrain_metric_scale()
    np.testing.assert_allclose(
        options.pose_prior.position_stddev, [2.0, 3.0, 4.0]
    )
    assert options.pose_prior.check()
    assert options.check()

    assert (
        options.ceres.prior_position_loss_function_type
        == pycolmap.LossFunctionType.TRIVIAL
    )
    options.ceres.prior_position_loss_function_type = (
        pycolmap.LossFunctionType.CAUCHY
    )
    options.ceres.prior_position_loss_scale = 3.5
    assert (
        options.ceres.prior_position_loss_function_type
        == pycolmap.LossFunctionType.CAUCHY
    )
    assert options.ceres.prior_position_loss_scale == 3.5
    assert options.ceres.check()


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


def test_bundle_adjustment_with_pose_priors(tmp_path: Path) -> None:
    pycolmap.set_random_seed(0)
    database_path = tmp_path / "database.db"
    with pycolmap.Database.open(database_path) as database:
        synth_options = pycolmap.SyntheticDatasetOptions()
        synth_options.num_rigs = 1
        synth_options.num_cameras_per_rig = 1
        synth_options.num_frames_per_rig = 5
        synth_options.num_points3D = 50
        synth_options.prior_position = True
        gt_reconstruction = pycolmap.synthesize_dataset(synth_options, database)
        pose_priors = database.read_all_pose_priors()

    priors = pycolmap.BundleAdjustmentPriors()
    priors.pose_priors = pose_priors

    reconstruction = pycolmap.Reconstruction(gt_reconstruction)
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in reconstruction.reg_image_ids():
        config.add_image(image_id)

    options = pycolmap.BundleAdjustmentOptions()
    options.pose_prior.use_position = True
    options.print_summary = False
    # Pass a temporary Reconstruction to verify py::keep_alive lifetime safety.
    adjuster = pycolmap.create_default_bundle_adjuster(
        options, config, pycolmap.Reconstruction(gt_reconstruction), priors
    )
    summary = adjuster.solve()
    assert summary.is_solution_usable()

    # Pipeline entry point with explicit priors and with database_path.
    rec_from_priors = pycolmap.Reconstruction(gt_reconstruction)
    pycolmap.bundle_adjustment(rec_from_priors, options, priors=priors)

    rec_from_db = pycolmap.Reconstruction(gt_reconstruction)
    pycolmap.bundle_adjustment(
        rec_from_db, options, database_path=database_path
    )

    with pytest.raises(ValueError, match="database_path must be specified"):
        pycolmap.bundle_adjustment(
            pycolmap.Reconstruction(gt_reconstruction), options
        )

    # Deprecated factory shims emit DeprecationWarning and still solve.
    prior_options = pycolmap.PosePriorBundleAdjustmentOptions()
    with pytest.warns(DeprecationWarning):
        deprecated_adjuster = pycolmap.create_pose_prior_bundle_adjuster(
            pycolmap.BundleAdjustmentOptions(),
            prior_options,
            config,
            pose_priors,
            pycolmap.Reconstruction(gt_reconstruction),
        )
    assert deprecated_adjuster.solve().is_solution_usable()
    with pytest.warns(DeprecationWarning):
        deprecated_ceres_adjuster = (
            pycolmap.create_pose_prior_ceres_bundle_adjuster(
                pycolmap.BundleAdjustmentOptions(),
                prior_options,
                config,
                pose_priors,
                pycolmap.Reconstruction(gt_reconstruction),
            )
        )
    assert deprecated_ceres_adjuster.solve().is_solution_usable()
