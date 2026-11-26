# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import pycolmap


def test_ba_covariance_options_params_enum() -> None:
    assert {
        k: int(v)
        for k, v in pycolmap.BACovarianceOptionsParams.__members__.items()
    } == {
        "POSES": 0,
        "POINTS": 1,
        "POSES_AND_POINTS": 2,
        "ALL": 3,
    }


def test_ba_covariance_options_variance_scaling_enum() -> None:
    assert {
        k: int(v)
        for k, v in (
            pycolmap.BACovarianceOptionsVarianceScaling.__members__.items()
        )
    } == {
        "NONE": 0,
        "POSTERIOR": 1,
        "SPATIAL_DEFF": 2,
    }


def test_ba_covariance_options_default_init() -> None:
    options = pycolmap.BACovarianceOptions()
    assert options is not None
    assert (
        options.variance_scaling
        == pycolmap.BACovarianceOptionsVarianceScaling.NONE
    )


def test_ba_covariance_options_params_readwrite() -> None:
    options = pycolmap.BACovarianceOptions()
    options.params = pycolmap.BACovarianceOptionsParams.POINTS
    assert options.params == pycolmap.BACovarianceOptionsParams.POINTS


def test_ba_covariance_options_variance_scaling_readwrite() -> None:
    options = pycolmap.BACovarianceOptions()
    options.variance_scaling = (
        pycolmap.BACovarianceOptionsVarianceScaling.SPATIAL_DEFF
    )
    assert (
        options.variance_scaling
        == pycolmap.BACovarianceOptionsVarianceScaling.SPATIAL_DEFF
    )


def test_ba_covariance_options_damping_readwrite() -> None:
    options = pycolmap.BACovarianceOptions()
    assert isinstance(options.damping, float)
    options.damping = 1e-6
    assert options.damping == 1e-6


def test_experimental_pose_param_default_init() -> None:
    param = pycolmap.ExperimentalPoseParam()
    assert param is not None


def test_experimental_pose_param_image_id_readwrite() -> None:
    param = pycolmap.ExperimentalPoseParam()
    param.image_id = 42
    assert param.image_id == 42


def test_estimate_ba_covariance_is_callable() -> None:
    assert callable(pycolmap.estimate_ba_covariance)


def test_estimate_ba_covariance_variance_scaling() -> None:
    pycolmap.set_random_seed(0)
    dataset_options = pycolmap.SyntheticDatasetOptions()
    dataset_options.num_rigs = 1
    dataset_options.num_cameras_per_rig = 1
    dataset_options.num_frames_per_rig = 8
    dataset_options.num_points3D = 200
    reconstruction = pycolmap.synthesize_dataset(dataset_options)

    noise_options = pycolmap.SyntheticNoiseOptions()
    noise_options.point2D_stddev = 2.0
    pycolmap.synthesize_noise(noise_options, reconstruction)

    config = pycolmap.BundleAdjustmentConfig()
    image_ids = sorted(reconstruction.reg_image_ids())
    for image_id in image_ids:
        config.add_image(image_id)
    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.print_summary = False
    ba = pycolmap.create_default_ceres_bundle_adjuster(
        ba_options, config, reconstruction
    )
    summary = ba.solve()
    assert summary.is_solution_usable()

    none_cov = pycolmap.estimate_ba_covariance(
        pycolmap.BACovarianceOptions(), reconstruction, ba
    )
    assert none_cov is not None
    assert none_cov.posterior_variance_factor == 1.0
    assert none_cov.spatial_design_effect == 1.0
    assert none_cov.variance_factor == 1.0

    post_opts = pycolmap.BACovarianceOptions()
    post_opts.variance_scaling = (
        pycolmap.BACovarianceOptionsVarianceScaling.POSTERIOR
    )
    post_cov = pycolmap.estimate_ba_covariance(post_opts, reconstruction, ba)
    assert post_cov is not None
    assert 2.5 < post_cov.posterior_variance_factor < 5.5
    assert post_cov.spatial_design_effect == 1.0
    assert post_cov.variance_factor == post_cov.posterior_variance_factor

    deff_opts = pycolmap.BACovarianceOptions()
    deff_opts.variance_scaling = (
        pycolmap.BACovarianceOptionsVarianceScaling.SPATIAL_DEFF
    )
    deff_cov = pycolmap.estimate_ba_covariance(deff_opts, reconstruction, ba)
    assert deff_cov is not None
    assert np.isclose(
        deff_cov.posterior_variance_factor, post_cov.posterior_variance_factor
    )
    assert 1.0 <= deff_cov.spatial_design_effect < 1.6
    assert np.isclose(
        deff_cov.variance_factor,
        deff_cov.posterior_variance_factor * deff_cov.spatial_design_effect,
    )

    id0 = image_ids[0]
    pt0 = next(iter(reconstruction.points3D.keys()))
    np.testing.assert_allclose(
        deff_cov.get_cam_cov_from_world(id0),
        deff_cov.variance_factor * none_cov.get_cam_cov_from_world(id0),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        deff_cov.get_point_cov(pt0),
        deff_cov.posterior_variance_factor * none_cov.get_point_cov(pt0),
        rtol=1e-10,
        atol=1e-12,
    )
