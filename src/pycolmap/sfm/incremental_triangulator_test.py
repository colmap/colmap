# SPDX-License-Identifier: BSD-3-Clause

import pycolmap


def test_incremental_triangulator_options_init() -> None:
    options = pycolmap.IncrementalTriangulatorOptions()
    assert options is not None


def test_incremental_triangulator_options_max_transitivity() -> None:
    options = pycolmap.IncrementalTriangulatorOptions()
    options.max_transitivity = 2
    assert options.max_transitivity == 2


def test_incremental_triangulator_options_create_max_angle_error() -> None:
    options = pycolmap.IncrementalTriangulatorOptions()
    options.create_max_angle_error = 3.0
    assert options.create_max_angle_error == 3.0


def test_incremental_triangulator_covariance_options_readwrite() -> None:
    options = pycolmap.IncrementalTriangulatorOptions()
    options.use_covariance = False
    options.inlier_chi2_threshold = 7.0
    options.re_chi2_threshold = 14.0
    options.max_relative_depth_uncertainty = 0.1
    assert options.measurement_noise_dof == 3.0
    assert options.covariance_legacy_gates
    assert options.covariance_legacy_fallback
    options.measurement_noise_dof = float("inf")
    options.covariance_legacy_gates = False
    options.covariance_legacy_fallback = False
    assert not options.use_covariance
    assert options.inlier_chi2_threshold == 7.0
    assert options.re_chi2_threshold == 14.0
    assert options.max_relative_depth_uncertainty == 0.1
    assert options.measurement_noise_dof == float("inf")
    assert not options.covariance_legacy_gates
    assert not options.covariance_legacy_fallback


def test_incremental_triangulator_options_check() -> None:
    options = pycolmap.IncrementalTriangulatorOptions()
    assert options.check()


def test_incremental_triangulator_class_exists() -> None:
    assert hasattr(pycolmap, "IncrementalTriangulator")
