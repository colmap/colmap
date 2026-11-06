# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pycolmap


def test_feature_matcher_type_enum() -> None:
    assert pycolmap.FeatureMatcherType.UNDEFINED is not None
    assert pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE is not None
    assert pycolmap.FeatureMatcherType.CUSIFT_GEMM is not None


def test_sift_matching_options_default_init() -> None:
    options = pycolmap.SiftMatchingOptions()
    assert options is not None


def test_sift_matching_options_check() -> None:
    options = pycolmap.SiftMatchingOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_cusift_matching_options() -> None:
    options = pycolmap.FeatureMatchingOptions(
        type=pycolmap.FeatureMatcherType.CUSIFT_GEMM
    )
    options.cusift.cross_check_reverse_ratio = False
    options.cusift.distance_type = pycolmap.CusiftDistanceType.L2
    assert options.cusift.cross_check_reverse_ratio is False
    assert options.cusift.distance_type == pycolmap.CusiftDistanceType.L2
    assert isinstance(options.cusift.check(), bool)


def test_feature_matching_options_default_init() -> None:
    options = pycolmap.FeatureMatchingOptions()
    assert options is not None


def test_feature_matching_options_type_readwrite() -> None:
    options = pycolmap.FeatureMatchingOptions()
    options.type = pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
    assert options.type == pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
    options.type = pycolmap.FeatureMatcherType.CUSIFT_GEMM
    assert options.type == pycolmap.FeatureMatcherType.CUSIFT_GEMM


def test_feature_matching_options_num_threads_readwrite() -> None:
    options = pycolmap.FeatureMatchingOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_feature_matching_options_use_gpu_readwrite() -> None:
    options = pycolmap.FeatureMatchingOptions()
    options.use_gpu = False
    assert options.use_gpu is False


def test_feature_matching_options_max_num_matches_readwrite() -> None:
    options = pycolmap.FeatureMatchingOptions()
    options.max_num_matches = 16384
    assert options.max_num_matches == 16384


def test_feature_matching_options_guided_matching_readwrite() -> None:
    options = pycolmap.FeatureMatchingOptions()
    options.guided_matching = True
    assert options.guided_matching is True
    options.guided_matching = False
    assert options.guided_matching is False


def test_feature_matching_options_skip_geometric_verification_readwrite() -> (
    None
):
    options = pycolmap.FeatureMatchingOptions()
    options.skip_geometric_verification = True
    assert options.skip_geometric_verification is True
    options.skip_geometric_verification = False
    assert options.skip_geometric_verification is False


def test_feature_matching_options_check() -> None:
    options = pycolmap.FeatureMatchingOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_feature_matcher_create() -> None:
    matcher = pycolmap.FeatureMatcher.create(device=pycolmap.Device.cpu)
    assert matcher is not None


@pytest.mark.usefixtures("requires_cusift")
def test_cusift_feature_matcher_create() -> None:
    options = pycolmap.FeatureMatchingOptions(
        type=pycolmap.FeatureMatcherType.CUSIFT_GEMM
    )
    matcher = pycolmap.FeatureMatcher.create(
        options=options, device=pycolmap.Device.cuda
    )
    assert matcher is not None
