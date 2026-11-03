# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pycolmap

requires_cusift = pytest.mark.skipif(
    not pycolmap.has_cuda, reason="cusift requires CUDA"
)


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


def test_cusift_matching_options_default_init():
    options = pycolmap.CusiftMatchingOptions()
    assert options is not None


def test_cusift_matching_options_check():
    options = pycolmap.CusiftMatchingOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_cusift_matching_options_field_readwrite():
    # The two cusift-specific axes. Both were added with the bidirectional
    # cross-check work and neither had a Python test, so the bindings could have
    # silently failed to expose them.
    options = pycolmap.CusiftMatchingOptions()
    assert options.cross_check_reverse_ratio is True, (
        "must default to the SiftGPU/COLMAP-CPU-equivalent bidirectional test"
    )
    options.cross_check_reverse_ratio = False
    assert options.cross_check_reverse_ratio is False
    options.max_ratio = 0.75
    assert options.max_ratio == 0.75
    options.max_distance = 0.6
    assert options.max_distance == 0.6
    options.cross_check = False
    assert options.cross_check is False


def test_cusift_distance_type_enum_and_default():
    assert pycolmap.CusiftDistanceType.L2 is not None
    assert pycolmap.CusiftDistanceType.ANGULAR is not None
    options = pycolmap.CusiftMatchingOptions()
    # ANGULAR is what COLMAP's exact brute-force reference and SiftGPU both do;
    # L2 reproduces the faiss/index convention instead.
    assert options.distance_type == pycolmap.CusiftDistanceType.ANGULAR
    options.distance_type = pycolmap.CusiftDistanceType.L2
    assert options.distance_type == pycolmap.CusiftDistanceType.L2


def test_feature_matching_options_exposes_cusift_block():
    options = pycolmap.FeatureMatchingOptions()
    assert options.cusift is not None
    options.cusift.max_ratio = 0.7
    assert options.cusift.max_ratio == 0.7


def test_cusift_matcher_requires_gpu():
    # cusift has no CPU implementation, so this must be rejected as a bad option
    # rather than surfacing later as an unknown-matcher-type error.
    options = pycolmap.FeatureMatchingOptions(
        type=pycolmap.FeatureMatcherType.CUSIFT_GEMM
    )
    options.use_gpu = False
    assert options.check() is False


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


@requires_cusift
def test_cusift_feature_matcher_create():
    options = pycolmap.FeatureMatchingOptions(
        type=pycolmap.FeatureMatcherType.CUSIFT_GEMM
    )
    matcher = pycolmap.FeatureMatcher.create(
        options=options, device=pycolmap.Device.cuda
    )
    assert matcher is not None
