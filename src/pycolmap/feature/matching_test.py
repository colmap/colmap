# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
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


def test_cusift_matching_options_defaults() -> None:
    # Ratio test on L2 distances, in both directions under cross-check, as the
    # default CPU matcher.
    options = pycolmap.CusiftMatchingOptions()
    assert options.cross_check_reverse_ratio is True
    assert options.distance_type == pycolmap.CusiftDistanceType.L2
    assert (
        pycolmap.CusiftDistanceType("ANGULAR")
        == pycolmap.CusiftDistanceType.ANGULAR
    )


def test_cusift_matching_options() -> None:
    options = pycolmap.FeatureMatchingOptions(
        type=pycolmap.FeatureMatcherType.CUSIFT_GEMM
    )
    options.cusift.cross_check_reverse_ratio = False
    options.cusift.distance_type = pycolmap.CusiftDistanceType.ANGULAR
    assert options.cusift.cross_check_reverse_ratio is False
    assert options.cusift.distance_type == pycolmap.CusiftDistanceType.ANGULAR
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


CusiftFeatures = tuple[pycolmap.FeatureKeypoints, pycolmap.FeatureDescriptors]


@pytest.fixture
def cusift_features(
    requires_cusift: None,
) -> CusiftFeatures:
    extractor = pycolmap.FeatureExtractor.create(
        options=pycolmap.FeatureExtractionOptions(
            type=pycolmap.FeatureExtractorType.CUSIFT
        ),
        device=pycolmap.Device.cuda,
    )
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(256, 320), dtype=np.uint8)
    return extractor.extract_from_uint8_array(image)


def create_cusift_matcher() -> pycolmap.FeatureMatcher:
    options = pycolmap.FeatureMatchingOptions(
        type=pycolmap.FeatureMatcherType.CUSIFT_GEMM
    )
    return pycolmap.FeatureMatcher.create(
        options=options, device=pycolmap.Device.cuda
    )


def test_cusift_feature_matcher_matches_a_self_pair(
    cusift_features: CusiftFeatures,
) -> None:
    keypoints, descriptors = cusift_features
    assert len(keypoints) > 100

    matches = create_cusift_matcher().match(
        keypoints, descriptors, keypoints, descriptors
    )

    assert matches.shape[1] == 2
    assert len(matches) > len(keypoints) // 2
    np.testing.assert_array_equal(matches[:, 0], matches[:, 1])


def test_cusift_feature_matcher_finds_nothing_in_empty_descriptors(
    cusift_features: CusiftFeatures,
) -> None:
    keypoints, descriptors = cusift_features
    empty = pycolmap.FeatureDescriptors(
        type=pycolmap.FeatureExtractorType.SIFT,
        data=np.zeros((0, 128), dtype=np.uint8),
    )

    matches = create_cusift_matcher().match(
        keypoints, descriptors, pycolmap.FeatureKeypoints(), empty
    )

    assert matches.shape == (0, 2)
