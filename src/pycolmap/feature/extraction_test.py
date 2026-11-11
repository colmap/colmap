# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable

import numpy as np
import pytest

import pycolmap


def test_normalization_enum() -> None:
    assert pycolmap.Normalization.L1_ROOT is not None
    assert pycolmap.Normalization.L2 is not None


def test_sift_extraction_options_default_init() -> None:
    options = pycolmap.SiftExtractionOptions()
    assert options is not None


def test_sift_extraction_options_num_octaves_readwrite() -> None:
    options = pycolmap.SiftExtractionOptions()
    original = options.num_octaves
    assert isinstance(original, int)
    options.num_octaves = 5
    assert options.num_octaves == 5


def test_sift_extraction_options_max_num_features_readwrite() -> None:
    options = pycolmap.SiftExtractionOptions()
    assert isinstance(options.max_num_features, int)
    options.max_num_features = 4096
    assert options.max_num_features == 4096


def test_sift_extraction_options_peak_threshold_readwrite() -> None:
    options = pycolmap.SiftExtractionOptions()
    assert isinstance(options.peak_threshold, float)
    options.peak_threshold = 0.01
    assert abs(options.peak_threshold - 0.01) < 1e-6


def test_sift_extraction_options_check() -> None:
    options = pycolmap.SiftExtractionOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_feature_extraction_options_default_init() -> None:
    options = pycolmap.FeatureExtractionOptions()
    assert options is not None


def test_feature_extraction_options_type_readwrite() -> None:
    options = pycolmap.FeatureExtractionOptions()
    options.type = pycolmap.FeatureExtractorType.SIFT
    assert options.type == pycolmap.FeatureExtractorType.SIFT


def test_feature_extraction_options_max_image_size_readwrite() -> None:
    options = pycolmap.FeatureExtractionOptions()
    options.max_image_size = 2048
    assert options.max_image_size == 2048


def test_feature_extraction_options_num_threads_readwrite() -> None:
    options = pycolmap.FeatureExtractionOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_feature_extraction_options_use_gpu_readwrite() -> None:
    options = pycolmap.FeatureExtractionOptions()
    options.use_gpu = False
    assert options.use_gpu is False


def test_feature_extraction_options_sift_property() -> None:
    options = pycolmap.FeatureExtractionOptions()
    sift = options.sift
    assert isinstance(sift, pycolmap.SiftExtractionOptions)


def test_feature_extraction_options_requires_rgb() -> None:
    options = pycolmap.FeatureExtractionOptions()
    result = options.requires_rgb()
    assert isinstance(result, bool)


def test_feature_extraction_options_requires_opengl() -> None:
    options = pycolmap.FeatureExtractionOptions()
    result = options.requires_opengl()
    assert isinstance(result, bool)


def test_feature_extraction_options_eff_max_image_size() -> None:
    options = pycolmap.FeatureExtractionOptions()
    result = options.eff_max_image_size()
    assert isinstance(result, int)
    assert result > 0


def test_feature_extraction_options_check() -> None:
    options = pycolmap.FeatureExtractionOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_feature_extractor_create() -> None:
    extractor = pycolmap.FeatureExtractor.create(device=pycolmap.Device.cpu)
    assert extractor is not None


# ------------------------------------------------------------------ cusift
# Binding-level checks only; cusift's behaviour is tested in C++
# (src/colmap/feature/cusift_test.cc).


def create_cusift_extraction_options() -> pycolmap.FeatureExtractionOptions:
    return pycolmap.FeatureExtractionOptions(
        type=pycolmap.FeatureExtractorType.CUSIFT
    )


def test_cusift_extractor_type_enum() -> None:
    assert pycolmap.FeatureExtractorType.CUSIFT is not None
    # SIFT must keep its integer identity: descriptor types are persisted in
    # COLMAP databases. CUSIFT's ordinal is deliberately not pinned -- cusift
    # tags its descriptors FeatureExtractorType::SIFT (cusift.cc), so its own
    # output never stores CUSIFT, and it is appended last, leaving the upstream
    # values untouched whenever new extractor types land in between.
    assert int(pycolmap.FeatureExtractorType.SIFT) == 0
    assert (
        pycolmap.FeatureExtractorType.CUSIFT
        != pycolmap.FeatureExtractorType.SIFT
    )


def test_cusift_descriptors_round_trip_through_database(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
    database, _, image_id = populated_database
    data = np.array([[0, 7, 255], [1, 128, 64]], dtype=np.uint8)
    database.write_descriptors(
        image_id,
        pycolmap.FeatureDescriptors(
            type=pycolmap.FeatureExtractorType.CUSIFT, data=data
        ),
    )
    read_back = database.read_descriptors(image_id)
    assert read_back.type == pycolmap.FeatureExtractorType.CUSIFT
    as_float = read_back.to_float()
    np.testing.assert_array_equal(as_float.data, data.astype(np.float32))
    np.testing.assert_array_equal(as_float.to_bytes().data, data)


def test_cusift_extraction_options_defaults() -> None:
    # The CPU (VLFeat) backend's octave count.
    assert (
        pycolmap.CusiftExtractionOptions().num_octaves
        == pycolmap.SiftExtractionOptions().num_octaves
    )


def test_cusift_extraction_options() -> None:
    options = pycolmap.FeatureExtractionOptions(
        type=pycolmap.FeatureExtractorType.CUSIFT
    )
    assert isinstance(options.sift, pycolmap.SiftExtractionOptions)
    options.cusift.num_octaves = -1
    assert options.cusift.num_octaves == -1
    assert options.cusift.check() is True


def use_cpu(options: pycolmap.FeatureExtractionOptions) -> None:
    options.use_gpu = False


def set_zero_octaves(options: pycolmap.FeatureExtractionOptions) -> None:
    options.cusift.num_octaves = 0


def enable_affine_shape(options: pycolmap.FeatureExtractionOptions) -> None:
    options.sift.estimate_affine_shape = True


# requires_cusift runs these only where the default options pass check(), so
# each rejection comes from its one change: to the device, to cusift's own
# options, or to a SIFT option cusift does not support.
@pytest.mark.usefixtures("requires_cusift")
@pytest.mark.parametrize(
    "invalidate",
    [
        pytest.param(use_cpu, id="cpu"),
        pytest.param(set_zero_octaves, id="zero_octaves"),
        pytest.param(enable_affine_shape, id="affine_shape"),
    ],
)
def test_cusift_feature_extraction_options_check_rejects(
    invalidate: Callable[[pycolmap.FeatureExtractionOptions], None],
) -> None:
    options = create_cusift_extraction_options()

    invalidate(options)

    assert options.check() is False


@pytest.mark.usefixtures("requires_cusift")
def test_cusift_extract_from_uint8_array(
    overlapping_squares_image: np.ndarray,
) -> None:
    extractor = pycolmap.FeatureExtractor.create(
        options=create_cusift_extraction_options(), device=pycolmap.Device.cuda
    )
    height, width = overlapping_squares_image.shape

    keypoints, descriptors = extractor.extract_from_uint8_array(
        overlapping_squares_image
    )

    assert len(keypoints) >= 200
    assert all(0 <= kp.x < width and 0 <= kp.y < height for kp in keypoints)
    assert descriptors.data.shape == (len(keypoints), 128)
    assert descriptors.data.dtype == np.uint8
    # COLMAP scales unit descriptors by 512 before quantizing them.
    norms = np.linalg.norm(descriptors.data.astype(np.float64), axis=1)
    np.testing.assert_allclose(norms, 512, atol=8)
    # Descriptors are plain SIFT descriptors, so the existing SIFT matchers and
    # databases consume them unchanged.
    assert descriptors.type == pycolmap.FeatureExtractorType.SIFT
