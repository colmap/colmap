# SPDX-License-Identifier: BSD-3-Clause

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
# cusift is a CUDA-only SIFT extractor that replicates COLMAP's VLFeat CPU
# backend. It is driven entirely by SiftExtractionOptions and tags its
# descriptors as SIFT, so there is no separate options class to bind and the
# generic FeatureExtractor API is the whole Python surface.

requires_cusift = pytest.mark.skipif(
    not pycolmap.has_cuda, reason="cusift requires CUDA"
)


def test_cusift_extractor_type_enum():
    assert pycolmap.FeatureExtractorType.CUSIFT is not None
    # SIFT must keep its integer identity: descriptor types are persisted in
    # COLMAP databases. CUSIFT's ordinal is deliberately not pinned -- cusift
    # tags its descriptors FeatureExtractorType::SIFT (cusift.cc), so CUSIFT
    # never reaches the database and is appended last, leaving the upstream
    # values untouched whenever new extractor types land in between.
    assert int(pycolmap.FeatureExtractorType.SIFT) == 0
    assert (
        pycolmap.FeatureExtractorType.CUSIFT
        != pycolmap.FeatureExtractorType.SIFT
    )


def test_cusift_options_use_sift_block():
    options = pycolmap.FeatureExtractionOptions(
        type=pycolmap.FeatureExtractorType.CUSIFT
    )
    assert options.type == pycolmap.FeatureExtractorType.CUSIFT
    assert isinstance(options.sift, pycolmap.SiftExtractionOptions)
    assert options.requires_rgb() is False
    assert options.requires_opengl() is False
    assert options.eff_max_image_size() == 3200


def test_cusift_options_check_requires_gpu():
    options = pycolmap.FeatureExtractionOptions(
        type=pycolmap.FeatureExtractorType.CUSIFT
    )
    options.use_gpu = False
    assert options.check() is False


@requires_cusift
def test_cusift_extractor_create():
    options = pycolmap.FeatureExtractionOptions(
        type=pycolmap.FeatureExtractorType.CUSIFT
    )
    extractor = pycolmap.FeatureExtractor.create(
        options=options, device=pycolmap.Device.cuda
    )
    assert extractor is not None


@requires_cusift
def test_cusift_extract_from_uint8_array():
    options = pycolmap.FeatureExtractionOptions(
        type=pycolmap.FeatureExtractorType.CUSIFT
    )
    extractor = pycolmap.FeatureExtractor.create(
        options=options, device=pycolmap.Device.cuda
    )
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(256, 320), dtype=np.uint8)
    keypoints, descriptors = extractor.extract_from_uint8_array(image)
    assert len(keypoints) > 0
    assert descriptors.data.shape == (len(keypoints), 128)
    assert descriptors.data.dtype == np.uint8
    # Descriptors are plain SIFT descriptors, so the existing SIFT matchers and
    # databases consume them unchanged.
    assert descriptors.type == pycolmap.FeatureExtractorType.SIFT


@requires_cusift
def test_cusift_matches_vlfeat_cpu_feature_count():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(480, 640), dtype=np.uint8)

    def extract(feature_type, device):
        options = pycolmap.FeatureExtractionOptions(type=feature_type)
        extractor = pycolmap.FeatureExtractor.create(
            options=options, device=device
        )
        return extractor.extract_from_uint8_array(image)

    kp_cusift, _ = extract(
        pycolmap.FeatureExtractorType.CUSIFT, pycolmap.Device.cuda
    )
    kp_vlfeat, _ = extract(
        pycolmap.FeatureExtractorType.SIFT, pycolmap.Device.cpu
    )
    # cusift is a VLFeat replica; HANDOVER.md §1 measures counts 0.016% apart on
    # real imagery. Allow 1% here, which still fails loudly on a real
    # regression while tolerating noise-image edge cases.
    assert abs(len(kp_cusift) - len(kp_vlfeat)) <= 0.01 * len(kp_vlfeat)


@requires_cusift
@pytest.mark.parametrize(
    "attr,value",
    [
        ("estimate_affine_shape", True),
        ("domain_size_pooling", True),
        ("upright", True),
        ("darkness_adaptivity", True),
        ("first_octave", -2),
        ("octave_resolution", 4),
    ],
)
def test_cusift_rejects_unsupported_options(attr, value):
    options = pycolmap.FeatureExtractionOptions(
        type=pycolmap.FeatureExtractorType.CUSIFT
    )
    setattr(options.sift, attr, value)
    # Rejected loudly rather than silently ignored: a no-op would produce a
    # database that looks fine and reconstructs worse. Match on the message so
    # an unrelated failure cannot make this pass.
    with pytest.raises(Exception, match="cusift"):
        pycolmap.FeatureExtractor.create(
            options=options, device=pycolmap.Device.cuda
        )


# --------------------------------------------------- GPU backend parity
# Every GPU extraction assertion is run against BOTH backends from one body, so
# CUSIFT is held to exactly what SIFT-GPU is held to. Parametrized rather than
# duplicated so the two can never drift apart.

GPU_EXTRACTOR_TYPES = [
    pycolmap.FeatureExtractorType.SIFT,
    pycolmap.FeatureExtractorType.CUSIFT,
]

# Same list, but the SIFT-GPU parameter is marked, for the tests that assert the
# >= 12 keypoint floor on the synthetic square. Upstream SiftGPU's detector is
# not content-deterministic: measured over 12 fresh processes it returned 24
# keypoints 9 times and 8, 16 and 20 once each -- four different descriptor sets
# for one input. Below 12 the floor fails, so those tests flake ~1 run in 10
# through no fault of cusift, which returned a bit-identical set of 22 in all 12
# (upstream colmap#1197, open since 2021; upstream CI excludes feature/sift_test
# on every CUDA config for the same reason).
#
# Marked rather than removed so the SIFT path keeps its coverage by default, and
# rather than applied to GPU_EXTRACTOR_TYPES wholesale so the three tests that
# do NOT assert the floor keep testing both backends unconditionally.
# Deselect with: pytest -m 'not siftgpu_flaky'
GPU_EXTRACTOR_TYPES_KEYPOINT_FLOOR = [
    pytest.param(
        pycolmap.FeatureExtractorType.SIFT, marks=pytest.mark.siftgpu_flaky
    ),
    pycolmap.FeatureExtractorType.CUSIFT,
]


def _gpu_options(feature_type):
    options = pycolmap.FeatureExtractionOptions(type=feature_type)
    options.use_gpu = True
    options.gpu_index = "0"
    return options


def _synthetic_square(size=256):
    """Same synthetic image the C++ feature tests use.

    NOT usable for MATCHER tests: the centred square has 4-fold symmetry, so
    every corner descriptor's second-best is as close as its best and Lowe's
    ratio test annihilates all of them. Measured: SIFT-GPU matched against
    itself, byte-identical descriptors, returns 0 matches here at the default
    max_ratio=0.8 with cross-checking. Use _synthetic_texture for matching.
    """
    image = np.zeros((size, size), dtype=np.uint8)
    lo, hi = size // 2 - size // 8, size // 2 + size // 8
    image[lo:hi, lo:hi] = 255
    return image


def _synthetic_texture(size=512):
    """Asymmetric, distinctive texture for MATCHING tests.

    _synthetic_square is deliberately left alone (five other tests depend on
    it), but it is unusable for matching: a centred square has 4-fold symmetry,
    so every corner descriptor has three near-identical twins and Lowe's ratio
    test correctly rejects them all. Measured on that fixture, SIFT-GPU matched
    against ITSELF -- byte-identical descriptors -- returned 0 matches at the
    default max_ratio=0.8 with cross-checking, which is the fixture failing, not
    the matcher.

    This pattern places blobs of varying size and intensity at irregular
    positions from a fixed seed, so each neighbourhood is locally unique and the
    ratio test has a genuine second-best to reject.
    """
    rng = np.random.default_rng(12345)
    image = np.zeros((size, size), dtype=np.uint8)
    for _ in range(45):
        h = int(rng.integers(size // 32, size // 10))
        w = int(rng.integers(size // 32, size // 10))
        r = int(rng.integers(0, size - h))
        c = int(rng.integers(0, size - w))
        image[r : r + h, c : c + w] = int(rng.integers(90, 256))
    return image


@requires_cusift
@pytest.mark.parametrize("feature_type", GPU_EXTRACTOR_TYPES, ids=str)
def test_gpu_extractor_create(feature_type):
    options = _gpu_options(feature_type)
    assert options.check() is True
    assert options.requires_opengl() is False
    extractor = pycolmap.FeatureExtractor.create(
        options=options, device=pycolmap.Device.cuda
    )
    assert extractor is not None


@requires_cusift
@pytest.mark.parametrize(
    "feature_type", GPU_EXTRACTOR_TYPES_KEYPOINT_FLOOR, ids=str
)
def test_gpu_extract_from_uint8_array(feature_type):
    extractor = pycolmap.FeatureExtractor.create(
        options=_gpu_options(feature_type), device=pycolmap.Device.cuda
    )
    keypoints, descriptors = extractor.extract_from_uint8_array(
        _synthetic_square()
    )
    assert len(keypoints) >= 12
    assert descriptors.data.shape == (len(keypoints), 128)
    assert descriptors.data.dtype == np.uint8
    # Both backends emit plain SIFT descriptors.
    assert descriptors.type == pycolmap.FeatureExtractorType.SIFT
    norms = np.linalg.norm(descriptors.data.astype(np.float64), axis=1)
    assert np.all(np.abs(norms - 512) < 1)


@requires_cusift
@pytest.mark.parametrize(
    "feature_type", GPU_EXTRACTOR_TYPES_KEYPOINT_FLOOR, ids=str
)
def test_gpu_extract_from_float32_array(feature_type):
    extractor = pycolmap.FeatureExtractor.create(
        options=_gpu_options(feature_type), device=pycolmap.Device.cuda
    )
    image = _synthetic_square().astype(np.float32) / 255.0
    keypoints, descriptors = extractor.extract_from_float32_array(image)
    assert len(keypoints) >= 12
    assert descriptors.data.shape == (len(keypoints), 128)


@requires_cusift
@pytest.mark.parametrize("feature_type", GPU_EXTRACTOR_TYPES, ids=str)
def test_gpu_keypoints_matrix_roundtrip(feature_type):
    extractor = pycolmap.FeatureExtractor.create(
        options=_gpu_options(feature_type), device=pycolmap.Device.cuda
    )
    keypoints, _ = extractor.extract_from_uint8_array(_synthetic_square())
    kp = np.asarray(pycolmap.keypoints_to_matrix(keypoints))
    assert kp.shape == (len(keypoints), 4)
    # x, y inside the image; positive scale; orientation in [-pi, pi].
    assert np.all(kp[:, 0] >= 0) and np.all(kp[:, 0] <= 256)
    assert np.all(kp[:, 1] >= 0) and np.all(kp[:, 1] <= 256)
    assert np.all(kp[:, 2] > 0)
    assert np.all(np.abs(kp[:, 3]) <= np.pi + 1e-6)


@requires_cusift
@pytest.mark.parametrize("feature_type", GPU_EXTRACTOR_TYPES, ids=str)
@pytest.mark.parametrize("device", ["cuda", "cpu"], ids=str)
def test_gpu_descriptors_matchable_by_sift_matcher(feature_type, device):
    """Either backend's descriptors go straight into SIFT_BRUTEFORCE."""
    extractor = pycolmap.FeatureExtractor.create(
        options=_gpu_options(feature_type), device=pycolmap.Device.cuda
    )
    keypoints, descriptors = extractor.extract_from_uint8_array(
        _synthetic_square()
    )

    match_device = getattr(pycolmap.Device, device)
    options = pycolmap.FeatureMatchingOptions(
        type=pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
    )
    options.use_gpu = device == "cuda"
    # image_id is not exposed by the binding, so the faiss index cache is
    # unusable from Python; force the exact brute-force CPU path.
    options.sift.cpu_brute_force_matcher = device == "cpu"
    matcher = pycolmap.FeatureMatcher.create(
        options=options, device=match_device
    )

    matches = np.asarray(
        matcher.match(keypoints, descriptors, keypoints, descriptors)
    )
    assert len(matches) > 0
    # Self-match: every accepted pair must be the identity.
    assert np.all(matches[:, 0] == matches[:, 1])


@requires_cusift
def test_cusift_and_siftgpu_cross_match():
    """CUSIFT features matched against SIFT-GPU features of the SAME image.

    This is the drop-in property: correct matches are the spatial identity map,
    so nearly every returned pair must connect coincident keypoints.
    """
    # Asymmetric fixture: see _synthetic_texture. A centred square cannot be
    # matched even against itself at default settings.
    image = _synthetic_texture(512)

    def extract(feature_type):
        extractor = pycolmap.FeatureExtractor.create(
            options=_gpu_options(feature_type), device=pycolmap.Device.cuda
        )
        kps, desc = extractor.extract_from_uint8_array(image)
        return kps, desc, np.asarray(pycolmap.keypoints_to_matrix(kps))

    cu_kp, cu_desc, cu_mat = extract(pycolmap.FeatureExtractorType.CUSIFT)
    sg_kp, sg_desc, sg_mat = extract(pycolmap.FeatureExtractorType.SIFT)

    options = pycolmap.FeatureMatchingOptions(
        type=pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
    )
    options.use_gpu = True
    matcher = pycolmap.FeatureMatcher.create(
        options=options, device=pycolmap.Device.cuda
    )
    matches = np.asarray(matcher.match(cu_kp, cu_desc, sg_kp, sg_desc))
    assert len(matches) > 0

    disp = np.linalg.norm(
        cu_mat[matches[:, 0], :2] - sg_mat[matches[:, 1], :2], axis=1
    )
    assert np.mean(disp <= 2.0) > 0.95
