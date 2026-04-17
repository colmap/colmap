import pycolmap


def test_normalization_enum():
    assert pycolmap.Normalization.L1_ROOT is not None
    assert pycolmap.Normalization.L2 is not None


def test_sift_extraction_options_default_init():
    options = pycolmap.SiftExtractionOptions()
    assert options is not None


def test_sift_extraction_options_num_octaves_readwrite():
    options = pycolmap.SiftExtractionOptions()
    original = options.num_octaves
    assert isinstance(original, int)
    options.num_octaves = 5
    assert options.num_octaves == 5


def test_sift_extraction_options_max_num_features_readwrite():
    options = pycolmap.SiftExtractionOptions()
    assert isinstance(options.max_num_features, int)
    options.max_num_features = 4096
    assert options.max_num_features == 4096


def test_sift_extraction_options_peak_threshold_readwrite():
    options = pycolmap.SiftExtractionOptions()
    assert isinstance(options.peak_threshold, float)
    options.peak_threshold = 0.01
    assert abs(options.peak_threshold - 0.01) < 1e-6


def test_sift_extraction_options_check():
    options = pycolmap.SiftExtractionOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_feature_extraction_options_default_init():
    options = pycolmap.FeatureExtractionOptions()
    assert options is not None


def test_feature_extraction_options_type_readwrite():
    options = pycolmap.FeatureExtractionOptions()
    options.type = pycolmap.FeatureExtractorType.SIFT
    assert options.type == pycolmap.FeatureExtractorType.SIFT


def test_feature_extraction_options_max_image_size_readwrite():
    options = pycolmap.FeatureExtractionOptions()
    options.max_image_size = 2048
    assert options.max_image_size == 2048


def test_feature_extraction_options_num_threads_readwrite():
    options = pycolmap.FeatureExtractionOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_feature_extraction_options_use_gpu_readwrite():
    options = pycolmap.FeatureExtractionOptions()
    options.use_gpu = False
    assert options.use_gpu is False


def test_feature_extraction_options_sift_property():
    options = pycolmap.FeatureExtractionOptions()
    sift = options.sift
    assert isinstance(sift, pycolmap.SiftExtractionOptions)


def test_feature_extraction_options_requires_rgb():
    options = pycolmap.FeatureExtractionOptions()
    result = options.requires_rgb()
    assert isinstance(result, bool)


def test_feature_extraction_options_requires_opengl():
    options = pycolmap.FeatureExtractionOptions()
    result = options.requires_opengl()
    assert isinstance(result, bool)


def test_feature_extraction_options_eff_max_image_size():
    options = pycolmap.FeatureExtractionOptions()
    result = options.eff_max_image_size()
    assert isinstance(result, int)
    assert result > 0


def test_feature_extraction_options_check():
    options = pycolmap.FeatureExtractionOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_feature_extractor_create():
    extractor = pycolmap.FeatureExtractor.create(device=pycolmap.Device.cpu)
    assert extractor is not None


def test_sift_deprecated_class():
    if hasattr(pycolmap, "Sift"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sift = pycolmap.Sift(device=pycolmap.Device.cpu)
            assert sift is not None
            assert sift.options is not None
            assert sift.device is not None
