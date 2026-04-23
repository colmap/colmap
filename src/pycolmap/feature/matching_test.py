import pycolmap


def test_feature_matcher_type_enum():
    assert pycolmap.FeatureMatcherType.UNDEFINED is not None
    assert pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE is not None


def test_sift_matching_options_default_init():
    options = pycolmap.SiftMatchingOptions()
    assert options is not None


def test_sift_matching_options_check():
    options = pycolmap.SiftMatchingOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_feature_matching_options_default_init():
    options = pycolmap.FeatureMatchingOptions()
    assert options is not None


def test_feature_matching_options_type_readwrite():
    options = pycolmap.FeatureMatchingOptions()
    options.type = pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
    assert options.type == pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE


def test_feature_matching_options_num_threads_readwrite():
    options = pycolmap.FeatureMatchingOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_feature_matching_options_use_gpu_readwrite():
    options = pycolmap.FeatureMatchingOptions()
    options.use_gpu = False
    assert options.use_gpu is False


def test_feature_matching_options_max_num_matches_readwrite():
    options = pycolmap.FeatureMatchingOptions()
    options.max_num_matches = 16384
    assert options.max_num_matches == 16384


def test_feature_matching_options_guided_matching_readwrite():
    options = pycolmap.FeatureMatchingOptions()
    options.guided_matching = True
    assert options.guided_matching is True
    options.guided_matching = False
    assert options.guided_matching is False


def test_feature_matching_options_skip_geometric_verification_readwrite():
    options = pycolmap.FeatureMatchingOptions()
    options.skip_geometric_verification = True
    assert options.skip_geometric_verification is True
    options.skip_geometric_verification = False
    assert options.skip_geometric_verification is False


def test_feature_matching_options_check():
    options = pycolmap.FeatureMatchingOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_feature_matcher_create():
    matcher = pycolmap.FeatureMatcher.create(device=pycolmap.Device.cpu)
    assert matcher is not None
