import pycolmap


def test_ransac_options_default_init() -> None:
    options = pycolmap.RANSACOptions()
    assert options is not None


def test_ransac_options_max_error_readwrite() -> None:
    options = pycolmap.RANSACOptions()
    assert isinstance(options.max_error, float)
    options.max_error = 8.0
    assert options.max_error == 8.0


def test_ransac_options_min_inlier_ratio_readwrite() -> None:
    options = pycolmap.RANSACOptions()
    assert isinstance(options.min_inlier_ratio, float)
    options.min_inlier_ratio = 0.05
    assert options.min_inlier_ratio == 0.05


def test_ransac_options_confidence_readwrite() -> None:
    options = pycolmap.RANSACOptions()
    assert isinstance(options.confidence, float)
    options.confidence = 0.999
    assert options.confidence == 0.999


def test_ransac_options_dyn_num_trials_multiplier_readwrite() -> None:
    options = pycolmap.RANSACOptions()
    assert isinstance(options.dyn_num_trials_multiplier, float)
    options.dyn_num_trials_multiplier = 5.0
    assert options.dyn_num_trials_multiplier == 5.0


def test_ransac_options_min_num_trials_readwrite() -> None:
    options = pycolmap.RANSACOptions()
    assert isinstance(options.min_num_trials, int)
    options.min_num_trials = 500
    assert options.min_num_trials == 500


def test_ransac_options_max_num_trials_readwrite() -> None:
    options = pycolmap.RANSACOptions()
    assert isinstance(options.max_num_trials, int)
    options.max_num_trials = 50000
    assert options.max_num_trials == 50000


def test_ransac_options_random_seed_readwrite() -> None:
    options = pycolmap.RANSACOptions()
    options.random_seed = 42
    assert options.random_seed == 42


def test_ransac_options_num_threads_readwrite() -> None:
    options = pycolmap.RANSACOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_ransac_options_check() -> None:
    options = pycolmap.RANSACOptions()
    options.check()
