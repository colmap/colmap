import pycolmap


def test_database_cache_options_default_init():
    options = pycolmap.DatabaseCacheOptions()
    assert options is not None


def test_database_cache_options_min_num_matches_readwrite():
    options = pycolmap.DatabaseCacheOptions()
    options.min_num_matches = 20
    assert options.min_num_matches == 20


def test_database_cache_options_ignore_watermarks_readwrite():
    options = pycolmap.DatabaseCacheOptions()
    options.ignore_watermarks = True
    assert options.ignore_watermarks is True


def test_database_cache_options_load_all_images_readwrite():
    options = pycolmap.DatabaseCacheOptions()
    options.load_all_images = True
    assert options.load_all_images is True


def test_database_cache_create(populated_database):
    database, _, _ = populated_database
    options = pycolmap.DatabaseCacheOptions()
    options.load_all_images = True
    cache = pycolmap.DatabaseCache.create(database, options)
    assert cache.num_cameras() > 0
    assert cache.num_images() > 0


def test_database_cache_cameras_property(populated_database):
    database, _, _ = populated_database
    options = pycolmap.DatabaseCacheOptions()
    options.load_all_images = True
    cache = pycolmap.DatabaseCache.create(database, options)
    assert cache.cameras is not None


def test_database_cache_images_property(populated_database):
    database, _, _ = populated_database
    options = pycolmap.DatabaseCacheOptions()
    options.load_all_images = True
    cache = pycolmap.DatabaseCache.create(database, options)
    assert cache.images is not None
