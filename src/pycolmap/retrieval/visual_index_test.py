import pycolmap


def test_image_score_class_exists():
    score = pycolmap.ImageScore()
    assert score is not None


def test_image_score_readonly_image_id():
    score = pycolmap.ImageScore()
    assert isinstance(score.image_id, int)


def test_image_score_readonly_score():
    score = pycolmap.ImageScore()
    assert isinstance(score.score, (int, float))


def test_visual_index_index_options_init():
    options = pycolmap.VisualIndex.IndexOptions()
    assert options is not None


def test_visual_index_index_options_num_neighbors():
    options = pycolmap.VisualIndex.IndexOptions()
    original = options.num_neighbors
    options.num_neighbors = 5
    assert options.num_neighbors == 5


def test_visual_index_index_options_num_checks():
    options = pycolmap.VisualIndex.IndexOptions()
    options.num_checks = 128
    assert options.num_checks == 128


def test_visual_index_index_options_num_threads():
    options = pycolmap.VisualIndex.IndexOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_visual_index_query_options_init():
    options = pycolmap.VisualIndex.QueryOptions()
    assert options is not None


def test_visual_index_query_options_max_num_images():
    options = pycolmap.VisualIndex.QueryOptions()
    options.max_num_images = 50
    assert options.max_num_images == 50


def test_visual_index_query_options_num_neighbors():
    options = pycolmap.VisualIndex.QueryOptions()
    options.num_neighbors = 3
    assert options.num_neighbors == 3


def test_visual_index_query_options_num_checks():
    options = pycolmap.VisualIndex.QueryOptions()
    options.num_checks = 64
    assert options.num_checks == 64


def test_visual_index_query_options_num_threads():
    options = pycolmap.VisualIndex.QueryOptions()
    options.num_threads = 2
    assert options.num_threads == 2


def test_visual_index_query_options_num_images_after_verification():
    options = pycolmap.VisualIndex.QueryOptions()
    options.num_images_after_verification = 10
    assert options.num_images_after_verification == 10


def test_visual_index_build_options_init():
    options = pycolmap.VisualIndex.BuildOptions()
    assert options is not None


def test_visual_index_build_options_num_visual_words():
    options = pycolmap.VisualIndex.BuildOptions()
    options.num_visual_words = 1024
    assert options.num_visual_words == 1024


def test_visual_index_build_options_num_iterations():
    options = pycolmap.VisualIndex.BuildOptions()
    options.num_iterations = 5
    assert options.num_iterations == 5


def test_visual_index_build_options_num_rounds():
    options = pycolmap.VisualIndex.BuildOptions()
    options.num_rounds = 3
    assert options.num_rounds == 3


def test_visual_index_build_options_num_checks():
    options = pycolmap.VisualIndex.BuildOptions()
    options.num_checks = 256
    assert options.num_checks == 256


def test_visual_index_build_options_num_threads():
    options = pycolmap.VisualIndex.BuildOptions()
    options.num_threads = 8
    assert options.num_threads == 8


def test_visual_index_create():
    index = pycolmap.VisualIndex.create(128, 64)
    assert index is not None
    assert index.num_visual_words() == 0
    assert index.num_images() == 0
