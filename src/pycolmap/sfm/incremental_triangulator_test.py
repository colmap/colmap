import pycolmap


def test_incremental_triangulator_options_init():
    options = pycolmap.IncrementalTriangulatorOptions()
    assert options is not None


def test_incremental_triangulator_options_max_transitivity():
    options = pycolmap.IncrementalTriangulatorOptions()
    options.max_transitivity = 2
    assert options.max_transitivity == 2


def test_incremental_triangulator_options_create_max_angle_error():
    options = pycolmap.IncrementalTriangulatorOptions()
    options.create_max_angle_error = 3.0
    assert options.create_max_angle_error == 3.0


def test_incremental_triangulator_options_check():
    options = pycolmap.IncrementalTriangulatorOptions()
    assert options.check()


def test_incremental_triangulator_class_exists():
    assert hasattr(pycolmap, "IncrementalTriangulator")
