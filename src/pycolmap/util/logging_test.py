import pycolmap


def test_logging_level_enum():
    assert pycolmap.logging.INFO is not None
    assert pycolmap.logging.WARNING is not None
    assert pycolmap.logging.ERROR is not None
    assert pycolmap.logging.FATAL is not None


def test_logging_minloglevel_readwrite():
    original = pycolmap.logging.minloglevel
    pycolmap.logging.minloglevel = 0
    assert pycolmap.logging.minloglevel == 0
    pycolmap.logging.minloglevel = original


def test_logging_stderrthreshold_readwrite():
    original = pycolmap.logging.stderrthreshold
    pycolmap.logging.stderrthreshold = 2
    assert pycolmap.logging.stderrthreshold == 2
    pycolmap.logging.stderrthreshold = original


def test_logging_log_dir_readwrite():
    original = pycolmap.logging.log_dir
    assert isinstance(original, str)
    pycolmap.logging.log_dir = original


def test_logging_logtostderr_readwrite():
    original = pycolmap.logging.logtostderr
    pycolmap.logging.logtostderr = True
    assert pycolmap.logging.logtostderr is True
    pycolmap.logging.logtostderr = original


def test_logging_alsologtostderr_readwrite():
    original = pycolmap.logging.alsologtostderr
    pycolmap.logging.alsologtostderr = True
    assert pycolmap.logging.alsologtostderr is True
    pycolmap.logging.alsologtostderr = original


def test_logging_verbose_level_readwrite():
    original = pycolmap.logging.verbose_level
    pycolmap.logging.verbose_level = 1
    assert pycolmap.logging.verbose_level == 1
    pycolmap.logging.verbose_level = original


def test_logging_info():
    pycolmap.logging.info("smoke test info message")


def test_logging_warning():
    pycolmap.logging.warning("smoke test warning message")


def test_logging_error():
    pycolmap.logging.error("smoke test error message")


def test_logging_verbose():
    pycolmap.logging.verbose(1, "smoke test verbose message")


def test_logging_set_log_destination(tmp_path):
    log_dir = str(tmp_path)
    pycolmap.logging.set_log_destination(pycolmap.logging.INFO, log_dir)
