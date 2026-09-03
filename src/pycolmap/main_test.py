import pycolmap


def test_version_is_str() -> None:
    assert isinstance(pycolmap.__version__, str)
    assert len(pycolmap.__version__) > 0


def test_ceres_version_is_str() -> None:
    assert isinstance(pycolmap.__ceres_version__, str)
    assert len(pycolmap.__ceres_version__) > 0


def test_hash_map_backend_is_known() -> None:
    assert pycolmap.__hash_map_backend__ in ("std", "boost")


def test_has_cuda_is_bool() -> None:
    assert isinstance(pycolmap.has_cuda, bool)


def test_colmap_version_is_str() -> None:
    assert isinstance(pycolmap.COLMAP_version, str)
    assert len(pycolmap.COLMAP_version) > 0


def test_colmap_build_is_str() -> None:
    assert isinstance(pycolmap.COLMAP_build, str)
    assert len(pycolmap.COLMAP_build) > 0


def test_device_enum() -> None:
    assert pycolmap.Device.auto is not None
    assert pycolmap.Device.cpu is not None
    assert pycolmap.Device.cuda is not None


def test_set_random_seed() -> None:
    pycolmap.set_random_seed(42)
