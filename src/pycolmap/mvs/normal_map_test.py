import numpy as np
import pytest

import pycolmap

pytestmark = pytest.mark.skipif(
    not hasattr(pycolmap, "NormalMap"),
    reason="NormalMap not available",
)


def test_normal_map_default_init():
    normal_map = pycolmap.NormalMap()
    assert normal_map is not None


def test_normal_map_init_with_params():
    normal_map = pycolmap.NormalMap(64, 48)
    assert normal_map.width == 64
    assert normal_map.height == 48


def test_normal_map_readonly_width():
    normal_map = pycolmap.NormalMap(64, 48)
    assert isinstance(normal_map.width, int)
    assert normal_map.width == 64


def test_normal_map_readonly_height():
    normal_map = pycolmap.NormalMap(64, 48)
    assert isinstance(normal_map.height, int)
    assert normal_map.height == 48


def test_normal_map_from_array_to_array_roundtrip():
    array = np.random.rand(48, 64, 3).astype(np.float32)
    normal_map = pycolmap.NormalMap.from_array(array)
    result = normal_map.to_array()
    np.testing.assert_array_almost_equal(array, result)


def test_normal_map_rescale():
    normal_map = pycolmap.NormalMap(64, 48)
    normal_map.rescale(0.5)
    assert normal_map.width == 32
    assert normal_map.height == 24


def test_normal_map_downsize():
    normal_map = pycolmap.NormalMap(64, 48)
    normal_map.downsize(32, 24)
    assert normal_map.width <= 32
    assert normal_map.height <= 24


def test_normal_map_to_bitmap():
    array = np.random.rand(48, 64, 3).astype(np.float32)
    normal_map = pycolmap.NormalMap.from_array(array)
    bitmap = normal_map.to_bitmap()
    assert bitmap is not None


def test_normal_map_write_read_roundtrip(tmp_path):
    array = np.random.rand(48, 64, 3).astype(np.float32)
    normal_map = pycolmap.NormalMap.from_array(array)
    filepath = str(tmp_path / "normal.bin")
    normal_map.write(filepath)
    loaded = pycolmap.NormalMap()
    loaded.read(filepath)
    result = loaded.to_array()
    np.testing.assert_array_almost_equal(array, result)
