import numpy as np
import pytest

import pycolmap

pytestmark = pytest.mark.skipif(
    not hasattr(pycolmap, "DepthMap"),
    reason="DepthMap not available",
)


def test_depth_map_default_init():
    depth_map = pycolmap.DepthMap()
    assert depth_map is not None


def test_depth_map_init_with_params():
    depth_map = pycolmap.DepthMap(64, 48, 0.1, 100.0)
    assert depth_map.width == 64
    assert depth_map.height == 48
    assert depth_map.depth_min == pytest.approx(0.1)
    assert depth_map.depth_max == pytest.approx(100.0)


def test_depth_map_readonly_width():
    depth_map = pycolmap.DepthMap(64, 48, 0.1, 100.0)
    assert isinstance(depth_map.width, int)
    assert depth_map.width == 64


def test_depth_map_readonly_height():
    depth_map = pycolmap.DepthMap(64, 48, 0.1, 100.0)
    assert isinstance(depth_map.height, int)
    assert depth_map.height == 48


def test_depth_map_readonly_depth_min():
    depth_map = pycolmap.DepthMap(64, 48, 0.5, 50.0)
    assert isinstance(depth_map.depth_min, float)
    assert depth_map.depth_min == pytest.approx(0.5)


def test_depth_map_readonly_depth_max():
    depth_map = pycolmap.DepthMap(64, 48, 0.5, 50.0)
    assert isinstance(depth_map.depth_max, float)
    assert depth_map.depth_max == pytest.approx(50.0)


def test_depth_map_from_array_to_array_roundtrip():
    array = np.random.rand(48, 64).astype(np.float32)
    depth_map = pycolmap.DepthMap.from_array(
        array, depth_min=0.1, depth_max=10.0
    )
    result = depth_map.to_array()
    np.testing.assert_array_almost_equal(array, result)


def test_depth_map_rescale():
    depth_map = pycolmap.DepthMap(64, 48, 0.1, 100.0)
    depth_map.rescale(0.5)
    assert depth_map.width == 32
    assert depth_map.height == 24


def test_depth_map_downsize():
    depth_map = pycolmap.DepthMap(64, 48, 0.1, 100.0)
    depth_map.downsize(32, 24)
    assert depth_map.width <= 32
    assert depth_map.height <= 24


def test_depth_map_to_bitmap():
    array = np.random.rand(48, 64).astype(np.float32)
    depth_map = pycolmap.DepthMap.from_array(
        array, depth_min=0.0, depth_max=1.0
    )
    bitmap = depth_map.to_bitmap(5.0, 95.0)
    assert bitmap is not None


def test_depth_map_write_read_roundtrip(tmp_path):
    array = np.random.rand(48, 64).astype(np.float32)
    depth_map = pycolmap.DepthMap.from_array(
        array, depth_min=0.1, depth_max=10.0
    )
    filepath = str(tmp_path / "depth.bin")
    depth_map.write(filepath)
    loaded = pycolmap.DepthMap()
    loaded.read(filepath)
    result = loaded.to_array()
    np.testing.assert_array_almost_equal(array, result)
