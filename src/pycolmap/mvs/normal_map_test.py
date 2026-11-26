# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest

import pycolmap

pytestmark = pytest.mark.skipif(
    not hasattr(pycolmap, "NormalMap"),
    reason="NormalMap not available",
)


def test_normal_map_default_init() -> None:
    normal_map = pycolmap.NormalMap()
    assert normal_map is not None


def test_normal_map_init_with_params() -> None:
    normal_map = pycolmap.NormalMap(64, 48)
    assert normal_map.width == 64
    assert normal_map.height == 48


def test_normal_map_readonly_width() -> None:
    normal_map = pycolmap.NormalMap(64, 48)
    assert isinstance(normal_map.width, int)
    assert normal_map.width == 64


def test_normal_map_readonly_height() -> None:
    normal_map = pycolmap.NormalMap(64, 48)
    assert isinstance(normal_map.height, int)
    assert normal_map.height == 48


def test_normal_map_from_array_to_array_roundtrip() -> None:
    array = np.random.rand(48, 64, 3).astype(np.float32)
    normal_map = pycolmap.NormalMap.from_array(array)
    result = normal_map.to_array()
    np.testing.assert_array_almost_equal(array, result)


def test_normal_map_rescale() -> None:
    normal_map = pycolmap.NormalMap(64, 48)
    normal_map.rescale(0.5)
    assert normal_map.width == 32
    assert normal_map.height == 24


def test_normal_map_downsize() -> None:
    normal_map = pycolmap.NormalMap(64, 48)
    normal_map.downsize(32, 24)
    assert normal_map.width <= 32
    assert normal_map.height <= 24


def test_normal_map_to_bitmap() -> None:
    array = np.random.rand(48, 64, 3).astype(np.float32)
    normal_map = pycolmap.NormalMap.from_array(array)
    bitmap = normal_map.to_bitmap()
    assert bitmap is not None


def _native_normal_map_bytes(normals: np.ndarray) -> bytes:
    # COLMAP stores normal maps as "width&height&depth&" + channel planes.
    height, width, depth = normals.shape
    planes = normals.transpose(2, 0, 1).astype("<f4")
    return f"{width}&{height}&{depth}&".encode() + planes.tobytes()


def test_normal_map_read_native_layout(tmp_path: Path) -> None:
    normals = np.random.rand(2, 3, 3).astype(np.float32)
    filepath = tmp_path / "normal.bin"
    filepath.write_bytes(_native_normal_map_bytes(normals))
    normal_map = pycolmap.NormalMap()
    normal_map.read(str(filepath))
    np.testing.assert_array_equal(normal_map.to_array(), normals)


def test_normal_map_write_native_layout(tmp_path: Path) -> None:
    normals = np.random.rand(2, 3, 3).astype(np.float32)
    filepath = tmp_path / "normal.bin"
    pycolmap.NormalMap.from_array(normals).write(str(filepath))
    assert filepath.read_bytes() == _native_normal_map_bytes(normals)


def test_normal_map_write_read_roundtrip(tmp_path: Path) -> None:
    array = np.random.rand(48, 64, 3).astype(np.float32)
    normal_map = pycolmap.NormalMap.from_array(array)
    filepath = str(tmp_path / "normal.bin")
    normal_map.write(filepath)
    loaded = pycolmap.NormalMap()
    loaded.read(filepath)
    result = loaded.to_array()
    np.testing.assert_array_almost_equal(array, result)
