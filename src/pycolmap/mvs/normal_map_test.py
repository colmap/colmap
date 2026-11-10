# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest

import pycolmap

pytestmark = pytest.mark.skipif(
    not hasattr(pycolmap, "NormalMap"),
    reason="NormalMap not available",
)


@pytest.fixture(params=[False, True], ids=["contiguous", "strided"])
def normals(request: pytest.FixtureRequest) -> np.ndarray:
    array = np.array(
        [
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[0.6, 0, -0.8], [0, 0.6, -0.8], [-0.6, 0, -0.8]],
        ],
        dtype=np.float32,
    )
    if not request.param:
        return array
    storage = np.zeros((2, 6, 3), dtype=np.float32)
    storage[:, ::2, :] = array
    return storage[:, ::2, :]


def test_normal_map_default_init() -> None:
    normal_map = pycolmap.NormalMap()
    assert normal_map is not None
    assert normal_map.to_array().shape == (0, 0, 3)


@pytest.mark.parametrize("width,height", [(0, 2), (3, 0)])
def test_normal_map_empty_array_shape(width: int, height: int) -> None:
    normal_map = pycolmap.NormalMap(width, height)
    assert normal_map.to_array().shape == (height, width, 3)


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


def test_normal_map_to_bitmap(normals: np.ndarray) -> None:
    normal_map = pycolmap.NormalMap.from_array(normals)
    bitmap = normal_map.to_bitmap()
    expected = np.array(
        [
            [[255, 128, 0], [128, 255, 0], [128, 128, 255]],
            [[51, 128, 204], [128, 51, 204], [204, 128, 204]],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(bitmap.to_array(), expected)


def test_normal_map_identity_rescale_preserves_vectors(
    normals: np.ndarray,
) -> None:
    normal_map = pycolmap.NormalMap.from_array(normals)
    normal_map.rescale(1.0)
    np.testing.assert_allclose(normal_map.to_array(), normals, atol=1e-6)


def test_normal_map_reads_native_planar_storage(
    tmp_path: Path, normals: np.ndarray
) -> None:
    filepath = tmp_path / "native.normal.bin"
    filepath.write_bytes(
        b"3&2&3&" + normals.transpose(2, 0, 1).astype("<f4").tobytes()
    )
    normal_map = pycolmap.NormalMap()
    normal_map.read(str(filepath))
    np.testing.assert_array_equal(normal_map.to_array(), normals)


def test_normal_map_writes_native_planar_storage(
    tmp_path: Path, normals: np.ndarray
) -> None:
    filepath = tmp_path / "native.normal.bin"
    normal_map = pycolmap.NormalMap.from_array(normals)
    normal_map.write(str(filepath))
    assert filepath.read_bytes() == (
        b"3&2&3&" + normals.transpose(2, 0, 1).astype("<f4").tobytes()
    )


def test_normal_map_write_read_roundtrip(tmp_path: Path) -> None:
    array = np.random.rand(48, 64, 3).astype(np.float32)
    normal_map = pycolmap.NormalMap.from_array(array)
    filepath = str(tmp_path / "normal.bin")
    normal_map.write(filepath)
    loaded = pycolmap.NormalMap()
    loaded.read(filepath)
    result = loaded.to_array()
    np.testing.assert_array_almost_equal(array, result)
