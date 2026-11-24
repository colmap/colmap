# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import pycolmap


def test_rotation3d_default_init() -> None:
    rotation = pycolmap.Rotation3d()
    assert rotation is not None


def test_rotation3d_init_from_xyzw() -> None:
    rotation = pycolmap.Rotation3d(xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
    assert rotation is not None


def test_rotation3d_init_from_matrix() -> None:
    rotation = pycolmap.Rotation3d(matrix=np.eye(3))
    assert rotation is not None


def test_rotation3d_init_from_axis_angle() -> None:
    rotation = pycolmap.Rotation3d(axis_angle=np.zeros(3))
    assert rotation is not None


def test_rotation3d_from_buffer() -> None:
    array = np.array([0.0, 0.0, 0.0, 1.0])
    rotation = pycolmap.Rotation3d.from_buffer(array)
    assert rotation is not None


@pytest.fixture(params=["contiguous", "strided", "reversed", "float32"])
def quaternion_array(request: pytest.FixtureRequest) -> np.ndarray:
    quat = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    if request.param == "strided":
        # A row in a Fortran-order pose array is a common strided input.
        return np.asfortranarray([quat, [1.0, 0.0, 0.0, 0.0]])[0]
    if request.param == "reversed":
        return quat[::-1].copy()[::-1]
    if request.param == "float32":
        return quat.astype(np.float32)
    return quat


def test_rotation3d_buffer_respects_array_layout(
    quaternion_array: np.ndarray,
) -> None:
    expected = pycolmap.Rotation3d(quaternion_array.copy()).matrix()
    rotation = pycolmap.Rotation3d.from_buffer(quaternion_array)
    np.testing.assert_allclose(rotation.matrix(), expected)
    shares_input = (
        quaternion_array.flags.c_contiguous
        and quaternion_array.dtype == np.float64
    )
    assert np.shares_memory(rotation.quat, quaternion_array) == shares_input
    # Updating a copied view must not write into neighboring pose coefficients.
    original = quaternion_array.copy()
    rotation.quat = np.array([0.0, 0.0, 0.0, 1.0])
    if shares_input:
        np.testing.assert_array_equal(quaternion_array, rotation.quat)
    else:
        np.testing.assert_array_equal(quaternion_array, original)


@pytest.mark.parametrize("kind", ["rigid", "similarity"])
def test_transform_quaternion_respects_array_layout(
    quaternion_array: np.ndarray, kind: str
) -> None:
    expected = pycolmap.Rotation3d(quaternion_array.copy()).matrix()
    # The custom quaternion caster accepts arrays as well as Rotation3d.
    if kind == "rigid":
        transform = pycolmap.Rigid3d(quaternion_array, np.zeros(3))  # type: ignore[call-overload]
    else:
        transform = pycolmap.Sim3d(1.0, quaternion_array, np.zeros(3))  # type: ignore[call-overload]
    np.testing.assert_allclose(transform.rotation.matrix(), expected)
    np.testing.assert_array_equal(transform.rotation.quat, quaternion_array)


@pytest.mark.parametrize("kind", ["rigid", "similarity"])
def test_transform_rotation_setter_respects_array_layout(
    quaternion_array: np.ndarray, kind: str
) -> None:
    expected = pycolmap.Rotation3d(quaternion_array.copy()).matrix()
    transform = pycolmap.Rigid3d() if kind == "rigid" else pycolmap.Sim3d()
    transform.rotation = quaternion_array  # type: ignore[assignment]
    np.testing.assert_allclose(transform.rotation.matrix(), expected)
    np.testing.assert_array_equal(transform.rotation.quat, quaternion_array)


@pytest.mark.parametrize("kind", ["rigid", "similarity"])
def test_transform_rotation_remains_a_view(kind: str) -> None:
    transform = pycolmap.Rigid3d() if kind == "rigid" else pycolmap.Sim3d()
    rotation = transform.rotation
    assert np.shares_memory(rotation.quat, transform.params)
    rotation.quat = np.array([0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(transform.params[:4], rotation.quat)


def test_rotation3d_quat_readwrite() -> None:
    rotation = pycolmap.Rotation3d()
    quat = rotation.quat
    assert quat.shape == (4,)
    rotation.quat = np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_almost_equal(rotation.quat, [0.0, 0.0, 0.0, 1.0])


def test_rotation3d_matrix() -> None:
    rotation = pycolmap.Rotation3d(xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
    matrix = rotation.matrix()
    assert matrix.shape == (3, 3)
    np.testing.assert_array_almost_equal(matrix, np.eye(3))


def test_rotation3d_norm() -> None:
    rotation = pycolmap.Rotation3d(xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
    assert isinstance(rotation.norm(), float)


def test_rotation3d_angle() -> None:
    rotation = pycolmap.Rotation3d(axis_angle=np.zeros(3))
    angle = rotation.angle()
    assert isinstance(angle, float)
    assert angle >= 0.0


def test_rotation3d_angle_to() -> None:
    rotation1 = pycolmap.Rotation3d()
    rotation2 = pycolmap.Rotation3d()
    angle = rotation1.angle_to(rotation2)
    assert isinstance(angle, float)


def test_rotation3d_inverse() -> None:
    rotation = pycolmap.Rotation3d(xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
    inverse = rotation.inverse()
    assert inverse is not None


def test_rotation3d_normalize() -> None:
    rotation = pycolmap.Rotation3d(xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
    rotation.normalize()
    assert abs(rotation.norm() - 1.0) < 1e-10


def test_rotation3d_multiply_rotation3d() -> None:
    rotation1 = pycolmap.Rotation3d()
    rotation2 = pycolmap.Rotation3d()
    result = rotation1 * rotation2
    assert result is not None


def test_rotation3d_multiply_vector3d() -> None:
    rotation = pycolmap.Rotation3d()
    vector = np.array([1.0, 2.0, 3.0])
    result = rotation * vector
    assert result.shape == (3,)


def test_rotation3d_multiply_nx3_matrix() -> None:
    rotation = pycolmap.Rotation3d()
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = rotation * points
    assert result.shape == (2, 3)


def test_alignedbox3d_default_init() -> None:
    bbox = pycolmap.AlignedBox3d()
    assert bbox is not None


def test_alignedbox3d_init_min_max() -> None:
    bbox = pycolmap.AlignedBox3d(
        min=np.array([0.0, 0.0, 0.0]),
        max=np.array([1.0, 1.0, 1.0]),
    )
    assert bbox is not None


def test_alignedbox3d_min_max_readwrite() -> None:
    bbox = pycolmap.AlignedBox3d(
        min=np.array([0.0, 0.0, 0.0]),
        max=np.array([1.0, 1.0, 1.0]),
    )
    np.testing.assert_array_almost_equal(bbox.min, [0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(bbox.max, [1.0, 1.0, 1.0])
    bbox.min = np.array([-1.0, -1.0, -1.0])
    bbox.max = np.array([2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(bbox.min, [-1.0, -1.0, -1.0])
    np.testing.assert_array_almost_equal(bbox.max, [2.0, 2.0, 2.0])


def test_alignedbox3d_diagonal() -> None:
    bbox = pycolmap.AlignedBox3d(
        min=np.array([0.0, 0.0, 0.0]),
        max=np.array([1.0, 2.0, 3.0]),
    )
    diagonal = bbox.diagonal()
    np.testing.assert_array_almost_equal(diagonal, [1.0, 2.0, 3.0])


def test_alignedbox3d_contains_point_inside() -> None:
    bbox = pycolmap.AlignedBox3d(
        min=np.array([0.0, 0.0, 0.0]),
        max=np.array([1.0, 1.0, 1.0]),
    )
    assert bbox.contains_point(np.array([0.5, 0.5, 0.5]))


def test_alignedbox3d_contains_point_outside() -> None:
    bbox = pycolmap.AlignedBox3d(
        min=np.array([0.0, 0.0, 0.0]),
        max=np.array([1.0, 1.0, 1.0]),
    )
    assert not bbox.contains_point(np.array([2.0, 2.0, 2.0]))


def test_alignedbox3d_contains_bbox() -> None:
    outer = pycolmap.AlignedBox3d(
        min=np.array([0.0, 0.0, 0.0]),
        max=np.array([10.0, 10.0, 10.0]),
    )
    inner = pycolmap.AlignedBox3d(
        min=np.array([1.0, 1.0, 1.0]),
        max=np.array([2.0, 2.0, 2.0]),
    )
    assert outer.contains_bbox(inner)
    assert not inner.contains_bbox(outer)
