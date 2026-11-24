# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import pycolmap


def _make_sensor(
    sensor_type: pycolmap.SensorType, sensor_id: int
) -> pycolmap.sensor_t:
    return pycolmap.sensor_t(type=sensor_type, id=sensor_id)


def test_rig_init() -> None:
    rig = pycolmap.Rig()
    assert rig is not None


def test_rig_rig_id_readwrite() -> None:
    rig = pycolmap.Rig()
    rig.rig_id = 5
    assert rig.rig_id == 5


def test_rig_add_ref_sensor() -> None:
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    rig.add_ref_sensor(ref_sensor)
    assert rig.has_sensor(ref_sensor)
    assert rig.is_ref_sensor(ref_sensor)


def test_rig_add_sensor() -> None:
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, None)
    assert rig.has_sensor(other_sensor)


def test_rig_add_ref_sensor_keyword() -> None:
    rig = pycolmap.Rig()
    sensor = _make_sensor(pycolmap.SensorType.CAMERA, 3)
    rig.add_ref_sensor(sensor_id=sensor)
    assert rig.ref_sensor_id == sensor
    assert rig.is_ref_sensor(sensor)


@pytest.mark.parametrize("keyword", [False, True])
def test_rig_add_sensor_default_transform(keyword: bool) -> None:
    rig = pycolmap.Rig()
    rig.add_ref_sensor(_make_sensor(pycolmap.SensorType.CAMERA, 3))
    sensor = _make_sensor(pycolmap.SensorType.CAMERA, 5)
    if keyword:
        rig.add_sensor(sensor_id=sensor)
    else:
        rig.add_sensor(sensor)
    assert rig.has_sensor(sensor)
    assert not rig.has_sensor_from_rig(sensor)
    assert rig.sensor_from_rig(sensor) is None


def test_rig_add_sensor_keyword_transform() -> None:
    rig = pycolmap.Rig()
    rig.rig_id = 7
    rig.add_ref_sensor(_make_sensor(pycolmap.SensorType.CAMERA, 3))
    sensor = _make_sensor(pycolmap.SensorType.CAMERA, 5)
    sensor_from_rig = pycolmap.Rigid3d(
        pycolmap.Rotation3d(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
        np.array([1, 2, 3]),
    )
    rig.add_sensor(sensor_id=sensor, sensor_from_rig=sensor_from_rig)
    frame = pycolmap.Frame(frame_id=9, rig_id=rig.rig_id)
    frame.rig = rig
    frame.rig_from_world = pycolmap.Rigid3d(
        pycolmap.Rotation3d(), np.array([-2, 1, 4])
    )
    np.testing.assert_allclose(
        frame.sensor_from_world(sensor).matrix(),
        np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 7]]),
        atol=1e-12,
    )


def test_rig_has_sensor() -> None:
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    missing_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 99)
    rig.add_ref_sensor(ref_sensor)
    assert rig.has_sensor(ref_sensor)
    assert not rig.has_sensor(missing_sensor)


def test_rig_is_ref_sensor() -> None:
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, None)
    assert rig.is_ref_sensor(ref_sensor)
    assert not rig.is_ref_sensor(other_sensor)


def test_rig_num_sensors() -> None:
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    rig.add_ref_sensor(ref_sensor)
    assert rig.num_sensors() == 1
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_sensor(other_sensor, None)
    assert rig.num_sensors() == 2


def test_rig_ref_sensor_id() -> None:
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    rig.add_ref_sensor(ref_sensor)
    assert rig.ref_sensor_id == ref_sensor


def test_rig_sensor_ids() -> None:
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, None)
    sensor_ids = rig.sensor_ids()
    assert ref_sensor in sensor_ids
    assert other_sensor in sensor_ids


def test_rig_non_ref_sensors() -> None:
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, None)
    non_ref = rig.non_ref_sensors
    assert other_sensor in non_ref


def test_rig_has_sensor_from_rig() -> None:
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, None)
    # No transform set yet.
    assert not rig.has_sensor_from_rig(other_sensor)


def test_rig_sensor_from_rig() -> None:
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, None)
    # Returns optional, should be None since not set.
    result = rig.sensor_from_rig(other_sensor)
    assert result is None


def test_rig_set_sensor_from_rig() -> None:
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, None)
    transform = pycolmap.Rigid3d()
    rig.set_sensor_from_rig(other_sensor, transform)
    assert rig.has_sensor_from_rig(other_sensor)


def test_rig_reset_sensor_from_rig() -> None:
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, pycolmap.Rigid3d())
    assert rig.has_sensor_from_rig(other_sensor)
    rig.reset_sensor_from_rig(other_sensor)
    assert not rig.has_sensor_from_rig(other_sensor)


def test_rig_map_empty() -> None:
    rig_map = pycolmap.RigMap()
    assert len(rig_map) == 0


def test_rig_map_insert_and_access() -> None:
    rig_map = pycolmap.RigMap()
    rig = pycolmap.Rig()
    rig.rig_id = 1
    rig_map[1] = rig
    assert len(rig_map) == 1
    assert rig_map[1].rig_id == 1
