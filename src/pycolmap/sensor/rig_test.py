import pycolmap


def _make_sensor(sensor_type, sensor_id):
    return pycolmap.sensor_t(type=sensor_type, id=sensor_id)


def test_rig_init():
    rig = pycolmap.Rig()
    assert rig is not None


def test_rig_rig_id_readwrite():
    rig = pycolmap.Rig()
    rig.rig_id = 5
    assert rig.rig_id == 5


def test_rig_add_ref_sensor():
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    rig.add_ref_sensor(ref_sensor)
    assert rig.has_sensor(ref_sensor)
    assert rig.is_ref_sensor(ref_sensor)


def test_rig_add_sensor():
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, None)
    assert rig.has_sensor(other_sensor)


def test_rig_has_sensor():
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    missing_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 99)
    rig.add_ref_sensor(ref_sensor)
    assert rig.has_sensor(ref_sensor)
    assert not rig.has_sensor(missing_sensor)


def test_rig_is_ref_sensor():
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, None)
    assert rig.is_ref_sensor(ref_sensor)
    assert not rig.is_ref_sensor(other_sensor)


def test_rig_num_sensors():
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    rig.add_ref_sensor(ref_sensor)
    assert rig.num_sensors() == 1
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_sensor(other_sensor, None)
    assert rig.num_sensors() == 2


def test_rig_ref_sensor_id():
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    rig.add_ref_sensor(ref_sensor)
    assert rig.ref_sensor_id == ref_sensor


def test_rig_sensor_ids():
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, None)
    sensor_ids = rig.sensor_ids()
    assert ref_sensor in sensor_ids
    assert other_sensor in sensor_ids


def test_rig_non_ref_sensors():
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, None)
    non_ref = rig.non_ref_sensors
    assert other_sensor in non_ref


def test_rig_has_sensor_from_rig():
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, None)
    # No transform set yet.
    assert not rig.has_sensor_from_rig(other_sensor)


def test_rig_sensor_from_rig():
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, None)
    # Returns optional, should be None since not set.
    result = rig.sensor_from_rig(other_sensor)
    assert result is None


def test_rig_set_sensor_from_rig():
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, None)
    transform = pycolmap.Rigid3d()
    rig.set_sensor_from_rig(other_sensor, transform)
    assert rig.has_sensor_from_rig(other_sensor)


def test_rig_reset_sensor_from_rig():
    rig = pycolmap.Rig()
    ref_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 0)
    other_sensor = _make_sensor(pycolmap.SensorType.CAMERA, 1)
    rig.add_ref_sensor(ref_sensor)
    rig.add_sensor(other_sensor, pycolmap.Rigid3d())
    assert rig.has_sensor_from_rig(other_sensor)
    rig.reset_sensor_from_rig(other_sensor)
    assert not rig.has_sensor_from_rig(other_sensor)


def test_rig_map_empty():
    rig_map = pycolmap.RigMap()
    assert len(rig_map) == 0


def test_rig_map_insert_and_access():
    rig_map = pycolmap.RigMap()
    rig = pycolmap.Rig()
    rig.rig_id = 1
    rig_map[1] = rig
    assert len(rig_map) == 1
    assert rig_map[1].rig_id == 1
    assert "RigMap{1: Rig(rig_id=1" in repr(rig_map)


def test_reconstruction_rig_map_views_and_iteration():
    reconstruction = pycolmap.Reconstruction()
    camera = pycolmap.Camera.create_from_model_id(
        camera_id=1,
        model=pycolmap.CameraModelId.SIMPLE_PINHOLE,
        focal_length=1.0,
        width=1,
        height=1,
    )
    reconstruction.add_camera_with_trivial_rig(camera)

    assert isinstance(repr(reconstruction.rigs.values()), str)
    assert [rig.rig_id for rig in reconstruction.rigs.values()] == [1]
    assert [
        (rig_id, rig.rig_id) for rig_id, rig in reconstruction.rigs.items()
    ] == [(1, 1)]
