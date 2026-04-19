import pycolmap


def test_sensor_type_enum():
    assert {m.name: int(m) for m in pycolmap.SensorType} == {
        "INVALID": -1,
        "CAMERA": 0,
        "IMU": 1,
    }


def test_sensor_type_string_construction():
    assert pycolmap.SensorType("CAMERA") == pycolmap.SensorType.CAMERA
    assert pycolmap.SensorType("IMU") == pycolmap.SensorType.IMU
    assert pycolmap.SensorType("INVALID") == pycolmap.SensorType.INVALID


def test_sensor_t_default_init():
    sensor = pycolmap.sensor_t()
    assert sensor is not None


def test_sensor_t_init_with_args():
    sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=42)
    assert sensor.type == pycolmap.SensorType.CAMERA
    assert sensor.id == 42


def test_sensor_t_readwrite_type():
    sensor = pycolmap.sensor_t()
    sensor.type = pycolmap.SensorType.IMU
    assert sensor.type == pycolmap.SensorType.IMU


def test_sensor_t_readwrite_id():
    sensor = pycolmap.sensor_t()
    sensor.id = 7
    assert sensor.id == 7


def test_data_t_default_init():
    data = pycolmap.data_t()
    assert data is not None


def test_data_t_init_with_args():
    sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1)
    data = pycolmap.data_t(sensor_id=sensor, id=99)
    assert data.id == 99


def test_data_t_readwrite_sensor_id():
    data = pycolmap.data_t()
    sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=5)
    data.sensor_id = sensor
    assert data.sensor_id.id == 5


def test_data_t_readwrite_id():
    data = pycolmap.data_t()
    data.id = 123
    assert data.id == 123


def test_image_pair_to_pair_id_and_roundtrip():
    pair_id = pycolmap.image_pair_to_pair_id(1, 2)
    assert isinstance(pair_id, int)
    image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
    assert image_id1 == 1
    assert image_id2 == 2


def test_should_swap_image_pair():
    result = pycolmap.should_swap_image_pair(2, 1)
    assert isinstance(result, bool)
