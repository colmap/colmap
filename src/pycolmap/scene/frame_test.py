import pycolmap


def test_frame_default_init() -> None:
    frame = pycolmap.Frame()
    assert frame is not None


def test_frame_frame_id_readwrite() -> None:
    frame = pycolmap.Frame()
    frame.frame_id = 5
    assert frame.frame_id == 5


def test_frame_rig_id_readwrite() -> None:
    frame = pycolmap.Frame()
    frame.rig_id = 3
    assert frame.rig_id == 3


def test_frame_has_rig_id() -> None:
    frame = pycolmap.Frame()
    frame.rig_id = 1
    assert frame.has_rig_id()


def test_frame_add_data_id() -> None:
    frame = pycolmap.Frame()
    data_id = pycolmap.data_t()
    data_id.sensor_id = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=0)
    data_id.id = 1
    frame.add_data_id(data_id)
    assert frame.num_data_ids() == 1


def test_frame_num_data_ids() -> None:
    frame = pycolmap.Frame()
    assert frame.num_data_ids() == 0


def test_frame_has_data() -> None:
    frame = pycolmap.Frame()
    data_id = pycolmap.data_t()
    data_id.sensor_id = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=0)
    data_id.id = 1
    frame.add_data_id(data_id)
    assert frame.has_data(data_id)


def test_frame_clear_data_ids() -> None:
    frame = pycolmap.Frame()
    data_id = pycolmap.data_t()
    data_id.sensor_id = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=0)
    data_id.id = 1
    frame.add_data_id(data_id)
    frame.clear_data_ids()
    assert frame.num_data_ids() == 0


def test_frame_finalize_data_ids() -> None:
    frame = pycolmap.Frame()
    data_id = pycolmap.data_t()
    data_id.sensor_id = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=0)
    data_id.id = 1
    frame.add_data_id(data_id)
    frame.finalize_data_ids()
    assert frame.has_final_data_ids()


def test_frame_has_final_data_ids() -> None:
    frame = pycolmap.Frame()
    assert not frame.has_final_data_ids()


def test_frame_data_ids_property() -> None:
    frame = pycolmap.Frame()
    data_ids = frame.data_ids
    assert hasattr(data_ids, "__len__")


def test_frame_data_ids_by_sensor() -> None:
    frame = pycolmap.Frame()
    data_ids = frame.data_ids_by_sensor(pycolmap.SensorType.CAMERA)
    assert isinstance(data_ids, list)


def test_frame_image_ids_property() -> None:
    frame = pycolmap.Frame()
    image_ids = frame.image_ids
    assert isinstance(image_ids, list)


def test_frame_has_pose() -> None:
    frame = pycolmap.Frame()
    assert not frame.has_pose()


def test_frame_reset_pose() -> None:
    frame = pycolmap.Frame()
    frame.reset_pose()
    assert not frame.has_pose()


def test_frame_map_empty() -> None:
    frame_map = pycolmap.FrameMap()
    assert len(frame_map) == 0


def test_frame_map_insert_and_access() -> None:
    frame_map = pycolmap.FrameMap()
    frame = pycolmap.Frame()
    frame.frame_id = 1
    frame_map[1] = frame
    assert len(frame_map) == 1
    assert frame_map[1].frame_id == 1
