import pycolmap


def test_rig_config_camera_default_init() -> None:
    rig_camera = pycolmap.RigConfigCamera()
    assert rig_camera is not None


def test_rig_config_camera_ref_sensor_readwrite() -> None:
    rig_camera = pycolmap.RigConfigCamera()
    rig_camera.ref_sensor = True
    assert rig_camera.ref_sensor is True
    rig_camera.ref_sensor = False
    assert rig_camera.ref_sensor is False


def test_rig_config_camera_image_prefix_readwrite() -> None:
    rig_camera = pycolmap.RigConfigCamera()
    rig_camera.image_prefix = "cam0_"
    assert rig_camera.image_prefix == "cam0_"


def test_rig_config_default_init() -> None:
    config = pycolmap.RigConfig()
    assert config is not None


def test_rig_config_cameras_readwrite() -> None:
    config = pycolmap.RigConfig()
    rig_camera = pycolmap.RigConfigCamera()
    rig_camera.ref_sensor = True
    rig_camera.image_prefix = "cam0_"
    config.cameras = [rig_camera]
    assert len(config.cameras) == 1
