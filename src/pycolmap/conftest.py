import pytest

import pycolmap


@pytest.fixture
def simple_camera():
    return pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.PINHOLE, 500.0, 1024, 768
    )


@pytest.fixture(scope="session")
def synthetic_reconstruction():
    options = pycolmap.SyntheticDatasetOptions()
    options.num_cameras_per_rig = 1
    options.num_frames_per_rig = 3
    options.num_points3D = 50
    return pycolmap.synthesize_dataset(options)


@pytest.fixture
def database(tmp_path):
    with pycolmap.Database.open(str(tmp_path / "test.db")) as db:
        yield db


@pytest.fixture
def populated_database(database):
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.PINHOLE, 500.0, 1024, 768
    )
    camera_id = database.write_camera(camera)
    image = pycolmap.Image()
    image.name = "test.jpg"
    image.camera_id = camera_id
    image_id = database.write_image(image)
    return database, camera_id, image_id
