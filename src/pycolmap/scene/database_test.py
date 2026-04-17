import numpy as np

import pycolmap


def test_database_open_and_context_manager(tmp_path):
    database_path = str(tmp_path / "test.db")
    with pycolmap.Database.open(database_path) as database:
        assert database is not None


def test_database_write_and_read_camera(database, simple_camera):
    camera_id = database.write_camera(simple_camera)
    assert camera_id > 0
    read_camera = database.read_camera(camera_id)
    assert read_camera.model == pycolmap.CameraModelId.PINHOLE


def test_database_update_camera(database, simple_camera):
    camera_id = database.write_camera(simple_camera)
    simple_camera.camera_id = camera_id
    simple_camera.width = 2048
    database.update_camera(simple_camera)
    updated_camera = database.read_camera(camera_id)
    assert updated_camera.width == 2048


def test_database_write_and_read_image(populated_database):
    database, camera_id, image_id = populated_database
    assert image_id > 0
    read_image = database.read_image(image_id)
    assert read_image.name == "test.jpg"


def test_database_update_image(populated_database):
    database, camera_id, image_id = populated_database
    image = database.read_image(image_id)
    image.name = "updated.jpg"
    database.update_image(image)
    updated_image = database.read_image(image_id)
    assert updated_image.name == "updated.jpg"


def test_database_clear_cameras(database, simple_camera):
    database.write_camera(simple_camera)
    assert database.num_cameras() > 0
    database.clear_cameras()
    assert database.num_cameras() == 0


def test_database_write_and_read_keypoints(populated_database):
    database, camera_id, image_id = populated_database
    keypoints = np.array(
        [[10.0, 20.0, 1.0, 0.0, 0.0, 1.0], [30.0, 40.0, 1.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    database.write_keypoints(image_id, keypoints)
    read_keypoints = database.read_keypoints(image_id)
    assert read_keypoints.shape[0] == 2


def test_database_write_and_read_descriptors(populated_database):
    database, camera_id, image_id = populated_database
    descriptors = pycolmap.FeatureDescriptors(
        type=pycolmap.FeatureExtractorType.SIFT,
        data=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
    )
    database.write_descriptors(image_id, descriptors)
    read_descriptors = database.read_descriptors(image_id)
    assert read_descriptors is not None


def test_database_write_and_read_matches(populated_database):
    database, camera_id, image_id = populated_database
    image2 = pycolmap.Image()
    image2.name = "test2.jpg"
    image2.camera_id = camera_id
    image_id2 = database.write_image(image2)
    matches = np.array([[0, 0], [1, 1]], dtype=np.uint32)
    database.write_matches(image_id, image_id2, matches)
    read_matches = database.read_matches(image_id, image_id2)
    assert read_matches.shape[0] == 2


def test_database_write_and_read_two_view_geometry(populated_database):
    database, camera_id, image_id = populated_database
    image2 = pycolmap.Image()
    image2.name = "test2.jpg"
    image2.camera_id = camera_id
    image_id2 = database.write_image(image2)
    two_view = pycolmap.TwoViewGeometry()
    two_view.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
    two_view.inlier_matches = np.array([[0, 1]], dtype=np.uint32)
    database.write_two_view_geometry(image_id, image_id2, two_view)
    read_two_view = database.read_two_view_geometry(image_id, image_id2)
    assert (
        read_two_view.config == pycolmap.TwoViewGeometryConfiguration.CALIBRATED
    )


def test_database_num_cameras(populated_database):
    database, _, _ = populated_database
    assert database.num_cameras() >= 1


def test_database_num_images(populated_database):
    database, _, _ = populated_database
    assert database.num_images() >= 1


def test_database_num_keypoints(database):
    assert database.num_keypoints() >= 0


def test_database_num_descriptors(database):
    assert database.num_descriptors() >= 0


def test_database_num_matches(database):
    assert database.num_matches() >= 0


def test_database_exists_camera(populated_database):
    database, camera_id, _ = populated_database
    assert database.exists_camera(camera_id)


def test_database_exists_image(populated_database):
    database, _, image_id = populated_database
    assert database.exists_image(image_id)


def test_database_exists_keypoints(populated_database):
    database, _, image_id = populated_database
    assert isinstance(database.exists_keypoints(image_id), bool)


def test_database_exists_descriptors(populated_database):
    database, _, image_id = populated_database
    assert isinstance(database.exists_descriptors(image_id), bool)


def test_database_read_all_cameras(populated_database):
    database, _, _ = populated_database
    assert len(database.read_all_cameras()) >= 1


def test_database_read_all_images(populated_database):
    database, _, _ = populated_database
    assert len(database.read_all_images()) >= 1


def test_database_clear_all_tables(database, simple_camera):
    database.write_camera(simple_camera)
    database.clear_all_tables()
    assert database.num_cameras() == 0


def test_database_merge(tmp_path):
    path1 = str(tmp_path / "db1.db")
    path2 = str(tmp_path / "db2.db")
    path_merged = str(tmp_path / "merged.db")
    with pycolmap.Database.open(path1) as database1:
        camera1 = pycolmap.Camera.create_from_model_id(
            1, pycolmap.CameraModelId.PINHOLE, 500.0, 1024, 768
        )
        database1.write_camera(camera1)
    with pycolmap.Database.open(path2) as database2:
        camera2 = pycolmap.Camera.create_from_model_id(
            1, pycolmap.CameraModelId.PINHOLE, 600.0, 800, 600
        )
        database2.write_camera(camera2)
    with pycolmap.Database.open(path1) as database1:
        with pycolmap.Database.open(path2) as database2:
            with pycolmap.Database.open(path_merged) as merged_database:
                pycolmap.Database.merge(database1, database2, merged_database)
                assert merged_database.num_cameras() == 2


def test_database_transaction(database, simple_camera):
    with pycolmap.DatabaseTransaction(database):
        database.write_camera(simple_camera)
    assert database.num_cameras() == 1
