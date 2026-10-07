# SPDX-License-Identifier: BSD-3-Clause

import copy
from pathlib import Path

import numpy as np
import pytest

import pycolmap


def test_database_direct_construction_is_disallowed() -> None:
    # Regression test for https://github.com/colmap/colmap/issues/4422:
    # Database is abstract, so direct construction is disallowed. Use
    # Database.open() instead.
    with pytest.raises(TypeError):
        pycolmap.Database()


def test_database_open_and_context_manager(tmp_path: Path) -> None:
    database_path = str(tmp_path / "test.db")
    with pycolmap.Database.open(database_path) as database:
        assert database is not None


def test_database_write_and_read_camera(
    database: pycolmap.Database, simple_camera: pycolmap.Camera
) -> None:
    camera_id = database.write_camera(simple_camera)
    assert camera_id > 0
    read_camera = database.read_camera(camera_id)
    assert read_camera.model == pycolmap.CameraModelId.PINHOLE


def test_database_update_camera(
    database: pycolmap.Database, simple_camera: pycolmap.Camera
) -> None:
    camera_id = database.write_camera(simple_camera)
    simple_camera.camera_id = camera_id
    simple_camera.width = 2048
    database.update_camera(simple_camera)
    updated_camera = database.read_camera(camera_id)
    assert updated_camera.width == 2048


def test_database_write_and_read_image(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
    database, camera_id, image_id = populated_database
    assert image_id > 0
    read_image = database.read_image(image_id)
    assert read_image.name == "test.jpg"


def test_database_update_image(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
    database, camera_id, image_id = populated_database
    image = database.read_image(image_id)
    image.name = "updated.jpg"
    database.update_image(image)
    updated_image = database.read_image(image_id)
    assert updated_image.name == "updated.jpg"


def test_database_clear_cameras(
    database: pycolmap.Database, simple_camera: pycolmap.Camera
) -> None:
    database.write_camera(simple_camera)
    assert database.num_cameras() > 0
    database.clear_cameras()
    assert database.num_cameras() == 0


def test_database_write_and_read_keypoints(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
    database, camera_id, image_id = populated_database
    keypoints = np.array(
        [[10.0, 20.0, 1.0, 0.0, 0.0, 1.0], [30.0, 40.0, 1.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    database.write_keypoints(image_id, keypoints)
    read_keypoints = database.read_keypoints(image_id)
    assert read_keypoints.shape[0] == 2


def test_database_write_and_read_descriptors(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
    database, camera_id, image_id = populated_database
    descriptors = pycolmap.FeatureDescriptors(
        type=pycolmap.FeatureExtractorType.SIFT,
        data=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
    )
    database.write_descriptors(image_id, descriptors)
    read_descriptors = database.read_descriptors(image_id)
    assert read_descriptors is not None


def test_database_write_and_read_matches(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
    database, camera_id, image_id = populated_database
    image2 = pycolmap.Image()
    image2.name = "test2.jpg"
    image2.camera_id = camera_id
    image_id2 = database.write_image(image2)
    matches = np.array([[0, 0], [1, 1]], dtype=np.uint32)
    database.write_matches(image_id, image_id2, matches)
    read_matches = database.read_matches(image_id, image_id2)
    assert read_matches.shape[0] == 2


def test_database_write_and_read_two_view_geometry(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
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


def test_database_num_cameras(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
    database, _, _ = populated_database
    assert database.num_cameras() >= 1


def test_database_num_images(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
    database, _, _ = populated_database
    assert database.num_images() >= 1


def test_database_num_keypoints(database: pycolmap.Database) -> None:
    assert database.num_keypoints() >= 0


def test_database_num_descriptors(database: pycolmap.Database) -> None:
    assert database.num_descriptors() >= 0


def test_database_num_matches(database: pycolmap.Database) -> None:
    assert database.num_matches() >= 0


def test_database_exists_camera(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
    database, camera_id, _ = populated_database
    assert database.exists_camera(camera_id)


def test_database_exists_image(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
    database, _, image_id = populated_database
    assert database.exists_image(image_id)


def test_database_exists_keypoints(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
    database, _, image_id = populated_database
    assert isinstance(database.exists_keypoints(image_id), bool)


def test_database_exists_descriptors(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
    database, _, image_id = populated_database
    assert isinstance(database.exists_descriptors(image_id), bool)


def test_database_read_all_cameras(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
    database, _, _ = populated_database
    assert len(database.read_all_cameras()) >= 1


def test_database_read_all_images(
    populated_database: tuple[pycolmap.Database, int, int],
) -> None:
    database, _, _ = populated_database
    assert len(database.read_all_images()) >= 1


def test_database_clear_all_tables(
    database: pycolmap.Database, simple_camera: pycolmap.Camera
) -> None:
    database.write_camera(simple_camera)
    database.clear_all_tables()
    assert database.num_cameras() == 0


def test_database_merge(tmp_path: Path) -> None:
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


def test_database_transaction(
    database: pycolmap.Database, simple_camera: pycolmap.Camera
) -> None:
    with pycolmap.DatabaseTransaction(database):
        database.write_camera(simple_camera)
    assert database.num_cameras() == 1


def test_database_camera_calibrations(
    database: pycolmap.Database, simple_camera: pycolmap.Camera
) -> None:
    simple_camera.source = pycolmap.CameraSource.GUESS
    camera_id = database.write_camera(simple_camera)
    assert camera_id > 0
    assert database.exists_camera(camera_id)
    with pytest.raises(ValueError):
        database.exists_camera_source(
            camera_id, pycolmap.CameraSource.BEST
        )
    assert database.exists_camera_source(
        camera_id, pycolmap.CameraSource.GUESS
    )
    assert not database.exists_camera_source(
        camera_id, pycolmap.CameraSource.EXIF
    )

    read_guess = database.read_camera(camera_id)
    assert read_guess.source == pycolmap.CameraSource.GUESS

    # Add EXIF calibration (EXIF > GUESS).
    camera_exif = copy.copy(read_guess)
    camera_exif.source = pycolmap.CameraSource.EXIF
    camera_exif.focal_length_x = 600.0
    camera_exif.focal_length_y = 600.0
    database.update_camera(camera_exif)

    assert database.exists_camera_source(
        camera_id, pycolmap.CameraSource.EXIF
    )
    assert database.read_camera(camera_id).source == pycolmap.CameraSource.EXIF
    assert database.read_camera(camera_id).focal_length_x == 600.0
    assert (
        database.read_camera(
            camera_id, pycolmap.CameraSource.GUESS
        ).focal_length_x
        != 600.0
    )

    # Add VIEW_GRAPH calibration (VIEW_GRAPH > EXIF).
    camera_vg = copy.copy(read_guess)
    camera_vg.source = pycolmap.CameraSource.VIEW_GRAPH
    camera_vg.focal_length_x = 700.0
    camera_vg.focal_length_y = 700.0
    database.update_camera(camera_vg)

    assert (
        database.read_camera(camera_id).source
        == pycolmap.CameraSource.VIEW_GRAPH
    )
    assert database.read_camera(camera_id).focal_length_x == 700.0

    # Read calibrations.
    calibs = database.read_all_camera_sources(camera_id)
    assert len(calibs) == 3
    assert pycolmap.CameraSource.GUESS in calibs
    assert pycolmap.CameraSource.EXIF in calibs
    assert pycolmap.CameraSource.VIEW_GRAPH in calibs
    assert calibs[pycolmap.CameraSource.EXIF].focal_length_x == 600.0
    assert calibs[pycolmap.CameraSource.VIEW_GRAPH].focal_length_x == 700.0

    all_calibs = database.read_all_camera_sources()
    assert len(all_calibs) == 1
    assert camera_id in all_calibs
    assert len(all_calibs[camera_id]) == 3

    assert len(database.read_all_cameras(pycolmap.CameraSource.BEST)) == 1
    assert len(database.read_all_cameras(pycolmap.CameraSource.VIEW_GRAPH)) == 1
    assert len(database.read_all_cameras(pycolmap.CameraSource.EXIF)) == 1
    assert len(database.read_all_cameras(pycolmap.CameraSource.GUESS)) == 1
    assert len(database.read_all_cameras(pycolmap.CameraSource.USER)) == 0

    # Test read_camera_excluding_sources and read_all_cameras_excluding_sources.
    cam_ex_vg = database.read_camera_excluding_sources(
        camera_id, [pycolmap.CameraSource.VIEW_GRAPH]
    )
    assert cam_ex_vg.source == pycolmap.CameraSource.EXIF
    assert cam_ex_vg.focal_length_x == 600.0

    cams_map = database.read_all_cameras_excluding_sources(
        [pycolmap.CameraSource.VIEW_GRAPH]
    )
    assert len(cams_map) == 1
    assert cams_map[camera_id].source == pycolmap.CameraSource.EXIF

    # Delete VIEW_GRAPH calibration: falls back to EXIF.
    database.delete_camera_source(
        camera_id, pycolmap.CameraSource.VIEW_GRAPH
    )
    assert not database.exists_camera_source(
        camera_id, pycolmap.CameraSource.VIEW_GRAPH
    )
    assert database.exists_camera(camera_id)
    assert database.read_camera(camera_id).source == pycolmap.CameraSource.EXIF
    assert database.read_camera(camera_id).focal_length_x == 600.0

    # Delete BEST deletes entire camera.
    database.delete_camera_source(camera_id, pycolmap.CameraSource.BEST)
    assert database.num_cameras() == 0
    assert not database.exists_camera(camera_id)


def test_database_merge_calibrations(tmp_path: Path) -> None:
    path1 = str(tmp_path / "db1.db")
    path2 = str(tmp_path / "db2.db")
    path_merged = str(tmp_path / "merged.db")
    with pycolmap.Database.open(path1) as database1:
        cam1 = pycolmap.Camera.create_from_model_id(
            1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 500.0, 1024, 768
        )
        cam1.source = pycolmap.CameraSource.GUESS
        database1.write_camera(cam1)
        cam1_exif = copy.copy(cam1)
        cam1_exif.source = pycolmap.CameraSource.EXIF
        cam1_exif.focal_length = 550.0
        database1.update_camera(cam1_exif)

    with pycolmap.Database.open(path2) as database2:
        cam2 = pycolmap.Camera.create_from_model_id(
            1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 600.0, 800, 600
        )
        cam2.source = pycolmap.CameraSource.GUESS
        database2.write_camera(cam2)
        cam2_user = copy.copy(cam2)
        cam2_user.source = pycolmap.CameraSource.USER
        cam2_user.focal_length = 650.0
        database2.update_camera(cam2_user)

    with pycolmap.Database.open(path1) as database1:
        with pycolmap.Database.open(path2) as database2:
            with pycolmap.Database.open(path_merged) as merged_database:
                pycolmap.Database.merge(database1, database2, merged_database)
                assert merged_database.num_cameras() == 2
                calibs1 = merged_database.read_all_camera_sources(1)
                assert len(calibs1) == 2
                assert (
                    merged_database.read_camera(1).source
                    == pycolmap.CameraSource.EXIF
                )
                assert (
                    merged_database.read_camera(
                        1, pycolmap.CameraSource.EXIF
                    ).focal_length
                    == 550.0
                )
                calibs2 = merged_database.read_all_camera_sources(2)
                assert len(calibs2) == 2
                assert (
                    merged_database.read_camera(2).source
                    == pycolmap.CameraSource.USER
                )
                assert (
                    merged_database.read_camera(
                        2, pycolmap.CameraSource.USER
                    ).focal_length
                    == 650.0
                )
