import os

import pycolmap


def test_reconstruction_default_init():
    reconstruction = pycolmap.Reconstruction()
    assert reconstruction is not None


def test_reconstruction_copy_init(synthetic_reconstruction):
    reconstruction_copy = pycolmap.Reconstruction(synthetic_reconstruction)
    assert (
        reconstruction_copy.num_cameras()
        == synthetic_reconstruction.num_cameras()
    )


def test_reconstruction_num_rigs(synthetic_reconstruction):
    assert isinstance(synthetic_reconstruction.num_rigs(), int)
    assert synthetic_reconstruction.num_rigs() > 0


def test_reconstruction_num_cameras(synthetic_reconstruction):
    assert isinstance(synthetic_reconstruction.num_cameras(), int)
    assert synthetic_reconstruction.num_cameras() > 0


def test_reconstruction_num_frames(synthetic_reconstruction):
    assert isinstance(synthetic_reconstruction.num_frames(), int)
    assert synthetic_reconstruction.num_frames() > 0


def test_reconstruction_num_images(synthetic_reconstruction):
    assert isinstance(synthetic_reconstruction.num_images(), int)
    assert synthetic_reconstruction.num_images() > 0


def test_reconstruction_num_points3d(synthetic_reconstruction):
    assert isinstance(synthetic_reconstruction.num_points3D(), int)
    assert synthetic_reconstruction.num_points3D() > 0


def test_reconstruction_num_reg_frames(synthetic_reconstruction):
    assert isinstance(synthetic_reconstruction.num_reg_frames(), int)


def test_reconstruction_num_reg_images(synthetic_reconstruction):
    assert isinstance(synthetic_reconstruction.num_reg_images(), int)


def test_reconstruction_cameras_property(synthetic_reconstruction):
    cameras = synthetic_reconstruction.cameras
    assert len(cameras) > 0


def test_reconstruction_frames_property(synthetic_reconstruction):
    frames = synthetic_reconstruction.frames
    assert frames is not None


def test_reconstruction_images_property(synthetic_reconstruction):
    images = synthetic_reconstruction.images
    assert len(images) > 0


def test_reconstruction_points3d_property(synthetic_reconstruction):
    points = synthetic_reconstruction.points3D
    assert len(points) > 0


def test_reconstruction_camera_access(synthetic_reconstruction):
    camera_ids = list(synthetic_reconstruction.cameras.keys())
    camera = synthetic_reconstruction.camera(camera_ids[0])
    assert camera is not None


def test_reconstruction_frame_access(synthetic_reconstruction):
    frame_ids = list(synthetic_reconstruction.frames.keys())
    frame = synthetic_reconstruction.frame(frame_ids[0])
    assert frame is not None


def test_reconstruction_image_access(synthetic_reconstruction):
    image_ids = list(synthetic_reconstruction.images.keys())
    image = synthetic_reconstruction.image(image_ids[0])
    assert image is not None


def test_reconstruction_point3d_access(synthetic_reconstruction):
    point_ids = list(synthetic_reconstruction.points3D.keys())
    point = synthetic_reconstruction.point3D(point_ids[0])
    assert point is not None


def test_reconstruction_exists_camera(synthetic_reconstruction):
    camera_ids = list(synthetic_reconstruction.cameras.keys())
    assert synthetic_reconstruction.exists_camera(camera_ids[0])
    assert not synthetic_reconstruction.exists_camera(99999)


def test_reconstruction_exists_frame(synthetic_reconstruction):
    frame_ids = list(synthetic_reconstruction.frames.keys())
    assert synthetic_reconstruction.exists_frame(frame_ids[0])


def test_reconstruction_exists_image(synthetic_reconstruction):
    image_ids = list(synthetic_reconstruction.images.keys())
    assert synthetic_reconstruction.exists_image(image_ids[0])
    assert not synthetic_reconstruction.exists_image(99999)


def test_reconstruction_exists_point3d(synthetic_reconstruction):
    point_ids = list(synthetic_reconstruction.points3D.keys())
    assert synthetic_reconstruction.exists_point3D(point_ids[0])
    assert not synthetic_reconstruction.exists_point3D(99999)


def test_reconstruction_reg_image_ids(synthetic_reconstruction):
    reg_ids = synthetic_reconstruction.reg_image_ids()
    assert isinstance(reg_ids, list)


def test_reconstruction_reg_frame_ids(synthetic_reconstruction):
    reg_ids = synthetic_reconstruction.reg_frame_ids()
    assert isinstance(reg_ids, list)


def test_reconstruction_point3d_ids(synthetic_reconstruction):
    point_ids = synthetic_reconstruction.point3D_ids()
    assert len(point_ids) > 0


def test_reconstruction_is_valid(synthetic_reconstruction):
    result = synthetic_reconstruction.is_valid()
    assert isinstance(result, bool)


def test_reconstruction_compute_num_observations(synthetic_reconstruction):
    count = synthetic_reconstruction.compute_num_observations()
    assert isinstance(count, int)


def test_reconstruction_compute_mean_track_length(synthetic_reconstruction):
    length = synthetic_reconstruction.compute_mean_track_length()
    assert isinstance(length, float)


def test_reconstruction_compute_mean_reprojection_error(
    synthetic_reconstruction,
):
    reconstruction_copy = pycolmap.Reconstruction(synthetic_reconstruction)
    error = reconstruction_copy.compute_mean_reprojection_error()
    assert isinstance(error, float)


def test_reconstruction_compute_mean_observations_per_reg_image(
    synthetic_reconstruction,
):
    mean_obs = (
        synthetic_reconstruction.compute_mean_observations_per_reg_image()
    )
    assert isinstance(mean_obs, float)


def test_reconstruction_summary(synthetic_reconstruction):
    summary = synthetic_reconstruction.summary()
    assert isinstance(summary, str)
    assert "Reconstruction" in summary


def test_reconstruction_compute_bounding_box(synthetic_reconstruction):
    bbox = synthetic_reconstruction.compute_bounding_box()
    assert bbox is not None


def test_reconstruction_compute_centroid(synthetic_reconstruction):
    centroid = synthetic_reconstruction.compute_centroid()
    assert centroid.shape == (3,)


def test_reconstruction_normalize(synthetic_reconstruction):
    reconstruction_copy = pycolmap.Reconstruction(synthetic_reconstruction)
    reconstruction_copy.normalize()
    assert reconstruction_copy.num_points3D() > 0


def test_reconstruction_transform(synthetic_reconstruction):
    reconstruction_copy = pycolmap.Reconstruction(synthetic_reconstruction)
    transform = pycolmap.Sim3d()
    reconstruction_copy.transform(transform)
    assert reconstruction_copy.num_points3D() > 0


def test_reconstruction_write_read_binary_roundtrip(
    synthetic_reconstruction, tmp_path
):
    output_dir = str(tmp_path / "binary")

    os.makedirs(output_dir)
    synthetic_reconstruction.write_binary(output_dir)
    loaded = pycolmap.Reconstruction()
    loaded.read_binary(output_dir)
    assert loaded.num_cameras() == synthetic_reconstruction.num_cameras()
    assert loaded.num_images() == synthetic_reconstruction.num_images()


def test_reconstruction_write_read_text_roundtrip(
    synthetic_reconstruction, tmp_path
):
    output_dir = str(tmp_path / "text")

    os.makedirs(output_dir)
    synthetic_reconstruction.write_text(output_dir)
    loaded = pycolmap.Reconstruction()
    loaded.read_text(output_dir)
    assert loaded.num_cameras() == synthetic_reconstruction.num_cameras()
    assert loaded.num_images() == synthetic_reconstruction.num_images()


def test_reconstruction_add_camera():
    reconstruction = pycolmap.Reconstruction()
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.PINHOLE, 500.0, 1024, 768
    )
    reconstruction.add_camera(camera)
    assert reconstruction.exists_camera(1)


def test_reconstruction_find_image_with_name(synthetic_reconstruction):
    image_ids = list(synthetic_reconstruction.images.keys())
    first_image = synthetic_reconstruction.image(image_ids[0])
    found = synthetic_reconstruction.find_image_with_name(first_image.name)
    assert found is not None


def test_reconstruction_find_common_reg_image_ids(synthetic_reconstruction):
    reconstruction_copy = pycolmap.Reconstruction(synthetic_reconstruction)
    common_ids = synthetic_reconstruction.find_common_reg_image_ids(
        reconstruction_copy
    )
    assert isinstance(common_ids, list)


def test_reconstruction_export_import_ply_roundtrip(
    synthetic_reconstruction, tmp_path
):
    ply_path = str(tmp_path / "points.ply")
    synthetic_reconstruction.export_PLY(ply_path)
    new_reconstruction = pycolmap.Reconstruction()
    new_reconstruction.import_PLY(ply_path)
    assert new_reconstruction.num_points3D() > 0


def test_reconstruction_crop(synthetic_reconstruction):
    reconstruction_copy = pycolmap.Reconstruction(synthetic_reconstruction)
    bbox = reconstruction_copy.compute_bounding_box()
    reconstruction_copy.crop(bbox)
    assert reconstruction_copy is not None


def test_reconstruction_delete_all_points2d_and_points3d(
    synthetic_reconstruction,
):
    reconstruction_copy = pycolmap.Reconstruction(synthetic_reconstruction)
    reconstruction_copy.delete_all_points2D_and_points3D()
    assert reconstruction_copy.num_points3D() == 0
