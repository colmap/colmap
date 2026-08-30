import pytest

import pycolmap


def test_cancellation_token():
    token = pycolmap.CancellationToken()
    assert not token.is_cancelled
    token.cancel()
    assert token.is_cancelled


def test_cancelled_feature_extraction_raises(tmp_path):
    image_path = tmp_path / "images"
    image_path.mkdir()
    token = pycolmap.CancellationToken()
    token.cancel()

    with pytest.raises(InterruptedError):
        pycolmap.extract_features(
            tmp_path / "database.db",
            image_path,
            device=pycolmap.Device.cpu,
            cancellation_token=token,
        )


def test_cancelled_bundle_adjustment_raises():
    options = pycolmap.SyntheticDatasetOptions()
    options.num_cameras_per_rig = 1
    options.num_frames_per_rig = 3
    options.num_points3D = 50
    reconstruction = pycolmap.synthesize_dataset(options)
    token = pycolmap.CancellationToken()
    token.cancel()

    with pytest.raises(InterruptedError):
        pycolmap.bundle_adjustment(reconstruction, cancellation_token=token)


def test_pre_cancelled_triangulation_preserves_reconstruction(tmp_path):
    database_path = tmp_path / "database.db"
    image_path = tmp_path / "images"
    image_path.mkdir()
    output_path = tmp_path / "output"
    options = pycolmap.SyntheticDatasetOptions()
    options.num_cameras_per_rig = 1
    options.num_frames_per_rig = 3
    options.num_points3D = 50
    with pycolmap.Database.open(database_path) as database:
        reconstruction = pycolmap.synthesize_dataset(options, database)

    num_points3D = reconstruction.num_points3D()
    token = pycolmap.CancellationToken()
    token.cancel()

    with pytest.raises(InterruptedError):
        pycolmap.triangulate_points(
            reconstruction,
            database_path,
            image_path,
            output_path,
            cancellation_token=token,
        )

    assert reconstruction.num_points3D() == num_points3D
    assert not (output_path / "cameras.bin").exists()
    assert not (output_path / "images.bin").exists()
    assert not (output_path / "points3D.bin").exists()


@pytest.mark.parametrize(
    "function_name",
    [
        "bundle_adjustment",
        "stereo_fusion",
        "triangulate_points",
        "undistort_images",
    ],
)
def test_pipeline_accepts_cancellation_token(function_name):
    assert "cancellation_token" in getattr(pycolmap, function_name).__doc__
