import pycolmap


def test_synthetic_dataset_match_config_enum():
    assert {m.name: int(m) for m in pycolmap.SyntheticDatasetMatchConfig} == {
        "EXHAUSTIVE": 1,
        "CHAINED": 2,
        "SPARSE": 3,
    }


def test_synthetic_dataset_options_default_init():
    options = pycolmap.SyntheticDatasetOptions()
    assert options is not None


def test_synthetic_dataset_options_num_rigs_readwrite():
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 2
    assert options.num_rigs == 2


def test_synthetic_dataset_options_num_cameras_per_rig_readwrite():
    options = pycolmap.SyntheticDatasetOptions()
    options.num_cameras_per_rig = 3
    assert options.num_cameras_per_rig == 3


def test_synthetic_dataset_options_num_frames_per_rig_readwrite():
    options = pycolmap.SyntheticDatasetOptions()
    options.num_frames_per_rig = 5
    assert options.num_frames_per_rig == 5


def test_synthetic_dataset_options_num_points3d_readwrite():
    options = pycolmap.SyntheticDatasetOptions()
    options.num_points3D = 100
    assert options.num_points3D == 100


def test_synthetic_dataset_options_track_length_readwrite():
    options = pycolmap.SyntheticDatasetOptions()
    options.track_length = 3
    assert options.track_length == 3


def test_synthetic_dataset_options_camera_model_id_readwrite():
    options = pycolmap.SyntheticDatasetOptions()
    options.camera_model_id = pycolmap.CameraModelId.PINHOLE
    assert options.camera_model_id == pycolmap.CameraModelId.PINHOLE


def test_synthetic_dataset_options_match_config_readwrite():
    options = pycolmap.SyntheticDatasetOptions()
    options.match_config = pycolmap.SyntheticDatasetMatchConfig.CHAINED
    assert options.match_config == pycolmap.SyntheticDatasetMatchConfig.CHAINED


def test_synthesize_dataset():
    options = pycolmap.SyntheticDatasetOptions()
    options.num_cameras_per_rig = 1
    options.num_frames_per_rig = 2
    options.num_points3D = 10
    reconstruction = pycolmap.synthesize_dataset(options)
    assert reconstruction.num_cameras() > 0
    assert reconstruction.num_images() > 0
    assert reconstruction.num_points3D() > 0


def test_synthetic_noise_options_default_init():
    options = pycolmap.SyntheticNoiseOptions()
    assert options is not None


def test_synthetic_noise_options_point2d_stddev_readwrite():
    options = pycolmap.SyntheticNoiseOptions()
    options.point2D_stddev = 0.5
    assert options.point2D_stddev == 0.5


def test_synthetic_noise_options_point3d_stddev_readwrite():
    options = pycolmap.SyntheticNoiseOptions()
    options.point3D_stddev = 0.1
    assert options.point3D_stddev == 0.1


def test_synthesize_noise():
    dataset_options = pycolmap.SyntheticDatasetOptions()
    dataset_options.num_cameras_per_rig = 1
    dataset_options.num_frames_per_rig = 2
    dataset_options.num_points3D = 10
    reconstruction = pycolmap.synthesize_dataset(dataset_options)
    noise_options = pycolmap.SyntheticNoiseOptions()
    noise_options.point2D_stddev = 1.0
    noise_options.point3D_stddev = 0.01
    pycolmap.synthesize_noise(noise_options, reconstruction)
    assert reconstruction.num_points3D() > 0


def test_synthetic_image_options_default_init():
    options = pycolmap.SyntheticImageOptions()
    assert options is not None


def test_synthetic_image_options_feature_peak_radius_readwrite():
    options = pycolmap.SyntheticImageOptions()
    options.feature_peak_radius = 5
    assert options.feature_peak_radius == 5
