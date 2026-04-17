import pycolmap


def test_camera_mode_auto():
    assert pycolmap.CameraMode.AUTO is not None


def test_camera_mode_single():
    assert pycolmap.CameraMode.SINGLE is not None


def test_camera_mode_per_folder():
    assert pycolmap.CameraMode.PER_FOLDER is not None


def test_camera_mode_per_image():
    assert pycolmap.CameraMode.PER_IMAGE is not None


def test_file_copy_type_copy():
    assert pycolmap.FileCopyType.copy is not None


def test_file_copy_type_softlink():
    assert pycolmap.FileCopyType.softlink is not None


def test_file_copy_type_hardlink():
    assert pycolmap.FileCopyType.hardlink is not None


def test_image_reader_options_init():
    options = pycolmap.ImageReaderOptions()
    assert options is not None


def test_image_reader_options_camera_model():
    options = pycolmap.ImageReaderOptions()
    options.camera_model = "SIMPLE_PINHOLE"
    assert options.camera_model == "SIMPLE_PINHOLE"


def test_image_reader_options_check():
    options = pycolmap.ImageReaderOptions()
    assert options.check()


def test_import_images_callable():
    assert callable(pycolmap.import_images)


def test_infer_camera_from_image_callable():
    assert callable(pycolmap.infer_camera_from_image)


def test_undistort_images_callable():
    assert callable(pycolmap.undistort_images)
