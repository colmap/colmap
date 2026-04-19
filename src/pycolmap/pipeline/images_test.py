import pytest

import pycolmap


def test_camera_mode_enum():
    assert {m.name: int(m) for m in pycolmap.CameraMode} == {
        "AUTO": 0,
        "SINGLE": 1,
        "PER_FOLDER": 2,
        "PER_IMAGE": 3,
    }


def test_file_copy_type_enum():
    assert {m.name: int(m) for m in pycolmap.FileCopyType} == {
        "copy": 0,
        "hardlink": 1,
        "softlink": 2,
    }


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


@pytest.mark.parametrize(
    "name",
    [
        "import_images",
        "infer_camera_from_image",
        "undistort_images",
    ],
)
def test_public_api_callable(name):
    assert callable(getattr(pycolmap, name))
