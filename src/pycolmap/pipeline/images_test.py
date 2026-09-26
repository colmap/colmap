# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest
from PIL import Image as PILImage

import pycolmap


def test_camera_mode_enum() -> None:
    assert {k: int(v) for k, v in pycolmap.CameraMode.__members__.items()} == {
        "AUTO": 0,
        "SINGLE": 1,
        "PER_FOLDER": 2,
        "PER_IMAGE": 3,
    }


def test_file_copy_type_enum() -> None:
    assert {
        k: int(v) for k, v in pycolmap.FileCopyType.__members__.items()
    } == {
        "copy": 0,
        "hardlink": 1,
        "softlink": 2,
    }


def test_image_reader_options_init() -> None:
    options = pycolmap.ImageReaderOptions()
    assert options is not None


def test_image_reader_options_camera_model() -> None:
    options = pycolmap.ImageReaderOptions()
    options.camera_model = "SIMPLE_PINHOLE"
    assert options.camera_model == "SIMPLE_PINHOLE"


def test_image_reader_options_check() -> None:
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
def test_public_api_callable(name: str) -> None:
    assert callable(getattr(pycolmap, name))


def test_import_images_uses_exif_focal_length(tmp_path: Path) -> None:
    image_path = tmp_path / "images"
    image_path.mkdir()
    image_file = image_path / "image.jpg"
    pil_image = PILImage.new("RGB", (64, 48))
    exif = pil_image.getexif()
    exif[41989] = 43  # FocalLengthIn35mmFilm.
    pil_image.save(image_file, exif=exif)

    database_path = tmp_path / "database.db"
    with pycolmap.Database.open(database_path):
        pass

    pycolmap.import_images(database_path, image_path)

    expected = pycolmap.infer_camera_from_image(image_file)
    with pycolmap.Database.open(database_path) as database:
        cameras = database.read_all_cameras()
    assert len(cameras) == 1
    assert cameras[0].has_prior_focal_length
    assert cameras[0].focal_length == pytest.approx(expected.focal_length)
