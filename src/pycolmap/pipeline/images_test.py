# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest

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


def test_undistort_images_with_masks(tmp_path: Path) -> None:
    image_path = tmp_path / "images"
    mask_path = tmp_path / "masks"
    input_path = tmp_path / "sparse"
    output_path = tmp_path / "dense"
    for path in (image_path, mask_path, input_path):
        path.mkdir()
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 1
    options.num_cameras_per_rig = 1
    options.num_frames_per_rig = 1
    options.camera_width = 100
    options.camera_height = 80
    options.camera_params = [70, 50, 40, 0.3]
    reconstruction = pycolmap.synthesize_dataset(options)
    reconstruction.write(input_path)
    image = next(iter(reconstruction.images.values()))
    pixels = np.full((80, 100, 3), 128, dtype=np.uint8)
    assert pycolmap.Bitmap.from_array(pixels).write(image_path / image.name)
    # The default must remain an empty input path, not pathlib.Path(".").
    unmasked_output_path = tmp_path / "unmasked"
    pycolmap.undistort_images(
        unmasked_output_path, input_path, image_path, num_threads=1
    )
    assert not (unmasked_output_path / "masks").exists()
    with pytest.raises(RuntimeError, match="Cannot read mask"):
        pycolmap.undistort_images(
            output_path,
            input_path,
            image_path,
            mask_path=mask_path,
            undistort_options={"max_image_size": 24},
            num_threads=1,
        )
    mask = ((np.indices((80, 100)).sum(axis=0) % 2) * 255).astype(np.uint8)
    assert pycolmap.Bitmap.from_array(mask).write(
        mask_path / (image.name + ".png")
    )
    pycolmap.undistort_images(
        output_path,
        input_path,
        image_path,
        mask_path=mask_path,
        undistort_options={"max_image_size": 24},
        num_threads=1,
    )
    output_mask = pycolmap.Bitmap.read(
        output_path / "masks" / (image.name + ".png"), as_rgb=False
    )
    assert output_mask is not None
    output_reconstruction = pycolmap.Reconstruction(output_path / "sparse")
    camera = next(iter(output_reconstruction.cameras.values()))
    assert (output_mask.width, output_mask.height) == (
        camera.width,
        camera.height,
    )
    assert set(np.unique(output_mask.to_array())) == {0, 255}


@pytest.mark.parametrize("output_type", ["PMVS", "CMP-MVS"])
def test_undistort_masks_require_colmap_output(
    tmp_path: Path, output_type: str
) -> None:
    with pytest.raises(ValueError, match="mask_path is only supported"):
        pycolmap.undistort_images(
            tmp_path / "output",
            tmp_path / "missing_model",
            tmp_path,
            output_type=output_type,
            mask_path=tmp_path,
        )
