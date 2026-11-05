# SPDX-License-Identifier: BSD-3-Clause

import functools
from collections.abc import Iterator
from pathlib import Path

import pytest

import pycolmap


@functools.cache
def _cusift_available() -> bool:
    # Check() rejects CUSIFT in builds without cusift, which also lack
    # get_num_cuda_devices when they lack CUDA, so it must come first.
    return (
        pycolmap.FeatureExtractionOptions(
            type=pycolmap.FeatureExtractorType.CUSIFT
        ).check()
        and pycolmap.get_num_cuda_devices() > 0
    )


@pytest.fixture
def requires_cusift() -> None:
    if not _cusift_available():
        pytest.skip("needs a build with cusift and a CUDA device")


@pytest.fixture
def simple_camera() -> pycolmap.Camera:
    return pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.PINHOLE, 500.0, 1024, 768
    )


@pytest.fixture(scope="session")
def synthetic_reconstruction() -> pycolmap.Reconstruction:
    options = pycolmap.SyntheticDatasetOptions()
    options.num_cameras_per_rig = 1
    options.num_frames_per_rig = 3
    options.num_points3D = 50
    return pycolmap.synthesize_dataset(options)


@pytest.fixture
def database(tmp_path: Path) -> Iterator[pycolmap.Database]:
    with pycolmap.Database.open(str(tmp_path / "test.db")) as db:
        yield db


@pytest.fixture
def populated_database(
    database: pycolmap.Database,
) -> tuple[pycolmap.Database, int, int]:
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.PINHOLE, 500.0, 1024, 768
    )
    camera_id = database.write_camera(camera)
    image = pycolmap.Image()
    image.name = "test.jpg"
    image.camera_id = camera_id
    image_id = database.write_image(image)
    return database, camera_id, image_id
