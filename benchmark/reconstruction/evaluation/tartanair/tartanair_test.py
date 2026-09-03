import json
from pathlib import Path

import numpy as np

import pycolmap

from . import (
    NED_FROM_COLMAP,
    DatasetTartanAirPerspective,
    tartanair_world_from_camera,
)


def test_tartanair_world_from_camera() -> None:
    translation = np.array([1.0, 2.0, 3.0])
    quaternion = np.array([0.0, 0.0, 0.0, 1.0])

    world_from_camera = tartanair_world_from_camera(translation, quaternion)

    np.testing.assert_allclose(
        world_from_camera.rotation.matrix(), NED_FROM_COLMAP, atol=1e-15
    )
    np.testing.assert_allclose(world_from_camera.translation, translation)


def test_dataset_prepares_ground_truth(tmp_path: Path) -> None:
    scene_path = (
        tmp_path / "data" / "tartanair-v2" / "domestic" / "House-easy-P000"
    )
    (scene_path / "images").mkdir(parents=True)
    metadata = {
        "image_size": [2048, 1024],
        "frames": [
            {"source_frame": 10, "image_name": "000010.png"},
            {"source_frame": 11, "image_name": "000011.png"},
        ],
    }
    (scene_path / "scene.json").write_text(json.dumps(metadata))
    (scene_path / "poses.txt").write_text(
        "10 1 2 3 0 0 0 1\n11 2 2 3 0 0 0 1\n"
    )
    dataset = DatasetTartanAirPerspective(
        data_path=tmp_path / "data",
        categories=[],
        scenes=[],
        run_path=tmp_path / "runs",
        run_name="test",
    )

    scene_info = dataset.list_scenes()[0]
    dataset.prepare_scene(scene_info)
    reconstruction = pycolmap.Reconstruction(scene_info.sparse_gt_path)

    assert reconstruction.num_images() == 2
    assert reconstruction.num_cameras() == 1
    assert reconstruction.cameras[1].model_name == "EQUIRECTANGULAR"
    centers = sorted(
        image.projection_center().tolist()
        for image in reconstruction.images.values()
    )
    np.testing.assert_allclose(centers, [[1, 2, 3], [2, 2, 3]])
