import numpy as np
import pytest

pytest.importorskip("cv2")
from panorama_sfm import filter_database_by_covisibility

import pycolmap


def test_filter_database_by_covisibility(tmp_path):
    database_path = tmp_path / "database.db"
    with pycolmap.Database.open(database_path) as database:
        camera = pycolmap.Camera.create_from_model_name(
            1, "SIMPLE_PINHOLE", 100.0, 100, 100
        )
        database.write_camera(camera)
        for image_id, name in enumerate(["a.png", "b.png", "c.png"], start=1):
            database.write_image(
                pycolmap.Image(
                    image_id=image_id, camera_id=camera.camera_id, name=name
                ),
                use_image_id=True,
            )
        geometry = pycolmap.TwoViewGeometry()
        geometry.inlier_matches = np.array([[0, 0]], dtype=np.uint32)
        database.write_two_view_geometry(1, 2, geometry)
        database.write_two_view_geometry(1, 3, geometry)

    covisibility_path = tmp_path / "covisibility.npz"
    np.savez(
        covisibility_path,
        image_names=np.array(["a.png", "b.png", "c.png"]),
        directed_overlap_counts=np.array(
            [[1, 10, 0], [10, 1, 0], [0, 0, 1]], dtype=np.uint32
        ),
    )
    filter_database_by_covisibility(
        database_path, covisibility_path, min_shared_points=1
    )

    with pycolmap.Database.open(database_path) as database:
        assert database.exists_two_view_geometry(1, 2)
        assert not database.exists_two_view_geometry(1, 3)
