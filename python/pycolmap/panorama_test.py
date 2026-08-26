import numpy as np

import pycolmap

from .panorama import (
    filter_database_by_covisibility,
    get_virtual_rotations,
)


def test_get_virtual_rotations():
    rotations = get_virtual_rotations(4, [-35.0, 0.0, 35.0])

    assert len(rotations) == 12
    np.testing.assert_allclose(
        rotations[4],
        np.eye(3),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        rotations[1],
        [
            [0.0, 0.0, -1.0],
            [-0.573576436351046, 0.8191520442889917, 0.0],
            [0.8191520442889917, 0.573576436351046, 0.0],
        ],
        atol=1e-15,
    )
    for rotation in rotations:
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-15)
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-15)


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
