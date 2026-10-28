# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pycolmap


@pytest.fixture
def scene() -> tuple[pycolmap.DatabaseCache, pycolmap.PoseGraph]:
    dataset_options = pycolmap.SyntheticDatasetOptions()
    dataset_options.num_rigs = 1
    dataset_options.num_cameras_per_rig = 1
    dataset_options.num_frames_per_rig = 4
    dataset_options.num_points3D = 50
    dataset_options.two_view_geometry_has_relative_pose = True
    with pycolmap.Database.open(":memory:") as database:
        reconstruction = pycolmap.synthesize_dataset(dataset_options, database)
        noise_options = pycolmap.SyntheticNoiseOptions()
        noise_options.point2D_stddev = 0.5
        pycolmap.synthesize_noise(noise_options, reconstruction, database)
        database_cache = pycolmap.DatabaseCache.create(
            database, pycolmap.DatabaseCacheOptions()
        )
    pose_graph = pycolmap.PoseGraph()
    pose_graph.load(database_cache.correspondence_graph)
    assert pose_graph.num_edges > 0
    return database_cache, pose_graph


def test_two_view_rotation_covariance_options() -> None:
    options = pycolmap.TwoViewRotationCovarianceOptions()
    options.point2D_stddev_px = 2.0
    options.max_rotation_sigma_deg = 30.0
    assert options.point2D_stddev_px == 2.0
    assert options.max_rotation_sigma_deg == 30.0


def test_estimate_two_view_rotation_covariance(
    scene: tuple[pycolmap.DatabaseCache, pycolmap.PoseGraph],
) -> None:
    database_cache, pose_graph = scene
    pair_id, edge = next(iter(pose_graph.edges.items()))
    image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
    image1 = database_cache.image(image_id1)
    image2 = database_cache.image(image_id2)
    geometry = database_cache.correspondence_graph.extract_two_view_geometry(
        image_id1, image_id2, extract_inlier_matches=True
    )
    geometry.cam2_from_cam1 = edge.cam2_from_cam1
    args = (
        database_cache.camera(image1.camera_id),
        [point.xy for point in image1.points2D],
        database_cache.camera(image2.camera_id),
        [point.xy for point in image2.points2D],
    )
    cov = pycolmap.estimate_two_view_rotation_covariance(
        *args, geometry, pycolmap.TwoViewRotationCovarianceOptions()
    )
    assert cov.shape == (3, 3)

    geometry.cam2_from_cam1 = None
    assert (
        pycolmap.estimate_two_view_rotation_covariance(*args, geometry) is None
    )


def test_estimate_pose_graph_rotation_covariances(
    scene: tuple[pycolmap.DatabaseCache, pycolmap.PoseGraph],
) -> None:
    database_cache, pose_graph = scene
    for edge in pose_graph.edges.values():
        assert edge.cam2_from_cam1_rotation_cov is None
    pycolmap.estimate_pose_graph_rotation_covariances(
        database_cache, pose_graph, num_threads=1
    )
    for edge in pose_graph.edges.values():
        assert edge.cam2_from_cam1_rotation_cov is not None
