# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import pycolmap


def _synthetic_two_view_geometry(
    num_points: int = 100, noise_px: float = 0.5, seed: int = 0
) -> tuple[pycolmap.Camera, list, list, pycolmap.TwoViewGeometry]:
    rng = np.random.default_rng(seed)
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 500.0, 640, 480
    )
    cam2_from_cam1 = pycolmap.Rigid3d(
        pycolmap.Rotation3d(np.array([0.02, 0.15, 0.01])),
        np.array([1.0, 0.0, 0.0]),
    )
    xy = rng.uniform([40, 40], [600, 440], size=(num_points, 2))
    depth = rng.uniform(4.0, 8.0, size=num_points)
    points3D = camera.cam_from_img(xy)
    points3D = np.c_[points3D, np.ones(num_points)] * depth[:, None]
    points1 = camera.img_from_cam(points3D)
    points2 = camera.img_from_cam(
        (cam2_from_cam1.matrix() @ np.c_[points3D, np.ones(num_points)].T).T
    )
    points1 = list(points1 + rng.normal(scale=noise_px, size=points1.shape))
    points2 = list(points2 + rng.normal(scale=noise_px, size=points2.shape))
    geometry = pycolmap.TwoViewGeometry()
    geometry.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
    geometry.inlier_matches = np.stack(
        [np.arange(num_points, dtype=np.uint32)] * 2, axis=1
    )
    geometry.cam2_from_cam1 = cam2_from_cam1
    return camera, points1, points2, geometry


def test_estimate_two_view_pose_covariance() -> None:
    camera, points1, points2, geometry = _synthetic_two_view_geometry()
    options = pycolmap.TwoViewPoseCovarianceOptions()
    cov = pycolmap.estimate_two_view_pose_covariance(
        camera, points1, camera, points2, geometry, options
    )
    assert cov is not None
    assert cov.shape == (3, 3)
    np.testing.assert_allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > 0)

    # Without a relative pose, no covariance can be estimated.
    geometry.cam2_from_cam1 = None
    assert (
        pycolmap.estimate_two_view_pose_covariance(
            camera, points1, camera, points2, geometry
        )
        is None
    )


def test_estimate_pose_graph_covariances() -> None:
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
    for edge in pose_graph.edges.values():
        assert edge.cam2_from_cam1_rotation_cov is None
    pycolmap.estimate_pose_graph_covariances(database_cache, pose_graph)
    for edge in pose_graph.edges.values():
        assert edge.cam2_from_cam1_rotation_cov is not None
        assert np.all(np.linalg.eigvalsh(edge.cam2_from_cam1_rotation_cov) > 0)
