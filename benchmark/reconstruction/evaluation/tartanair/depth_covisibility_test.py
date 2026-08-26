import numpy as np

from .depth_covisibility import compute_covisibility_counts


def test_compute_covisibility_counts_identical_views():
    depths = np.full((2, 4, 8), 80, dtype=np.uint16)
    world_from_cameras = np.repeat(np.eye(4)[np.newaxis], 2, axis=0)
    counts = compute_covisibility_counts(
        depths, world_from_cameras, depth_scale=8.0, stride=1
    )
    np.testing.assert_array_equal(counts, np.full((2, 2), 32))


def test_compute_covisibility_counts_rejects_disjoint_views():
    depths = np.full((2, 4, 8), 80, dtype=np.uint16)
    world_from_cameras = np.repeat(np.eye(4)[np.newaxis], 2, axis=0)
    world_from_cameras[1, 0, 3] = 100.0
    counts = compute_covisibility_counts(
        depths, world_from_cameras, depth_scale=8.0, stride=1
    )
    assert counts[0, 1] == 0
    assert counts[1, 0] == 0
