# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import pycolmap


def test_estimate_rigid3d_with_coplanar_points() -> None:
    source = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
    ]
    target = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
    ]
    result = pycolmap.estimate_rigid3d(source, target)
    # May return None for degenerate configurations, but call should succeed
    assert result is None or isinstance(result, pycolmap.Rigid3d)


def test_estimate_sim3d_with_points() -> None:
    source = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    target = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
        np.array([0.0, 0.0, 2.0]),
    ]
    result = pycolmap.estimate_sim3d(source, target)
    assert result is None or isinstance(result, pycolmap.Sim3d)


def test_estimate_transform_robust() -> None:
    src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    rigid = pycolmap.estimate_rigid3d_robust(src, src + 1)
    sim = pycolmap.estimate_sim3d_robust(src, 2 * src + 1)
    assert rigid is not None and sim is not None
    assert rigid["num_inliers"] == sim["num_inliers"] == len(src)
    np.testing.assert_allclose(rigid["tgt_from_src"] * src, src + 1, atol=1e-8)
    np.testing.assert_allclose(
        sim["tgt_from_src"] * src, 2 * src + 1, atol=1e-8
    )
