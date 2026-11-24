# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

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


def test_robust_transform_reports() -> None:
    source = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
        dtype=float,
    )
    translation = np.array([1, 2, 3])
    options = pycolmap.RANSACOptions(
        max_error=1e-5, min_num_trials=10, random_seed=0, num_threads=1
    )
    for result, scale in (
        (
            pycolmap.estimate_rigid3d_robust(
                source, source + translation, options
            ),
            1.0,
        ),
        (
            pycolmap.estimate_sim3d_robust(
                source, 2 * source + translation, options
            ),
            2.0,
        ),
    ):
        assert result is not None
        assert result["num_inliers"] == len(source)
        np.testing.assert_array_equal(
            result["inlier_mask"], np.ones(len(source), dtype=bool)
        )
        np.testing.assert_allclose(
            result["tgt_from_src"] * source,
            scale * source + translation,
            atol=1e-10,
        )


@pytest.mark.parametrize(
    "name",
    [
        "estimate_rigid3d_robust",
        "estimate_sim3d_robust",
    ],
)
def test_public_api_callable(name: str) -> None:
    assert callable(getattr(pycolmap, name))
