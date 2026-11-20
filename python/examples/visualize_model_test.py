# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

pytest.importorskip("open3d")

import visualize_model  # noqa: E402


@pytest.mark.parametrize(
    "principal_point",
    [
        (320.0, 240.0),
        (100.0, 240.0),
        (0.0, 0.0),
        (-50.0, -40.0),
        (700.0, 540.0),
    ],
)
@pytest.mark.parametrize("scale", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("transformed", [False, True])
def test_draw_camera_image_plane_matches_frustum(
    principal_point: tuple[float, float], scale: float, transformed: bool
) -> None:
    K = np.array(
        [
            [500.0, 0.0, principal_point[0]],
            [0.0, 400.0, principal_point[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    R = (
        np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        if transformed
        else np.eye(3)
    )
    t = np.array([1.0, -2.0, 3.0]) if transformed else np.zeros(3)

    _, plane, frustum = visualize_model.draw_camera(K, R, t, 640, 480, scale)
    plane_vertices = (np.asarray(plane.vertices) - t) @ R
    frustum_corners = (np.asarray(frustum.points)[1:] - t) @ R
    expected_corners = (
        scale
        * np.array(
            [
                [0.0, 0.0, 1.0],
                [640.0, 0.0, 1.0],
                [0.0, 480.0, 1.0],
                [640.0, 480.0, 1.0],
            ]
        )
        @ np.linalg.inv(K).T
    )

    np.testing.assert_allclose(frustum_corners, expected_corners, atol=1e-12)
    np.testing.assert_allclose(
        plane_vertices[:, :2].min(axis=0),
        expected_corners[:, :2].min(axis=0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        plane_vertices[:, :2].max(axis=0),
        expected_corners[:, :2].max(axis=0),
        atol=1e-12,
    )
    np.testing.assert_allclose(plane_vertices[:, 2].min(), scale, atol=1e-12)
    np.testing.assert_allclose(
        plane_vertices[:, 2].max(), scale + 1e-6, atol=1e-12
    )
