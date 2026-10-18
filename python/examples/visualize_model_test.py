# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest


# Keep this geometry-only test independent of the optional Open3D runtime.
class _TriangleMesh:
    def __init__(self, vertices: np.ndarray) -> None:
        self.vertices = vertices

    @staticmethod
    def create_coordinate_frame(size: float) -> "_TriangleMesh":
        return _TriangleMesh(np.zeros((1, 3)))

    @staticmethod
    def create_box(
        width: float, height: float, depth: float
    ) -> "_TriangleMesh":
        vertices = np.array(
            [
                [x, y, z]
                for x in (0, width)
                for y in (0, height)
                for z in (0, depth)
            ]
        )
        return _TriangleMesh(vertices)

    def paint_uniform_color(self, color: list[float]) -> None:
        pass

    def translate(self, translation: np.ndarray) -> None:
        self.vertices += translation

    def transform(self, transform: np.ndarray) -> None:
        homogeneous = np.column_stack(
            (self.vertices, np.ones(len(self.vertices)))
        )
        self.vertices = (transform @ homogeneous.T).T[:, :3]

    def get_min_bound(self) -> np.ndarray:
        return self.vertices.min(axis=0)

    def get_max_bound(self) -> np.ndarray:
        return self.vertices.max(axis=0)


class _LineSet:
    def __init__(self, points: np.ndarray, lines: np.ndarray) -> None:
        self.points = points
        self.lines = lines
        self.colors: np.ndarray | None = None


def _load_draw_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Any]:
    open3d = cast(Any, ModuleType("open3d"))
    open3d.geometry = ModuleType("open3d.geometry")
    open3d.geometry.Geometry = object
    open3d.geometry.TriangleMesh = _TriangleMesh
    open3d.geometry.LineSet = _LineSet
    open3d.utility = ModuleType("open3d.utility")
    open3d.utility.Vector3dVector = np.asarray
    open3d.utility.Vector2iVector = np.asarray
    open3d.visualization = ModuleType("open3d.visualization")
    open3d.visualization.Visualizer = object

    pycolmap = cast(Any, ModuleType("pycolmap"))
    pycolmap.Reconstruction = object

    monkeypatch.setitem(sys.modules, "open3d", open3d)
    monkeypatch.setitem(sys.modules, "pycolmap", pycolmap)

    script_path = Path(__file__).with_name("visualize_model.py")
    spec = importlib.util.spec_from_file_location(
        "visualize_model", script_path
    )
    assert spec is not None and spec.loader is not None
    module = cast(Any, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module.draw_camera


@pytest.mark.parametrize("cx, cy", [(100.0, 240.0), (320.0, -100.0)])
def test_draw_camera_plane_matches_frustum_bounds(
    monkeypatch: pytest.MonkeyPatch, cx: float, cy: float
) -> None:
    draw_camera = _load_draw_camera(monkeypatch)
    intrinsics = np.array([[500.0, 0.0, cx], [0.0, 500.0, cy], [0.0, 0.0, 1.0]])

    _, plane, frustum = draw_camera(
        intrinsics, np.eye(3), np.zeros(3), 640, 480
    )

    frustum_corners = np.asarray(frustum.points)[1:]
    plane_bounds = np.array([plane.get_min_bound(), plane.get_max_bound()])[
        :, :2
    ]
    frustum_bounds = np.array(
        [
            frustum_corners[:, :2].min(axis=0),
            frustum_corners[:, :2].max(axis=0),
        ]
    )
    np.testing.assert_allclose(plane_bounds, frustum_bounds)
