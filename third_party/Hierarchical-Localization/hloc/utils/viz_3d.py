"""
3D visualization based on plotly.
Works for a small number of points and cameras, might be slow otherwise.

1) Initialize a figure with `init_figure`
2) Add 3D points, camera frustums, or both as a pycolmap.Reconstruction

Written by Paul-Edouard Sarlin and Philipp Lindenberger.
"""

from typing import Optional

import numpy as np
import plotly.graph_objects as go
import pycolmap


def to_homogeneous(points):
    pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)


def init_figure(height: int = 800) -> go.Figure:
    """Initialize a 3D figure."""
    fig = go.Figure()
    axes = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=True,
        autorange=True,
    )
    fig.update_layout(
        template="plotly_dark",
        height=height,
        scene_camera=dict(
            eye=dict(x=0.0, y=-0.1, z=-2),
            up=dict(x=0, y=-1.0, z=0),
            projection=dict(type="orthographic"),
        ),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.1),
    )
    return fig


def plot_points(
    fig: go.Figure,
    pts: np.ndarray,
    color: str = "rgba(255, 0, 0, 1)",
    ps: int = 2,
    colorscale: Optional[str] = None,
    name: Optional[str] = None,
):
    """Plot a set of 3D points."""
    x, y, z = pts.T
    tr = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        name=name,
        legendgroup=name,
        marker=dict(size=ps, color=color, line_width=0.0, colorscale=colorscale),
    )
    fig.add_trace(tr)


def plot_camera(
    fig: go.Figure,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    color: str = "rgb(0, 0, 255)",
    name: Optional[str] = None,
    legendgroup: Optional[str] = None,
    fill: bool = False,
    size: float = 1.0,
    text: Optional[str] = None,
):
    """Plot a camera frustum from pose and intrinsic matrix."""
    W, H = K[0, 2] * 2, K[1, 2] * 2
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
    if size is not None:
        image_extent = max(size * W / 1024.0, size * H / 1024.0)
        world_extent = max(W, H) / (K[0, 0] + K[1, 1]) / 0.5
        scale = 0.5 * image_extent / world_extent
    else:
        scale = 1.0
    corners = to_homogeneous(corners) @ np.linalg.inv(K).T
    corners = (corners / 2 * scale) @ R.T + t
    legendgroup = legendgroup if legendgroup is not None else name

    x, y, z = np.concatenate(([t], corners)).T
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    if fill:
        pyramid = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            color=color,
            i=i,
            j=j,
            k=k,
            legendgroup=legendgroup,
            name=name,
            showlegend=False,
            hovertemplate=text.replace("\n", "<br>") if text else None,
        )
        fig.add_trace(pyramid)

    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([t], corners))
    tri_points = np.array([vertices[i] for i in triangles.reshape(-1)])
    x, y, z = tri_points.T

    pyramid = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        legendgroup=legendgroup,
        name=name,
        line=dict(color=color, width=1),
        showlegend=False,
        hovertemplate=text.replace("\n", "<br>") if text else None,
    )
    fig.add_trace(pyramid)


def plot_camera_colmap(
    fig: go.Figure, cam_from_world: pycolmap.Rigid3d, camera: pycolmap.Camera, **kwargs
):
    """Plot a camera frustum from PyCOLMAP objects"""
    world_t_camera = cam_from_world.inverse()
    plot_camera(
        fig,
        world_t_camera.rotation.matrix(),
        world_t_camera.translation,
        camera.calibration_matrix(),
        **kwargs
    )


def plot_image_colmap(
    fig: go.Figure,
    image: pycolmap.Image,
    camera: pycolmap.Camera,
    name: Optional[str] = None,
    **kwargs
):
    """Plot a camera frustum from a PyCOLMAP image."""
    plot_camera_colmap(
        fig,
        image.cam_from_world(),
        camera,
        name=name or str(image.image_id),
        text=str(image),
        **kwargs
    )


def plot_cameras(fig: go.Figure, reconstruction: pycolmap.Reconstruction, **kwargs):
    """Plot a camera as a cone with camera frustum."""
    for image_id, image in reconstruction.images.items():
        plot_image_colmap(fig, image, reconstruction.cameras[image.camera_id], **kwargs)


def plot_reconstruction(
    fig: go.Figure,
    rec: pycolmap.Reconstruction,
    max_reproj_error: float = 6.0,
    color: str = "rgb(0, 0, 255)",
    name: Optional[str] = None,
    min_track_length: int = 2,
    points: bool = True,
    cameras: bool = True,
    points_rgb: bool = True,
    cs: float = 1.0,
):
    # Filter outliers
    bbs = rec.compute_bounding_box(0.001, 0.999)
    # Filter points, use original reproj error here
    p3Ds = [
        p3D
        for _, p3D in rec.points3D.items()
        if (
            bbs.contains_point(p3D.xyz)
            and p3D.error <= max_reproj_error
            and p3D.track.length() >= min_track_length
        )
    ]
    xyzs = [p3D.xyz for p3D in p3Ds]
    if points_rgb:
        pcolor = [p3D.color for p3D in p3Ds]
    else:
        pcolor = color
    if points:
        plot_points(fig, np.array(xyzs), color=pcolor, ps=1, name=name)
    if cameras:
        plot_cameras(fig, rec, color=color, legendgroup=name, size=cs)
