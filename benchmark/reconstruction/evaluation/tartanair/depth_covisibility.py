"""Depth-based covisibility for TartanAir equirectangular images."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

SAMPLE_STRIDE = 4
MAX_DEPTH = 256.0
ABSOLUTE_TOLERANCE = 0.25
RELATIVE_TOLERANCE = 0.01


def equirectangular_rays(
    width: int, height: int, stride: int
) -> npt.NDArray[np.float64]:
    x = np.arange(0, width, stride, dtype=np.float64) + 0.5
    y = np.arange(0, height, stride, dtype=np.float64) + 0.5
    u: npt.NDArray[np.float64]
    v: npt.NDArray[np.float64]
    u, v = np.meshgrid(x / width, y / height)
    yaw = (2.0 * u - 1.0) * np.pi
    pitch = (1.0 - 2.0 * v) * np.pi / 2.0
    cos_pitch = np.cos(pitch)
    return np.column_stack(
        [
            np.sin(yaw).ravel() * cos_pitch.ravel(),
            -np.sin(pitch).ravel(),
            np.cos(yaw).ravel() * cos_pitch.ravel(),
        ]
    )


def _project_equirectangular(
    points_in_camera: npt.NDArray[np.float64], width: int, height: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    yaw = np.arctan2(points_in_camera[:, 0], points_in_camera[:, 2])
    pitch = -np.arctan2(
        points_in_camera[:, 1],
        np.linalg.norm(points_in_camera[:, [0, 2]], axis=1),
    )
    x = np.floor((1.0 + yaw / np.pi) * width / 2.0).astype(np.int64)
    y = np.floor((1.0 - pitch * 2.0 / np.pi) * height / 2.0).astype(np.int64)
    return np.mod(x, width), np.clip(y, 0, height - 1)


def compute_covisibility_counts(
    depths: npt.NDArray[np.uint16],
    world_from_cameras: npt.NDArray[np.float64],
    depth_scale: float,
    *,
    stride: int = SAMPLE_STRIDE,
    max_depth: float = MAX_DEPTH,
    absolute_tolerance: float = ABSOLUTE_TOLERANCE,
    relative_tolerance: float = RELATIVE_TOLERANCE,
) -> npt.NDArray[np.uint32]:
    if depths.ndim != 3:
        raise ValueError(f"Expected depth array (N,H,W), got {depths.shape}")
    if world_from_cameras.shape != (len(depths), 4, 4):
        raise ValueError(
            "Expected one 4x4 world_from_camera matrix per depth image"
        )
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")

    _, height, width = depths.shape
    rays = equirectangular_rays(width, height, stride)
    sampled_depths = (
        depths[:, ::stride, ::stride]
        .reshape(len(depths), -1)
        .astype(np.float64)
        / depth_scale
    )
    counts: npt.NDArray[np.uint32] = np.zeros(
        (len(depths), len(depths)), dtype=np.uint32
    )

    for source_idx in range(len(depths)):
        source_depth = sampled_depths[source_idx]
        valid = (source_depth > 0) & (source_depth <= max_depth)
        if not np.any(valid):
            continue
        source_points = rays[valid] * source_depth[valid, np.newaxis]
        source_pose = world_from_cameras[source_idx]
        points_in_world = (
            source_pose[:3, :3] @ source_points.T
            + source_pose[:3, 3, np.newaxis]
        ).T
        for target_idx in range(len(depths)):
            if source_idx == target_idx:
                counts[source_idx, target_idx] = np.count_nonzero(valid)
                continue
            target_pose = world_from_cameras[target_idx]
            points_in_target = (
                target_pose[:3, :3].T @ (points_in_world - target_pose[:3, 3]).T
            ).T
            projected_depth = np.linalg.norm(points_in_target, axis=1)
            x, y = _project_equirectangular(points_in_target, width, height)
            target_depth = depths[target_idx, y, x].astype(np.float64)
            target_depth /= depth_scale
            tolerance = np.maximum(
                absolute_tolerance, relative_tolerance * projected_depth
            )
            visible = (
                (target_depth > 0)
                & (target_depth <= max_depth)
                & (np.abs(target_depth - projected_depth) <= tolerance)
            )
            counts[source_idx, target_idx] = np.count_nonzero(visible)
    return counts


def covisibility_is_current(path: Path, image_names: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as covisibility:
            return (
                covisibility["image_names"].tolist() == image_names
                and covisibility["sample_stride"].item() == SAMPLE_STRIDE
                and covisibility["max_depth"].item() == MAX_DEPTH
                and covisibility["absolute_tolerance"].item()
                == ABSOLUTE_TOLERANCE
                and covisibility["relative_tolerance"].item()
                == RELATIVE_TOLERANCE
            )
    except (KeyError, OSError, ValueError):
        return False


def write_covisibility(
    output_path: Path,
    depth_paths: list[Path],
    image_names: list[str],
    world_from_cameras: npt.NDArray[np.float64],
    depth_scale: float,
) -> None:
    from PIL import Image

    depths = np.stack(
        [np.asarray(Image.open(path), dtype=np.uint16) for path in depth_paths]
    )
    counts = compute_covisibility_counts(
        depths, world_from_cameras, depth_scale
    )
    np.savez_compressed(
        output_path,
        image_names=np.asarray(image_names),
        directed_overlap_counts=counts,
        sample_stride=SAMPLE_STRIDE,
        max_depth=MAX_DEPTH,
        absolute_tolerance=ABSOLUTE_TOLERANCE,
        relative_tolerance=RELATIVE_TOLERANCE,
    )
