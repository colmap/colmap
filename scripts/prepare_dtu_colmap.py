#!/usr/bin/env python3
"""Prepare an official DTU SampleSet scan as a COLMAP dense workspace.

The DTU SampleSet supplies rectified images and 3x4 projection matrices. This
script decomposes those matrices into COLMAP PINHOLE cameras, resizes the
images, estimates a global depth interval from the structured-light reference,
and chooses source images using measured frustum overlap and triangulation
angle. No feature extraction or pose estimation is involved.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Camera:
    image_id: int
    name: str
    width: int
    height: int
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    center: np.ndarray
    projection_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=Path("data/metal-benchmark/dtu/SampleSet"),
        help="Extracted official DTU SampleSet root",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("data/metal-benchmark/dtu/scan6-metal-800"),
        help="Dense COLMAP workspace to create",
    )
    parser.add_argument("--scan", type=int, default=6)
    parser.add_argument(
        "--light",
        type=int,
        default=3,
        help="DTU light index; 3 is the diffuse all-lights condition",
    )
    parser.add_argument("--max-image-size", type=int, default=800)
    parser.add_argument("--num-sources", type=int, default=10)
    parser.add_argument(
        "--point-stride",
        type=int,
        default=100,
        help="Use every Nth reference point for bounds and overlap scoring",
    )
    parser.add_argument(
        "--depth-tail",
        type=float,
        default=0.001,
        help="Fraction trimmed from each end of sampled visible depths",
    )
    return parser.parse_args()


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Return a normalized COLMAP quaternion in qw, qx, qy, qz order."""
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s,
             (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            quat = np.array(
                [(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                 (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
            )
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            quat = np.array(
                [(R[0, 2] - R[2, 0]) / s,
                 (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                 (R[1, 2] + R[2, 1]) / s]
            )
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            quat = np.array(
                [(R[1, 0] - R[0, 1]) / s,
                 (R[0, 2] + R[2, 0]) / s,
                 (R[1, 2] + R[2, 1]) / s, 0.25 * s]
            )
    quat /= np.linalg.norm(quat)
    if quat[0] < 0:
        quat *= -1
    return quat


def read_reference_points(path: Path, stride: int) -> np.ndarray:
    """Memory-map XYZ from the binary-little-endian DTU reference PLY."""
    properties: list[tuple[str, str]] = []
    type_map = {
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "uchar": "u1",
        "uint8": "u1",
        "char": "i1",
        "int8": "i1",
        "ushort": "<u2",
        "uint16": "<u2",
        "short": "<i2",
        "int16": "<i2",
        "uint": "<u4",
        "uint32": "<u4",
        "int": "<i4",
        "int32": "<i4",
    }
    vertex_count = None
    with path.open("rb") as ply_file:
        if ply_file.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        in_vertex = False
        while True:
            raw_line = ply_file.readline()
            if not raw_line:
                raise ValueError(f"Missing end_header in {path}")
            line = raw_line.decode("ascii").strip()
            fields = line.split()
            if fields[:2] == ["format", "binary_little_endian"]:
                pass
            elif fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                in_vertex = True
            elif fields and fields[0] == "element":
                in_vertex = False
            elif in_vertex and fields[:1] == ["property"]:
                if fields[1] == "list":
                    raise ValueError("List property in PLY vertex is unsupported")
                properties.append((fields[2], type_map[fields[1]]))
            elif line == "end_header":
                data_offset = ply_file.tell()
                break
    if vertex_count is None:
        raise ValueError(f"No vertex element in {path}")
    dtype = np.dtype(properties)
    vertices = np.memmap(
        path, dtype=dtype, mode="r", offset=data_offset, shape=(vertex_count,)
    )[::stride]
    return np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float64
    )


def decompose_camera(
    image_id: int,
    image_path: Path,
    projection_path: Path,
    max_image_size: int,
) -> tuple[Camera, Image.Image]:
    projection = np.loadtxt(projection_path, dtype=np.float64)
    if projection.shape != (3, 4):
        raise ValueError(f"Expected 3x4 projection matrix: {projection_path}")

    K, R, center_h, *_ = cv2.decomposeProjectionMatrix(projection)
    K /= K[2, 2]
    center = (center_h[:3] / center_h[3]).reshape(3)
    t = -R @ center

    recomposed = K @ np.column_stack((R, t))
    scale = float(np.vdot(recomposed, projection) / np.vdot(recomposed, recomposed))
    error = float(
        np.linalg.norm(scale * recomposed - projection) / np.linalg.norm(projection)
    )
    if error > 1e-8 or np.linalg.det(R) < 0.999999:
        raise ValueError(
            f"Unstable projection decomposition for {projection_path}: {error:.3g}"
        )

    with Image.open(image_path) as source:
        source.load()
        width, height = source.size
        scale_factor = min(1.0, max_image_size / max(width, height))
        output_size = (
            max(1, round(width * scale_factor)),
            max(1, round(height * scale_factor)),
        )
        if output_size != source.size:
            output_image = source.resize(output_size, Image.Resampling.LANCZOS)
        else:
            output_image = source.copy()

    scaled_K = K.copy()
    scaled_K[0, :] *= output_size[0] / width
    scaled_K[1, :] *= output_size[1] / height
    camera = Camera(
        image_id=image_id,
        name=image_path.name,
        width=output_size[0],
        height=output_size[1],
        K=scaled_K,
        R=R,
        t=t,
        center=center,
        projection_error=error,
    )
    return camera, output_image


def visible_points(camera: Camera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_in_camera = points @ camera.R.T + camera.t
    depth = points_in_camera[:, 2]
    pixels = points_in_camera[:, :2] / depth[:, None]
    pixels[:, 0] = camera.K[0, 0] * pixels[:, 0] + camera.K[0, 2]
    pixels[:, 1] = camera.K[1, 1] * pixels[:, 1] + camera.K[1, 2]
    visible = (
        (depth > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < camera.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < camera.height)
    )
    return visible, depth


def choose_source_images(
    cameras: list[Camera],
    points: np.ndarray,
    visibility: np.ndarray,
    num_sources: int,
) -> tuple[list[list[int]], list[list[dict[str, float]]]]:
    source_ids: list[list[int]] = []
    source_reports: list[list[dict[str, float]]] = []
    for ref_idx, ref_camera in enumerate(cameras):
        candidates: list[tuple[float, int, dict[str, float]]] = []
        ref_count = int(np.count_nonzero(visibility[ref_idx]))
        for src_idx, src_camera in enumerate(cameras):
            if src_idx == ref_idx:
                continue
            common = visibility[ref_idx] & visibility[src_idx]
            common_count = int(np.count_nonzero(common))
            if common_count < 100:
                continue
            common_points = points[common]
            if len(common_points) > 5000:
                common_points = common_points[:: math.ceil(len(common_points) / 5000)]
            ref_rays = common_points - ref_camera.center
            src_rays = common_points - src_camera.center
            cosines = np.einsum("ij,ij->i", ref_rays, src_rays)
            cosines /= np.linalg.norm(ref_rays, axis=1) * np.linalg.norm(
                src_rays, axis=1
            )
            angle = float(
                np.degrees(np.median(np.arccos(np.clip(cosines, -1.0, 1.0))))
            )
            overlap = common_count / max(1, ref_count)
            angle_reward = min(angle / 8.0, 1.0)
            wide_angle_penalty = math.exp(-max(0.0, angle - 35.0) / 15.0)
            score = overlap * angle_reward * wide_angle_penalty
            report = {
                "source_id": src_camera.image_id,
                "overlap": overlap,
                "median_angle_deg": angle,
                "score": score,
            }
            candidates.append((score, src_idx, report))
        candidates.sort(reverse=True, key=lambda item: item[0])
        selected = candidates[:num_sources]
        if len(selected) != num_sources:
            raise ValueError(
                f"Only found {len(selected)} sources for {ref_camera.name}"
            )
        source_ids.append([src_idx for _, src_idx, _ in selected])
        source_reports.append([report for _, _, report in selected])
    return source_ids, source_reports


def write_sparse_model(
    workspace: Path, cameras: list[Camera], connectivity_point: np.ndarray
) -> None:
    """Write cameras plus one track used only to connect views during fusion.

    COLMAP's stereo fusion derives its view-adjacency graph exclusively from
    sparse tracks, even when patch-match.cfg explicitly supplies source views.
    A direct calibrated DTU workspace otherwise has no sparse reconstruction
    and fusion cannot traverse from one depth map to another. One point near
    the scan centre, observed by every camera, supplies that connectivity; it
    does not seed or otherwise participate in PatchMatch depth estimation.
    """
    sparse = workspace / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    cameras_lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    images_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(cameras)}, mean observations per image: 1",
    ]
    for camera in cameras:
        fx, fy = camera.K[0, 0], camera.K[1, 1]
        cx, cy = camera.K[0, 2], camera.K[1, 2]
        cameras_lines.append(
            f"{camera.image_id} PINHOLE {camera.width} {camera.height} "
            f"{fx:.15g} {fy:.15g} {cx:.15g} {cy:.15g}"
        )
        quat = rotation_matrix_to_quaternion(camera.R)
        pose = " ".join(f"{value:.17g}" for value in (*quat, *camera.t))
        point_in_camera = camera.R @ connectivity_point + camera.t
        pixel = camera.K @ point_in_camera
        pixel /= pixel[2]
        if not (0 <= pixel[0] < camera.width and 0 <= pixel[1] < camera.height):
            raise ValueError(
                f"Connectivity point is outside camera {camera.image_id}"
            )
        images_lines.extend(
            [
                f"{camera.image_id} {pose} {camera.image_id} {camera.name}",
                f"{pixel[0]:.12g} {pixel[1]:.12g} 1",
            ]
        )
    (sparse / "cameras.txt").write_text("\n".join(cameras_lines) + "\n")
    (sparse / "images.txt").write_text("\n".join(images_lines) + "\n")
    track = " ".join(f"{camera.image_id} 0" for camera in cameras)
    xyz = " ".join(f"{value:.17g}" for value in connectivity_point)
    (sparse / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
        f"# Number of points: 1, mean track length: {len(cameras)}\n"
        f"1 {xyz} 255 255 255 0 {track}\n"
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.depth_tail < 0.5:
        raise ValueError("--depth-tail must be in [0, 0.5)")
    if args.point_stride < 1 or args.num_sources < 1:
        raise ValueError("--point-stride and --num-sources must be positive")

    mvs_root = args.sample_root / "MVS Data"
    image_root = mvs_root / "Rectified" / f"scan{args.scan}"
    calibration_root = mvs_root / "Calibration" / "cal18"
    reference_path = mvs_root / "Points" / "stl" / f"stl{args.scan:03d}_total.ply"
    image_paths = sorted(image_root.glob(f"rect_*_{args.light}_r5000.png"))
    if not image_paths:
        raise FileNotFoundError(f"No DTU images found in {image_root}")

    args.workspace.mkdir(parents=True, exist_ok=True)
    output_images = args.workspace / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    stereo = args.workspace / "stereo"
    for folder in ("depth_maps", "normal_maps", "consistency_graphs"):
        (stereo / folder).mkdir(parents=True, exist_ok=True)

    cameras: list[Camera] = []
    for image_id, image_path in enumerate(image_paths, start=1):
        position = int(image_path.name.split("_")[1])
        camera, output_image = decompose_camera(
            image_id,
            image_path,
            calibration_root / f"pos_{position:03d}.txt",
            args.max_image_size,
        )
        output_image.save(output_images / camera.name, optimize=True)
        cameras.append(camera)

    points = read_reference_points(reference_path, args.point_stride)
    visible_masks: list[np.ndarray] = []
    visible_depths: list[np.ndarray] = []
    for camera in cameras:
        mask, depth = visible_points(camera, points)
        visible_masks.append(mask)
        visible_depths.append(depth[mask])
    visibility = np.stack(visible_masks)
    depths = np.concatenate(visible_depths)
    depth_min, depth_max = np.quantile(
        depths, [args.depth_tail, 1.0 - args.depth_tail]
    )
    depth_padding = 0.1 * (depth_max - depth_min)
    depth_min = max(0.01, float(depth_min - depth_padding))
    depth_max = float(depth_max + depth_padding)

    source_ids, source_reports = choose_source_images(
        cameras, points, visibility, args.num_sources
    )
    connectivity_point = np.median(points, axis=0)
    write_sparse_model(args.workspace, cameras, connectivity_point)

    patch_match_lines: list[str] = []
    for ref_idx, sources in enumerate(source_ids):
        patch_match_lines.extend(
            [
                cameras[ref_idx].name,
                ", ".join(cameras[src_idx].name for src_idx in sources),
            ]
        )
    (stereo / "patch-match.cfg").write_text("\n".join(patch_match_lines) + "\n")
    (stereo / "fusion.cfg").write_text(
        "\n".join(camera.name for camera in cameras) + "\n"
    )

    report = {
        "scan": args.scan,
        "light": args.light,
        "num_images": len(cameras),
        "image_size": [cameras[0].width, cameras[0].height],
        "sampled_reference_points": len(points),
        "point_stride": args.point_stride,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "max_projection_decomposition_error": max(
            camera.projection_error for camera in cameras
        ),
        "fusion_connectivity_point": connectivity_point.tolist(),
        "source_pairs": {
            cameras[idx].name: reports
            for idx, reports in enumerate(source_reports)
        },
    }
    report_path = args.workspace / "dtu-preparation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prepared {len(cameras)} images in {args.workspace}")
    print(f"Image size: {cameras[0].width}x{cameras[0].height}")
    print(f"Depth range: {depth_min:.6f} .. {depth_max:.6f} mm")
    print(
        "Maximum projection decomposition error: "
        f"{report['max_projection_decomposition_error']:.3g}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
