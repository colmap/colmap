#!/usr/bin/env python3
"""Evaluate COLMAP dense maps against sparse reconstruction checkpoints.

Sparse observations are not ground truth, but they provide a useful smoke test
for gross depth, indexing, pose, and filtering errors. The script also
summarizes consistency graphs and a fused PLY when they are available.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_dense_map(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        header = bytearray()
        while header.count(b"&") < 3:
            value = file.read(1)
            if not value:
                raise ValueError(f"Incomplete dense-map header: {path}")
            header.extend(value)
        width, height, depth = map(int, header[:-1].split(b"&"))
        values = np.fromfile(file, dtype="<f4")
    expected_size = width * height * depth
    if values.size != expected_size:
        raise ValueError(
            f"Dense-map size mismatch for {path}: "
            f"expected {expected_size}, got {values.size}"
        )
    return values.reshape((depth, height, width))


def read_consistency_graph(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        header = bytearray()
        while header.count(b"&") < 3:
            value = file.read(1)
            if not value:
                raise ValueError(f"Incomplete consistency-graph header: {path}")
            header.extend(value)
        width, height, depth = map(int, header[:-1].split(b"&"))
        values = np.fromfile(file, dtype="<i4")

    offset = 0
    record_count = 0
    link_count = 0
    while offset < values.size:
        if offset + 3 > values.size:
            raise ValueError(f"Truncated consistency graph: {path}")
        col, row, num_images = map(int, values[offset : offset + 3])
        if not (0 <= col < width and 0 <= row < height) or num_images < 0:
            raise ValueError(
                f"Invalid consistency-graph record at integer {offset}: {path}"
            )
        offset += 3 + num_images
        if offset > values.size:
            raise ValueError(f"Truncated consistency-graph image list: {path}")
        record_count += 1
        link_count += num_images

    return {
        "width": width,
        "height": height,
        "depth": depth,
        "pixels_with_consistency": record_count,
        "consistent_pixel_fraction": record_count / (width * height),
        "source_links": link_count,
        "mean_sources_per_consistent_pixel": (
            link_count / record_count if record_count else 0.0
        ),
    }


def read_cameras(path: Path) -> dict[int, tuple[int, int]]:
    cameras = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            cameras[int(tokens[0])] = (int(tokens[2]), int(tokens[3]))
    return cameras


def quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qw * qz,
                2 * qx * qz + 2 * qw * qy,
            ],
            [
                2 * qx * qy + 2 * qw * qz,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qw * qx,
            ],
            [
                2 * qx * qz - 2 * qw * qy,
                2 * qy * qz + 2 * qw * qx,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def read_images(path: Path) -> dict[str, dict[str, Any]]:
    images: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        while True:
            header = file.readline()
            if not header:
                break
            header = header.strip()
            if not header or header.startswith("#"):
                continue

            tokens = header.split(maxsplit=9)
            if len(tokens) != 10:
                raise ValueError(f"Invalid image header in {path}: {header}")
            observations_line = file.readline()
            if not observations_line:
                raise ValueError(
                    f"Missing observations after image header in {path}: {header}"
                )
            observation_tokens = observations_line.split()
            if len(observation_tokens) % 3 != 0:
                raise ValueError(f"Invalid POINTS2D line for {tokens[9]} in {path}")

            observations = []
            for index in range(0, len(observation_tokens), 3):
                point_id = int(observation_tokens[index + 2])
                if point_id >= 0:
                    observations.append(
                        (
                            float(observation_tokens[index]),
                            float(observation_tokens[index + 1]),
                            point_id,
                        )
                    )
            images[tokens[9]] = {
                "qvec": np.asarray(tokens[1:5], dtype=np.float64),
                "tvec": np.asarray(tokens[5:8], dtype=np.float64),
                "camera_id": int(tokens[8]),
                "observations": observations,
            }
    return images


def read_points3d(path: Path) -> dict[int, np.ndarray]:
    points = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split(maxsplit=4)
            points[int(tokens[0])] = np.asarray(tokens[1:4], dtype=np.float64)
    return points


def percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def evaluate_sparse_observations(
    depth_map: np.ndarray,
    image: dict[str, Any],
    camera_size: tuple[int, int],
    points3d: dict[int, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    height, width = depth_map.shape
    camera_width, camera_height = camera_size
    scale_x = width / camera_width
    scale_y = height / camera_height
    rotation = quaternion_to_rotation(image["qvec"])

    eligible = 0
    abs_errors = []
    rel_errors = []
    for x, y, point_id in image["observations"]:
        point = points3d.get(point_id)
        if point is None:
            continue
        expected_depth = float((rotation @ point + image["tvec"])[2])
        col = int(round(x * scale_x))
        row = int(round(y * scale_y))
        if expected_depth <= 0 or not (0 <= col < width and 0 <= row < height):
            continue
        eligible += 1
        estimated_depth = float(depth_map[row, col])
        if not math.isfinite(estimated_depth) or estimated_depth <= 0:
            continue
        abs_error = abs(estimated_depth - expected_depth)
        abs_errors.append(abs_error)
        rel_errors.append(abs_error / expected_depth)

    abs_array = np.asarray(abs_errors, dtype=np.float64)
    rel_array = np.asarray(rel_errors, dtype=np.float64)
    metrics = {
        "eligible_sparse_observations": eligible,
        "valid_dense_samples": int(abs_array.size),
        "valid_dense_sample_fraction": (
            float(abs_array.size / eligible) if eligible else 0.0
        ),
        "absolute_error": percentiles(abs_array),
        "relative_error": percentiles(rel_array),
        "relative_error_within_1_percent": (
            float(np.mean(rel_array <= 0.01)) if rel_array.size else 0.0
        ),
        "relative_error_within_5_percent": (
            float(np.mean(rel_array <= 0.05)) if rel_array.size else 0.0
        ),
        "relative_error_within_10_percent": (
            float(np.mean(rel_array <= 0.10)) if rel_array.size else 0.0
        ),
    }
    return metrics, abs_array, rel_array


def read_ply_vertex_count(path: Path) -> int:
    with path.open("rb") as file:
        for raw_line in file:
            line = raw_line.decode("ascii").strip()
            if line.startswith("element vertex "):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise ValueError(f"PLY has no vertex count: {path}")


def discover_image_names(workspace: Path, input_type: str) -> list[str]:
    suffix = f".{input_type}.bin"
    paths = (workspace / "stereo" / "depth_maps").glob(f"*{suffix}")
    return sorted(path.name[: -len(suffix)] for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model-text", type=Path)
    parser.add_argument(
        "--input-type",
        choices=("photometric", "geometric"),
        default="geometric",
    )
    parser.add_argument("--images", nargs="+")
    parser.add_argument("--ply", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    image_names = args.images or discover_image_names(workspace, args.input_type)
    if not image_names:
        parser.error(f"No {args.input_type} depth maps found in {workspace}")

    model_text = (args.model_text or workspace / "sparse-text").resolve()
    model_files = ("cameras.txt", "images.txt", "points3D.txt")
    model_available = all((model_text / name).is_file() for name in model_files)
    cameras = read_cameras(model_text / "cameras.txt") if model_available else {}
    images = read_images(model_text / "images.txt") if model_available else {}
    points3d = read_points3d(model_text / "points3D.txt") if model_available else {}

    result: dict[str, Any] = {
        "workspace": str(workspace),
        "input_type": args.input_type,
        "num_images": len(image_names),
        "sparse_model_text": str(model_text) if model_available else None,
        "images": {},
    }
    all_abs_errors = []
    all_rel_errors = []
    eligible_total = 0
    valid_sample_total = 0
    valid_pixel_total = 0
    pixel_total = 0
    graph_count = 0
    graph_pixel_total = 0
    graph_consistent_pixel_total = 0
    graph_link_total = 0

    for image_name in image_names:
        map_path = (
            workspace
            / "stereo"
            / "depth_maps"
            / f"{image_name}.{args.input_type}.bin"
        )
        depth_map = read_dense_map(map_path)[0]
        valid = np.isfinite(depth_map) & (depth_map > 0)
        valid_depths = depth_map[valid]
        image_result: dict[str, Any] = {
            "width": int(depth_map.shape[1]),
            "height": int(depth_map.shape[0]),
            "valid_pixels": int(valid.sum()),
            "valid_pixel_fraction": float(valid.mean()),
            "valid_depth": percentiles(valid_depths),
        }
        valid_pixel_total += int(valid.sum())
        pixel_total += int(valid.size)

        graph_path = (
            workspace
            / "stereo"
            / "consistency_graphs"
            / f"{image_name}.{args.input_type}.bin"
        )
        if graph_path.is_file():
            graph = read_consistency_graph(graph_path)
            image_result["consistency_graph"] = graph
            graph_count += 1
            graph_pixel_total += graph["width"] * graph["height"]
            graph_consistent_pixel_total += graph["pixels_with_consistency"]
            graph_link_total += graph["source_links"]

        if model_available and image_name in images:
            image = images[image_name]
            camera_size = cameras[image["camera_id"]]
            metrics, abs_errors, rel_errors = evaluate_sparse_observations(
                depth_map, image, camera_size, points3d
            )
            image_result["sparse_checkpoint"] = metrics
            eligible_total += metrics["eligible_sparse_observations"]
            valid_sample_total += metrics["valid_dense_samples"]
            all_abs_errors.append(abs_errors)
            all_rel_errors.append(rel_errors)

        result["images"][image_name] = image_result

    result["aggregate"] = {
        "valid_pixels": valid_pixel_total,
        "pixels": pixel_total,
        "valid_pixel_fraction": valid_pixel_total / pixel_total,
    }
    if graph_count:
        result["aggregate"]["consistency_graph"] = {
            "num_graphs": graph_count,
            "pixels_with_consistency": graph_consistent_pixel_total,
            "consistent_pixel_fraction": (
                graph_consistent_pixel_total / graph_pixel_total
            ),
            "source_links": graph_link_total,
            "mean_sources_per_consistent_pixel": (
                graph_link_total / graph_consistent_pixel_total
            ),
        }
    if all_abs_errors:
        abs_errors = np.concatenate(all_abs_errors)
        rel_errors = np.concatenate(all_rel_errors)
        result["aggregate"]["sparse_checkpoint"] = {
            "eligible_sparse_observations": eligible_total,
            "valid_dense_samples": valid_sample_total,
            "valid_dense_sample_fraction": valid_sample_total / eligible_total,
            "absolute_error": percentiles(abs_errors),
            "relative_error": percentiles(rel_errors),
            "relative_error_within_1_percent": float(
                np.mean(rel_errors <= 0.01)
            ),
            "relative_error_within_5_percent": float(
                np.mean(rel_errors <= 0.05)
            ),
            "relative_error_within_10_percent": float(
                np.mean(rel_errors <= 0.10)
            ),
        }

    if args.ply:
        ply_path = args.ply.resolve()
        result["fused_ply"] = {
            "path": str(ply_path),
            "vertices": read_ply_vertex_count(ply_path),
            "bytes": ply_path.stat().st_size,
        }

    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
