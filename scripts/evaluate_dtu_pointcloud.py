#!/usr/bin/env python3
"""Evaluate a point cloud with the official DTU SampleSet protocol.

This is a Python port of the relevant PointCompareMain.m and
ComputeStat_web.m operations distributed in the official DTU SampleSet:

* greedily reduce reconstructed points to 0.2 mm spacing;
* measure reconstruction-to-reference accuracy inside ObsMask;
* measure reference-to-reconstruction completeness above the ground plane;
* discard distances at or above the official 20 mm statistics threshold.

The official MATLAB reduction is stochastic. This port fixes its seed so that
benchmark results are reproducible.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree


PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=Path("data/metal-benchmark/dtu/SampleSet"),
    )
    parser.add_argument("--scan", type=int, default=6)
    parser.add_argument("--mask-margin", type=int, default=10)
    parser.add_argument("--min-distance", type=float, default=0.2)
    parser.add_argument("--max-distance", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--query-chunk-size", type=int, default=500_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_ply_xyz(path: Path) -> np.ndarray:
    properties: list[tuple[str, str]] = []
    vertex_count = None
    ply_format = None
    header_lines = 0
    with path.open("rb") as ply_file:
        if ply_file.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        header_lines += 1
        in_vertex = False
        while True:
            raw_line = ply_file.readline()
            if not raw_line:
                raise ValueError(f"Missing end_header in {path}")
            header_lines += 1
            line = raw_line.decode("ascii").strip()
            fields = line.split()
            if fields[:1] == ["format"]:
                ply_format = fields[1]
            elif fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                in_vertex = True
            elif fields and fields[0] == "element":
                in_vertex = False
            elif in_vertex and fields[:1] == ["property"]:
                if fields[1] == "list":
                    raise ValueError("List property in PLY vertex is unsupported")
                properties.append((fields[2], PLY_TYPES[fields[1]]))
            elif line == "end_header":
                data_offset = ply_file.tell()
                break
    if vertex_count is None or ply_format is None:
        raise ValueError(f"Incomplete PLY header: {path}")
    names = [name for name, _ in properties]
    if not all(name in names for name in ("x", "y", "z")):
        raise ValueError(f"PLY has no XYZ properties: {path}")

    if ply_format == "binary_little_endian":
        vertices = np.memmap(
            path,
            dtype=np.dtype(properties),
            mode="r",
            offset=data_offset,
            shape=(vertex_count,),
        )
        points = np.column_stack(
            (vertices["x"], vertices["y"], vertices["z"])
        )
    elif ply_format == "ascii":
        xyz_columns = tuple(names.index(name) for name in ("x", "y", "z"))
        points = np.loadtxt(
            path,
            skiprows=header_lines,
            max_rows=vertex_count,
            usecols=xyz_columns,
        )
    else:
        raise ValueError(f"Unsupported PLY format {ply_format}: {path}")
    points = np.asarray(points, dtype=np.float64)
    finite = np.all(np.isfinite(points), axis=1)
    return points[finite]


def reduce_points(
    points: np.ndarray, min_distance: float, seed: int
) -> np.ndarray:
    """Random-order greedy radius suppression, matching reducePts_haa.m."""
    if min_distance <= 0 or len(points) < 2:
        return points
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))
    cell_origin = np.min(points, axis=0)
    inverse_cell_size = 1.0 / min_distance
    squared_distance = min_distance * min_distance
    neighbor_offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
    cells: dict[tuple[int, int, int], list[int]] = {}
    kept: list[int] = []

    for point_idx in order:
        point = points[point_idx]
        cell_array = np.floor((point - cell_origin) * inverse_cell_size).astype(
            np.int64
        )
        cell = (int(cell_array[0]), int(cell_array[1]), int(cell_array[2]))
        has_close_point = False
        for dx, dy, dz in neighbor_offsets:
            for other_idx in cells.get(
                (cell[0] + dx, cell[1] + dy, cell[2] + dz), ()
            ):
                delta = point - points[other_idx]
                if float(delta @ delta) <= squared_distance:
                    has_close_point = True
                    break
            if has_close_point:
                break
        if not has_close_point:
            kept.append(int(point_idx))
            cells.setdefault(cell, []).append(int(point_idx))
    return points[np.asarray(kept, dtype=np.int64)]


def points_in_observation_mask(
    points: np.ndarray, mask: np.ndarray, bounding_box: np.ndarray, resolution: float
) -> np.ndarray:
    # MATLAB uses round((Q - BB(1)) / Res + 1) and one-based array indices.
    # Coordinates inside this mask are non-negative, so floor(x + 0.5) is the
    # equivalent zero-based operation without NumPy's ties-to-even behavior.
    voxels = np.floor((points - bounding_box[0]) / resolution + 0.5).astype(
        np.int64
    )
    in_bounds = np.all(voxels >= 0, axis=1) & np.all(
        voxels < np.asarray(mask.shape), axis=1
    )
    result = np.zeros(len(points), dtype=bool)
    valid_voxels = voxels[in_bounds]
    result[in_bounds] = mask[
        valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]
    ].astype(bool)
    return result


def nearest_distances(
    target: np.ndarray,
    query: np.ndarray,
    workers: int,
    chunk_size: int,
) -> np.ndarray:
    tree = cKDTree(target)
    distances = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), chunk_size):
        stop = min(len(query), start + chunk_size)
        distances[start:stop] = tree.query(
            query[start:stop], k=1, workers=workers
        )[0]
    return distances


def summarize_distances(
    distances: np.ndarray, max_distance: float
) -> dict[str, Any]:
    inliers = distances < max_distance
    values = distances[inliers]
    if not len(values):
        raise ValueError("No distances survived the DTU outlier threshold")
    return {
        "eligible_points": int(len(distances)),
        "inlier_points": int(len(values)),
        "inlier_fraction": float(np.mean(inliers)),
        "mean_mm": float(np.mean(values)),
        "median_mm": float(np.median(values)),
        "p90_mm": float(np.percentile(values, 90)),
        "p95_mm": float(np.percentile(values, 95)),
        "max_inlier_mm": float(np.max(values)),
        "within_threshold_fraction": {
            f"{threshold:g}_mm": float(np.mean(distances < threshold))
            for threshold in (0.5, 1.0, 2.0, 5.0)
        },
    }


def main() -> None:
    args = parse_args()
    if args.min_distance < 0 or args.max_distance <= 0:
        raise ValueError("Distance thresholds must be positive")
    if args.query_chunk_size < 1:
        raise ValueError("--query-chunk-size must be positive")

    mvs_root = args.sample_root / "MVS Data"
    reference_path = (
        mvs_root / "Points" / "stl" / f"stl{args.scan:03d}_total.ply"
    )
    mask_path = (
        mvs_root
        / "ObsMask"
        / f"ObsMask{args.scan}_{args.mask_margin}.mat"
    )
    plane_path = mvs_root / "ObsMask" / f"Plane{args.scan}.mat"

    print(f"Reading reconstruction: {args.reconstruction}")
    reconstructed_raw = read_ply_xyz(args.reconstruction)
    print(
        f"Reducing {len(reconstructed_raw):,} points to "
        f"{args.min_distance:g} mm spacing..."
    )
    reconstructed = reduce_points(
        reconstructed_raw, args.min_distance, args.seed
    )
    print(f"Retained {len(reconstructed):,} reconstructed points")
    reference = read_ply_xyz(reference_path)

    mask_data = loadmat(mask_path)
    observation_mask = np.asarray(mask_data["ObsMask"])
    bounding_box = np.asarray(mask_data["BB"], dtype=np.float64)
    resolution = float(np.asarray(mask_data["Res"]).item())
    reconstruction_in_mask = points_in_observation_mask(
        reconstructed, observation_mask, bounding_box, resolution
    )

    plane = np.asarray(loadmat(plane_path)["P"], dtype=np.float64).reshape(4)
    reference_above_plane = (
        reference @ plane[:3] + plane[3]
    ) > 0

    print("Computing reconstruction-to-reference accuracy...")
    accuracy_distances = nearest_distances(
        reference,
        reconstructed[reconstruction_in_mask],
        args.workers,
        args.query_chunk_size,
    )
    print("Computing reference-to-reconstruction completeness...")
    completeness_distances = nearest_distances(
        reconstructed,
        reference[reference_above_plane],
        args.workers,
        args.query_chunk_size,
    )
    accuracy = summarize_distances(accuracy_distances, args.max_distance)
    completeness = summarize_distances(completeness_distances, args.max_distance)

    result = {
        "protocol": {
            "source": "DTU SampleSet PointCompareMain.m and ComputeStat_web.m",
            "scan": args.scan,
            "mask_margin_mm": args.mask_margin,
            "minimum_point_spacing_mm": args.min_distance,
            "statistics_outlier_threshold_mm": args.max_distance,
            "reduction_seed": args.seed,
        },
        "paths": {
            "reconstruction": str(args.reconstruction.resolve()),
            "reference": str(reference_path.resolve()),
            "observation_mask": str(mask_path.resolve()),
            "ground_plane": str(plane_path.resolve()),
        },
        "counts": {
            "reconstruction_raw": int(len(reconstructed_raw)),
            "reconstruction_reduced": int(len(reconstructed)),
            "reconstruction_in_mask": int(np.count_nonzero(reconstruction_in_mask)),
            "reference": int(len(reference)),
            "reference_above_plane": int(np.count_nonzero(reference_above_plane)),
        },
        "accuracy": accuracy,
        "completeness": completeness,
        "overall_mean_mm": (accuracy["mean_mm"] + completeness["mean_mm"]) / 2,
    }
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(f"Wrote {args.output}")
    print(output, end="")


if __name__ == "__main__":
    main()
