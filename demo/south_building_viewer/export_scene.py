#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Export a COLMAP sparse model for the South Building web viewer.

Writes scene.json next to this script and symlinks points.ply.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def parse_cameras(path):
    cameras = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(v) for v in parts[4:]]
        if model == "SIMPLE_RADIAL":
            focal, cx, cy, k1 = params
        elif model == "SIMPLE_PINHOLE":
            focal, cx, cy = params
            k1 = 0.0
        elif model == "PINHOLE":
            fx, fy, cx, cy = params
            focal = (fx + fy) / 2
            k1 = 0.0
        elif model in ("OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
            fx, fy, cx, cy = params[:4]
            focal = (fx + fy) / 2
            k1 = params[4] if len(params) > 4 else 0.0
        else:
            focal = params[0]
            cx = params[1] if len(params) > 1 else width / 2
            cy = params[2] if len(params) > 2 else height / 2
            k1 = 0.0
        cameras[camera_id] = {
            "model": model,
            "width": width,
            "height": height,
            "focal": focal,
            "cx": cx,
            "cy": cy,
            "k1": k1,
        }
    return cameras


def parse_images(path, cameras):
    images = []
    lines = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for line in lines:
        parts = line.split()
        if len(parts) < 10 or "." not in parts[-1]:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        rotation = qvec_to_rotmat([qw, qx, qy, qz])
        translation = np.array([tx, ty, tz], dtype=np.float64)
        center = -rotation.T @ translation
        world_from_cam = rotation.T
        opengl = world_from_cam @ np.diag([1.0, -1.0, -1.0])
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = opengl
        matrix[:3, 3] = center
        camera = cameras[camera_id]
        images.append(
            {
                "id": int(parts[0]),
                "name": name,
                "camera_id": camera_id,
                "width": camera["width"],
                "height": camera["height"],
                "focal": camera["focal"],
                "position": center.tolist(),
                "matrix": matrix.flatten(order="F").tolist(),
            }
        )
    images.sort(key=lambda item: item["name"])
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap", required=True)
    parser.add_argument("--model", required=True, help="sparse model directory")
    parser.add_argument("--ply", required=True)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent),
        help="viewer directory",
    )
    parser.add_argument("--label", default="South Building")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                args.colmap,
                "model_converter",
                "--input_path",
                args.model,
                "--output_path",
                tmp,
                "--output_type",
                "TXT",
            ],
            check=True,
        )
        cameras = parse_cameras(Path(tmp) / "cameras.txt")
        images = parse_images(Path(tmp) / "images.txt", cameras)
        points_header = (Path(tmp) / "points3D.txt").read_text().splitlines()
        n_points = sum(
            1 for line in points_header if line.strip() and not line.startswith("#")
        )

    ply_link = out / "points.ply"
    ply_src = Path(args.ply).resolve()
    if ply_link.is_symlink() or ply_link.exists():
        ply_link.unlink()
    os.symlink(ply_src, ply_link)

    scene = {
        "label": args.label,
        "num_points": n_points,
        "num_images": len(images),
        "num_cameras": len(cameras),
        "ply": "points.ply",
        "images": images,
    }
    (out / "scene.json").write_text(json.dumps(scene))
    print(f"wrote {out / 'scene.json'} images={len(images)} points={n_points}")


if __name__ == "__main__":
    main()
