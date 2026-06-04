#!/usr/bin/env python3
"""
Native equirectangular SfM on 360° panorama images.

Unlike ``panorama_sfm.py``, which renders perspective cube/fisheye views and
reconstructs them as a rig, this example feeds the equirectangular panoramas
directly to COLMAP using the native ``SPHERICAL`` camera model: features are
extracted on the panorama and unprojected to bearings via ``CamRayFromImg``,
so no fisheye conversion, virtual cameras or rig are needed.

Reconstruct with the incremental ``mapper`` and/or the global ``global_mapper``
(formerly GLOMAP, now incorporated into COLMAP). Optionally writes a Rerun
visualization (requires ``rerun-sdk``).

NOTE: This is an interim CLI-driven example that shells out to the ``colmap``
executable. It is intended to be ported to the pycolmap API and to then
supersede ``panorama_sfm.py``.
"""

import argparse
import os
import time
import subprocess
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Optional imports for visualization
try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    print("Warning: rerun-sdk not installed. Visualization will be skipped.")
    print("Install with: pip install rerun-sdk")


def select_largest_model(sparse_path: Path) -> Path:
    """Return the reconstruction sub-model with the most images.

    The incremental mapper can emit several disconnected models (0/, 1/, ...);
    model 0 is not necessarily the largest. The size of images.bin is a good
    proxy for the model with the most images/observations.
    """
    candidates = [
        d for d in sparse_path.iterdir()
        if d.is_dir() and (d / "images.bin").exists()
    ]
    if not candidates:
        return sparse_path / "0"
    return max(candidates, key=lambda d: (d / "images.bin").stat().st_size)


def find_files(folder, types):
    """Find files with given extensions in folder."""
    return [entry.path for entry in os.scandir(folder)
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in types]


def video_to_frames(video_path: Path, dest: Path, increment: int = 4, downscale: int = 1):
    """Extract frames from video file."""
    extraction_start = time.time()

    if not dest.exists():
        dest.mkdir(parents=True)
    else:
        # Clear existing files
        for file in find_files(dest, [".jpg", ".png"]):
            os.remove(file)

    cmd = f'ffmpeg -i {video_path} -qmin 1 -q:v 1 -vf "select=not(mod(n\\,{increment})),scale=iw/{downscale}:ih/{downscale}" -fps_mode vfr {dest}/frame%06d.jpg'
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=False)

    paths = find_files(dest, [".jpg"])
    extraction_end = time.time()
    print(f"Extracted {len(paths)} frames in {extraction_end - extraction_start:.2f} seconds")
    return paths


def run_colmap_sfm(colmap_path: Path, database_path: Path, image_path: Path,
                   output_path: Path, use_gpu: bool = True):
    """Run COLMAP SfM pipeline with SPHERICAL camera model."""
    print("\n" + "="*60)
    print("Running COLMAP SfM Pipeline (SPHERICAL)")
    print("="*60)

    gpu_flag = "--FeatureExtraction.use_gpu 1" if use_gpu else "--FeatureExtraction.use_gpu 0"

    # Feature extraction
    print("\n[COLMAP] Feature Extraction...")
    t0 = time.perf_counter()
    cmd = f"{colmap_path} feature_extractor --database_path {database_path} --image_path {image_path} --ImageReader.camera_model SPHERICAL --ImageReader.single_camera 1 {gpu_flag}"
    print(f"  Command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print(f"  Completed in {time.perf_counter() - t0:.2f}s")

    gpu_match_flag = "--FeatureMatching.use_gpu 1" if use_gpu else "--FeatureMatching.use_gpu 0"

    # Sequential matching
    print("\n[COLMAP] Sequential Matching...")
    t0 = time.perf_counter()
    cmd = f"{colmap_path} sequential_matcher --database_path {database_path} {gpu_match_flag}"
    print(f"  Command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print(f"  Completed in {time.perf_counter() - t0:.2f}s")

    # Incremental mapping (GPU bundle adjustment when available: DENSE_SCHUR on
    # CUDA for >= 50 images).
    print("\n[COLMAP] Incremental Mapping...")
    t0 = time.perf_counter()
    output_path.mkdir(parents=True, exist_ok=True)
    ba_gpu_flag = "--Mapper.ba_use_gpu 1" if use_gpu else "--Mapper.ba_use_gpu 0"
    cmd = f"{colmap_path} mapper --database_path {database_path} --image_path {image_path} --output_path {output_path} {ba_gpu_flag}"
    print(f"  Command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print(f"  Completed in {time.perf_counter() - t0:.2f}s")

    return select_largest_model(output_path)


def run_global_sfm(colmap_path: Path, database_path: Path, image_path: Path,
                   output_path: Path, use_gpu: bool = True):
    """Run COLMAP global SfM with the SPHERICAL camera model.

    GLOMAP's global mapper is now incorporated into COLMAP as the
    ``global_mapper`` subcommand, so no separate glomap binary is needed.
    """
    print("\n" + "="*60)
    print("Running COLMAP Global SfM Pipeline (SPHERICAL)")
    print("="*60)

    gpu_flag = "--FeatureExtraction.use_gpu 1" if use_gpu else "--FeatureExtraction.use_gpu 0"

    # Feature extraction
    print("\n[Global] Feature Extraction...")
    t0 = time.perf_counter()
    cmd = f"{colmap_path} feature_extractor --database_path {database_path} --image_path {image_path} --ImageReader.camera_model SPHERICAL --ImageReader.single_camera 1 {gpu_flag}"
    print(f"  Command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print(f"  Completed in {time.perf_counter() - t0:.2f}s")

    gpu_match_flag = "--FeatureMatching.use_gpu 1" if use_gpu else "--FeatureMatching.use_gpu 0"

    # Sequential matching
    print("\n[Global] Sequential Matching...")
    t0 = time.perf_counter()
    cmd = f"{colmap_path} sequential_matcher --database_path {database_path} {gpu_match_flag}"
    print(f"  Command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print(f"  Completed in {time.perf_counter() - t0:.2f}s")

    # Global mapping (GLOMAP, now built into COLMAP as `global_mapper`)
    print("\n[Global] Global Mapping...")
    t0 = time.perf_counter()
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = f"{colmap_path} global_mapper --database_path {database_path} --image_path {image_path} --output_path {output_path}"
    print(f"  Command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print(f"  Completed in {time.perf_counter() - t0:.2f}s")

    return select_largest_model(output_path)


def export_to_txt(colmap_path: Path, model_path: Path, output_path: Path):
    """Export COLMAP model to text format."""
    print(f"\n[Export] Converting model to TXT format...")
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = f"{colmap_path} model_converter --input_path {model_path} --output_path {output_path} --output_type TXT"
    subprocess.run(cmd, shell=True, check=True)
    print(f"  Exported to {output_path}")


def analyze_model(colmap_path: Path, model_path: Path):
    """Analyze reconstruction quality."""
    print(f"\n[Analysis] Model statistics:")
    cmd = f"{colmap_path} model_analyzer --path {model_path}"
    subprocess.run(cmd, shell=True, check=True)


# ============================================================================
# Rerun Visualization Functions
# ============================================================================

def translation_and_quaternion_to_matrix(translation, quaternion):
    """Convert translation and quaternion to a 4x4 transformation matrix.
    COLMAP quaternion order: (w, x, y, z), but Scipy expects (x, y, z, w)
    """
    rotation_matrix = R.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation
    return transformation_matrix


def project_to_panorama(points, pano_width, pano_height):
    """Project 3D points to equirectangular image coordinates.

    Mirrors SphericalCameraModel::ImgFromCam in COLMAP exactly so the overlay
    lines up with the image (note the -Y / atan2 elevation convention, which
    is what fixes the vertical flip):
        theta = atan2(X, Z); phi = atan2(-Y, sqrt(X^2 + Z^2))
        x = (theta/(2*pi) + 0.5) * w;   y = (0.5 - phi/pi) * h
    """
    dirs = points / np.linalg.norm(points, axis=1, keepdims=True)
    x_c, y_c, z_c = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    # Azimuth in [-pi, pi].
    theta = np.arctan2(x_c, z_c)
    # Elevation in [-pi/2, pi/2], measured with -Y as "up" to match the model.
    horizontal = np.sqrt(x_c * x_c + z_c * z_c)
    phi = np.arctan2(-y_c, horizontal)

    u = (theta / (2 * np.pi) + 0.5) * pano_width
    v = (0.5 - phi / np.pi) * pano_height

    return np.column_stack([u, v]).astype(np.float32)


def parse_cameras_txt(filepath):
    """Parse COLMAP cameras.txt format."""
    cameras = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]
                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params
                }
    return cameras


def parse_images_txt(filepath):
    """Parse COLMAP images.txt format."""
    images = {}
    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            name = parts[9]

            images[image_id] = {
                'quaternion': np.array([qw, qx, qy, qz]),
                'translation': np.array([tx, ty, tz]),
                'camera_id': camera_id,
                'name': name
            }
            i += 2  # Skip the points2D line
        else:
            i += 1

    return images


def parse_points3d_txt(filepath):
    """Parse COLMAP points3D.txt format."""
    points = []
    colors = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) >= 7:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                points.append([x, y, z])
                colors.append([r, g, b])

    return np.array(points) if points else np.zeros((0, 3)), np.array(colors) if colors else np.zeros((0, 3), dtype=np.uint8)


def generate_rerun_visualization(txt_path: Path, images_path: Path, output_rrd: Path,
                                  name: str = "equirectangular_reconstruction"):
    """Generate Rerun visualization from COLMAP TXT output."""
    if not RERUN_AVAILABLE:
        print("Skipping visualization - rerun-sdk not installed")
        return

    print(f"\n[Visualization] Generating Rerun visualization...")
    print(f"  Input: {txt_path}")
    print(f"  Output: {output_rrd}")

    # Parse reconstruction data
    cameras_file = txt_path / "cameras.txt"
    images_file = txt_path / "images.txt"
    points_file = txt_path / "points3D.txt"

    if not images_file.exists() or not points_file.exists():
        print(f"  Error: Missing reconstruction files in {txt_path}")
        return

    cameras = parse_cameras_txt(cameras_file) if cameras_file.exists() else {}
    images = parse_images_txt(images_file)
    points, colors = parse_points3d_txt(points_file)

    print(f"  Loaded {len(images)} camera poses, {len(points)} 3D points")

    if len(images) == 0 or len(points) == 0:
        print("  Warning: Empty reconstruction, skipping visualization")
        return

    # Get panorama dimensions from cameras or use default
    pano_width, pano_height = 7680, 3840
    if cameras:
        cam = list(cameras.values())[0]
        pano_width, pano_height = cam['width'], cam['height']

    # Compute camera transforms and world positions
    camera_transforms = {}
    camera_positions = {}
    for img_id, img_data in images.items():
        transform = translation_and_quaternion_to_matrix(
            img_data['translation'],
            img_data['quaternion']
        )
        camera_transforms[img_id] = transform
        inv_transform = np.linalg.inv(transform)
        camera_positions[img_id] = inv_transform[:3, 3]

    # Sort by image name
    sorted_ids = sorted(images.keys(), key=lambda x: images[x]['name'])

    # Build trajectory
    trajectory = np.array([camera_positions[img_id] for img_id in sorted_ids])

    # Initialize Rerun
    rr.init(name, spawn=False)
    rr.save(str(output_rrd))

    # Log static data
    rr.set_time("frame", sequence=0)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    rr.log("/world/trajectory", rr.LineStrips3D([trajectory], colors=[[0, 255, 0]]))
    rr.log("/world/points", rr.Points3D(points, colors=colors, radii=0.02), static=True)

    # Prepare 4D points for transformation
    vec4_points = np.ones((points.shape[0], 4), dtype=np.float32)
    vec4_points[:, :3] = points

    # Log per-frame data
    print("  Generating per-frame visualizations...")
    for frame_idx, img_id in enumerate(tqdm(sorted_ids, desc="  Frames")):
        img_data = images[img_id]
        transform = camera_transforms[img_id]

        # Transform points to camera local coordinates
        local_pts = (vec4_points @ transform.T)[:, :3]
        local_distances = np.linalg.norm(local_pts, axis=1)

        # For equirectangular cameras, we see in ALL directions (360 degrees)
        valid_mask = (local_distances > 0.1) & (local_distances < 100)
        valid_local_pts = local_pts[valid_mask]
        valid_distances = local_distances[valid_mask]
        valid_colors = colors[valid_mask]

        if len(valid_local_pts) == 0:
            continue

        # Sort by distance (far to near)
        sorted_indices = np.argsort(-valid_distances)
        valid_local_pts = valid_local_pts[sorted_indices]
        valid_distances = valid_distances[sorted_indices]
        valid_colors = valid_colors[sorted_indices]

        # Project to panorama coordinates
        pano_coords = project_to_panorama(valid_local_pts, pano_width, pano_height)

        # Scale point radii by distance
        radii = np.clip(30.0 / valid_distances, 2.0, 15.0)

        # Set time and log
        rr.set_time("frame", sequence=frame_idx)
        rr.log("/world/camera", rr.Points3D([camera_positions[img_id]], radii=0.3, colors=[[255, 0, 0]]))

        # Log image with projected points
        image_path = images_path / img_data['name']
        if image_path.exists():
            rr.log("/camera/image", rr.EncodedImage(path=str(image_path)))
            rr.log("/camera/image/projected_points", rr.Points2D(pano_coords, colors=valid_colors, radii=radii))

    print(f"  Visualization saved to: {output_rrd}")
    print(f"  Open with: rerun {output_rrd}")


# ============================================================================
# Main Pipeline
# ============================================================================

def run(args):
    total_start = time.perf_counter()

    # Setup paths
    base_path = Path(args.output_path)
    base_path.mkdir(parents=True, exist_ok=True)

    colmap_db_path = base_path / "colmap_database.db"
    colmap_sparse_path = base_path / "colmap_sparse"
    colmap_txt_path = base_path / "colmap_txt"
    colmap_rrd_path = base_path / "colmap_visualization.rrd"

    global_db_path = base_path / "global_database.db"
    global_sparse_path = base_path / "global_sparse"
    global_txt_path = base_path / "global_txt"
    global_rrd_path = base_path / "global_visualization.rrd"

    colmap_path = Path(args.colmap_path)

    # Determine the panorama image directory: either extract frames from a
    # video, or use a directory of equirectangular panoramas directly.
    if args.input_video:
        pano_image_path = base_path / "pano_images"
        video_to_frames(Path(args.input_video), pano_image_path,
                        increment=args.increment, downscale=args.downscale)
    elif args.input_image_path:
        pano_image_path = Path(args.input_image_path)
    else:
        print("Error: provide --input_image_path (a directory of "
              "equirectangular panoramas) or --input_video")
        return

    if not pano_image_path.exists():
        print(f"Error: panorama image directory not found at {pano_image_path}")
        return

    num_images = len(find_files(pano_image_path, [".jpg", ".png"]))
    print(f"\nFound {num_images} panorama images in {pano_image_path}")

    # Run COLMAP
    if args.run_colmap:
        if colmap_db_path.exists():
            colmap_db_path.unlink()
        if colmap_sparse_path.exists():
            shutil.rmtree(colmap_sparse_path)

        colmap_model = run_colmap_sfm(
            colmap_path, colmap_db_path, pano_image_path,
            colmap_sparse_path, use_gpu=args.use_gpu
        )

        if colmap_model.exists():
            export_to_txt(colmap_path, colmap_model, colmap_txt_path)
            analyze_model(colmap_path, colmap_model)

            if args.visualize and RERUN_AVAILABLE:
                generate_rerun_visualization(
                    colmap_txt_path, pano_image_path, colmap_rrd_path,
                    name="colmap_equirectangular"
                )

    # Run COLMAP global mapper (formerly GLOMAP, now built into COLMAP)
    if args.run_global:
        if global_db_path.exists():
            global_db_path.unlink()
        if global_sparse_path.exists():
            shutil.rmtree(global_sparse_path)

        global_model = run_global_sfm(
            colmap_path, global_db_path, pano_image_path,
            global_sparse_path, use_gpu=args.use_gpu
        )

        if global_model.exists():
            export_to_txt(colmap_path, global_model, global_txt_path)
            analyze_model(colmap_path, global_model)

            if args.visualize and RERUN_AVAILABLE:
                generate_rerun_visualization(
                    global_txt_path, pano_image_path, global_rrd_path,
                    name="colmap_global_spherical"
                )

    total_time = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"Total pipeline time: {total_time:.2f} seconds")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Native Equirectangular SfM Pipeline using COLMAP "
                    "(incremental mapper and/or global_mapper)"
    )

    # Input/Output
    parser.add_argument("--input_image_path", type=str, default=None,
                       help="Directory of equirectangular (360°) panorama images")
    parser.add_argument("--input_video", type=str, default=None,
                       help="Input 360° video; frames are extracted to "
                            "<output_path>/pano_images (alternative to "
                            "--input_image_path)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output directory for the database and models")

    # Tool paths
    parser.add_argument("--colmap_path", type=str, default="colmap",
                       help="Path to the COLMAP executable")

    # Pipeline options
    parser.add_argument("--run_colmap", action="store_true", default=True,
                       help="Run COLMAP incremental SfM pipeline")
    parser.add_argument("--no_colmap", action="store_true",
                       help="Skip COLMAP incremental pipeline")
    parser.add_argument("--run_global", action="store_true", default=False,
                       help="Run COLMAP global SfM (global_mapper, formerly GLOMAP)")

    # Video extraction options
    parser.add_argument("--increment", type=int, default=4,
                       help="Keep every Nth video frame (with --input_video)")
    parser.add_argument("--downscale", type=int, default=1,
                       help="Downscale factor for extracted video frames")

    # GPU options
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="Use GPU for feature extraction/matching")
    parser.add_argument("--no_gpu", action="store_true",
                       help="Disable GPU")

    # Visualization
    parser.add_argument("--visualize", action="store_true", default=False,
                       help="Write a Rerun visualization (requires rerun-sdk)")

    args = parser.parse_args()

    # Handle negation flags
    if args.no_colmap:
        args.run_colmap = False
    if args.no_gpu:
        args.use_gpu = False

    run(args)
