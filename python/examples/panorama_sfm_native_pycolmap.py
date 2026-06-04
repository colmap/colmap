#!/usr/bin/env python3
"""
Native equirectangular SfM on 360° panoramas, driven via the pycolmap API.

Same goal as ``panorama_sfm_native.py`` (feed equirectangular images directly
to COLMAP using the native ``SPHERICAL`` camera model — no fisheye/cubemap
rendering or rig), but this version uses the **pycolmap** Python API instead of
shelling out to the ``colmap`` executable.

It keeps the fast ffmpeg frame extraction, with **NVDEC GPU decode** when the
source codec supports it (HEVC / AV1 / VP9 at any size; H.264 only up to
4096 px wide — NVDEC's H.264 limit). Reconstruct with the incremental
``mapper`` and/or the global ``global_mapper`` (formerly GLOMAP). Optionally
writes a Rerun visualization (requires ``rerun-sdk``).

Because extraction + matching are shared, both mappers reuse a single database.

This is intended to eventually supersede ``panorama_sfm.py``.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import pycolmap
from pycolmap import logging

try:
    import rerun as rr  # pip install rerun-sdk
    from tqdm import tqdm
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False


# ============================================================================
# Fast ffmpeg frame extraction (NVDEC GPU decode when supported)
# ============================================================================

# NVDEC decodes these at 8K; H.264 is limited to 4096 px wide.
NVDEC_8K_CODECS = {"hevc", "av1", "vp9"}


def probe_video(path: Path):
    """Return (codec_name, width, height) for the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    return s["codec_name"], int(s["width"]), int(s["height"])


def _ffmpeg_extract(video_path, out_dir, tw, th, codec, w, hwaccel,
                    select_expr=None, skip_nonkey=False):
    """Run one ffmpeg extraction (NVDEC when codec/size allow, else CPU)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nvdec_ok = codec in NVDEC_8K_CODECS or (codec == "h264" and w <= 4096)
    use_nvdec = hwaccel == "cuda" or (hwaccel == "auto" and nvdec_ok)
    if hwaccel == "cuda" and not nvdec_ok:
        logging.warning(
            f"NVDEC not available for codec={codec} {w} px wide; CPU decode")
        use_nvdec = False

    chain = []
    if select_expr:  # escaped comma so ffmpeg doesn't split the filtergraph
        chain.append(f"select={select_expr}")
    input_opts = []
    if skip_nonkey:  # decode only keyframes (I-frames)
        input_opts += ["-skip_frame", "nokey"]
    if use_nvdec:
        input_opts += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                       "-c:v", f"{codec}_cuvid"]
        chain += [f"scale_cuda={tw}:{th}:interp_algo=lanczos",
                  "hwdownload", "format=nv12"]
    else:
        chain += [f"scale={tw}:{th}"]
    cmd = (["ffmpeg", "-hide_banner", "-loglevel", "error"] + input_opts +
           ["-i", str(video_path), "-vf", ",".join(chain),
            "-qmin", "1", "-q:v", "1", "-fps_mode", "vfr",
            f"{out_dir}/frame%06d.jpg"])
    subprocess.run(cmd, check=True)
    return "NVDEC" if use_nvdec else "CPU"


def video_to_frames(video_path: Path, dest: Path, increment: int = 4,
                    downscale: int = 1, hwaccel: str = "auto",
                    select: str = "stride"):
    """Extract frames from a video, downscaled by ``downscale``.

    ``select`` chooses the sampling strategy:
      - "stride":    every ``increment``-th frame (default).
      - "keyframes": only codec keyframes (I-frames); fast, but GOP-spaced and
                     fewer frames — not actually a sharpness criterion.

    NVDEC GPU decode is used when the codec/size allow (HEVC/AV1/VP9 at any
    size, H.264 up to 4096 px); ``hwaccel="none"`` forces CPU.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for f in list(dest.glob("*.jpg")) + list(dest.glob("*.png")):
        f.unlink()

    codec, w, h = probe_video(video_path)
    tw = max(2, (w // downscale) & ~1)  # even target dims
    th = max(2, (h // downscale) & ~1)
    t0 = time.perf_counter()

    if select == "keyframes":
        mode = _ffmpeg_extract(video_path, dest, tw, th, codec, w, hwaccel,
                               skip_nonkey=True)
    else:  # stride
        mode = _ffmpeg_extract(video_path, dest, tw, th, codec, w, hwaccel,
                               select_expr=f"not(mod(n\\,{increment}))")

    n = len(list(dest.glob("*.jpg")))
    logging.info(f"Extracted {n} frames ({codec} {w}x{h} -> {tw}x{th}, "
                 f"{mode} decode, select={select}) in "
                 f"{time.perf_counter() - t0:.2f}s")
    return n


# ============================================================================
# Reconstruction helpers
# ============================================================================

def largest_reconstruction(recs):
    """Pick the sub-model with the most registered images (mappers may emit
    several disconnected models)."""
    if not recs:
        return None
    return max(recs.values(), key=lambda r: r.num_reg_images())


# ============================================================================
# Rerun visualization (optional)
# ============================================================================

def translation_and_quaternion_to_matrix(translation, quaternion):
    """4x4 world-from-... matrix. COLMAP quaternion is (w, x, y, z)."""
    rotation_matrix = R.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    m = np.eye(4)
    m[:3, :3] = rotation_matrix
    m[:3, 3] = translation
    return m


def project_to_panorama(points, pano_width, pano_height):
    """Project 3D points to equirectangular image coordinates.

    Mirrors SphericalCameraModel::ImgFromCam exactly (note the -Y / atan2
    elevation convention) so the 2D overlay lines up with the image:
        theta = atan2(X, Z); phi = atan2(-Y, sqrt(X^2 + Z^2))
        x = (theta/(2*pi) + 0.5) * w;   y = (0.5 - phi/pi) * h
    """
    dirs = points / np.linalg.norm(points, axis=1, keepdims=True)
    x_c, y_c, z_c = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    theta = np.arctan2(x_c, z_c)
    horizontal = np.sqrt(x_c * x_c + z_c * z_c)
    phi = np.arctan2(-y_c, horizontal)
    u = (theta / (2 * np.pi) + 0.5) * pano_width
    v = (0.5 - phi / np.pi) * pano_height
    return np.column_stack([u, v]).astype(np.float32)


def _parse_cameras_txt(filepath):
    cameras = {}
    for line in open(filepath):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        cameras[int(p[0])] = {"width": int(p[2]), "height": int(p[3])}
    return cameras


def _parse_images_txt(filepath):
    images = {}
    lines = open(filepath).readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        p = line.split()
        if len(p) >= 10:
            images[int(p[0])] = {
                "quaternion": np.array([float(x) for x in p[1:5]]),  # w,x,y,z
                "translation": np.array([float(x) for x in p[5:8]]),
                "name": p[9],
            }
            i += 2  # skip the points2D line
        else:
            i += 1
    return images


def _parse_points3d_txt(filepath):
    pts, cols = [], []
    for line in open(filepath):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 7:
            pts.append([float(p[1]), float(p[2]), float(p[3])])
            cols.append([int(p[4]), int(p[5]), int(p[6])])
    return (np.array(pts) if pts else np.zeros((0, 3)),
            np.array(cols, dtype=np.uint8) if cols else np.zeros((0, 3), np.uint8))


def generate_rerun_visualization(txt_path: Path, images_path: Path,
                                  output_rrd: Path, name: str):
    """Write a Rerun .rrd from a COLMAP TXT model: 3D points, camera
    trajectory, and per-frame equirectangular overlay of projected points."""
    if not RERUN_AVAILABLE:
        logging.info("Skipping visualization - rerun-sdk not installed")
        return
    images = _parse_images_txt(txt_path / "images.txt")
    points, colors = _parse_points3d_txt(txt_path / "points3D.txt")
    cameras = _parse_cameras_txt(txt_path / "cameras.txt")
    if len(images) == 0 or len(points) == 0:
        logging.info("Empty reconstruction, skipping visualization")
        return
    cam = next(iter(cameras.values()))
    pano_width, pano_height = cam["width"], cam["height"]

    transforms, positions = {}, {}
    for img_id, img in images.items():
        m = translation_and_quaternion_to_matrix(img["translation"],
                                                 img["quaternion"])
        transforms[img_id] = m
        positions[img_id] = np.linalg.inv(m)[:3, 3]

    sorted_ids = sorted(images.keys(), key=lambda i: images[i]["name"])
    trajectory = np.array([positions[i] for i in sorted_ids])

    rr.init(name, spawn=False)
    rr.save(str(output_rrd))
    rr.set_time("frame", sequence=0)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    rr.log("/world/trajectory", rr.LineStrips3D([trajectory], colors=[[0, 255, 0]]))
    rr.log("/world/points", rr.Points3D(points, colors=colors, radii=0.02), static=True)

    vec4 = np.ones((points.shape[0], 4), dtype=np.float32)
    vec4[:, :3] = points
    for frame_idx, img_id in enumerate(tqdm(sorted_ids, desc="  Frames")):
        local = (vec4 @ transforms[img_id].T)[:, :3]
        dist = np.linalg.norm(local, axis=1)
        mask = (dist > 0.1) & (dist < 100)
        if not np.any(mask):
            continue
        order = np.argsort(-dist[mask])
        lp, ld, lc = local[mask][order], dist[mask][order], colors[mask][order]
        coords = project_to_panorama(lp, pano_width, pano_height)
        radii = np.clip(30.0 / ld, 2.0, 15.0)
        rr.set_time("frame", sequence=frame_idx)
        rr.log("/world/camera", rr.Points3D([positions[img_id]], radii=0.3,
                                            colors=[[255, 0, 0]]))
        image_path = images_path / images[img_id]["name"]
        if image_path.exists():
            rr.log("/camera/image", rr.EncodedImage(path=str(image_path)))
            rr.log("/camera/image/projected_points",
                   rr.Points2D(coords, colors=lc, radii=radii))
    logging.info(f"Visualization saved to {output_rrd}")


# ============================================================================
# Pipeline
# ============================================================================

def run(args):
    pycolmap.set_random_seed(0)
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve the panorama image directory.
    if args.input_video:
        image_dir = out / "pano_images"
        video_to_frames(Path(args.input_video), image_dir,
                        increment=args.increment, downscale=args.downscale,
                        hwaccel=args.hwaccel, select=args.select)
    elif args.input_image_path:
        image_dir = Path(args.input_image_path)
    else:
        logging.fatal("Provide --input_image_path or --input_video")
        return

    database_path = out / "database.db"
    if database_path.exists():
        database_path.unlink()

    device = pycolmap.Device.auto if args.use_gpu else pycolmap.Device.cpu

    # Feature extraction: native SPHERICAL model, one shared camera. Features
    # are unprojected to 3D bearings via CamRayFromImg downstream.
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "SPHERICAL"
    t0 = time.perf_counter()
    pycolmap.extract_features(
        database_path,
        image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        device=device,
    )
    logging.info(f"Feature extraction: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    pycolmap.match_sequential(
        database_path,
        pairing_options=pycolmap.SequentialPairingOptions(loop_detection=True),
        device=device,
    )
    logging.info(f"Sequential matching: {time.perf_counter() - t0:.2f}s")

    # Incremental mapper.
    if args.run_incremental:
        sparse = out / "sparse"
        sparse.mkdir(parents=True, exist_ok=True)
        opts = pycolmap.IncrementalPipelineOptions()
        opts.ba_use_gpu = args.use_gpu  # CUDA dense Schur for >= 50 images
        t0 = time.perf_counter()
        recs = pycolmap.incremental_mapping(database_path, image_dir, sparse, opts)
        logging.info(f"Incremental mapping: {time.perf_counter() - t0:.2f}s")
        rec = largest_reconstruction(recs)
        if rec is not None:
            logging.info(f"[incremental] {rec.summary()}")
            txt = out / "sparse_txt"
            txt.mkdir(parents=True, exist_ok=True)
            rec.write_text(str(txt))
            if args.visualize:
                generate_rerun_visualization(
                    txt, image_dir, out / "incremental.rrd",
                    "incremental_spherical")

    # Global mapper (formerly GLOMAP); BA runs on GPU by default.
    if args.run_global:
        gsparse = out / "global_sparse"
        gsparse.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        recs = pycolmap.global_mapping(
            database_path, image_dir, gsparse, pycolmap.GlobalPipelineOptions())
        logging.info(f"Global mapping: {time.perf_counter() - t0:.2f}s")
        rec = largest_reconstruction(recs)
        if rec is not None:
            logging.info(f"[global] {rec.summary()}")
            txt = out / "global_txt"
            txt.mkdir(parents=True, exist_ok=True)
            rec.write_text(str(txt))
            if args.visualize:
                generate_rerun_visualization(
                    txt, image_dir, out / "global.rrd", "global_spherical")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Native equirectangular SfM via pycolmap "
                    "(SPHERICAL model; incremental and/or global_mapper)")
    parser.add_argument("--input_image_path", type=str, default=None,
                        help="Directory of equirectangular (360°) panoramas")
    parser.add_argument("--input_video", type=str, default=None,
                        help="Input 360° video; frames extracted to "
                             "<output_path>/pano_images")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for the database and models")
    parser.add_argument("--run_incremental", action="store_true", default=True,
                        help="Run the incremental mapper")
    parser.add_argument("--no_incremental", action="store_true",
                        help="Skip the incremental mapper")
    parser.add_argument("--run_global", action="store_true", default=False,
                        help="Run the global mapper (formerly GLOMAP)")
    parser.add_argument("--increment", type=int, default=4,
                        help="Keep every Nth source frame (stride mode)")
    parser.add_argument("--select", choices=["stride", "keyframes"],
                        default="stride",
                        help="Frame selection: every Nth (stride) or codec "
                             "keyframes (fast but GOP-spaced/fewer frames)")
    parser.add_argument("--downscale", type=int, default=1,
                        help="Downscale factor for extracted frames")
    parser.add_argument("--hwaccel", choices=["auto", "cuda", "none"],
                        default="auto",
                        help="Video decode: auto (NVDEC if codec/size allow), "
                             "cuda (force NVDEC), or none (CPU)")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="Use GPU for SIFT, matching, and bundle adjustment")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Write a Rerun visualization (requires rerun-sdk)")
    args = parser.parse_args()

    if args.no_incremental:
        args.run_incremental = False
    if args.no_gpu:
        args.use_gpu = False

    run(args)
