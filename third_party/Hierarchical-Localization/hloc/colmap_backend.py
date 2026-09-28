"""COLMAP geometry for hloc, via this repository's colmap binary.

Feature extraction and matching stay in hloc. Cameras, keypoints, and matches
are written into a database this COLMAP build can read. Geometric verification,
triangulation, and absolute pose then call the binary instead of pycolmap.
"""

import os
import sqlite3
import subprocess
from pathlib import Path

import numpy as np

from . import logger
from .utils.io import get_keypoints, get_matches

# int32 max. ImagePairToPairId multiplies the smaller image id by this.
_MAX_NUM_IMAGES = np.iinfo(np.int32).max

_CAMERA_MODELS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}
_SENSOR_CAMERA = 0
# TwoViewGeometry::CALIBRATED. PnP RANSAC drops the outliers.
_CALIBRATED = 2


def colmap_binary() -> Path:
    env = os.environ.get("COLMAP_BIN")
    if env:
        binary = Path(env)
    else:
        root = Path(__file__).resolve().parents[3]
        binary = root / "build" / "src" / "colmap" / "exe" / "colmap"
    if not binary.is_file():
        raise FileNotFoundError(
            f"COLMAP binary not found at {binary}. Set COLMAP_BIN to this "
            "repository's build/src/colmap/exe/colmap."
        )
    return binary


def run_colmap(args, cwd=None):
    cmd = [str(colmap_binary()), *map(str, args)]
    logger.info("Running %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def _comment_lines(path):
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        yield line


def read_cameras_txt(path):
    cameras = {}
    for line in _comment_lines(path):
        camera_id, model, width, height, *params = line.split()
        cameras[int(camera_id)] = {
            "model": model,
            "model_id": _CAMERA_MODELS[model],
            "width": int(width),
            "height": int(height),
            "params": np.array(params, np.float64),
        }
    return cameras


def _image_records(path):
    """Header and POINTS2D line for each image.

    The second line is empty when an image has no observations. Skipping blank
    lines would pair two headers together.
    """
    records = []
    header = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            continue
        if header is None:
            if not line.strip():
                continue
            header = line
        else:
            records.append((header, line))
            header = None
    if header is not None:
        raise ValueError(f"{path} is not a two-line COLMAP images.txt")
    return records


def read_images_txt(path):
    """Return name -> pose/camera, preserving the file's image ids."""
    images = {}
    for header, _points in _image_records(path):
        parts = header.split()
        image_id = int(parts[0])
        qvec = np.array(parts[1:5], np.float64)
        tvec = np.array(parts[5:8], np.float64)
        camera_id = int(parts[8])
        name = parts[9]
        images[name] = {
            "image_id": image_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
        }
    return images


def read_frames_txt(path):
    frames = []
    for line in _comment_lines(path):
        parts = line.split()
        frame_id = int(parts[0])
        rig_id = int(parts[1])
        qvec = np.array(parts[2:6], np.float64)
        tvec = np.array(parts[6:9], np.float64)
        num_data = int(parts[9])
        data = parts[10:]
        sensors = []
        for i in range(num_data):
            sensor_type, sensor_id, data_id = data[3 * i : 3 * i + 3]
            sensors.append((sensor_type, int(sensor_id), int(data_id)))
        frames.append(
            {
                "frame_id": frame_id,
                "rig_id": rig_id,
                "qvec": qvec,
                "tvec": tvec,
                "sensors": sensors,
            }
        )
    return frames


def read_rigs_txt(path):
    rigs = []
    for line in _comment_lines(path):
        parts = line.split()
        rigs.append(
            {
                "rig_id": int(parts[0]),
                "ref_type": parts[2],
                "ref_id": int(parts[3]),
            }
        )
    return rigs


def model_to_txt(model_dir, txt_dir):
    txt_dir = Path(txt_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)
    run_colmap(
        [
            "model_converter",
            "--input_path",
            model_dir,
            "--output_path",
            txt_dir,
            "--output_type",
            "TXT",
        ]
    )
    return txt_dir


def txt_to_model(txt_dir, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_colmap(
        [
            "model_converter",
            "--input_path",
            txt_dir,
            "--output_path",
            model_dir,
            "--output_type",
            "BIN",
        ]
    )
    return model_dir


def export_subset_model(model_dir, output_dir, keep_names):
    """Copy a posed model down to keep_names, without the old sparse points.

    Image ids are preserved so a database written from the same text model
    lines up with guided_geometric_verifier, which matches on image id.
    """
    keep_names = set(keep_names)
    txt_dir = Path(output_dir) / "txt"
    if txt_dir.exists():
        for child in txt_dir.iterdir():
            child.unlink()
    else:
        txt_dir.mkdir(parents=True)
    model_to_txt(model_dir, txt_dir)
    images = read_images_txt(txt_dir / "images.txt")
    keep_ids = {images[name]["image_id"] for name in keep_names}
    frames = [
        frame
        for frame in read_frames_txt(txt_dir / "frames.txt")
        if any(data_id in keep_ids for _, _, data_id in frame["sensors"])
    ]
    kept_frames = []
    for frame in frames:
        sensors = [
            (sensor_type, sensor_id, data_id)
            for sensor_type, sensor_id, data_id in frame["sensors"]
            if data_id in keep_ids
        ]
        if sensors:
            frame = dict(frame)
            frame["sensors"] = sensors
            kept_frames.append(frame)

    _write_cameras_copy(txt_dir / "cameras.txt")
    _write_rigs_copy(txt_dir / "rigs.txt")
    _write_frames(txt_dir / "frames.txt", kept_frames)
    _write_images(txt_dir / "images.txt", images, keep_names)
    (txt_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as "
        "(IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0\n"
    )
    bin_dir = Path(output_dir) / "sparse"
    txt_to_model(txt_dir, bin_dir)
    return bin_dir


def _write_cameras_copy(path):
    # model_converter already wrote this file; leave it in place.
    if not Path(path).exists():
        raise FileNotFoundError(path)


def _write_rigs_copy(path):
    if not Path(path).exists():
        raise FileNotFoundError(path)


def _write_frames(path, frames):
    lines = [
        "# Frame list with one line of data per frame:",
        "#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], "
        "NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)",
        f"# Number of frames: {len(frames)}",
    ]
    for frame in frames:
        pose = " ".join(f"{v:.17g}" for v in list(frame["qvec"]) + list(frame["tvec"]))
        sensors = " ".join(
            f"{sensor_type} {sensor_id} {data_id}"
            for sensor_type, sensor_id, data_id in frame["sensors"]
        )
        lines.append(
            f"{frame['frame_id']} {frame['rig_id']} {pose} {len(frame['sensors'])} {sensors}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def _write_images(path, images, keep_names):
    kept = [images[name] | {"name": name} for name in images if name in keep_names]
    kept.sort(key=lambda image: image["image_id"])
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(kept)}",
    ]
    for image in kept:
        pose = " ".join(
            f"{v:.17g}" for v in list(image["qvec"]) + list(image["tvec"])
        )
        lines.append(
            f"{image['image_id']} {pose} {image['camera_id']} {image['name']}"
        )
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")


def create_database(database_path):
    database_path = Path(database_path)
    if database_path.exists():
        database_path.unlink()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    run_colmap(["database_creator", "--database_path", database_path])


def _connect(database_path):
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def fill_database_from_model(database_path, txt_dir, keep_names):
    """Insert the posed subset with the same ids as the text model."""
    txt_dir = Path(txt_dir)
    cameras = read_cameras_txt(txt_dir / "cameras.txt")
    images = read_images_txt(txt_dir / "images.txt")
    rigs = read_rigs_txt(txt_dir / "rigs.txt")
    frames = read_frames_txt(txt_dir / "frames.txt")
    keep_names = set(keep_names) if keep_names is not None else set(images)
    images = {name: image for name, image in images.items() if name in keep_names}
    keep_ids = {image["image_id"] for image in images.values()}
    frames = [
        frame
        for frame in frames
        if any(data_id in keep_ids for _, _, data_id in frame["sensors"])
    ]

    connection = _connect(database_path)
    with connection:
        for camera_id, camera in cameras.items():
            connection.execute(
                "INSERT INTO cameras(camera_id, model, width, height, params, "
                "prior_focal_length) VALUES (?, ?, ?, ?, ?, 1)",
                (
                    camera_id,
                    camera["model_id"],
                    camera["width"],
                    camera["height"],
                    camera["params"].tobytes(),
                ),
            )
        for rig in rigs:
            connection.execute(
                "INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) "
                "VALUES (?, ?, ?)",
                (rig["rig_id"], rig["ref_id"], _SENSOR_CAMERA),
            )
        for frame in frames:
            connection.execute(
                "INSERT INTO frames(frame_id, rig_id) VALUES (?, ?)",
                (frame["frame_id"], frame["rig_id"]),
            )
            for _sensor_type, sensor_id, data_id in frame["sensors"]:
                if data_id not in keep_ids:
                    continue
                connection.execute(
                    "INSERT INTO frame_data(frame_id, data_id, sensor_id, "
                    "sensor_type) VALUES (?, ?, ?, ?)",
                    (frame["frame_id"], data_id, sensor_id, _SENSOR_CAMERA),
                )
        for name, image in images.items():
            connection.execute(
                "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
                (image["image_id"], name, image["camera_id"]),
            )
    connection.close()
    return {name: image["image_id"] for name, image in images.items()}


def add_query_images(database_path, query_names, rig_id=1, camera_id=1):
    """Append query images on the map's rig. They have no pose yet."""
    connection = _connect(database_path)
    next_image_id = connection.execute("SELECT MAX(image_id) FROM images").fetchone()[0] + 1
    next_frame_id = connection.execute("SELECT MAX(frame_id) FROM frames").fetchone()[0] + 1
    connection.close()
    connection = _connect(database_path)
    name_to_id = {}
    with connection:
        for name in query_names:
            image_id = next_image_id
            frame_id = next_frame_id
            next_image_id += 1
            next_frame_id += 1
            connection.execute(
                "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
                (image_id, name, camera_id),
            )
            connection.execute(
                "INSERT INTO frames(frame_id, rig_id) VALUES (?, ?)",
                (frame_id, rig_id),
            )
            connection.execute(
                "INSERT INTO frame_data(frame_id, data_id, sensor_id, sensor_type) "
                "VALUES (?, ?, ?, ?)",
                (frame_id, image_id, camera_id, _SENSOR_CAMERA),
            )
            name_to_id[name] = image_id
    connection.close()
    return name_to_id


def write_keypoints(database_path, name_to_id, features_path):
    connection = _connect(database_path)
    with connection:
        for name, image_id in name_to_id.items():
            keypoints = np.asarray(get_keypoints(features_path, name), np.float32)
            keypoints = keypoints + np.float32(0.5)
            blob = np.ascontiguousarray(keypoints[:, :2])
            connection.execute(
                "INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (image_id, blob.shape[0], blob.shape[1], blob.tobytes()),
            )
    connection.close()


def _pair_id(image_id1, image_id2):
    if image_id1 == image_id2:
        raise ValueError("A match pair needs two different images")
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
        swapped = True
    else:
        swapped = False
    return int(_MAX_NUM_IMAGES) * image_id1 + image_id2, swapped


def write_matches(database_path, name_to_id, pairs_path, matches_path, min_score=None):
    pairs = _read_pairs(pairs_path)
    connection = _connect(database_path)
    written = set()
    with connection:
        for name0, name1 in pairs:
            id0, id1 = name_to_id[name0], name_to_id[name1]
            pair_id, swapped = _pair_id(id0, id1)
            if pair_id in written:
                continue
            matches, scores = get_matches(matches_path, name0, name1)
            if min_score is not None:
                matches = matches[scores > min_score]
            if swapped:
                matches = matches[:, ::-1]
            matches = np.ascontiguousarray(matches, np.uint32)
            connection.execute(
                "INSERT INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (pair_id, matches.shape[0], 2, matches.tobytes()),
            )
            written.add(pair_id)
    connection.close()
    return len(written)


def write_query_geometries(database_path, name_to_id, pairs_path, matches_path):
    """Store LightGlue matches as calibrated inliers.

    The query has no pose, so guided verification cannot use one. RegisterNextImage
    runs P3P RANSAC on these correspondences.
    """
    pairs = _read_pairs(pairs_path)
    connection = _connect(database_path)
    written = set()
    with connection:
        for name0, name1 in pairs:
            id0, id1 = name_to_id[name0], name_to_id[name1]
            pair_id, swapped = _pair_id(id0, id1)
            if pair_id in written:
                continue
            matches, _scores = get_matches(matches_path, name0, name1)
            if swapped:
                matches = matches[:, ::-1]
            matches = np.ascontiguousarray(matches, np.uint32)
            if matches.shape[0] == 0:
                continue
            connection.execute(
                "INSERT INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (pair_id, matches.shape[0], 2, matches.tobytes()),
            )
            connection.execute(
                "INSERT INTO two_view_geometries(pair_id, rows, cols, data, config, "
                "F, E, H, qvec, tvec, camera1, camera2) "
                "VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
                (pair_id, matches.shape[0], 2, matches.tobytes(), _CALIBRATED),
            )
            written.add(pair_id)
    connection.close()
    return len(written)


def _read_pairs(pairs_path):
    pairs = []
    for line in Path(pairs_path).read_text().splitlines():
        if not line.strip():
            continue
        name0, name1 = line.split()
        pairs.append((name0, name1))
    return pairs


def guided_geometric_verification(database_path, reference_model):
    run_colmap(
        [
            "guided_geometric_verifier",
            "--database_path",
            database_path,
            "--input_path",
            reference_model,
            "--TwoViewGeometry.min_num_inliers",
            15,
            "--TwoViewGeometry.max_error",
            4,
        ]
    )


def triangulate(database_path, image_dir, input_model, output_model):
    output_model = Path(output_model)
    output_model.mkdir(parents=True, exist_ok=True)
    run_colmap(
        [
            "point_triangulator",
            "--database_path",
            database_path,
            "--image_path",
            image_dir,
            "--input_path",
            input_model,
            "--output_path",
            output_model,
            "--clear_points",
            1,
            "--refine_intrinsics",
            0,
            "--Mapper.ba_refine_focal_length",
            0,
            "--Mapper.ba_refine_principal_point",
            0,
            "--Mapper.ba_refine_extra_params",
            0,
        ]
    )
    return output_model


def register_queries(database_path, image_dir, input_model, output_model):
    """Register every database image that is not already in input_model.

    Existing frames stay fixed. Each new image is posed by P3P RANSAC inside
    this COLMAP's incremental mapper.
    """
    output_model = Path(output_model)
    output_model.mkdir(parents=True, exist_ok=True)
    run_colmap(
        [
            "mapper",
            "--database_path",
            database_path,
            "--image_path",
            image_dir,
            "--input_path",
            input_model,
            "--output_path",
            output_model,
            "--Mapper.fix_existing_frames",
            1,
            "--Mapper.ba_refine_focal_length",
            0,
            "--Mapper.ba_refine_principal_point",
            0,
            "--Mapper.ba_refine_extra_params",
            0,
            "--Mapper.abs_pose_max_error",
            12,
            "--Mapper.abs_pose_min_num_inliers",
            15,
            "--Mapper.multiple_models",
            0,
        ]
    )
    return output_model


def incremental_mapping(database_path, image_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_colmap(
        [
            "mapper",
            "--database_path",
            database_path,
            "--image_path",
            image_dir,
            "--output_path",
            output_dir,
            "--Mapper.ba_refine_principal_point",
            0,
            "--Mapper.multiple_models",
            0,
        ]
    )
    return output_dir


def geometric_verification(database_path):
    run_colmap(
        [
            "geometric_verifier",
            "--database_path",
            database_path,
            "--TwoViewGeometry.min_num_inliers",
            15,
            "--TwoViewGeometry.max_error",
            4,
        ]
    )
