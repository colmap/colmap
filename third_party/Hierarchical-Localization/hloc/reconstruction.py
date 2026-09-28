import argparse
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from . import logger
from .colmap_backend import (
    create_database,
    geometric_verification,
)
from .colmap_backend import (
    incremental_mapping as run_incremental_mapper,
)
from .triangulation import OutputCapture, import_features, import_matches

# COLMAP's prior when EXIF has no focal length: 1.2 * max(width, height).
_PINHOLE = 1


def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
    logger.info("Creating an empty database...")
    create_database(database_path)


def import_images(
    image_dir: Path,
    database_path: Path,
    camera_mode: str = "AUTO",
    image_list: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,
):
    """Register images with this COLMAP's database schema.

    A shared PINHOLE is used when ``options`` contains ``camera_params``.
    Otherwise each image gets COLMAP's missing-EXIF focal prior. ``camera_mode``
    is accepted so the old command line still parses; PER_IMAGE and AUTO both
    take the per-image prior when no parameters are given.
    """
    del camera_mode
    logger.info("Importing images into the database...")
    options = options or {}
    names = image_list
    if not names:
        names = sorted(
            path.name
            for path in Path(image_dir).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        )
    if not names:
        raise IOError(f"No images found in {image_dir}.")

    shared = options.get("camera_params")
    connection = sqlite3.connect(database_path)
    with connection:
        if shared:
            params = [float(value) for value in str(shared).split(",")]
            sample = cv2.imread(str(Path(image_dir) / names[0]))
            if sample is None:
                raise IOError(f"Cannot read {names[0]}")
            height, width = sample.shape[:2]
            connection.execute(
                "INSERT INTO cameras(camera_id, model, width, height, params, "
                "prior_focal_length) VALUES (1, ?, ?, ?, ?, 1)",
                (_PINHOLE, width, height, _params_blob(params)),
            )
            connection.execute(
                "INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) "
                "VALUES (1, 1, 0)"
            )
        for index, name in enumerate(names, start=1):
            image = cv2.imread(str(Path(image_dir) / name))
            if image is None:
                raise IOError(f"Cannot read {name}")
            height, width = image.shape[:2]
            if shared:
                camera_id = 1
                rig_id = 1
            else:
                focal = 1.2 * max(width, height)
                params = [focal, focal, width / 2.0, height / 2.0]
                camera_id = index
                rig_id = index
                connection.execute(
                    "INSERT INTO cameras(camera_id, model, width, height, params, "
                    "prior_focal_length) VALUES (?, ?, ?, ?, ?, 0)",
                    (camera_id, _PINHOLE, width, height, _params_blob(params)),
                )
                connection.execute(
                    "INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) "
                    "VALUES (?, ?, 0)",
                    (rig_id, camera_id),
                )
            connection.execute(
                "INSERT INTO frames(frame_id, rig_id) VALUES (?, ?)",
                (index, rig_id),
            )
            connection.execute(
                "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
                (index, name, camera_id),
            )
            connection.execute(
                "INSERT INTO frame_data(frame_id, data_id, sensor_id, sensor_type) "
                "VALUES (?, ?, ?, 0)",
                (index, index, camera_id),
            )
    connection.close()


def _params_blob(params):
    import numpy as np

    return np.asarray(params, np.float64).tobytes()


def get_image_ids(database_path: Path) -> Dict[str, int]:
    connection = sqlite3.connect(database_path)
    rows = connection.execute("SELECT name, image_id FROM images").fetchall()
    connection.close()
    return {name: image_id for name, image_id in rows}


def run_reconstruction(
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    verbose: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> Path:
    del options
    models_path = sfm_dir / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info("Running 3D reconstruction...")
    with OutputCapture(verbose):
        run_incremental_mapper(database_path, image_dir, models_path)

    model_dir = models_path / "0"
    if not model_dir.is_dir():
        logger.error("Could not reconstruct any model!")
        return None
    for filename in [
        "images.bin",
        "cameras.bin",
        "points3D.bin",
        "frames.bin",
        "rigs.bin",
    ]:
        source = model_dir / filename
        if not source.exists():
            continue
        target = sfm_dir / filename
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))
    return sfm_dir


def main(
    sfm_dir: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    camera_mode: str = "AUTO",
    verbose: bool = False,
    skip_geometric_verification: bool = False,
    min_match_score: Optional[float] = None,
    image_list: Optional[List[str]] = None,
    image_options: Optional[Dict[str, Any]] = None,
    mapper_options: Optional[Dict[str, Any]] = None,
) -> Path:
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches
    del mapper_options

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"

    create_empty_db(database)
    import_images(image_dir, database, camera_mode, image_list, image_options)
    image_ids = get_image_ids(database)
    import_features(database, features, image_ids)
    import_matches(
        database,
        image_ids,
        pairs,
        matches,
        min_match_score,
        skip_geometric_verification,
    )
    if not skip_geometric_verification:
        logger.info("Performing geometric verification of the matches...")
        geometric_verification(database)
    reconstruction = run_reconstruction(sfm_dir, database, image_dir, verbose)
    if reconstruction is not None:
        logger.info("Reconstruction written to %s", reconstruction)
    return reconstruction


def _parse_cli_options(args: List[str]) -> Dict[str, Any]:
    options = {}
    for arg in args:
        key, value = arg.split("=", 1)
        options[key] = eval(value)
    return options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm_dir", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument(
        "--camera_mode",
        type=str,
        default="AUTO",
        choices=["AUTO", "SINGLE", "PER_IMAGE", "PER_FOLDER"],
    )
    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--min_match_score", type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--image_options", nargs="+", default=[])
    parser.add_argument("--mapper_options", nargs="+", default=[])
    args = parser.parse_args().__dict__
    args["image_options"] = _parse_cli_options(args.pop("image_options"))
    args["mapper_options"] = _parse_cli_options(args.pop("mapper_options"))
    main(**args)
