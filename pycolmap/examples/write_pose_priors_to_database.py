"""
Python reimplementation of the C++ incremental mapper with equivalent logic.
"""

import argparse
from pathlib import Path

import numpy as np

import pycolmap
from pycolmap import logging


def get_pose_prior_from_line(
    line: str,
    delimiter: str,
    coordinate_system: int,
    position_covariance: np.array,
):
    els = line.split(delimiter)

    if len(els) == 13 and position_covariance is None:
        # Get image_name, postion and covariance (in row major format)
        image_name = els[0]
        position = np.array([float(els[1]), float(els[2]), float(els[3])])
        covariance_values = list(map(float, els[4:13]))
        position_covariance = np.array(covariance_values).reshape(3, 3)
        return (image_name, position, coordinate_system, position_covariance)
    elif len(els) >= 4 and position_covariance is not None:
        # Get image_name, position
        image_name = els[0]
        position = np.array([float(els[1]), float(els[2]), float(els[3])])
        return (image_name, position, coordinate_system, position_covariance)
    else:
        print("ERROR: Pose priors file lines should contain 4 or 13 elements.")
        print("Current line contains: {0}: #{1} elements".format(els, len(els)))

    return None


def update_pose_prior_from_image_name(
    colmap_db: pycolmap.Database,
    image_name: str,
    position: np.array,
    coordinate_system: int = -1,
    position_covariance: np.array = None,
):
    """
    Update the pose prior for a specific image name in the database.
    If the pose prior doesn't exist, insert a new one.

    Args:
        colmap_db (pycolmap.Database): colmap database to update.
        image_name (str): name of the image to update.
        position (np.array): Position as a 3-element array (x, y, z).
        coordinate_system (int): Coordinate system index (default: -1).
        position_covariance (np.array): 3x3 position covariance matrix (default: None).
    """
    # Get image_id from image_name
    if colmap_db.exists_image(image_name):
        position = np.asarray(position, dtype=np.float64)
        if position_covariance is None:
            position_covariance = np.full((3, 3), np.nan, dtype=np.float64)
        image = colmap_db.read_image(image_name)
        if colmap_db.exists_pose_prior(image.image_id):
            colmap_db.update_pose_prior(image.image_id, pycolmap.PosePrior(position, position_covariance, coordinate_system))
        else:
            colmap_db.write_pose_prior(image.image_id, pycolmap.PosePrior(position, position_covariance, coordinate_system))
    else:
        logging.warning(f"Image at path {image_name} not found in database.")


def write_pose_priors_to_database():
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, required=True)
    parser.add_argument("--pose_priors_path", type=str, required=True)
    parser.add_argument("--pose_priors_delimiter", type=str, default=" ")
    parser.add_argument(
        "--coordinate_system",
        type=int,
        default=0,
        help="(-1: unknwon, 0: WGS84, 1: Cartesian)",
    )
    parser.add_argument(
        "--use_covariance_from_pose_priors_file",
        type=bool,
        default=False,
        help="If False, use prior_position_std options to set a common covariance to the priors.",
    )
    parser.add_argument("--prior_position_std_x", type=float, default=1.0)
    parser.add_argument("--prior_position_std_y", type=float, default=1.0)
    parser.add_argument("--prior_position_std_z", type=float, default=1.0)
    args = parser.parse_args()

    database_path = Path(args.database_path)
    pose_priors_path = Path(args.pose_priors_path)

    if not database_path.exists():
        print("ERROR: database path does not exist.")
        return

    if not pose_priors_path.exists():
        print("ERROR: pose priors path already does not exist.")
        return

    colmap_db = pycolmap.Database(database_path)

    # Setup covariance matrix if required
    position_covariance = None
    if args.use_covariance_from_pose_priors_file is False:
        position_covariance = np.diag(
            [
                args.prior_position_std_x**2,
                args.prior_position_std_y**2,
                args.prior_position_std_z**2,
            ]
        )

    # Add pose priors from file.
    pose_prior_file = open(args.pose_priors_path, "r")
    for line in pose_prior_file:
        if line[0] == "#":
            continue
        pose_prior = get_pose_prior_from_line(
            line,
            args.pose_priors_delimiter,
            args.coordinate_system,
            position_covariance,
        )
        if pose_prior is not None:
            update_pose_prior_from_image_name(colmap_db, *pose_prior)

    # Close database.
    colmap_db.close()


if __name__ == "__main__":
    write_pose_priors_to_database()
