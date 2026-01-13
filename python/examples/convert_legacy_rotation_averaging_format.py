"""
Convert legacy rotation averaging file formats to a COLMAP database.

This script provides a migration path from deprecated file-based input
to database-based input for the rotation averaging CLI.

Legacy file formats:
- Relative poses: IMAGE_NAME_1 IMAGE_NAME_2 QW QX QY QZ TX TY TZ
- Gravity priors: IMAGE_NAME GX GY GZ (optional)

After conversion, use the database with:
    colmap rotation_averager --database_path database.db --output_path output/

Usage:
    python legacy_rotation_averaging_io.py \
        --relpose_path relative_poses.txt \
        --database_path database.db \
        [--gravity_path gravity_priors.txt]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import pycolmap
from pycolmap import logging


@dataclass
class RelativePose:
    """Relative pose between two images."""

    image_name1: str
    image_name2: str
    cam2_from_cam1: pycolmap.Rigid3d


@dataclass
class GravityPrior:
    """Gravity prior for an image."""

    image_name: str
    gravity: NDArray[np.float64]


def read_relative_poses(file_path: Path | str) -> list[RelativePose]:
    """Read relative poses from a file.

    Format: IMAGE_NAME_1 IMAGE_NAME_2 QW QX QY QZ TX TY TZ

    Args:
        file_path: Path to the relative poses file.

    Returns:
        List of RelativePose objects.
    """
    file_path = Path(file_path)
    relative_poses = []

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 9:
                continue

            name1, name2 = parts[0], parts[1]
            qw, qx, qy, qz = map(float, parts[2:6])
            tx, ty, tz = map(float, parts[6:9])

            # pycolmap.Rotation3d takes quaternion in xyzw format
            rotation = pycolmap.Rotation3d(np.array([qx, qy, qz, qw]))
            translation = np.array([tx, ty, tz])
            cam2_from_cam1 = pycolmap.Rigid3d(rotation, translation)

            relative_poses.append(
                RelativePose(
                    image_name1=name1,
                    image_name2=name2,
                    cam2_from_cam1=cam2_from_cam1,
                )
            )

    return relative_poses


def read_gravity_priors(file_path: Path | str) -> list[GravityPrior]:
    """Read gravity priors from a file.

    Format: IMAGE_NAME GX GY GZ

    Args:
        file_path: Path to the gravity priors file.

    Returns:
        List of GravityPrior objects.
    """
    file_path = Path(file_path)
    gravity_priors = []

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            name = parts[0]
            gx, gy, gz = map(float, parts[1:4])

            gravity_priors.append(
                GravityPrior(
                    image_name=name,
                    gravity=np.array([gx, gy, gz], dtype=np.float64),
                )
            )

    return gravity_priors


def get_image_names_from_relative_poses(
    relative_poses: list[RelativePose],
) -> dict[str, int]:
    """Extract unique image names from relative poses and assign IDs.

    Args:
        relative_poses: List of RelativePose objects.

    Returns:
        Dictionary mapping image names to IDs.
    """
    image_names: dict[str, int] = {}
    next_id = 1  # COLMAP uses 1-based IDs

    for pose in relative_poses:
        for name in [pose.image_name1, pose.image_name2]:
            if name not in image_names:
                image_names[name] = next_id
                next_id += 1

    return image_names


def create_database_from_relative_poses(
    database_path: Path,
    relative_poses: list[RelativePose],
    gravity_priors: list[GravityPrior] | None = None,
) -> dict[str, int]:
    """Create a COLMAP database from relative poses and gravity priors.

    Args:
        database_path: Path to the output database.
        relative_poses: List of relative poses between image pairs.
        gravity_priors: Optional list of gravity priors for images.

    Returns:
        Dictionary mapping image names to their database image IDs.
    """
    image_name_to_id = get_image_names_from_relative_poses(relative_poses)

    if database_path.exists():
        database_path.unlink()

    # Use Reconstruction to create cameras, rigs, frames, and images
    # with the trivial rig/frame helper methods
    reconstruction = pycolmap.Reconstruction()
    for image_name, image_id in image_name_to_id.items():
        camera_id = image_id

        # Create camera with trivial rig (rig_id = camera_id)
        camera = pycolmap.Camera.create(
            camera_id=camera_id,
            model=pycolmap.CameraModelId.SIMPLE_PINHOLE,
            focal_length=1.0,
            width=1,
            height=1,
        )
        reconstruction.add_camera_with_trivial_rig(camera)

        # Create image with trivial frame (frame_id = image_id)
        image = pycolmap.Image()
        image.image_id = image_id
        image.name = image_name
        image.camera_id = camera_id
        reconstruction.add_image_with_trivial_frame(image)

    # Write to database
    with pycolmap.Database.open(database_path) as db:
        for camera in reconstruction.cameras.values():
            db.write_camera(camera, use_camera_id=True)

        for rig in reconstruction.rigs.values():
            db.write_rig(rig, use_rig_id=True)

        for frame in reconstruction.frames.values():
            db.write_frame(frame, use_frame_id=True)

        for image in reconstruction.images.values():
            db.write_image(image, use_image_id=True)

        # Write two-view geometries with relative poses
        for rel_pose in relative_poses:
            id1 = image_name_to_id[rel_pose.image_name1]
            id2 = image_name_to_id[rel_pose.image_name2]

            two_view_geom = pycolmap.TwoViewGeometry()
            two_view_geom.config = (
                pycolmap.TwoViewGeometryConfiguration.CALIBRATED
            )
            two_view_geom.cam2_from_cam1 = rel_pose.cam2_from_cam1

            db.write_two_view_geometry(id1, id2, two_view_geom)

        # Write gravity priors if provided
        if gravity_priors:
            gravity_by_name = {
                gp.image_name: gp.gravity for gp in gravity_priors
            }
            for image in reconstruction.images.values():
                if image.name in gravity_by_name:
                    pose_prior = pycolmap.PosePrior()
                    pose_prior.pose_prior_id = image.image_id
                    pose_prior.corr_data_id = image.data_id
                    pose_prior.gravity = gravity_by_name[image.name]
                    db.write_pose_prior(pose_prior, use_pose_prior_id=True)

    return image_name_to_id


def main():
    parser = argparse.ArgumentParser(
        description="Convert rotation averaging files to database format"
    )
    parser.add_argument(
        "--relpose_path",
        type=Path,
        required=True,
        help="Path to relative poses file",
    )
    parser.add_argument(
        "--database_path",
        type=Path,
        required=True,
        help="Path for output database",
    )
    parser.add_argument(
        "--gravity_path",
        type=Path,
        default=None,
        help="Optional path to gravity priors file",
    )
    args = parser.parse_args()

    if not args.relpose_path.exists():
        logging.error(f"Relative poses file not found: {args.relpose_path}")
        return 1

    logging.info(f"Reading relative poses from {args.relpose_path}")
    relative_poses = read_relative_poses(args.relpose_path)
    logging.info(f"Loaded {len(relative_poses)} relative poses")

    gravity_priors = None
    if args.gravity_path is not None and args.gravity_path.exists():
        logging.info(f"Reading gravity priors from {args.gravity_path}")
        gravity_priors = read_gravity_priors(args.gravity_path)
        logging.info(f"Loaded {len(gravity_priors)} gravity priors")

    logging.info(f"Creating database at {args.database_path}")
    image_name_to_id = create_database_from_relative_poses(
        args.database_path, relative_poses, gravity_priors
    )
    logging.info(f"Database created with {len(image_name_to_id)} images")

    logging.info(
        f"Conversion complete. Use the database with:\n"
        f"  colmap rotation_averager --database_path {args.database_path} "
        f"--output_path <output_path>"
    )
    return 0


if __name__ == "__main__":
    exit(main())
