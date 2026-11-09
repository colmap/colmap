# SPDX-License-Identifier: BSD-3-Clause

"""Build the eth3d-undistorted benchmark from the distorted ETH3D DSLR images.

The official undistorted ETH3D images keep the principal point of each
distorted GT camera. Their size thus differs between GT cameras and their
principal point is off-center by about 13 px on average. Here, the distorted
JPEGs are instead undistorted by COLMAP to a single PINHOLE camera shared by
all images of all scenes, with the principal point at the image center. The GT
poses and 3D points are unchanged, and the GT 2D observations are undistorted
together with the images.

Example (from benchmark/reconstruction):
  python download.py --datasets eth3d-distorted
  python -m evaluation.package_eth3d_undistorted \\
    --input_path data/eth3d-distorted --output_path data/eth3d-undistorted \\
    --archive_path data/eth3d-undistorted-archives
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import pycolmap

RELEASE_REPOSITORY = "colmap/colmap"
RELEASE_TAG = "datasets-eth3d-undistorted-v1"
ARCHIVE_PREFIX = "eth3d-undistorted-v1"
CHECKSUMS_PATH = Path(__file__).with_name("eth3d_undistorted_checksums.json")

# Over all distorted GT cameras of all scenes, the largest centered 3:2 image
# without blank pixels is 6141x4094 at the median GT focal length of 3412 px.
TARGET_WIDTH = 6120
TARGET_HEIGHT = 4080
TARGET_FOCAL_LENGTH = 3410.0
JPEG_QUALITY = 97

SOURCE_CALIBRATION = "dslr_calibration_jpg"
TARGET_CALIBRATION = "dslr_calibration_undistorted"

README = """\
# eth3d-undistorted

This package is derived from the distorted DSLR JPEG images of the ETH3D
high-resolution multi-view benchmark for evaluating structure-from-motion in
COLMAP. All images are undistorted to the same PINHOLE camera with the
principal point at the image center and stored as quality-{quality} JPEG. The
ground-truth poses and 3D points are those of the distorted images, and the 2D
observations are undistorted together with the images.

Dataset: https://www.eth3d.net/

ETH3D is distributed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International license
(CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/). This
derived package is distributed under the same license. Cite the original
dataset when publishing results based on this package:

> Thomas Schöps et al., "A Multi-View Stereo Benchmark with High-Resolution
> Images and Multi-Camera Videos," CVPR 2017.
"""


def create_target_camera() -> pycolmap.Camera:
    return pycolmap.Camera(
        camera_id=1,
        model="PINHOLE",
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
        params=[
            TARGET_FOCAL_LENGTH,
            TARGET_FOCAL_LENGTH,
            TARGET_WIDTH / 2,
            TARGET_HEIGHT / 2,
        ],
    )


def check_no_blank_pixels(
    source_camera: pycolmap.Camera, target_camera: pycolmap.Camera
) -> None:
    """Checks that the border of the target image maps inside the source."""
    w, h = target_camera.width, target_camera.height
    xs, ys = np.arange(w + 1.0), np.arange(h + 1.0)
    border = np.concatenate(
        [
            np.stack([xs, np.zeros_like(xs)], axis=1),
            np.stack([xs, np.full_like(xs, h)], axis=1),
            np.stack([np.zeros_like(ys), ys], axis=1),
            np.stack([np.full_like(ys, w), ys], axis=1),
        ]
    )
    rays = target_camera.cam_from_img(border)
    mapped = source_camera.img_from_cam(
        np.concatenate([rays, np.ones((len(rays), 1))], axis=1)
    )
    # Keep a margin of one pixel for bilinear interpolation.
    if not np.all(
        (mapped >= 1)
        & (mapped <= [source_camera.width - 1, source_camera.height - 1])
    ):
        raise ValueError(f"Blank pixels for source camera {source_camera}")


def merge_cameras(
    reconstruction: pycolmap.Reconstruction, camera: pycolmap.Camera
) -> pycolmap.Reconstruction:
    """Returns a copy of the reconstruction with all images in one camera."""
    merged = pycolmap.Reconstruction()
    merged.add_camera_with_trivial_rig(camera)
    for image_id, image in reconstruction.images.items():
        keypoints = np.array([p.xy for p in image.points2D]).reshape(-1, 2)
        merged.add_image_with_trivial_frame(
            pycolmap.Image(
                name=image.name,
                keypoints=keypoints,
                camera_id=camera.camera_id,
                image_id=image_id,
            ),
            image.cam_from_world(),
        )
    # Keep the GT point IDs, and update the errors for the merged camera.
    for point3D_id, point3D in reconstruction.points3D.items():
        merged.add_point3D_with_id(point3D_id, point3D)
    merged.update_point_3d_errors()
    return merged


def undistort_scene(
    input_scene_path: Path,
    output_scene_path: Path,
    target_camera: pycolmap.Camera,
    num_threads: int = -1,
) -> None:
    source_path = input_scene_path / SOURCE_CALIBRATION
    for camera in pycolmap.Reconstruction(source_path).cameras.values():
        check_no_blank_pixels(camera, target_camera)

    options = pycolmap.UndistortCameraOptions()
    options.target_camera = target_camera
    output_scene_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_scene_path) as tmp_dir:
        pycolmap.undistort_images(
            output_path=tmp_dir,
            input_path=source_path,
            image_path=input_scene_path / "images",
            undistort_options=options,
            jpeg_quality=JPEG_QUALITY,
            num_threads=num_threads,
        )
        # The undistorter keeps one copy of the target camera per GT camera.
        undistorted = pycolmap.Reconstruction(Path(tmp_dir) / "sparse")
        calibration_path = output_scene_path / TARGET_CALIBRATION
        calibration_path.mkdir(exist_ok=True)
        merge_cameras(undistorted, target_camera).write_text(calibration_path)
        if (output_scene_path / "images").exists():
            shutil.rmtree(output_scene_path / "images")
        shutil.move(Path(tmp_dir) / "images", output_scene_path / "images")


def package_scene(
    dataset_path: Path, scene: str, archive_path: Path
) -> tuple[str, str]:
    """Writes the archive of a scene and returns its name and SHA256."""
    # Only required for packaging, unlike the rest of the module.
    import py7zr

    archive_path.mkdir(parents=True, exist_ok=True)
    filename = f"{ARCHIVE_PREFIX}-{scene}.7z"
    # JPEGs do not compress further, so only copy them into the archive.
    with py7zr.SevenZipFile(
        archive_path / filename, "w", filters=[{"id": py7zr.FILTER_COPY}]
    ) as archive:
        archive.writestr(README.format(quality=JPEG_QUALITY), "README.md")
        archive.writeall(dataset_path / scene, scene)
    digest = hashlib.sha256()
    with open(archive_path / filename, "rb") as fid:
        while chunk := fid.read(1024 * 1024):
            digest.update(chunk)
    return filename, digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Path to the eth3d-distorted dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to the eth3d-undistorted dataset to write.",
    )
    parser.add_argument(
        "--archive_path",
        type=Path,
        default=None,
        help="If set, write one release archive per scene to this path and "
        "update the release checksums.",
    )
    parser.add_argument("--scenes", nargs="+", default=[])
    parser.add_argument("--num_threads", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path / "dslr"
    output_path = args.output_path / "dslr"
    scenes = args.scenes or sorted(
        path.name
        for path in input_path.iterdir()
        if (path / SOURCE_CALIBRATION).is_dir()
    )
    checksums = (
        json.loads(CHECKSUMS_PATH.read_text())
        if CHECKSUMS_PATH.exists()
        else {}
    )
    for scene in scenes:
        pycolmap.logging.info(f"Processing scene {scene}")
        undistort_scene(
            input_path / scene,
            output_path / scene,
            create_target_camera(),
            args.num_threads,
        )
        if args.archive_path is not None:
            filename, checksum = package_scene(
                output_path, scene, args.archive_path
            )
            checksums[filename] = checksum
    if args.archive_path is not None:
        CHECKSUMS_PATH.write_text(
            json.dumps(dict(sorted(checksums.items())), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
