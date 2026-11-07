# SPDX-License-Identifier: BSD-3-Clause

"""Build the eth3d-centered benchmark from the distorted ETH3D DSLR images.

The official undistorted ETH3D images keep the principal point of each
distorted GT camera. Their size thus differs between GT cameras and their
principal point is off-center by about 13 px on average. Here, the distorted
JPEGs are instead warped to a single PINHOLE camera shared by all images of
all scenes, with the principal point at the image center. The GT poses and 3D
points are unchanged, and the GT 2D observations are mapped to the new camera.
Of the EXIF metadata, only the orientation is kept: the EXIF focal length of
the distorted images does not apply to the warped images.

Example (from benchmark/reconstruction):
  python download.py --datasets eth3d-distorted
  python -m evaluation.package_eth3d_centered \\
    --input_path data/eth3d-distorted --output_path data/eth3d-centered \\
    --archive_path data/eth3d-centered-archives
"""

import argparse
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
import py7zr

import pycolmap

RELEASE_REPOSITORY = "colmap/colmap"
RELEASE_TAG = "datasets-eth3d-centered-v1"
ARCHIVE_PREFIX = "eth3d-centered-v1"
CHECKSUMS_PATH = Path(__file__).with_name("eth3d_centered_checksums.json")

# Over all distorted GT cameras of all scenes, the largest centered 3:2 image
# without blank pixels is 6141x4094 at the median GT focal length of 3412 px.
TARGET_WIDTH = 6120
TARGET_HEIGHT = 4080
TARGET_FOCAL_LENGTH = 3410.0
JPEG_QUALITY = 97

SOURCE_CALIBRATION = "dslr_calibration_jpg"
TARGET_CALIBRATION = "dslr_calibration_undistorted"
TARGET_IMAGE_DIR = "dslr_images_undistorted"

README = """\
# eth3d-centered

This package is derived from the distorted DSLR JPEG images of the ETH3D
high-resolution multi-view benchmark for evaluating structure-from-motion in
COLMAP. All images are warped to the same PINHOLE camera with the principal
point at the image center and stored as quality-{quality} JPEG. Only the EXIF
orientation is kept. The ground-truth poses and 3D points are those of the
distorted images, and the 2D observations are mapped to the new camera.

Dataset: https://www.eth3d.net/

ETH3D is distributed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International license
(CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/). This
derived package is distributed under the same license. Cite the original
dataset when publishing results based on this package:

> Thomas Schöps et al., "A Multi-View Stereo Benchmark with High-Resolution
> Images and Multi-Camera Videos," CVPR 2017.
"""


def create_target_camera(camera_id: int = 1) -> pycolmap.Camera:
    return pycolmap.Camera(
        camera_id=camera_id,
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


def map_pixels(
    source_camera: pycolmap.Camera,
    target_camera: pycolmap.Camera,
    points2D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Maps pixels between two cameras with the same pose.

    Returns the mapped pixels and whether they are inside the target image.
    """
    rays = source_camera.cam_from_img(points2D.reshape(-1, 2))
    rays = np.concatenate([rays, np.ones((len(rays), 1))], axis=1)
    mapped = target_camera.img_from_cam(rays).reshape(-1, 2)
    valid = np.all(np.isfinite(mapped), axis=1)
    valid[valid] &= np.all(
        (mapped[valid] >= 0)
        & (mapped[valid] < [target_camera.width, target_camera.height]),
        axis=1,
    )
    return mapped, valid


def check_no_blank_pixels(
    source_camera: pycolmap.Camera,
    target_camera: pycolmap.Camera,
    margin: float = 2.0,
) -> None:
    """Checks that the target image border maps inside the source image.

    The margin keeps bilinear interpolation away from the source border.
    """
    w, h = target_camera.width, target_camera.height
    xs = np.arange(w + 1, dtype=float)
    ys = np.arange(h + 1, dtype=float)
    border = np.concatenate(
        [
            np.stack([xs, np.zeros_like(xs)], axis=1),
            np.stack([xs, np.full_like(xs, h)], axis=1),
            np.stack([np.zeros_like(ys), ys], axis=1),
            np.stack([np.full_like(ys, w), ys], axis=1),
        ]
    )
    mapped, _ = map_pixels(target_camera, source_camera, border)
    if not (
        np.all(np.isfinite(mapped))
        and np.all(mapped >= margin)
        and np.all(
            mapped
            <= [source_camera.width - margin, source_camera.height - margin]
        )
    ):
        raise ValueError(
            f"Target camera {target_camera} has blank pixels "
            f"for source camera {source_camera}"
        )


def undistort_reconstruction(
    source: pycolmap.Reconstruction, target_camera: pycolmap.Camera
) -> pycolmap.Reconstruction:
    """Maps a reconstruction to the target camera with the same poses.

    Observations that fall outside of the target image are dropped, and so
    are the 3D points that are then observed by fewer than two images.
    """
    target = pycolmap.Reconstruction()
    target.add_camera_with_trivial_rig(target_camera)
    track_elements: dict[int, list[pycolmap.TrackElement]] = (
        collections.defaultdict(list)
    )
    for image_id, image in sorted(source.images.items()):
        observations = [p for p in image.points2D if p.has_point3D()]
        points2D = np.array([p.xy for p in observations]).reshape(-1, 2)
        mapped, valid = map_pixels(
            source.cameras[image.camera_id], target_camera, points2D
        )
        target_image = pycolmap.Image(
            name=f"{TARGET_IMAGE_DIR}/{Path(image.name).name}",
            keypoints=mapped[valid],
            camera_id=target_camera.camera_id,
            image_id=image_id,
        )
        target.add_image_with_trivial_frame(
            target_image, image.cam_from_world()
        )
        point3D_ids = [
            p.point3D_id for p, v in zip(observations, valid, strict=True) if v
        ]
        for point2D_idx, point3D_id in enumerate(point3D_ids):
            track_elements[point3D_id].append(
                pycolmap.TrackElement(image_id, point2D_idx)
            )
    for point3D_id, elements in sorted(track_elements.items()):
        if len(elements) < 2:
            continue
        point3D = source.points3D[point3D_id]
        target.add_point3D(point3D.xyz, pycolmap.Track(elements), point3D.color)
    return target


def undistort_image(
    source_path: Path,
    target_path: Path,
    source_camera: pycolmap.Camera,
    target_camera: pycolmap.Camera,
) -> None:
    source_image = pycolmap.Bitmap.read(source_path, as_rgb=True)
    if source_image is None:
        raise RuntimeError(f"Failed to read {source_path}")
    if (source_image.width, source_image.height) != (
        source_camera.width,
        source_camera.height,
    ):
        raise RuntimeError(f"Unexpected image size of {source_path}")
    target_image = pycolmap.warp_image_between_cameras(
        pycolmap.WarpImageOptions(),
        source_camera,
        target_camera,
        source_image,
    )
    orientation = source_image.exif_orientation()
    if orientation is not None:
        target_image.set_metadata("Orientation", orientation)
    target_image.set_jpeg_quality(JPEG_QUALITY)
    if not target_image.write(target_path):
        raise RuntimeError(f"Failed to write {target_path}")


def process_scene(
    input_scene_path: Path, output_scene_path: Path, num_threads: int
) -> None:
    source = pycolmap.Reconstruction(input_scene_path / SOURCE_CALIBRATION)
    target_camera = create_target_camera()
    for camera in source.cameras.values():
        check_no_blank_pixels(camera, target_camera)

    target = undistort_reconstruction(source, target_camera)
    calibration_path = output_scene_path / TARGET_CALIBRATION
    calibration_path.mkdir(parents=True, exist_ok=True)
    target.write_text(calibration_path)

    image_path = output_scene_path / "images" / TARGET_IMAGE_DIR
    image_path.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
        futures = [
            executor.submit(
                undistort_image,
                input_scene_path / "images" / image.name,
                image_path / Path(image.name).name,
                source.cameras[image.camera_id],
                target_camera,
            )
            for image in source.images.values()
        ]
        for future in futures:
            future.result()


def package_scene(
    dataset_path: Path, scene: str, archive_path: Path
) -> tuple[str, str]:
    """Writes the archive of a scene and returns its name and SHA256."""
    archive_path.mkdir(parents=True, exist_ok=True)
    filename = f"{ARCHIVE_PREFIX}-{scene}.7z"
    readme_path = archive_path / "README.md"
    readme_path.write_text(README.format(quality=JPEG_QUALITY))
    # JPEGs do not compress further, so only copy them into the archive.
    with py7zr.SevenZipFile(
        archive_path / filename, "w", filters=[{"id": py7zr.FILTER_COPY}]
    ) as archive:
        archive.write(readme_path, "README.md")
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
        help="Path to the eth3d-centered dataset to write.",
    )
    parser.add_argument(
        "--archive_path",
        type=Path,
        default=None,
        help="If set, write one release archive per scene to this path and "
        "update the release checksums.",
    )
    parser.add_argument("--scenes", nargs="+", default=[])
    parser.add_argument("--num_threads", type=int, default=8)
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
        process_scene(input_path / scene, output_path / scene, args.num_threads)
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
