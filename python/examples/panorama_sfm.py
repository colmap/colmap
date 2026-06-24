"""
An example for running incremental SfM on 360 spherical panorama images.

Two modes are available via --pano_render_type:
  * "perspective_overlapping" / "perspective_non_overlapping": render the
    panoramas into a rig of perspective virtual cameras (cubemap-style)
    and reconstruct from those.
  * "spherical": reconstruct directly on the panoramas using the native
    equirectangular camera model, without rendering perspective images.
The "perspective_*" modes are generally more accurate but slower. For these
modes, the script additionally writes a reconstruction that maps the perspective
virtual cameras back onto the original equirectangular input images.
"""

import argparse
import collections
import enum
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal, TypeVar, cast

import cv2
import numpy as np
import numpy.typing as npt
import PIL.ExifTags
import PIL.Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pycolmap
from pycolmap import logging


class Matcher(enum.StrEnum):
    SEQUENTIAL = enum.auto()
    EXHAUSTIVE = enum.auto()
    VOCABTREE = enum.auto()
    SPATIAL = enum.auto()


class Mapper(enum.StrEnum):
    INCREMENTAL = enum.auto()
    GLOBAL = enum.auto()


class PanoRenderType(enum.StrEnum):
    PERSPECTIVE_OVERLAPPING = enum.auto()
    PERSPECTIVE_NON_OVERLAPPING = enum.auto()
    # Reconstruct directly on the panoramas with the native EQUIRECTANGULAR
    # camera model instead of rendering perspective images.
    SPHERICAL = enum.auto()


N = TypeVar("N", bound=int)
NDArrayNx2 = np.ndarray[tuple[N, Literal[2]], np.dtype[np.float64]]
NDArray3x1 = np.ndarray[tuple[Literal[3], Literal[1]], np.dtype[np.float64]]
NDArray3x3 = np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float64]]


@dataclass
class PanoRenderOptions:
    num_steps_yaw: int
    pitches_deg: Sequence[float]
    hfov_deg: float
    vfov_deg: float


PANO_RENDER_OPTIONS: dict[PanoRenderType, PanoRenderOptions] = {
    PanoRenderType.PERSPECTIVE_OVERLAPPING: PanoRenderOptions(
        num_steps_yaw=4,
        pitches_deg=(-35.0, 0.0, 35.0),
        hfov_deg=90.0,
        vfov_deg=90.0,
    ),
    # Cubemap without top and bottom images.
    PanoRenderType.PERSPECTIVE_NON_OVERLAPPING: PanoRenderOptions(
        num_steps_yaw=4,
        pitches_deg=(0.0,),
        hfov_deg=90.0,
        vfov_deg=90.0,
    ),
}


def create_virtual_camera(
    *,
    pano_width: int,
    pano_height: int,
    hfov_deg: float,
    vfov_deg: float,
) -> pycolmap.Camera:
    """Create a virtual perspective camera."""
    image_width = int(pano_width * hfov_deg / 360)
    image_height = int(pano_height * vfov_deg / 180)
    focal = image_width / (2 * np.tan(np.deg2rad(hfov_deg) / 2))
    camera = pycolmap.Camera.create_from_model_id(
        camera_id=0,
        model=pycolmap.CameraModelId.SIMPLE_PINHOLE,
        focal_length=focal,
        width=image_width,
        height=image_height,
    )
    # Not set by create_from_model_id.
    camera.has_prior_focal_length = True
    return camera


def get_virtual_camera_rays(
    camera: pycolmap.Camera,
) -> npt.NDArray[np.floating]:
    size = (camera.width, camera.height)
    x, y = np.indices(size).astype(np.float32)
    xy: NDArrayNx2 = np.column_stack([x.ravel(), y.ravel()])
    # The center of the upper left most pixel has coordinate (0.5, 0.5)
    xy += 0.5
    xy_norm: NDArrayNx2 = camera.cam_from_img(image_points=xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def spherical_img_from_cam(
    image_size: tuple[int, int], rays_in_cam: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Project rays into a 360 panorama (spherical) image."""
    if image_size[0] != image_size[1] * 2:
        raise ValueError("Only 360° panoramas are supported.")
    if rays_in_cam.ndim != 2 or rays_in_cam.shape[1] != 3:
        raise ValueError(f"{rays_in_cam.shape=} but expected (N,3).")
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size


def get_virtual_rotations(
    num_steps_yaw: int, pitches_deg: Sequence[float]
) -> Sequence[npt.NDArray[np.floating]]:
    """Get the relative rotations of the virtual cameras w.r.t. the panorama."""
    # Assuming that the panos are approximately upright.
    cams_from_pano_r = []
    yaws = np.linspace(0, 360, num_steps_yaw, endpoint=False)
    for pitch_deg in pitches_deg:
        yaw_offset = (360 / num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            cam_from_pano_r = Rotation.from_euler(
                "XY", [-pitch_deg, -yaw_deg], degrees=True
            ).as_matrix()
            cams_from_pano_r.append(cam_from_pano_r)
    return cams_from_pano_r


def create_pano_rig_config(
    cams_from_pano_rotation: Sequence[npt.NDArray[np.floating]],
    ref_idx: int = 0,
) -> pycolmap.RigConfig:
    """Create a RigConfig for the given virtual rotations."""
    rig_cameras = []
    zero_translation = cast(NDArray3x1, np.zeros((3, 1), dtype=np.float64))
    for idx, cam_from_pano_rotation in enumerate(cams_from_pano_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = (
                cam_from_pano_rotation @ cams_from_pano_rotation[ref_idx].T
            )
            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_rotation),
                zero_translation,
            )
        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=idx == ref_idx,
                image_prefix=f"pano_camera{idx}/",
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)


class PanoProcessor:
    def __init__(
        self,
        pano_image_dir: Path,
        output_image_dir: Path,
        mask_dir: Path,
        render_options: PanoRenderOptions,
    ):
        self.render_options = render_options
        self.pano_image_dir = pano_image_dir
        self.output_image_dir = output_image_dir
        self.mask_dir = mask_dir

        self.cams_from_pano_rotation = get_virtual_rotations(
            num_steps_yaw=render_options.num_steps_yaw,
            pitches_deg=render_options.pitches_deg,
        )
        self.rig_config = create_pano_rig_config(self.cams_from_pano_rotation)

        # We assign each pano pixel to the virtual camera
        # with the closest camera center.
        self.cam_centers_in_pano = np.einsum(
            "nij,i->nj", self.cams_from_pano_rotation, [0, 0, 1]
        )

        self._lock = Lock()

        # These are initialized on the first pano image
        # to avoid recomputing the rays for each pano image.
        self._camera: pycolmap.Camera | None = None
        self._pano_size: tuple[int, int] | None = None
        self._rays_in_cam: npt.NDArray[np.floating] | None = None

    def process(self, pano_name: str) -> None:
        pano_path = self.pano_image_dir / pano_name
        try:
            pano_pil_image = PIL.Image.open(pano_path)
        except PIL.Image.UnidentifiedImageError:
            logging.info(f"Skipping file {pano_path} as it cannot be read.")
            return

        pano_exif = pano_pil_image.getexif()
        gpsonly_exif = PIL.Image.Exif()
        gpsonly_exif[PIL.ExifTags.IFD.GPSInfo] = pano_exif.get_ifd(
            PIL.ExifTags.IFD.GPSInfo
        )

        pano_image = np.asarray(pano_pil_image)
        pano_height, pano_width, *_ = pano_image.shape
        if pano_width != pano_height * 2:
            raise ValueError("Only 360° panoramas are supported.")

        with self._lock:
            if self._camera is None:  # First image, precompute rays once.
                self._camera = create_virtual_camera(
                    pano_width=pano_width,
                    pano_height=pano_height,
                    hfov_deg=self.render_options.hfov_deg,
                    vfov_deg=self.render_options.vfov_deg,
                )
                for rig_camera in self.rig_config.cameras:
                    rig_camera.camera = self._camera
                self._pano_size = (pano_width, pano_height)
                self._rays_in_cam = get_virtual_camera_rays(self._camera)
            else:  # Later images, verify consistent panoramas.
                if (pano_width, pano_height) != self._pano_size:
                    raise ValueError(
                        "Panoramas of different sizes are not supported."
                    )

        for cam_idx, cam_from_pano_r in enumerate(self.cams_from_pano_rotation):
            assert self._rays_in_cam is not None
            rays_in_pano = self._rays_in_cam @ cam_from_pano_r
            xy_in_pano = spherical_img_from_cam(self._pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(
                self._camera.width, self._camera.height, 2
            ).astype(np.float32)
            xy_in_pano -= 0.5  # COLMAP to OpenCV pixel origin.
            x_coords, y_coords = np.moveaxis(xy_in_pano, [0, 1, 2], [2, 1, 0])
            image = cv2.remap(
                pano_image,
                x_coords,
                y_coords,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
            # We define a mask such that each pixel of the panorama has its
            # features extracted only in a single virtual camera.
            closest_camera = np.argmax(
                rays_in_pano @ self.cam_centers_in_pano.T, -1
            )
            mask = (
                ((closest_camera == cam_idx) * 255)
                .astype(np.uint8)
                .reshape(self._camera.width, self._camera.height)
                .transpose()
            )

            image_name = (
                self.rig_config.cameras[cam_idx].image_prefix + pano_name
            )
            mask_name = f"{image_name}.png"

            image_path = self.output_image_dir / image_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            PIL.Image.fromarray(image).save(image_path, exif=gpsonly_exif)

            mask_path = self.mask_dir / mask_name
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            if not pycolmap.Bitmap.from_array(mask).write(mask_path):
                raise RuntimeError(f"Cannot write {mask_path}")

    def split_image_name(self, image_name: str) -> tuple[int, str]:
        """Split a rendered image name into (virtual camera idx, pano name)."""
        for cam_idx, rig_camera in enumerate(self.rig_config.cameras):
            prefix = rig_camera.image_prefix
            if image_name.startswith(prefix):
                return cam_idx, image_name[len(prefix) :]
        raise ValueError(f"Unknown virtual camera for image {image_name!r}.")

    def convert_to_equirectangular(
        self, reconstruction: pycolmap.Reconstruction
    ) -> pycolmap.Reconstruction:
        """Convert a reconstruction built from the rig of perspective virtual
        cameras back to one equirectangular camera/image per input panorama.

        The output reconstruction references the original panorama images with
        the native EQUIRECTANGULAR camera model. Frame poses, 3D points, and
        all keypoints (including those without a 3D observation) are carried
        over by re-projecting the perspective keypoints onto the panorama
        through the same spherical mapping used for rendering, so the result is
        a valid, bundle-adjustable reconstruction.
        """
        if self._camera is None or self._pano_size is None:
            raise RuntimeError("No panorama was rendered yet.")
        pano_width, pano_height = self._pano_size

        equirect = pycolmap.Reconstruction()
        equirect_camera = pycolmap.Camera.create_from_model_id(
            camera_id=1,
            model=pycolmap.CameraModelId.EQUIRECTANGULAR,
            focal_length=0.0,
            width=pano_width,
            height=pano_height,
        )
        equirect.add_camera_with_trivial_rig(equirect_camera)

        # The rig reference sensor is virtual camera 0 (see
        # create_pano_rig_config), so rig_from_world == cam0_from_world and
        # pano_from_world = pano_from_cam0 @ cam0_from_world. The virtual
        # cameras share the panorama center, hence the zero translation.
        pano_from_ref = pycolmap.Rigid3d(
            pycolmap.Rotation3d(
                cast(NDArray3x3, self.cams_from_pano_rotation[0])
            ),
            cast(NDArray3x1, np.zeros((3, 1), dtype=np.float64)),
        ).inverse()

        # Group the registered virtual cameras by frame. All virtual cameras of
        # a frame observe the same panorama and share its pose, so we can
        # accumulate their keypoints into a single equirectangular image.
        images_by_frame: dict[int, list[pycolmap.Image]] = (
            collections.defaultdict(list)
        )
        for image in reconstruction.images.values():
            if image.has_pose:
                images_by_frame[image.frame_id].append(image)

        # Maps an image_id of a virtual camera to a dict from its old point2D
        # index to the (new point2D index, pano_name) in the equirectangular
        # image, so we can later rebuild the 3D point tracks.
        old_to_new_point2D: dict[int, dict[int, tuple[int, str]]] = {}
        pano_to_image_id: dict[str, int] = {}

        frame_images = sorted(
            images_by_frame.values(),
            key=lambda images: self.split_image_name(images[0].name)[1],
        )
        for image_id, images in enumerate(frame_images, start=1):
            pano_name = self.split_image_name(images[0].name)[1]
            pano_to_image_id[pano_name] = image_id
            frame = images[0].frame
            assert frame is not None
            rig_from_world = frame.rig_from_world
            assert rig_from_world is not None
            pano_from_world = pano_from_ref * rig_from_world

            # Concatenate the keypoints of all virtual cameras of this panorama.
            keypoints: list[npt.NDArray[np.floating]] = []
            for image in images:
                cam_idx = self.split_image_name(image.name)[0]
                num_points2D = len(image.points2D)
                if num_points2D == 0:
                    old_to_new_point2D[image.image_id] = {}
                    continue
                xy = np.array([point2D.xy for point2D in image.points2D])
                rays_in_cam: npt.NDArray[np.floating] = np.asarray(
                    self._camera.cam_ray_from_img(image_points=xy)
                )
                rays_in_cam /= np.linalg.norm(
                    rays_in_cam, axis=-1, keepdims=True
                )
                rays_in_pano = (
                    rays_in_cam @ self.cams_from_pano_rotation[cam_idx]
                )
                xy_in_pano = spherical_img_from_cam(
                    self._pano_size, rays_in_pano
                )

                base_idx = len(keypoints)
                keypoints.extend(xy_in_pano)
                old_to_new_point2D[image.image_id] = {
                    point2D_idx: (base_idx + point2D_idx, pano_name)
                    for point2D_idx in range(num_points2D)
                }

            equirect.add_image_with_trivial_frame(
                pycolmap.Image(
                    name=pano_name,
                    keypoints=keypoints,
                    camera_id=equirect_camera.camera_id,
                    image_id=image_id,
                ),
                pano_from_world,
            )

        for point3D_id, point3D in reconstruction.points3D.items():
            track = pycolmap.Track()
            for element in point3D.track.elements:
                new_point2D_idx, pano_name = old_to_new_point2D[
                    element.image_id
                ][element.point2D_idx]
                track.add_element(pano_to_image_id[pano_name], new_point2D_idx)
            equirect.add_point3D_with_id(
                point3D_id,
                pycolmap.Point3D(
                    xyz=point3D.xyz, color=point3D.color, track=track
                ),
            )

        return equirect


def render_perspective_images(
    pano_image_names: Sequence[str],
    pano_image_dir: Path,
    output_image_dir: Path,
    mask_dir: Path,
    render_options: PanoRenderOptions,
) -> PanoProcessor:
    processor = PanoProcessor(
        pano_image_dir, output_image_dir, mask_dir, render_options
    )

    num_panos = len(pano_image_names)
    max_workers = min(32, (os.cpu_count() or 2) - 1)

    with tqdm(total=num_panos) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as thread_pool:
            futures = [
                thread_pool.submit(processor.process, pano_name)
                for pano_name in pano_image_names
            ]
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    return processor


def run_matcher(
    args: argparse.Namespace,
    database_path: Path,
    matching_options: pycolmap.FeatureMatchingOptions,
) -> None:
    if args.matcher == "sequential":
        pycolmap.match_sequential(
            database_path,
            pairing_options=pycolmap.SequentialPairingOptions(
                loop_detection=True
            ),
            matching_options=matching_options,
        )
    elif args.matcher == "exhaustive":
        pycolmap.match_exhaustive(
            database_path, matching_options=matching_options
        )
    elif args.matcher == "vocabtree":
        pycolmap.match_vocabtree(
            database_path, matching_options=matching_options
        )
    elif args.matcher == "spatial":
        pycolmap.match_spatial(database_path, matching_options=matching_options)
    else:
        logging.fatal(f"Unknown matcher: {args.matcher}")


def run_spherical(
    args: argparse.Namespace, database_path: Path, rec_path: Path
) -> None:
    """Reconstruct directly on the equirectangular panoramas with the native
    EQUIRECTANGULAR camera model, without rendering perspective images."""

    logging.info("Reconstructing with spherical camera")

    reader_options = pycolmap.ImageReaderOptions(camera_model="EQUIRECTANGULAR")
    pycolmap.extract_features(
        database_path,
        args.input_image_path,
        reader_options=reader_options,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )

    # A single EQUIRECTANGULAR camera observes the whole sphere from one
    # center, so there is no rig and no per-frame image-pair skipping.
    run_matcher(args, database_path, pycolmap.FeatureMatchingOptions())

    # The EQUIRECTANGULAR model has no focal length, principal point, or
    # distortion to refine; its (w, h) params are held constant in bundle
    # adjustment.
    recs = pycolmap.incremental_mapping(
        database_path, args.input_image_path, rec_path
    )
    for idx, rec in recs.items():
        logging.info(f"#{idx} {rec.summary()}")


def run_perspective(
    args: argparse.Namespace, database_path: Path, rec_path: Path
) -> None:
    """Render the panoramas into a rig of perspective virtual cameras and
    reconstruct from those."""

    logging.info("Reconstructing with rig of perspective virtual cameras")

    image_dir = args.output_path / "images"
    mask_dir = args.output_path / "masks"
    image_dir.mkdir(exist_ok=True, parents=True)
    mask_dir.mkdir(exist_ok=True, parents=True)

    # Search for input images.
    pano_image_dir = args.input_image_path
    pano_image_names = sorted(
        p.relative_to(pano_image_dir).as_posix()
        for p in pano_image_dir.rglob("*")
        if not p.is_dir()
    )
    logging.info(f"Found {len(pano_image_names)} images in {pano_image_dir}.")

    processor = render_perspective_images(
        pano_image_names,
        pano_image_dir,
        image_dir,
        mask_dir,
        PANO_RENDER_OPTIONS[args.pano_render_type],
    )
    rig_config = processor.rig_config

    rendered_camera = rig_config.cameras[0].camera
    assert rendered_camera is not None  # Make mypy happy.
    pycolmap.extract_features(
        database_path,
        image_dir,
        reader_options=pycolmap.ImageReaderOptions(
            mask_path=mask_dir,
            camera_model=rendered_camera.model_name,
            camera_params=rendered_camera.params_to_string(),
        ),
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
    )

    with pycolmap.Database.open(database_path) as db:
        pycolmap.apply_rig_config([rig_config], db)

    matching_options = pycolmap.FeatureMatchingOptions()
    # We have perfect sensor_from_rig poses (except for potential stitching
    # artifacts by the spherical image provider), so we can perform geometric
    # verification using rig constraints.
    matching_options.rig_verification = True
    # The images within a frame do not have overlap due to the provided masks.
    matching_options.skip_image_pairs_in_same_frame = True
    run_matcher(args, database_path, matching_options)

    if args.mapper == Mapper.INCREMENTAL:
        opts = pycolmap.IncrementalPipelineOptions(
            ba_refine_sensor_from_rig=False,
            ba_refine_focal_length=False,
            ba_refine_principal_point=False,
            ba_refine_extra_params=False,
        )
        recs = pycolmap.incremental_mapping(
            database_path, image_dir, rec_path, opts
        )
    elif args.mapper == Mapper.GLOBAL:
        global_opts = pycolmap.GlobalPipelineOptions(
            mapper=pycolmap.GlobalMapperOptions(refine_sensor_from_rig=False)
        )
        # Don't set these in the init to not overwrite custom default options.
        global_opts.mapper.bundle_adjustment.refine_focal_length = False
        global_opts.mapper.bundle_adjustment.refine_principal_point = False
        global_opts.mapper.bundle_adjustment.refine_extra_params = False
        recs = pycolmap.global_mapping(
            database_path, image_dir, rec_path, global_opts
        )
    else:
        logging.fatal(f"Unknown mapper: {args.mapper}")

    for idx, rec in recs.items():
        logging.info(f"#{idx} {rec.summary()}")

    logging.info("Converting virtual cameras back to equirectangular")
    equirect_rec_path = args.output_path / "sparse_equirectangular"
    for idx, rec in recs.items():
        equirect_rec = processor.convert_to_equirectangular(rec)
        output_path = equirect_rec_path / str(idx)
        output_path.mkdir(exist_ok=True, parents=True)
        equirect_rec.write(output_path)
        logging.info(f"#{idx} {equirect_rec.summary()}")


def run(args: argparse.Namespace) -> None:
    pycolmap.set_random_seed(0)

    database_path = args.output_path / "database.db"
    if database_path.exists():
        database_path.unlink()

    rec_path = args.output_path / "sparse"
    rec_path.mkdir(exist_ok=True, parents=True)

    if args.pano_render_type == PanoRenderType.SPHERICAL:
        run_spherical(args, database_path, rec_path)
    else:
        run_perspective(args, database_path, rec_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--matcher",
        type=Matcher,
        default=Matcher.SEQUENTIAL,
        choices=list(Matcher),
    )
    parser.add_argument(
        "--mapper",
        type=Mapper,
        default=Mapper.INCREMENTAL,
        choices=list(Mapper),
    )
    parser.add_argument(
        "--pano_render_type",
        type=PanoRenderType,
        default=PanoRenderType.PERSPECTIVE_OVERLAPPING,
        choices=list(PanoRenderType),
    )
    args = parser.parse_args()
    run(args)
