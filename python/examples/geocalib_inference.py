"""
An example for using GeoCalib to extract gravity and camera intrinsics.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import enlighten
import geocalib
import numpy as np
import torch

import pycolmap
from pycolmap import logging

NDArray3x1 = np.ndarray[tuple[Literal[3], Literal[1]], np.dtype[np.float64]]
NDArray2x2 = np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.float64]]


def rot90_vec(vec: np.ndarray, k: int) -> np.ndarray:
    """Rotates a 3D gravity vector by k * 90 degrees CCW around the Z axis."""
    k = k % 4
    if k == 0:
        return vec
    x, y, z = vec
    if k == 1:  # 90 deg CCW
        return np.array([y, -x, z])
    elif k == 2:  # 180 deg CCW
        return np.array([-x, -y, z])
    else:  # 270 deg CCW
        return np.array([-y, x, z])


def write_focal(camera: pycolmap.Camera, focal: float) -> None:
    if len(camera.focal_length_idxs()) == 1:
        camera.focal_length = focal
    elif len(camera.focal_length_idxs()) == 2:
        camera.focal_length_x = focal
        camera.focal_length_y = focal
    camera.has_prior_focal_length = True


@dataclass
class CalibrationResult:
    gravity_direction: NDArray3x1
    gravity_covariance: NDArray2x2
    focal_length: float


class GeoCalibProcessor:
    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logging.info(f"Loading GeoCalib model on {self.device}")
        self.model = geocalib.GeoCalib().to(self.device)

    def _process_image_tensor(
        self,
        img_tensor: torch.Tensor,
        prior_focal_length: float | None,
        rot90: int = 0,
    ) -> CalibrationResult:
        rotated_img = torch.rot90(img_tensor, k=rot90, dims=[1, 2])
        priors = {}
        if prior_focal_length is not None:
            priors["focal"] = torch.tensor(
                prior_focal_length, dtype=torch.float32, device=self.device
            )
        results = self.model.calibrate(rotated_img, priors=priors)

        pred_gravity = results["gravity"].squeeze(0).cpu().numpy()
        orig_gravity = rot90_vec(pred_gravity, -rot90)
        pred_focal_length = results["camera"].f.mean().item()

        cov = results["covariance"].squeeze(0).cpu().numpy()
        gravity_covariance = cov[:2, :2]

        return CalibrationResult(
            orig_gravity, gravity_covariance, pred_focal_length
        )

    def process_known_orientation(
        self, image_path: Path, rot90: int, prior_focal_length: float | None
    ) -> CalibrationResult:
        img_tensor = self.model.load_image(image_path).to(self.device)
        result = self._process_image_tensor(
            img_tensor, prior_focal_length, rot90
        )
        uncertainty = float(np.trace(result.gravity_covariance))
        logging.verbose(
            1, f"Gravity uncertainty for known orientation: {uncertainty}"
        )
        return result

    def process_unknown_orientation(
        self, image_path: Path, prior_focal_length: float | None
    ) -> CalibrationResult:
        img_tensor = self.model.load_image(image_path).to(self.device)
        score_result_rot90 = []
        for rot90 in range(4):
            result = self._process_image_tensor(
                img_tensor, prior_focal_length, rot90
            )
            uncertainty = np.trace(result.gravity_covariance)
            score_result_rot90.append((uncertainty, result, rot90))
        _, best_result, best_rot90 = min(score_result_rot90)
        uncertainties = [float(u) for u, *_ in score_result_rot90]
        logging.verbose(
            1,
            "Gravity uncertainties for orientations (0, 90, 180, 270)° CCW:"
            f" {uncertainties}.",
        )
        logging.verbose(1, f"Best orientation found: {best_rot90 * 90}° CCW.")
        return best_result


def process_image(
    image: pycolmap.Image,
    camera: pycolmap.Camera,
    db: pycolmap.Database,
    processor: GeoCalibProcessor,
    pose_prior: pycolmap.PosePrior | None,
    input_path: Path,
) -> None:
    if "PINHOLE" not in camera.model_name:
        logging.warning(
            f"Image has camera model {camera.model_name}, "
            "will calibrate only the focal length."
        )

    prior_focal_length = None
    if camera.has_prior_focal_length:
        prior_focal_length = camera.mean_focal_length()
        logging.verbose(
            1, f"Image has prior focal length: {prior_focal_length} px."
        )
    else:
        logging.verbose(1, "Image has no prior focal length.")
    image_path = input_path / image.name

    if pose_prior is None:
        pose_prior = pycolmap.PosePrior()
        pose_prior.corr_data_id = image.data_id
        pose_prior.pose_prior_id = db.write_pose_prior(pose_prior)

    if pose_prior.has_gravity():
        gravity_np = pose_prior.gravity
        rot90 = pycolmap.compute_rot90_from_gravity(gravity_np)
        logging.verbose(
            1, f"Image has prior gravity, rotating by {rot90 * 90}° CCW."
        )
        result = processor.process_known_orientation(
            image_path, rot90, prior_focal_length
        )
    else:
        logging.verbose(
            1, "Image has no prior gravity. Searching 4 orientations."
        )
        result = processor.process_unknown_orientation(
            image_path, prior_focal_length
        )

    pose_prior.gravity = result.gravity_direction
    db.update_pose_prior(pose_prior)

    if not camera.has_prior_focal_length:
        write_focal(camera, result.focal_length)
        db.update_camera(camera)


def run(input_path: Path, output_path: Path) -> None:
    output_path.mkdir(exist_ok=True, parents=True)
    database_path = output_path / "database.db"
    if database_path.exists():
        database_path.unlink()
    pycolmap.Database.open(database_path).close()
    logging.info(f"Importing images from {input_path} into {database_path}")
    pycolmap.import_images(
        database_path,
        input_path,
        options=pycolmap.ImageReaderOptions(camera_model="SIMPLE_PINHOLE"),
    )

    processor = GeoCalibProcessor()

    with pycolmap.Database.open(database_path) as db:
        images = db.read_all_images()
        cameras = {cam.camera_id: cam for cam in db.read_all_cameras()}
        pose_priors = {
            prior.corr_data_id.id: prior
            for prior in db.read_all_pose_priors()
            if prior.corr_data_id.sensor_id.type == pycolmap.SensorType.CAMERA
        }

        manager = enlighten.get_manager()
        with manager.counter(
            total=len(images), desc="Processed images:"
        ) as pbar:
            for image in images:
                logging.verbose(1, f">> Processing image {image.name}")
                process_image(
                    image=image,
                    camera=cameras[image.camera_id],
                    db=db,
                    processor=processor,
                    pose_prior=pose_priors.get(image.data_id.id),
                    input_path=input_path,
                )
                pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()
    run(input_path=args.input_image_path, output_path=args.output_path)
