from dataclasses import dataclass
from pathlib import Path
import numpy as np

import symforce


symforce.set_epsilon_to_number(float(10 * np.finfo(np.float32).eps))

import symforce.symbolic as sf
from symforce import typing as T
from symforce.experimental.caspar import CasparLibrary
from symforce.experimental.caspar import memory as mem
from symforce.codegen import codegen_util

@dataclass
class PinholeCamera:
    cam_T_world: sf.Pose3
    calibration: sf.V3

class Point(sf.V3):
    pass

class Pixel(sf.V2):
    pass

caslib = CasparLibrary()


@caslib.add_factor
def reprojection_constant_intrinsics(
    cam: T.Annotated[PinholeCamera, mem.Constant],
    point: T.Annotated[Point, mem.Tunable],
    pixel: T.Annotated[Pixel, mem.Constant],
)->sf.V2:
    focal_length, k1, k2 = cam.calibration
    point_cam = cam.cam_T_world * point
    depth = point_cam[2]
    point_ideal_camera_coords = -sf.V2(point_cam[:2])/(depth + sf.epsilon() * sf.sign_no_zero(depth))
    radial_distortion = 1 + k1*point_ideal_camera_coords.squared_norm() + k2*point_ideal_camera_coords.squared_norm()**2
    pixel_projected = focal_length * radial_distortion * point_ideal_camera_coords
    reprojection_error = pixel_projected - pixel
    return reprojection_error

@caslib.add_factor
def reprojection_variable_intrinsics(
    cam: T.Annotated[PinholeCamera, mem.Tunable],
    point: T.Annotated[Point, mem.Tunable],
    pixel: T.Annotated[Pixel, mem.Constant],
)->sf.V2:
    focal_length, k1, k2 = cam.calibration
    point_cam = cam.cam_T_world * point
    depth = point_cam[2]
    point_ideal_camera_coords = -sf.V2(point_cam[:2])/(depth + sf.epsilon() * sf.sign_no_zero(depth))
    radial_distortion = 1 + k1*point_ideal_camera_coords.squared_norm() + k2*point_ideal_camera_coords.squared_norm()**2
    pixel_projected = focal_length * radial_distortion * point_ideal_camera_coords
    reprojection_error = pixel_projected - pixel
    return reprojection_error




out_dir = Path(__file__).resolve().parent / "generated"
caslib.generate(out_dir)
caslib.compile(out_dir)

lib = caslib.import_lib(out_dir)
