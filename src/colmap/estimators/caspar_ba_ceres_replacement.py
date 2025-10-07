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
def reprojection_simple_pinhole_constant(
    cam: T.Annotated[PinholeCamera, mem.Constant],
    point: T.Annotated[Point, mem.Tunable],
    pixel: T.Annotated[Pixel, mem.Constant],
)->sf.V2:
    focal_length, cx, cy = cam.calibration
    center_pixel = sf.V2([cx, cy])
    point_cam = cam.cam_T_world * point
    depth = point_cam[2]
    point_ideal_camera_coords = -sf.V2(point_cam[:2] - center_pixel)/(depth + sf.epsilon() * sf.sign_no_zero(depth))
    pixel_projected = focal_length  * point_ideal_camera_coords
    reprojection_error = pixel_projected
    return reprojection_error




out_dir = Path(__file__).resolve().parent / "generated"
caslib.generate(out_dir)
caslib.compile(out_dir)

lib = caslib.import_lib(out_dir)
