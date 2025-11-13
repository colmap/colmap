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
class SimplePinholeCamera:
  cam_T_world: sf.Pose3
  calibration: sf.V3 # f, cx, cy


@dataclass
class PinholeCamera:
    cam_T_world: sf.Pose3
    calibration: sf.V4 # fx, fy, cx, cy

@dataclass
class SimpleRadialCamera:
    cam_T_world: sf.Pose3
    calibration: sf.V4 # f, cx, cy, k

@dataclass
class RadialCamera:
    cam_T_world: sf.Pose3
    calibration: sf.V5 # f, cx, cy, k1, k2

class Point(sf.V3):
  pass

class Pixel(sf.V2):
  pass

caslib = CasparLibrary()

# @caslib.add_factor
# def simple_pinhole(
#     cam: T.Annotated[SimplePinholeCamera, mem.Tunable],
#     point: T.Annotated[Point, mem.Tunable],
#     pixel: T.Annotated[Pixel, mem.Constant],
# )->sf.V2:
#     cam_T_world = cam.cam_T_world
#     intrinsics = cam.calibration
#     focal_length, cx, cy = intrinsics
#     principal_point = sf.V2([cx, cy])
#     point_cam = cam_T_world * point
#     depth = point_cam[2]
#     p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
#     pixel_projected = focal_length  * p + principal_point
#     reprojection_error = pixel_projected - pixel
#     return reprojection_error

# @caslib.add_factor
# def simple_pinhole_fixed_cam(
#     cam_calib: T.Annotated[sf.V3, mem.Tunable],
#     cam_T_world: T.Annotated[sf.Pose3, mem.Constant], # Pose is constant 
#     point: T.Annotated[Point, mem.Tunable],
#     pixel: T.Annotated[Pixel, mem.Constant],
# )->sf.V2:
#     focal_length, cx, cy = cam_calib
#     principal_point = sf.V2([cx, cy]) 
#     point_cam = cam_T_world * point
#     depth = point_cam[2]
#     p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
#     pixel_projected = focal_length * p + principal_point
#     reprojection_error = pixel_projected - pixel
#     return reprojection_error


# @caslib.add_factor
# def pinhole(
#     cam: T.Annotated[PinholeCamera, mem.Tunable],
#     point: T.Annotated[Point, mem.Tunable],
#     pixel: T.Annotated[Pixel, mem.Constant],
# )->sf.V2:
#     cam_T_world = cam.cam_T_world
#     intrinsics = cam.calibration
#     fx, fy, cx, cy = intrinsics
#     principal_point = sf.V2([cx, cy])
#     focal_length = sf.V2([fx, fy])
#     point_cam = cam_T_world * point
#     depth = point_cam[2]
#     p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
#     pixel_projected = focal_length.multiply_elementwise(p) + principal_point
#     reprojection_error = pixel_projected - pixel
#     return reprojection_error

# @caslib.add_factor
# def pinhole_fixed_cam(
#     cam_calib: T.Annotated[sf.V4, mem.Tunable],
#     cam_T_world: T.Annotated[sf.Pose3, mem.Constant],
#     point: T.Annotated[Point, mem.Tunable],
#     pixel: T.Annotated[Pixel, mem.Constant],
# )->sf.V2:
#     fx, fy, cx, cy = cam_calib
#     principal_point = sf.V2([cx, cy])
#     focal_length = sf.V2([fx, fy])
#     point_cam = cam_T_world * point
#     depth = point_cam[2]
#     p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
#     pixel_projected = focal_length.multiply_elementwise(p) + principal_point
#     reprojection_error = pixel_projected - pixel
#     return reprojection_error


@caslib.add_factor
def simple_radial(
    cam: T.Annotated[SimpleRadialCamera, mem.Tunable],
    point: T.Annotated[Point, mem.Tunable],
    pixel: T.Annotated[Pixel, mem.Constant],
)->sf.V2:
    cam_T_world = cam.cam_T_world
    intrinsics = cam.calibration
    focal_length, cx, cy, k = intrinsics
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
    r = 1 + k * p.squared_norm()
    pixel_projected = focal_length  * r * p + principal_point
    reprojection_error = pixel_projected - pixel
    return reprojection_error

# @caslib.add_factor
# def simple_radial_fixed_cam(
#     cam_calib: T.Annotated[sf.V4, mem.Tunable],
#     cam_T_world: T.Annotated[sf.Pose3, mem.Constant],
#     point: T.Annotated[Point, mem.Tunable],
#     pixel: T.Annotated[Pixel, mem.Constant],
# )->sf.V2:
#     focal_length, cx, cy, k = cam_calib
#     principal_point = sf.V2([cx, cy])
#     point_cam = cam_T_world * point
#     depth = point_cam[2]
#     p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
#     r = 1 + k * p.squared_norm()
#     pixel_projected = focal_length  * r * p + principal_point
#     reprojection_error = pixel_projected - pixel
#     return reprojection_error

# @caslib.add_factor
# def radial(
#     cam: T.Annotated[RadialCamera, mem.Tunable],
#     point: T.Annotated[Point, mem.Tunable],
#     pixel: T.Annotated[Pixel, mem.Constant],
# ) -> sf.V2:
#     cam_T_world = cam.cam_T_world
#     intrinsics = cam.calibration
#     focal_length, cx, cy, k1, k2 = intrinsics
#     principal_point = sf.V2([cx, cy])
#     point_cam = cam_T_world * point
#     depth = point_cam[2]
#     p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
#     r = 1 + k1 * p.squared_norm() + k2 * p.squared_norm() ** 2
#     pixel_projected = focal_length * r * p + principal_point
#     err = pixel_projected - pixel
#     return err

# @caslib.add_factor
# def radial_fixed_cam(
#     cam_calib: T.Annotated[sf.V5, mem.Tunable],
#     cam_T_world: T.Annotated[sf.Pose3, mem.Constant],
#     point: T.Annotated[Point, mem.Tunable],
#     pixel: T.Annotated[Pixel, mem.Constant],
# ) -> sf.V2:
#     focal_length, cx, cy, k1, k2 = cam_calib
#     principal_point = sf.V2([cx, cy])
#     point_cam = cam_T_world * point
#     depth = point_cam[2]
#     p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
#     r = 1 + k1 * p.squared_norm() + k2 * p.squared_norm() ** 2
#     pixel_projected = focal_length * r * p + principal_point
#     err = pixel_projected - pixel
#     return err




out_dir = Path(__file__).resolve().parent / "generated"
caslib.generate(out_dir)
caslib.compile(out_dir)