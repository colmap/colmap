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

class SimpleRadialCalib(sf.V4):
  pass

class Point(sf.V3):
  pass

class Pixel(sf.V2):
  pass

caslib = CasparLibrary()


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


# Adding large weights would enable us to reuse factors, 
# but we will do the Ceres-way of parameter fixing for now
@caslib.add_factor
def simple_radial_fixed_intrinsics(
    cam_T_world: T.Annotated[sf.Pose3, mem.Tunable],
    point: T.Annotated[Point, mem.Tunable],
    cam_calib: T.Annotated[SimpleRadialCalib, mem.Constant],
    pixel: T.Annotated[Pixel, mem.Constant],
)->sf.V2:
    focal_length, cx, cy, k = cam_calib
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
    r = 1 + k * p.squared_norm()
    pixel_projected = focal_length  * r * p + principal_point
    reprojection_error = pixel_projected - pixel
    return reprojection_error


@caslib.add_factor
def simple_radial_fixed_pose(
    cam_calib: T.Annotated[SimpleRadialCalib, mem.Tunable],
    point: T.Annotated[Point, mem.Tunable],
    cam_T_world: T.Annotated[sf.Pose3, mem.Constant],
    pixel: T.Annotated[Pixel, mem.Constant],
)->sf.V2:
    focal_length, cx, cy, k = cam_calib
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
    r = 1 + k * p.squared_norm()
    pixel_projected = focal_length  * r * p + principal_point
    reprojection_error = pixel_projected - pixel
    return reprojection_error


@caslib.add_factor
def simple_radial_fixed_cam(
    point: T.Annotated[Point, mem.Tunable],
    cam: T.Annotated[SimpleRadialCamera, mem.Constant],
    pixel: T.Annotated[Pixel, mem.Constant],
)->sf.V2:
    cam_T_world = cam.cam_T_world
    focal_length, cx, cy, k = cam.calibration
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
    r = 1 + k * p.squared_norm()
    pixel_projected = focal_length  * r * p + principal_point
    reprojection_error = pixel_projected - pixel
    return reprojection_error


@caslib.add_factor
def simple_radial_fixed_point(
    cam: T.Annotated[SimpleRadialCamera, mem.Tunable],
    point: T.Annotated[Point, mem.Constant],
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



@caslib.add_factor
def simple_radial_fixed_intrinsics_and_point(
    cam_T_world: T.Annotated[sf.Pose3, mem.Tunable],
    cam_calib: T.Annotated[SimpleRadialCalib, mem.Constant],
    point: T.Annotated[Point, mem.Constant],
    pixel: T.Annotated[Pixel, mem.Constant],
)->sf.V2:
    focal_length, cx, cy, k = cam_calib
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
    r = 1 + k * p.squared_norm()
    pixel_projected = focal_length  * r * p + principal_point
    reprojection_error = pixel_projected - pixel
    return reprojection_error


@caslib.add_factor
def simple_radial_fixed_pose_and_point(
    cam_calib: T.Annotated[SimpleRadialCalib, mem.Tunable],
    cam_T_world: T.Annotated[sf.Pose3, mem.Constant],
    point: T.Annotated[sf.V3, mem.Constant],
    pixel: T.Annotated[Pixel, mem.Constant],
)->sf.V2:
    focal_length, cx, cy, k = cam_calib
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
    r = 1 + k * p.squared_norm()
    pixel_projected = focal_length  * r * p + principal_point
    reprojection_error = pixel_projected - pixel
    return reprojection_error

@caslib.add_factor
def simple_radial_fixed_translation_norm(
    rotation: T.Annotated[sf.Rot3, mem.Tunable],
    translation_direction: T.Annotated[sf.Unit3, mem.Tunable],
    translation_norm: T.Annotated[sf.V1, mem.Constant],
    cam_calib: T.Annotated[SimpleRadialCalib, mem.Tunable],
    point: T.Annotated[Point, mem.Tunable],
    pixel: T.Annotated[Pixel, mem.Constant],
) -> sf.V2:
    """Reprojection with pose that has fixed translation magnitude"""
    # Convert Unit3 to 3D vector and scale by fixed norm
    translation = translation_direction.to_unit_vector() * translation_norm[0]
    cam_T_world = sf.Pose3(R=rotation, t=translation)
    
    # Standard simple radial reprojection
    focal_length, cx, cy, k = cam_calib
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    r_radial = 1 + k * p.squared_norm()
    pixel_projected = focal_length * r_radial * p + principal_point
    return pixel_projected - pixel

@caslib.add_factor
def simple_radial_with_separate_calib(
    cam_T_world: T.Annotated[sf.Pose3, mem.Tunable],
    cam_calib: T.Annotated[SimpleRadialCalib, mem.Tunable],
    point: T.Annotated[Point, mem.Tunable],
    pixel: T.Annotated[Pixel, mem.Constant],
)->sf.V2:
    focal_length, cx, cy, k = cam_calib
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
    r = 1 + k * p.squared_norm()
    pixel_projected = focal_length * r * p + principal_point
    reprojection_error = pixel_projected - pixel
    return reprojection_error

# New factors can easily be added
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



out_dir = Path(__file__).resolve().parent / "generated"
caslib.generate(out_dir)
caslib.compile(out_dir)