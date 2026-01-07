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


# Tunables
class SimpleRadialCalib(sf.V4):
    pass

@dataclass
class SimpleRadialCamera:
    cam_T_world: sf.Pose3
    calibration: SimpleRadialCalib # f, cx, cy, k

@dataclass
class SimpleRadialCameraFixedPose:
    calibration: SimpleRadialCalib  # f, cx, cy, k

@dataclass
class SimpleRadialCameraFixedCalib:
    cam_T_world: sf.Pose3

@dataclass
class SimpleRadialCameraFixedTranslationNorm:
    rotation: sf.Rot3
    translation_direction: sf.Unit3
    cam_calib: SimpleRadialCalib


class Point(sf.V3):
    pass



# Constant type variations
class ConstPixel(sf.V2):
    pass

class ConstPoint(sf.V3):
    pass

class ConstSimpleRadialCalib(sf.V4):
    pass

class ConstPose(sf.Pose3):
    pass

@dataclass
class ConstSimpleRadialCamera:
    cam_T_world: sf.Pose3
    calibration: SimpleRadialCalib

class ConstScalar(sf.V1):
    pass

caslib = CasparLibrary()


@caslib.add_factor
def simple_radial(
    cam: T.Annotated[SimpleRadialCamera, mem.Tunable],
    point: T.Annotated[Point, mem.Tunable],
    pixel: T.Annotated[ConstPixel, mem.Constant],
) -> sf.V2:
    cam_T_world = cam.cam_T_world
    focal_length, cx, cy, k = cam.calibration
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
    r = 1 + k * p.squared_norm()
    pixel_projected = focal_length * r * p + principal_point
    return pixel_projected - pixel


@caslib.add_factor
def simple_radial_fixed_pose(
    cam_fixed_pose: T.Annotated[SimpleRadialCameraFixedPose, mem.Tunable],
    point: T.Annotated[Point, mem.Tunable],
    cam_T_world: T.Annotated[ConstPose, mem.Constant],
    pixel: T.Annotated[ConstPixel, mem.Constant],
) -> sf.V2:
    focal_length, cx, cy, k = cam_fixed_pose.calibration
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
    r = 1 + k * p.squared_norm()
    pixel_projected = focal_length * r * p + principal_point
    return pixel_projected - pixel


@caslib.add_factor
def simple_radial_fixed_point(
    cam: T.Annotated[SimpleRadialCamera, mem.Tunable],
    point: T.Annotated[ConstPoint, mem.Constant],
    pixel: T.Annotated[ConstPixel, mem.Constant],
) -> sf.V2:
    cam_T_world = cam.cam_T_world
    focal_length, cx, cy, k = cam.calibration
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
    r = 1 + k * p.squared_norm()
    pixel_projected = focal_length * r * p + principal_point
    return pixel_projected - pixel


@caslib.add_factor
def simple_radial_fixed_translation_norm(
    cam_fixed_norm: T.Annotated[SimpleRadialCameraFixedTranslationNorm, mem.Tunable],
    point: T.Annotated[Point, mem.Tunable],
    translation_norm: T.Annotated[ConstScalar, mem.Constant],
    pixel: T.Annotated[ConstPixel, mem.Constant],
) -> sf.V2:
    translation = cam_fixed_norm.translation_direction.to_unit_vector() * translation_norm[0]
    cam_T_world = sf.Pose3(R=cam_fixed_norm.rotation, t=translation)
    
    focal_length, cx, cy, k = cam_fixed_norm.cam_calib
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
    r_radial = 1 + k * p.squared_norm()
    pixel_projected = focal_length * r_radial * p + principal_point
    return pixel_projected - pixel

@caslib.add_factor
def simple_radial_fixed_translation_norm_and_point(
    cam_fixed_norm: T.Annotated[SimpleRadialCameraFixedTranslationNorm, mem.Tunable],
    translation_norm: T.Annotated[ConstScalar, mem.Constant],
    point: T.Annotated[ConstPoint, mem.Constant],
    pixel: T.Annotated[ConstPixel, mem.Constant],
) -> sf.V2:
    """Fixed translation norm + fixed point, variable intrinsics"""
    translation = cam_fixed_norm.translation_direction.to_unit_vector() * translation_norm[0]
    cam_T_world = sf.Pose3(R=cam_fixed_norm.rotation, t=translation)
    
    focal_length, cx, cy, k = cam_fixed_norm.cam_calib
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign(depth))
    r_radial = 1 + k * p.squared_norm()
    pixel_projected = focal_length * r_radial * p + principal_point
    return pixel_projected - pixel


out_dir = Path(__file__).resolve().parent / "generated"
caslib.generate(out_dir)
caslib.compile(out_dir)