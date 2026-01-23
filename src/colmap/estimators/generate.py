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

class PinholeCalib(sf.V4):
    pass

class Pose(sf.Pose3):
    pass

class Point(sf.V3):
    pass

# Constant type variations
class ConstPixel(sf.V2):
    pass

class ConstPoint(sf.V3):
    pass

class ConstSimpleRadialCalib(sf.V4):
    pass

class ConstPinholeCalib(sf.V4):
    pass

class ConstPose(sf.Pose3):
    pass

caslib = CasparLibrary()


# Simple Radial factors
@caslib.add_factor
def simple_radial(
    pose: T.Annotated[Pose, mem.TunableShared],
    calib: T.Annotated[SimpleRadialCalib, mem.TunableShared],
    point: T.Annotated[Point, mem.TunableShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
) -> sf.V2:
    cam_T_world = pose
    focal_length, cx, cy, k = calib
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    r = 1 + k * p.squared_norm()
    pixel_projected = focal_length * r * p + principal_point
    return pixel_projected - pixel


@caslib.add_factor
def simple_radial_fixed_pose(
    calib: T.Annotated[SimpleRadialCalib, mem.TunableShared],
    point: T.Annotated[Point, mem.TunableShared],
    cam_T_world: T.Annotated[ConstPose, mem.ConstantSequential],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
) -> sf.V2:
    focal_length, cx, cy, k = calib
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    r = 1 + k * p.squared_norm()
    pixel_projected = focal_length * r * p + principal_point
    return pixel_projected - pixel


@caslib.add_factor
def simple_radial_fixed_point(
    pose: T.Annotated[Pose, mem.TunableShared],
    calib: T.Annotated[SimpleRadialCalib, mem.TunableShared],
    point: T.Annotated[ConstPoint, mem.ConstantSequential],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
) -> sf.V2:
    cam_T_world = pose
    focal_length, cx, cy, k = calib
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    r = 1 + k * p.squared_norm()
    pixel_projected = focal_length * r * p + principal_point
    return pixel_projected - pixel


# Pinhole factors
@caslib.add_factor
def pinhole(
    pose: T.Annotated[Pose, mem.TunableShared],
    calib: T.Annotated[PinholeCalib, mem.TunableShared],
    point: T.Annotated[Point, mem.TunableShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
) -> sf.V2:
    cam_T_world = pose
    fx, fy, cx, cy = calib
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    pixel_projected = sf.V2([fx * p[0], fy * p[1]]) + principal_point
    return pixel_projected - pixel


@caslib.add_factor
def pinhole_fixed_pose(
    calib: T.Annotated[PinholeCalib, mem.TunableShared],
    point: T.Annotated[Point, mem.TunableShared],
    cam_T_world: T.Annotated[ConstPose, mem.ConstantSequential],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
) -> sf.V2:
    fx, fy, cx, cy = calib
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    pixel_projected = sf.V2([fx * p[0], fy * p[1]]) + principal_point
    return pixel_projected - pixel


@caslib.add_factor
def pinhole_fixed_point(
    pose: T.Annotated[Pose, mem.TunableShared],
    calib: T.Annotated[PinholeCalib, mem.TunableShared],
    point: T.Annotated[ConstPoint, mem.ConstantSequential],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
) -> sf.V2:
    cam_T_world = pose
    fx, fy, cx, cy = calib
    principal_point = sf.V2([cx, cy])
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    pixel_projected = sf.V2([fx * p[0], fy * p[1]]) + principal_point
    return pixel_projected - pixel


out_dir = Path(__file__).resolve().parent / "generated"
caslib.generate(out_dir)
caslib.compile(out_dir)