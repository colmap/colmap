from pathlib import Path
import sys
import symforce

USE_DOUBLE = (len(sys.argv) > 2 and sys.argv[2] == "--double")
symforce.set_epsilon_to_number(1e-15 if USE_DOUBLE else 1e-6)

import symforce.symbolic as sf
from symforce import typing as T
from symforce.experimental.caspar import CasparLibrary
from symforce.experimental.caspar import memory as mem



class SimpleRadialCalib(sf.V4): pass
class Pose(sf.Pose3): pass
class Point(sf.V3): pass
class ConstPixel(sf.V2): pass
class ConstPoint(sf.V3): pass
class ConstPose(sf.Pose3): pass

caslib = CasparLibrary(dtype=mem.DType.DOUBLE) if USE_DOUBLE else CasparLibrary(dtype=mem.DType.FLOAT)

    
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

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
          Path(__file__).resolve().parent / "generated"
caslib.generate(out_dir)
caslib.compile(out_dir)