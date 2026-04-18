# Usage:
#   python generate_caspar.py <out_dir> [f32|f64]

from pathlib import Path
from itertools import combinations
import sys
import inspect
from typing import Annotated, get_type_hints

# Must be set before importing symforce.symbolic
precision = sys.argv[2] if len(sys.argv) > 2 else "f32"

if precision not in ("f32", "f64"):
    print(f"ERROR: Unknown precision '{precision}'. Expected f32 or f64.")
    sys.exit(1)

import symforce
symforce.set_epsilon_to_number(1e-15 if precision == "f64" else 1e-6)

import symforce.symbolic as sf
from symforce import typing as T
from symforce.experimental.caspar import CasparLibrary
from symforce.experimental.caspar import memory as mem


# --- Shared geometric nodes ---
class Pose(sf.Pose3):      pass
class Point(sf.V3):        pass
class ConstPose(sf.Pose3): pass
class ConstPoint(sf.V3):   pass
class ConstPixel(sf.V2):   pass


# Focal length and the remaining calibration parameters (principal point +
# extra params) are split into separate nodes so each group can be fixed or
# refined independently via BundleAdjustmentOptions.refine_focal_length and
# refine_principal_point / refine_extra_params.
#
# Splitting increases the variant count from 2^3-1=7 to 2^4-1=15 per model.
# Variants where both focal and extra_calib are fixed simultaneously are
# generated but not dispatched (passed as 0-sized to the solver constructor).
# Verify the generated kernels stay within the 48KB GPU shared memory limit
# per thread block if adding further split groups.

# SimpleRadial: params = [f, cx, cy, k]
class SimpleRadialFocal(sf.V1):           pass  # [f]
class ConstSimpleRadialFocal(sf.V1):      pass
class SimpleRadialExtraCalib(sf.V3):      pass  # [cx, cy, k]
class ConstSimpleRadialExtraCalib(sf.V3): pass

# Pinhole: params = [fx, fy, cx, cy]
class PinholeFocal(sf.V2):                pass  # [fx, fy]
class ConstPinholeFocal(sf.V2):           pass
class PinholeExtraCalib(sf.V2):           pass  # [cx, cy]
class ConstPinholeExtraCalib(sf.V2):      pass


# --- Registrar ---
def _make_variant(core_fn, name: str, base_params: list, hints: dict, fixed: dict):
    new_hints = {}
    for p in base_params:
        if p in fixed:
            new_hints[p] = Annotated[fixed[p], mem.ConstantSequential]
        else:
            new_hints[p] = hints[p]

    tunable_params = [p for p in base_params if p not in fixed]
    const_params   = [p for p in base_params if p in fixed]
    ordered = tunable_params + const_params

    # Accept both positional and keyword args, Caspar calls fn(**symbolic_args)
    def wrapper(*args, **kwargs):
        merged = {**dict(zip(ordered, args)), **kwargs}
        return core_fn(*[merged[p] for p in base_params])

    wrapper.__name__ = name
    wrapper.__annotations__ = {p: new_hints[p] for p in ordered}
    wrapper.__annotations__['return'] = hints.get('return')
    wrapper.__signature__ = inspect.Signature([
        inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in ordered
    ])
    return wrapper


def register_camera_model(caslib, model_name: str, core_fn, fixable_params: dict):
    hints         = get_type_hints(core_fn, include_extras=True)
    base_params   = list(inspect.signature(core_fn).parameters.keys())
    fixable_items = list(fixable_params.items())

    # Generate all subsets of size 0..N-1, excluding the all-fixed case
    for r in range(len(fixable_items)):
        for combo in combinations(fixable_items, r):
            fixed = dict(combo)

            if fixed:
                # Build suffix in fixable_params definition order for
                # deterministic naming — must match C++ dispatch order
                suffix = "_".join(
                    f"fixed_{p}" for p, _ in fixable_items if p in fixed
                )
                name = f"{model_name}_{suffix}"
            else:
                name = model_name

            caslib.add_factor(
                _make_variant(core_fn, name, base_params, hints, fixed)
            )


# --- Camera models ---
def simple_radial_core(
    pose:        T.Annotated[Pose,                    mem.TunableShared],
    focal:       T.Annotated[SimpleRadialFocal,       mem.TunableShared],
    extra_calib: T.Annotated[SimpleRadialExtraCalib,  mem.TunableShared],
    point:       T.Annotated[Point,                   mem.TunableShared],
    pixel:       T.Annotated[ConstPixel,              mem.ConstantSequential],
) -> sf.V2:
    cam_T_world = pose
    f = focal[0]
    cx, cy, k = extra_calib
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    r = 1 + k * p.squared_norm()
    return f * r * p + sf.V2([cx, cy]) - pixel


def pinhole_core(
    pose:        T.Annotated[Pose,               mem.TunableShared],
    focal:       T.Annotated[PinholeFocal,       mem.TunableShared],
    extra_calib: T.Annotated[PinholeExtraCalib,  mem.TunableShared],
    point:       T.Annotated[Point,              mem.TunableShared],
    pixel:       T.Annotated[ConstPixel,         mem.ConstantSequential],
) -> sf.V2:
    cam_T_world = pose
    fx, fy = focal
    cx, cy = extra_calib
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    return sf.V2([fx * p[0] + cx, fy * p[1] + cy]) - pixel


dtype  = mem.DType.DOUBLE if precision == "f64" else mem.DType.FLOAT
caslib = CasparLibrary(name="caspar_lib", dtype=dtype)


# Suffix order here defines the generated variant names and must match the
# C++ dispatch function that builds variant names from BundleAdjustmentOptions.
#
# COLMAP flag mapping:
#   refine_rig_from_world   -> pose
#   refine_focal_length     -> focal
#   refine_principal_point
#     || refine_extra_params -> extra_calib
#   refine_points3D         -> point
#
# Known limitations:
#   - constant_rig_from_world_rotation not supported (requires splitting
#     Pose into rotation and translation sub-nodes)
#   - refine_sensor_from_rig not supported (single camera per rig assumed)

# All 15 subsets are generated per model. C++ dispatches the 12 where at
# least one of {focal, extra_calib} is tunable. The 3 fully-fixed-calib
# variants (fixed_focal_fixed_extra_calib, fixed_pose_fixed_focal_fixed_extra_calib,
# fixed_focal_fixed_extra_calib_fixed_point) are generated but passed as
# 0-sized to the solver constructor, so they allocate no GPU memory.

FIXABLE_SIMPLE_RADIAL = {
    'pose':        ConstPose,
    'focal':       ConstSimpleRadialFocal,
    'extra_calib': ConstSimpleRadialExtraCalib,
    'point':       ConstPoint,
}

FIXABLE_PINHOLE = {
    'pose':        ConstPose,
    'focal':       ConstPinholeFocal,
    'extra_calib': ConstPinholeExtraCalib,
    'point':       ConstPoint,
}

register_camera_model(caslib, "simple_radial", simple_radial_core, FIXABLE_SIMPLE_RADIAL)
register_camera_model(caslib, "pinhole",        pinhole_core,        FIXABLE_PINHOLE)

out_dir = Path(f"{sys.argv[1]}")
print(f"Generating Caspar kernels with precision {precision}: {out_dir}")
caslib.generate(out_dir)
