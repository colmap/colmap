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


# Calibration parameters are kept as a single packed node rather than split
# into focal_length / principal_point / extra_params sub-nodes.
#  Splitting also exceeds the 48KB GPU shared memory
# limit per thread block in Caspar's kernel design.
#
# If a future camera model requires one sub-group to be independently
# fixable (e.g. fixing only principal point), the pattern is:
#   1. Define separate node classes for each sub-group:
#        class ModelFocalLength(sf.V1): pass
#        class ConstModelFocalLength(sf.V1): pass
#   2. Split the core function signature accordingly.
#   3. Add each sub-group to FIXABLE_MODEL with its Const counterpart.
#   This will increase variant count (2^N - 1) and shared memory usage,
#   verify the kernel stays within the shared memory limit.

class SimpleRadialCalib(sf.V4):      pass  # [f, cx, cy, k]
class ConstSimpleRadialCalib(sf.V4): pass

class PinholeCalib(sf.V4):           pass  # [fx, fy, cx, cy]
class ConstPinholeCalib(sf.V4):      pass


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

    # Accept both positional and keyword args — Caspar calls fn(**symbolic_args)
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
    pose:  T.Annotated[Pose,               mem.TunableShared],
    calib: T.Annotated[SimpleRadialCalib,  mem.TunableShared],
    point: T.Annotated[Point,              mem.TunableShared],
    pixel: T.Annotated[ConstPixel,         mem.ConstantSequential],
) -> sf.V2:
    cam_T_world = pose
    f, cx, cy, k = calib
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    r = 1 + k * p.squared_norm()
    return f * r * p + sf.V2([cx, cy]) - pixel


def pinhole_core(
    pose:  T.Annotated[Pose,          mem.TunableShared],
    calib: T.Annotated[PinholeCalib,  mem.TunableShared],
    point: T.Annotated[Point,         mem.TunableShared],
    pixel: T.Annotated[ConstPixel,    mem.ConstantSequential],
) -> sf.V2:
    cam_T_world = pose
    fx, fy, cx, cy = calib
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
#   refine_rig_from_world -> pose
#   refine_focal_length && refine_principal_point
#     && refine_extra_params (all three) -> calib
#   refine_points3D -> point
#
# Known limitations:
#   - calib is all-or-nothing (see comment above)
#   - constant_rig_from_world_rotation not supported (requires splitting
#     Pose into rotation and translation sub-nodes)
#   - refine_sensor_from_rig not supported (single camera per rig assumed)

FIXABLE_SIMPLE_RADIAL = {
    'pose':  ConstPose,
    'point': ConstPoint,
}

FIXABLE_PINHOLE = {
    'pose':  ConstPose,
    'point': ConstPoint,
}

register_camera_model(caslib, "simple_radial", simple_radial_core, FIXABLE_SIMPLE_RADIAL)
register_camera_model(caslib, "pinhole",        pinhole_core,        FIXABLE_PINHOLE)

out_dir = Path(f"{sys.argv[1]}")
print(f"Generating Caspar kernels with precision {precision}: {out_dir}")
caslib.generate(out_dir)
