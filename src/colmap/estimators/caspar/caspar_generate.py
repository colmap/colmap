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


# Focal length and distortion are merged into one node (focal_and_extra) and
# principal point forms a separate node.  This means the typical default
# (refine_focal=True, refine_pp=False, refine_extra=True) hits the
# FIXED_PRINCIPAL_POINT variant instead of BASE, saving GPU work.
#
# All 2^4-1 = 15 non-fully-fixed subsets are generated and dispatched.
# Verify the generated kernels stay within the 48 KB GPU shared memory limit
# per thread block if adding further split groups.

# SimpleRadial: params = [f, cx, cy, k]
class SimpleRadialFocalAndExtra(sf.V2):         pass  # [f, k]
class ConstSimpleRadialFocalAndExtra(sf.V2):    pass
class SimpleRadialPrincipalPoint(sf.V2):        pass  # [cx, cy]
class ConstSimpleRadialPrincipalPoint(sf.V2):   pass

# Pinhole: params = [fx, fy, cx, cy]
class PinholeFocalAndExtra(sf.V2):              pass  # [fx, fy]
class ConstPinholeFocalAndExtra(sf.V2):         pass
class PinholePrincipalPoint(sf.V2):             pass  # [cx, cy]
class ConstPinholePrincipalPoint(sf.V2):        pass


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
    pose:            T.Annotated[Pose,                       mem.TunableShared],
    focal_and_extra: T.Annotated[SimpleRadialFocalAndExtra,  mem.TunableShared],
    principal_point: T.Annotated[SimpleRadialPrincipalPoint, mem.TunableShared],
    point:           T.Annotated[Point,                      mem.TunableShared],
    pixel:           T.Annotated[ConstPixel,                 mem.ConstantSequential],
) -> sf.V2:
    cam_T_world = pose
    f, k = focal_and_extra
    cx, cy = principal_point
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    r = 1 + k * p.squared_norm()
    return f * r * p + sf.V2([cx, cy]) - pixel


def pinhole_core(
    pose:            T.Annotated[Pose,                    mem.TunableShared],
    focal_and_extra: T.Annotated[PinholeFocalAndExtra,    mem.TunableShared],
    principal_point: T.Annotated[PinholePrincipalPoint,   mem.TunableShared],
    point:           T.Annotated[Point,                   mem.TunableShared],
    pixel:           T.Annotated[ConstPixel,              mem.ConstantSequential],
) -> sf.V2:
    cam_T_world = pose
    fx, fy = focal_and_extra
    cx, cy = principal_point
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
#   refine_rig_from_world                       -> pose
#   refine_focal_length && refine_extra_params  -> focal_and_extra
#   refine_principal_point                      -> principal_point
#   refine_points3D                             -> point
#
# Known limitations:
#   - constant_rig_from_world_rotation not supported (requires splitting
#     Pose into rotation and translation sub-nodes)
#   - refine_sensor_from_rig not supported (single camera per rig assumed)
#   - refine_focal_length != refine_extra_params not supported (observations
#     skipped with a warning; cannot split the merged focal_and_extra block)

FIXABLE_SIMPLE_RADIAL = {
    'pose':            ConstPose,
    'focal_and_extra': ConstSimpleRadialFocalAndExtra,
    'principal_point': ConstSimpleRadialPrincipalPoint,
    'point':           ConstPoint,
}

FIXABLE_PINHOLE = {
    'pose':            ConstPose,
    'focal_and_extra': ConstPinholeFocalAndExtra,
    'principal_point': ConstPinholePrincipalPoint,
    'point':           ConstPoint,
}

register_camera_model(caslib, "simple_radial", simple_radial_core, FIXABLE_SIMPLE_RADIAL)
register_camera_model(caslib, "pinhole",        pinhole_core,        FIXABLE_PINHOLE)

out_dir = Path(f"{sys.argv[1]}")
print(f"Generating Caspar kernels with precision {precision}: {out_dir}")
caslib.generate(out_dir)
