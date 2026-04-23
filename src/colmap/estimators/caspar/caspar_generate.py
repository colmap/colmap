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
# Point and ConstPoint are shared: the same 3D point can be observed by
# cameras of different types, so they must live in the same node pool.
class Point(sf.V3):     pass
class ConstPoint(sf.V3): pass
class ConstPixel(sf.V2): pass


# Calibration node strategy:
#
# When both focal_and_extra and principal_point are tunable, they are merged
# into a single shared-memory node (Calib) to reduce per-block node overhead.
# This saves one shared-memory node slot in the four variants where both
# intrinsic groups are variable:
#   BASE, FIXED_POSE, FIXED_POINT, FIXED_POSE_FIXED_POINT
# (using merged calib node)
#
# When either group is fixed it leaves shared memory, so the remaining tunable
# group keeps its dedicated node.  The 11 split variants cover all cases where
# at least one of focal_and_extra / principal_point is fixed.
#
# All 4 + 11 = 15 non-fully-fixed subsets are generated and dispatched.
# Verify the generated kernels stay within the 48 KB GPU shared memory limit
# per thread block if adding further split groups.
#
# Pose and Calib/FocalAndExtra/PrincipalPoint are camera-model-specific so
# that factors from different camera models are never batched into the same
# Caspar block, avoiding cross-model shared memory costs.

# SimpleRadial: params = [f, cx, cy, k]
class SimpleRadialPose(sf.Pose3):               pass
class ConstSimpleRadialPose(sf.Pose3):          pass
class SimpleRadialCalib(sf.V4):                 pass  # [f, k, cx, cy]  (merged)
class ConstSimpleRadialCalib(sf.V4):            pass
class SimpleRadialPrincipalPoint(sf.V2):        pass  # [cx, cy]  (split: pp tunable)
class ConstSimpleRadialPrincipalPoint(sf.V2):   pass
class SimpleRadialFocalAndExtra(sf.V2):         pass  # [f, k]    (split: focal tunable)
class ConstSimpleRadialFocalAndExtra(sf.V2):    pass


# Pinhole: params = [fx, fy, cx, cy]
class PinholePose(sf.Pose3):                    pass
class ConstPinholePose(sf.Pose3):               pass
class PinholeCalib(sf.V4):                      pass  # [fx, fy, cx, cy]  (merged)
class ConstPinholeCalib(sf.V4):                 pass
class PinholePrincipalPoint(sf.V2):             pass  # [cx, cy]  (split: pp tunable)
class ConstPinholePrincipalPoint(sf.V2):        pass
class PinholeFocalAndExtra(sf.V2):              pass  # [fx, fy]  (split: focal tunable)
class ConstPinholeFocalAndExtra(sf.V2):         pass


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


def register_camera_model(caslib, model_name: str, core_fn, fixable_params: dict,
                          must_fix_one_of: T.Optional[set] = None,
                          include_all_fixed: bool = False):
    hints         = get_type_hints(core_fn, include_extras=True)
    base_params   = list(inspect.signature(core_fn).parameters.keys())
    fixable_items = list(fixable_params.items())

    # Generate all subsets of size 0..N-1, excluding the all-fixed case unless
    # include_all_fixed=True (used for merged calib where the all-fixed subset
    # still has a tunable calib node).
    max_r = len(fixable_items) + (1 if include_all_fixed else 0)
    for r in range(max_r):
        for combo in combinations(fixable_items, r):
            fixed = dict(combo)

            # Skip variants where none of the required params are fixed.
            # Used to exclude merged-calib cases from the split registration.
            if must_fix_one_of and not any(p in fixed for p in must_fix_one_of):
                continue

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

# Merged cores: canonical math used by both merged and split variants.
# TODO: temporary merged-calib node until shared-memory layout is profiled/tuned.
# When both focal_and_extra and principal_point are tunable, a single V4 Calib
# node replaces two V2 nodes, saving one per-block shared-memory slot.

def simple_radial_merged_core(
    pose:  T.Annotated[SimpleRadialPose,  mem.TunableShared],
    calib: T.Annotated[SimpleRadialCalib, mem.TunableShared],  # [f, k, cx, cy]
    point: T.Annotated[Point,             mem.TunableShared],
    pixel: T.Annotated[ConstPixel,        mem.ConstantSequential],
) -> sf.V2:
    cam_T_world = pose
    f, k, cx, cy = calib
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    r = 1 + k * p.squared_norm()
    return f * r * p + sf.V2([cx, cy]) - pixel


def pinhole_merged_core(
    pose:  T.Annotated[PinholePose,  mem.TunableShared],
    calib: T.Annotated[PinholeCalib, mem.TunableShared],  # [fx, fy, cx, cy]
    point: T.Annotated[Point,        mem.TunableShared],
    pixel: T.Annotated[ConstPixel,   mem.ConstantSequential],
) -> sf.V2:
    cam_T_world = pose
    fx, fy, cx, cy = calib
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    return sf.V2([fx * p[0] + cx, fy * p[1] + cy]) - pixel


# Split cores: delegate to merged cores to avoid math duplication.
# Used only for variants where at least one of focal_and_extra / principal_point
# is fixed (must_fix_one_of constraint applied at registration).

def simple_radial_core(
    pose:            T.Annotated[SimpleRadialPose,            mem.TunableShared],
    focal_and_extra: T.Annotated[SimpleRadialFocalAndExtra,   mem.TunableShared],
    principal_point: T.Annotated[SimpleRadialPrincipalPoint,  mem.TunableShared],
    point:           T.Annotated[Point,                       mem.TunableShared],
    pixel:           T.Annotated[ConstPixel,                  mem.ConstantSequential],
) -> sf.V2:
    calib = sf.V4([focal_and_extra[0], focal_and_extra[1],
                   principal_point[0], principal_point[1]])
    return simple_radial_merged_core(pose, calib, point, pixel)


def pinhole_core(
    pose:            T.Annotated[PinholePose,                 mem.TunableShared],
    focal_and_extra: T.Annotated[PinholeFocalAndExtra,        mem.TunableShared],
    principal_point: T.Annotated[PinholePrincipalPoint,       mem.TunableShared],
    point:           T.Annotated[Point,                       mem.TunableShared],
    pixel:           T.Annotated[ConstPixel,                  mem.ConstantSequential],
) -> sf.V2:
    calib = sf.V4([focal_and_extra[0], focal_and_extra[1],
                   principal_point[0], principal_point[1]])
    return pinhole_merged_core(pose, calib, point, pixel)


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

FIXABLE_SIMPLE_RADIAL_MERGED = {
    'pose':  ConstSimpleRadialPose,
    'point': ConstPoint,
}

FIXABLE_PINHOLE_MERGED = {
    'pose':  ConstPinholePose,
    'point': ConstPoint,
}

FIXABLE_SIMPLE_RADIAL = {
    'pose':            ConstSimpleRadialPose,
    'focal_and_extra': ConstSimpleRadialFocalAndExtra,
    'principal_point': ConstSimpleRadialPrincipalPoint,
    'point':           ConstPoint,
}

FIXABLE_PINHOLE = {
    'pose':            ConstPinholePose,
    'focal_and_extra': ConstPinholeFocalAndExtra,
    'principal_point': ConstPinholePrincipalPoint,
    'point':           ConstPoint,
}

# Merged: 4 variants per model (r=0..2 over {pose, point}, including all-fixed).
# Covers BASE, FIXED_POSE, FIXED_POINT, FIXED_POSE_FIXED_POINT — the 4 cases
# where both focal_and_extra and principal_point are tunable.
register_camera_model(caslib, "simple_radial_merged",
                      simple_radial_merged_core, FIXABLE_SIMPLE_RADIAL_MERGED,
                      include_all_fixed=True)
register_camera_model(caslib, "pinhole_merged",
                      pinhole_merged_core, FIXABLE_PINHOLE_MERGED,
                      include_all_fixed=True)

# Split: 11 variants per model — all cases where at least one of
# {focal_and_extra, principal_point} is fixed.
register_camera_model(caslib, "simple_radial", simple_radial_core, FIXABLE_SIMPLE_RADIAL,
                      must_fix_one_of={'focal_and_extra', 'principal_point'})
register_camera_model(caslib, "pinhole", pinhole_core, FIXABLE_PINHOLE,
                      must_fix_one_of={'focal_and_extra', 'principal_point'})

out_dir = Path(f"{sys.argv[1]}")
print(f"Generating Caspar kernels with precision {precision}: {out_dir}")
caslib.generate(out_dir)
