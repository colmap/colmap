# Telemetry-native COLMAP cleanup: implementation contract

This is the complete handoff for the agent that will clean, rebuild, and
validate this fork. It is intentionally a planning artifact and **must not be
committed**. The implementation is complete only when every acceptance gate in
this document passes.

## 1. Product goal and non-negotiable boundary

The fork has one job:

> Import per-image WGS84 camera positions, position uncertainty, a camera-frame
> gravity-down vector, and optionally an uncertainty-weighted true-north camera
> heading; use all supplied telemetry as robust weighted constraints while global SfM establishes and
> preserves a metric ENU gauge; write a georeferenced reconstruction that
> LichtFeld Studio can train without changing coordinates; and publish the
> WGS84/ECEF-to-geometry transforms a phone viewer needs to place live GPS
> positions in the trained splat.

Plain statement of what the weighted constraints are for, since every
implementation choice below serves it:

> GPS says *where* the camera was. Gravity says *which way was down* and
> constrains pitch/roll without constraining yaw. An optional compass heading
> says which horizontal direction the camera faced and constrains only yaw.
> Every constraint is weighted by stated uncertainty and protected by a robust
> loss; none is a hard camera-pose assignment.

Keep a branch-added behavior only if it is required by that sentence and is
covered by a focused test or exercised by the end-to-end run. Remove every
other branch-added behavior. Do not retain speculative generality for a future
PR.

These constraints are final:

- Input positions are WGS84 latitude, longitude, and **ellipsoidal** altitude.
- Position uncertainty is supplied per image as either three standard
  deviations or one full symmetric covariance, expressed in the measurement's
  local ENU tangent frame.
- Gravity is a unit vector pointing down in the COLMAP camera sensor frame.
- Optional heading is the clockwise azimuth of the COLMAP camera `+Z` optical
  axis projected onto the local horizontal plane, measured from **true north**.
  It is a scalar one-degree-of-freedom observation, not a full quaternion or a
  pitch/roll prior. Each heading row carries its own `1 sigma` uncertainty; a
  reading believed accurate to about five degrees uses `HEADING_STD_DEG=5`.
- Magnetic-compass data must be corrected for magnetic declination before
  import. Body/drone headings must be converted through a known body-to-camera
  extrinsic before import. The archive always describes the camera, never an
  unspecified device body.
- Heading import/storage/reporting are supported, while optimization is
  explicitly requested with `pose_prior_use_heading`. The production street
  run has no trusted compass source and sets this to `0`; a later run can enable
  it without changing the database schema or solver design.
- The production mapper is `global_mapper`. Do not extend the incremental
  mapper or `pose_prior_mapper` for this fork.
- Position priors are either off or optimized as constraints. The unweighted
  `initialize` mode is not part of this product.
- Gravity is a soft, yaw-invariant bundle-adjustment constraint. Upstream
  `ra_use_gravity` remains available but the production run sets it to `0` so
  the same gravity observation is not also imposed as a hard rotation-averaging
  reduction.
- The delivered geometry frame is `LICHTFELD_COLMAP`. `ENU_Z_UP` remains as a
  diagnostic/reference output. No GLTF-specific output is needed.
- The command-line executable is the deliverable. Remove all branch-added
  pycolmap API and tests.
- Build CUDA only. The RTX 3070 Ti is compute capability 8.6, so `sm_86` is the
  exact target. Do not produce a CPU-only build or a multi-architecture binary.
- The CUDA build must include SiftGPU, Caspar, and cuDSS. A CPU linear-solver
  retry inside that GPU binary is required as the fail-safe for a cuDSS solve;
  it is not a separate CPU build.
- **Caspar is built and validated but is not exercised by the production run,
  and that is expected.** The current `OPENCV_FISHEYE` solve selects
  `SPARSE_SCHUR / CUDA_SPARSE`; Caspar remains a validated GPU capability for
  supported camera models. Any build comment must state only that general
  capability boundary, never a private dataset or future-capture story.
- The few gross GPS outliers, including kilometre-scale observations, are
  valid archive rows. Robust estimation must reject them and the residual CSV
  must identify them. Do not add a hard five-kilometre import bound.

## 2. Known-good baseline that the cleanup must preserve

Comparison base: upstream commit `7e7b86ec`.

Current fork baseline: `ad48af9c4fd18f04c79238435b58a7b3b6288a17`.

The current branch differs from that upstream base in 65 files by
`+10,630/-140`. The existing successful run at
`E:\street\colmap_test3_ad48af9c` established this regression baseline:

| Measurement | Baseline |
|---|---:|
| Images registered | 965 / 965 |
| Sparse points | 203,296 |
| Mean reprojection error | 0.9117 px |
| Quadratic sequential pairs vetoed by GPS | 4,508 |
| Position priors used | 965 |
| Initial position-gauge RANSAC inliers | 899 |
| Position RMSE | 3.9942 m to 1.6582 m |
| Position residuals in each later BA | 965 |
| Gravity residuals in each later BA | 965 |
| Final `model_aligner` support | 965 / 965 |
| Metres per input SfM unit | 1.000000391888 |
| Gravity consistency | 1.698 degrees |
| Final correction | 0.000123 degrees, 0.00011 m, scale delta about 3.9e-7 |

The historical `initialize` run produced about 21.19 metres per SfM unit. That
is evidence for deleting `initialize`, not a behavior to preserve.

The current summary-level `georeference.json` omits horizontal residual
median/P90 fields needed by the pipeline. The cleanup must always emit one
complete report; retaining the current summary/full split is not acceptable.

## 3. Final public surface

### 3.1 Retain

Retain and harden only the following branch-added surface:

| Area | Retained public surface |
|---|---|
| Import | `colmap pose_prior_importer --database_path ... --pose_prior_path ... --existing {error,replace}` |
| Sequential matching | `--SequentialMatching.max_prior_distance`; negative disables it |
| Global mapping | `--GlobalMapper.pose_prior_position_mode {off,optimize}` |
| Global mapping | `--GlobalMapper.pose_prior_use_gravity {0,1}` |
| Global mapping | `--GlobalMapper.pose_prior_gravity_stddev_deg`; production runner passes `5.0` explicitly |
| Global mapping | `--GlobalMapper.pose_prior_use_heading {0,1}`; default `0`, with uncertainty supplied per heading row |
| Determinism | existing mapper/matcher seed options plus branch-added `--alignment_random_seed` |
| Report output | `--scene_id`, `--georeference_json`, `--camera_residuals_csv` |
| Geometry output | `--output_coordinate_frame {ENU_Z_UP,LICHTFELD_COLMAP}` |

Existing upstream `model_aligner` options remain for its existing non-report
paths. Do not delete or reinterpret upstream coordinate systems or standard
alignment modes. The narrowed rules apply only to the branch-added archive,
weighted-prior, report, and output-frame path.

### 3.2 Delete completely

Delete implementation, parsing, serialization, bindings, tests, and
documentation for every item below. A deleted option must become an unknown
option; do not leave ignored compatibility aliases.

- Full absolute-orientation machinery. Delete `PosePrior::rotation`,
  `rotation_covariance`, quaternion/covariance archive fields, their BLOB
  serialization and migrations, `PosePriorRotationMode`, rotation
  initialization, absolute-rotation BA costs, orientation-assisted
  `model_aligner` refinement, and their options/tests/docs. A scalar weighted
  heading prior replaces none of those objects and must not be implemented by
  disguising a yaw measurement as a full quaternion.
- Cartesian pose-prior archives and report assumptions:
  - Archive `TX/TY/TZ`, `CartesianFrame`, `enu_origin`, and translation/rotation
    convention machinery.
  - `--pose_prior_cartesian_frame`.
  - Do **not** remove upstream `PosePrior::CoordinateSystem::CARTESIAN`; only
    the new JSON importer/report workflow is WGS84-only.
- Flexible/partial archive behavior:
  - Unknown-column ignore policy.
  - Partial position, covariance-only, gravity-only, and horizontal-only rows.
  - Merge semantics, `UpdatePosePriors`, unresolved-row skipping, and any
    partial update of an existing prior.
  - The `merge` value of `--existing`.
- Unused/tuning-only position modes and knobs:
  - `PosePriorPositionMode::initialize`.
  - `pose_prior_position_fallback_stddev` — the **branch-added** knob on
    `GlobalPositionerOptions`/`GlobalMapper` (absent from upstream). Do **not**
    touch upstream's similarly-named `prior_position_fallback_stddev`: it is a
    mandatory parameter of upstream `AlignReconstructionToPosePriors` guarded by
    `THROW_CHECK_GT(..., 0.0)`, and also a field of upstream
    `PosePriorBundleAdjustmentOptions`. Preserve its default and all upstream
    callers; the strict branch entry point does not accept it. See §4.6a.
  - `pose_prior_position_loss_scale` and
    `pose_prior_position_ransac_max_error` CLI options.
  - Duplicate BA knobs such as `ba_use_robust_loss_on_prior_position` and
    `ba_prior_position_loss_scale` when their only purpose is to make the
    required robust loss optional/tunable.
- Unused/tuning-only gravity surface:
  - `PosePriorGravityBAMode`; replace its two-state behavior with the boolean
    `pose_prior_use_gravity`.
  - `pose_prior_gravity_loss_scale` CLI/API. The standardized robust-loss
    radius is a named fixed constant.
  - Gravity hard-admission caps, consensus/MAD rejection, admission state, and
    all related CLI/report fields. Valid nonzero direction observations remain
    visible and are handled by the fixed robust loss.
- Unused/tuning-only heading surface:
  - Global fallback heading uncertainty. Every present heading row must state
    `HEADING_STD_DEG`; there is no CLI value that silently substitutes for it.
  - Hard heading initialization and hard heading rejection thresholds.
- Alternate alignment/report paths:
  - `AnisotropicEnuPositionEstimator`, `AnisotropicPositionGate`,
    `--alignment_max_horizontal_error`, and
    `--alignment_max_vertical_error`. Full covariance already supplies the
    correct anisotropic metric.
  - Explicit report origins: `--enu_origin_lat/lon/alt`.
  - `GeoreferenceReportLevel`, `--georeference_report_level`, and all
    summary/full branches.
  - CLI overrides for quality-warning and material-realignment thresholds.
    Keep one named constant for each policy and test its boundary.
  - `frame_contract.targets[]`.
  - `GLTF_Y_UP` and all GLTF-only transforms/tests.
- Branch changes to `IncrementalPipeline`, `IncrementalMapper`, and their
  tests. Revert those files to the upstream comparison base unless a line is
  independently required by upstream code after the cleanup.
- Every branch-added change under `src/pycolmap/`. The actual workflow uses
  the CLI and does not build or ship a pycolmap wheel.

### 3.3 Fixed policy constants

Define each value once next to the code that applies it; serialize the same
value in the report. Do not expose these as CLI options:

| Constant | Value | Meaning |
|---|---:|---|
| Position RANSAC/robust radius | `sqrt(7.814727903251179)` | 95% chi-square radius, 3 DoF, after covariance whitening |
| Gravity robust radius | `sqrt(5.991464547107979)` | 95% chi-square radius, 2 local DoF, after angular whitening |
| Heading robust radius | `sqrt(3.841458820694124)` | 95% chi-square radius, 1 DoF, after angular whitening |
| Collinearity warning | `0.1` | second/first horizontal singular-value ratio below this warns |
| Gravity warning/failure | `3.0 deg` | median registered-camera gravity disagreement above this fails delivery |
| Minimum position-inlier ratio | `0.8` | lower support fails delivery |
| Material rotation | `0.5 deg` | final correction above this fails delivery |
| Material translation | `1.0 m` | final correction above this fails delivery |
| Material scale delta | `0.01` | `abs(scale - 1)` above this fails delivery |
| Minimum horizontal heading projection norm | `1e-3` | below this, the camera forward direction is too close to vertical for yaw to be defined |

Collinearity is informational because a street capture is naturally elongated.
The position, gravity, and material-realignment limits are hard delivery gates.

## 4. Required implementation

Execute the following sections in order. Compile and run the directly affected
tests after each section; do not postpone all integration work until the end.

### 4.1 Narrow the archive to the actual sensor contract

Refactor `src/colmap/geometry/pose_prior_io.h` and `.cc` to accept exactly one
versioned JSON contract:

```json
{
  "schema_version": 1,
  "coordinate_system": "WGS84",
  "sensor_type": "CAMERA",
  "ellipsoid": "WGS84",
  "height_datum": "ELLIPSOIDAL",
  "position_covariance_frame": "LOCAL_ENU",
  "gravity_frame": "CAMERA",
  "gravity_direction": "DOWN",
  "heading_reference": "TRUE_NORTH",
  "heading_axis": "CAMERA_FORWARD_PROJECTED_HORIZONTAL",
  "heading_rotation": "CLOCKWISE_FROM_NORTH",
  "schema": [
    "NAME", "LAT", "LON", "ALT",
    "STD_TX", "STD_TY", "STD_TZ",
    "GX", "GY", "GZ",
    "HEADING_DEG", "HEADING_STD_DEG"
  ],
  "data": [
    ["image.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, 0.0, 1.0, 0.0, 92.0, 5.0],
    ["image-2.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, null, null, null, null, null]
  ]
}
```

The covariance form replaces `STD_TX/STD_TY/STD_TZ` with
`COV_TXX/COV_TXY/COV_TXZ/COV_TYY/COV_TYZ/COV_TZZ`. Column order may vary, but
the required columns must contain exactly one uncertainty form. The only
optional schema groups are `GX/GY/GZ` and
`HEADING_DEG/HEADING_STD_DEG`; a group is either wholly present or wholly
absent from `schema`. No other column is legal.

Metadata is conditional and exact:

- `gravity_frame` and `gravity_direction` are required exactly as shown when
  the gravity group is in `schema`, and forbidden otherwise.
- `heading_reference`, `heading_axis`, and `heading_rotation` are required
  exactly as shown when the heading group is in `schema`, and forbidden
  otherwise.
- The heading group requires the gravity group. A heading is defined relative
  to the horizontal plane established by the measured camera-frame down vector.
- A magnetic sensor must be declination-corrected to true north before writing
  the archive. A non-camera sensor must be transformed through a known rigid
  body-to-camera extrinsic before writing the archive.

Validation is fail-closed:

- Reject unknown, missing, or conditionally forbidden top-level keys and any
  unsupported `schema_version`.
- Require at least one data row and exactly one cell per schema column.
- Require a non-empty unique `NAME` for every row.
- Require every name to resolve to a database image. Report all unresolved
  names in one error before modifying the database.
- Require finite latitude, longitude, altitude, and uncertainty on every row.
- Require latitude in `[-90, 90]` and longitude in `[-180, 180]`.
- Require each standard deviation to be strictly positive.
- Reconstruct the symmetric covariance and require it to be strictly positive
  definite. A merely finite, zero, singular, or indefinite matrix is invalid.
- For each optional row group, accept either all finite numeric values or JSON
  `null` in every member; reject partial-null groups and non-finite values. The
  existing property-tree reader exposes JSON `null` as an empty cell, so handle
  that representation intentionally and test it. Do not add a second JSON
  dependency solely to distinguish lexical number tokens produced by the
  trusted adapter.
- For a present gravity row, accept a norm within `0.01` of one and normalize
  it; reject zero or a larger norm error. `GoPro:GravityVector` is already a
  normalized direction; do not compare its norm to `9.80665`.
- For a present heading row, require a present gravity row,
  `0 <= HEADING_DEG < 360`, and `0 < HEADING_STD_DEG <= 180`.
- Do not reject a geographically distant but otherwise valid row. Robust
  fitting, not the archive parser, decides whether it is an outlier.

`ToPosePriors` must return exactly one prior with complete position data per row
or fail. Optional gravity/heading groups may be absent only according to the
rules above. Remove APIs whose purpose was partial position merging or optional
image resolution.

Refactor `RunPosePriorImporter` in `src/colmap/exe/database.cc` so it validates
the complete archive and all image bindings before starting writes, then uses a
single SQLite transaction:

- `--existing=error`: fail if any target image already has a prior; write
  nothing.
- `--existing=replace`: update the full existing record or insert a full new
  record; no field preservation and no partial merge. A JSON-null optional
  group clears any previously stored gravity/heading values for that image.
- On any error, roll back every row.
- On success, log one machine-assertable line containing archive rows,
  resolved images, inserted rows, and replaced rows.

Update the telemetry adapter that produces the archive so it writes this exact
schema. Do not put producer-specific names, capture history, or run commentary
in COLMAP source or documentation.

### 4.2 Keep the database extension minimal and migration-safe

In `src/colmap/geometry/pose_prior.h` and `.cc`:

- Retain upstream position, position covariance, coordinate system, and
  gravity fields.
- Retain the stricter `HasGravity()` behavior that rejects finite zero/near-zero
  vectors and its focused test.
- Retain `IsValidPositionCovariance`, but give it the short contract: finite,
  symmetric, and strictly positive definite.
- Add scalar `heading_rad` and `heading_stddev_rad`, initialized to NaN, plus
  `HasHeading()`. It returns true only when both are finite, heading is in the
  canonical wrapped range `[0, 2*pi)`, and `0 < heading_stddev_rad <= pi`.
- Delete branch-added quaternion orientation members, helpers, stream output,
  and serialization. Heading is intentionally not represented as a quaternion.

In `src/colmap/scene/database_sqlite.cc` and database tests:

- Add nullable REAL columns `heading_rad` and `heading_stddev_rad`, bind/read
  them together, and cover absent/present values with round-trip tests.
- Keep each migration guarded by `ExistsColumn` and additive, matching the
  adjacent `maybe_add_two_view_geometries_blob_column` idiom. Nullable columns
  that nothing writes cost one NULL per row.
- Keep reads on explicitly named columns. Never `SELECT *` on `pose_priors`:
  named columns are what makes a database written by one build readable by
  another regardless of which optional columns exist.

Databases created by the pre-cleanup fork may physically retain unused
`rotation` BLOB columns. Do not destructively rebuild those databases merely to
remove them. New named-column queries ignore them safely; new databases and new
code use only the scalar heading columns.

### 4.3 Use one deterministic WGS84-to-ENU implementation

Create a small shared utility named `PosePriorEnuFrame` in
`src/colmap/geometry/pose_prior_transform.h/.cc`. It must be the only
branch-added implementation used by `DatabaseCache`, global positioning, and
the report path.

The utility must:

1. Sort usable priors by stable image/data identifier.
2. Convert all WGS84 positions to ECEF.
3. Choose the ECEF geometric median and convert it back to WGS84 as the shared
   tangent origin.
4. Store `origin_wgs84`, `origin_ecef`, `enu_from_ecef`, and its inverse.
5. Convert a prior position into the shared ENU frame.
6. Return `SharedFromLocalEnu(position)` for that observation's WGS84 tangent
   frame. Heading/gravity use this to obtain per-row true north and down in the
   shared gauge; do not approximate them with one constant direction over the
   whole scene.
7. Rotate a covariance from the prior's local ENU frame into the shared ENU
   frame using
   `R_shared_from_local * C_local * R_shared_from_local.transpose()`.

Retain the branch's `ENUFromECEF`/`ECEFFromENU` helpers in `gps.cc`; this is a
small useful deduplication and fixes the transform naming convention in one
place.

Replace all duplicate origin selection and covariance conversion in
`database_cache.cc` and `model_georeference.cc` with this utility. Remove
explicit-origin branches. Given identical priors, separate mapper and report
processes must compute bitwise-identical origins and transforms on the same
platform.

If more than one usable camera prior maps to the same rig frame, fail with a
clear unsupported-input error. Do not keep inverse-trace selection or invent
multi-camera fusion for a single-camera workflow.

### 4.4 Replace `GeometricMedian` with the robust algorithm it claims to be

In `src/colmap/math/geometric_median.h`:

- Throw on an empty input.
- Use the modified Weiszfeld/Vardi-Zhang update so an iterate coinciding with a
  sample is handled mathematically instead of silently dropping that sample.
- Use finite convergence and iteration limits, and fail clearly if the input or
  result is non-finite.
- Make the result deterministic for a fixed ordered input.
- Keep the implementation generic only to the extent already needed for
  `Eigen::Vector3d`; remove unused template flexibility if it complicates the
  implementation.

Add `src/colmap/math/geometric_median_test.cc` and register it in CMake. Test:
empty input, one point, identical points, two points, a symmetric set, a
coincident dominant sample, collinear data, a gross outlier, ECEF-scale
coordinates, input permutations, convergence, and finite output.

### 4.5 Retain GPS-gated sequential matching, but make it WGS84-specific

Keep `SequentialMatching.max_prior_distance` because it eliminated 4,508
unhelpful quadratic probes without losing registrations.

Clean `controllers/pairing.h/.cc` as follows:

- Always keep the base sequential-overlap window. Gate only quadratic probes
  whose index offset is larger than `overlap`.
- Convert usable WGS84 positions to ECEF and compare 3D metric distance.
- If either prior is absent or is not WGS84, do not gate that pair.
- Remove the current behavior that treats `UNDEFINED` as Cartesian.
- Preserve deterministic pair order.
- Keep the veto counter and one concise end-of-pass log line.
- Keep comments limited to the continuity invariant, units, and missing-prior
  behavior.

Tests must cover disabled gating, the base window, a near quadratic pair, a far
quadratic pair, a missing prior, a non-WGS84 prior, reset/count behavior, and
deterministic pair order.

### 4.6 Make covariance-weighted position optimization the only active mode

In `estimators/global_positioning.*`, `cost_functions/motion_averaging.h`, and
`sfm/global_mapper.*`:

- Reduce `PosePriorPositionMode` to `off` and `optimize`.
- In `off`, preserve upstream global-mapper behavior.
- In `optimize`, require at least three non-degenerate registered priors with
  valid covariance. If the constraint cannot engage, fail the mapper instead
  of continuing as visual-only SfM.
- Build the initial Sim3 gauge with a deterministic covariance-aware RANSAC.
  Its squared residual is the per-row Mahalanobis distance in shared ENU and
  its fixed inlier threshold is the 3-DoF 95% value above.
- Refine the inlier Sim3 with covariance-whitened position residuals and the
  fixed Cauchy radius.
- Jointly optimize the gauge and global frame positions using
  `PositionPriorViaSim3CostFunctor`; do not pre-warp measurements or use an
  isotropic fallback.
- Mark `engaged=true` only after the actual solve returns a usable solution.
- If a GPU global-positioning solve fails, retry once with the CPU sparse
  solver in the same CUDA binary. Log both termination summaries. If the retry
  fails, return failure and write no reconstruction.
- After metric engagement, preserve a metric scale and add all valid position
  residuals to every later global BA/retriangulation refinement stage.
- Always use the covariance-whitened Cauchy loss. Remove switches that permit a
  non-robust position constraint.
- Emit one stable summary line with requested mode, engaged state, usable
  priors, RANSAC inliers, initial RMSE, final RMSE, and solver backend.

Create one branch-specific
`AlignReconstructionToPosePriorsWeighted` entry point that requires a valid
covariance for every correspondence. Use it for the GLOMAP initial metric
gauge and report-mode `model_aligner`. Delete branch-added equal-weight and
horizontal/vertical-gate alternatives rather than maintaining multiple
definitions of a position inlier.

### 4.6a Upstream's mandatory fallback parameter

Do not change upstream `AlignReconstructionToPosePriors` or its
`prior_position_fallback_stddev` semantics. Existing upstream incremental and
`pose_prior_mapper` callers must retain their prior behavior and tests.

The strict branch path never calls that fallback-bearing function. Its weighted
entry point validates every covariance, refuses the whole request if any
selected correspondence lacks one, and has no fallback parameter. Likewise,
add a non-CLI internal
`PosePriorBundleAdjustmentOptions::require_valid_position_covariance` flag,
default `false`; GLOMAP sets it to `true` and fails before adding any residual
that would use an isotropic fallback. Upstream callers leave it false. Tests
must cover both unchanged upstream fallback behavior and the branch path's
strict refusal.

### 4.7 Keep gravity as one weighted, yaw-free constraint

Retain `AbsoluteGravityPriorCostFunctor` and its focused mathematical comment.
The retained behavior is:

- Measured gravity is down in camera coordinates.
- For each row, compute local down as `(0,0,-1)` in that observation's local
  ENU frame, rotate it with `SharedFromLocalEnu(position)`, then rotate the
  result into the camera. Do not use one constant shared-ENU down vector.
- The residual constrains roll/pitch and is invariant to yaw.
- It is weighted by the explicit global sensor uncertainty in degrees.
- It uses a fixed Cauchy radius based on 2 local DoF.
- It is added only after position `optimize` has established the ENU metric
  gauge.
- It is added to every subsequent pose-prior BA stage for every registered
  image with gravity.

Replace `PosePriorGravityBAMode` with
`GlobalMapperOptions::pose_prior_use_gravity`. If it is requested but position
optimization did not engage, or fewer than three registered usable gravity
residuals can be added, fail instead of silently disabling it. Validate
`0 < pose_prior_gravity_stddev_deg <= 180`.

Do not remove upstream `ra_use_gravity`; set it to `0` in the production
runner. The soft BA path and hard rotation-averaging reduction must not consume
the same readings in the production configuration.

### 4.7a Add a weighted one-DoF true-north heading constraint

Implement heading as a wrap-safe angular BA residual, never as absolute camera
orientation. COLMAP camera axes are `+X` right, `+Y` down, `+Z` forward. For a
measured heading `h` in radians clockwise from true north and normalized
measured camera-frame down `d`:

```text
f = (0, 0, 1)
f_h = normalize(f - d * dot(d, f))
r_h = normalize(cross(d, f_h))
n_measured = cos(h) * f_h - sin(h) * r_h
```

Reject the residual when `norm(f - d * dot(d, f)) < 1e-3`; camera forward is
then too close to vertical for its azimuth to be defined. For the row's WGS84
position:

```text
R_shared_from_local = enu_frame.SharedFromLocalEnu(position)
north_world = R_shared_from_local * (0, 1, 0)
n_pred = cam_from_world.rotation * north_world
n_pred_h = normalize(n_pred - d * dot(d, n_pred))
residual_angle = atan2(
    dot(d, cross(n_measured, n_pred_h)),
    dot(n_measured, n_pred_h))
```

The residual is in `[-pi, pi]`, divided by that row's
`heading_stddev_rad`, and passed through the fixed one-DoF Cauchy radius from
§3.3. The `atan2` form must return magnitude `pi`, not a false zero, for an
opposite heading. Implement camera and rig pose composition consistently with
the gravity cost.

Expose only `GlobalMapperOptions::pose_prior_use_heading` as
`--GlobalMapper.pose_prior_use_heading {0,1}`, default `0`. Import and retain
valid heading data regardless of this flag. When requested:

- Require position `optimize` to have established the metric ENU gauge.
- Require `pose_prior_use_gravity=1` and gravity engagement. The heading basis
  consumes the measured down direction; this fork does not support a
  heading-only mapper configuration.
- Require at least one registered heading row with valid gravity, uncertainty,
  and non-degenerate horizontal projection; otherwise fail loudly.
- Add heading to every subsequent pose-prior BA stage in which gravity is
  available. Do not seed or hard-set camera rotations from heading.
- Emit one stable summary containing `requested`, `engaged`, available rows,
  usable residuals, and residual mean/median/P90/maximum in degrees.

The production archive contains no heading and its runner sets the flag to `0`.
A future trusted compass run adds the two archive columns and the three heading
metadata keys, puts the estimated one-sigma accuracy (for example `5.0`) in
each `HEADING_STD_DEG` cell, and changes only the flag to `1`.

The final delivery `model_aligner` remains position/covariance based. It reports
gravity and heading consistency after its small Sim3 correction but performs no
second orientation refinement. The material-correction gate prevents that
final position alignment from silently overriding the mapper orientation.

### 4.8 Reduce report-mode `model_aligner` to the delivery contract

Keep normal upstream `model_aligner` behavior unchanged when no georeference
sidecar is requested. For the branch-added report path:

- Require `--database_path`, `--scene_id`, `--georeference_json`, and
  `--camera_residuals_csv`.
- Require WGS84 priors with valid covariance and at least
  `--min_common_images` registered correspondences.
- Use the shared deterministic ENU frame and covariance-aware Sim3 estimator.
- Honor `--alignment_random_seed`.
- Always enforce the fixed position-support and material-correction gates.
  Enforce the gravity-quality gate when gravity observations exist. Heading is
  diagnostic in report mode; its weighting was already enforced in mapper BA.
  Remove `--reject_material_realignment`; a dangerous correction is never a
  valid production output.
- A collinearity warning is recorded but does not fail the command.
- Support only `ENU_Z_UP` and `LICHTFELD_COLMAP` output frames.
- Apply the chosen output transform to the reconstruction actually written,
  not only to report metadata.
- Write the reconstruction first, verify that `cameras.bin`, `images.bin`, and
  `points3D.bin` can be reopened and match the in-memory counts, then publish
  JSON and CSV through checked temporary files and atomic same-directory
  renames. Any stream, parse, or rename failure must return nonzero.
- Never write a sidecar claiming success when reconstruction publication
  failed. The runner treats a nonzero exit or a missing sidecar as failure.

`LICHTFELD_COLMAP` remains a proper rotation from ENU with this exact raw-data
mapping:

```text
raw X = East
raw Y = -Up
raw Z = North
```

LichtFeld's COLMAP-loader boundary rotation `diag(1,-1,-1)` then displays
East as `+X`, Up as `+Y`, and North as `-Z`. Keep one concise source comment
beside the matrix stating this invariant. Do not mention a particular run,
planning document, or manual experiment there.

### 4.9 Emit one complete, versioned report

Delete report-level branching. Every `georeference.json` must contain the
complete aggregate diagnostics and transform contract below under one
top-level `schema_version: 1`:

- Identity/provenance: `scene_id`, COLMAP version, exact source commit, binary
  SHA-256 when provided by the runner, creation UTC, input/output paths.
- CRS: WGS84 ellipsoid, ellipsoidal height, selected ENU origin.
- Support: database priors, registered images, registered prior
  correspondences, position inliers/outliers, gravity observations, and heading
  observations.
- Initial/final alignment: input-SfM to ENU Sim3, metres per input unit,
  RANSAC seed and fixed standardized gate.
- Position diagnostics in metres for 3D, horizontal, and vertical residuals:
  mean, median, P90, maximum, support, and inlier support.
- Gravity diagnostics in degrees: mean, median, P90, maximum, and support.
- Heading diagnostics in degrees: mean, median, P90, maximum, and support,
  emitted with zero support when no headings exist. The report shape does not
  change when the optional archive group is absent.
- Geometry contract:
  - `geometry_frame`, handedness, raw up axis, and metres.
  - `geometry_from_enu` and `enu_from_geometry`.
  - `ecef_from_geometry` and `geometry_from_ecef`.
  - For LichtFeld only, `visualizer_from_geometry` and displayed up axis.
- Quality results: value, fixed threshold, fired state, and whether it is a
  warning or failure.
- Final-realignment check: rotation, translation, scale delta, fixed
  thresholds, and `is_material`.

Remove `targets`, GLTF fields, report-level metadata, duplicated top-level
transforms, and historical compatibility claims. This schema has no released
consumers requiring a legacy shape. Heading fields remain present with zero
support so adopting a trusted compass later does not change the report shape.

The residual CSV must contain one stable row per registered image, ordered by
image name:

```text
image_name,image_id,has_position_prior,position_fit_inlier,
residual_east_m,residual_north_m,residual_up_m,
residual_horizontal_m,residual_3d_m,
has_gravity_prior,gravity_residual_deg,
has_heading_prior,heading_stddev_deg,heading_residual_deg
```

Use an empty cell when a quantity does not exist for that row. This CSV is the
operator's outlier-cleanup input, so it includes every registered image and the
gross valid GPS rows even when robust fitting rejects them. Gravity and heading
rows remain observable through their raw post-solve residuals; robust losses do
not erase rows from the report.

### 4.10 Remove unrelated pipeline and Python work

Revert branch changes in:

- `src/colmap/controllers/incremental_pipeline.h`
- `src/colmap/sfm/incremental_mapper.cc`
- `src/colmap/sfm/incremental_mapper.h`
- `src/colmap/sfm/incremental_mapper_test.cc`
- every changed file under `src/pycolmap/`

Do not modify upstream `pose_prior_mapper` merely to make it resemble the
global path. The production runner does not call it. Do not build pycolmap
wheels as part of this work.

### 4.11 Enforce the heading scope boundary

The supported compass feature is exactly the scalar contract in §4.1 and the
one-DoF weighted cost in §4.7a. Code review and static gates must reject any
branch-added full quaternion orientation field, rotation-covariance field,
rotation initialization mode, or orientation-assisted alignment option. Those
surfaces add ambiguity without improving the stated phone-geolocation workflow.

## 5. File-by-file end state

Use this table as the implementation checklist. A file may be deleted if all
its retained responsibilities move to the named shared utility.

| File/group | Required end state |
|---|---|
| `doc/pose_priors.rst` | Authoritative concise archive, math, CLI, failure, and report contract; no capture history |
| `doc/cli.rst`, `doc/database.rst`, `doc/index.rst` | Link to the authoritative page; no duplicate schema prose |
| affected `CMakeLists.txt`, `math/math.h`, `exe/colmap.cc` | Register only retained source, tests, utility headers, and `pose_prior_importer`; remove deleted registrations |
| `controllers/global_pipeline.cc` | Convert priors only for position `optimize`; add gravity/heading only after that gauge engages; no rotation-initialization branch |
| `controllers/option_manager.cc` | Expose only the retained mapper options |
| `controllers/pairing.*` | WGS84-only optional quadratic-pair gate and focused tests |
| `estimators/alignment.*` | Dedicated strict covariance-aware position Sim3 for this workflow; upstream fallback API unchanged; no orientation-assisted refinement |
| `estimators/bundle_adjustment*` | Mandatory robust covariance position plus explicitly requested gravity and one-DoF heading residuals; fixed standardized radii |
| `cost_functions/motion_averaging.h` | Retain the Sim3 position-prior cost only |
| `cost_functions/pose_prior.h` | Retain yaw-free gravity and add the wrap-safe scalar heading cost; no full-orientation cost |
| `estimators/global_positioning.*` | `off/optimize`, fail-closed engagement, GPU then CPU retry |
| `exe/database.*` | Strict atomic importer, `error/replace` only |
| `exe/model.*`, `exe/model_georeference.*` | Narrow report path, two frames, complete report, checked publication |
| `geometry/gps.*` | Shared named ENU/ECEF rotations and tests |
| `geometry/pose_prior.*` | Upstream position/gravity, valid-covariance/nondegenerate-gravity checks, and scalar heading/uncertainty only |
| `geometry/pose_prior_io.*` | Strict version-1 WGS84/covariance reader with optional all-or-none gravity and heading row groups |
| `geometry/pose_prior_transform.*` | Only shared ENU origin/position/covariance transform implementation |
| `math/geometric_median.*` | Modified Weiszfeld implementation plus dedicated tests |
| `scene/database_cache.*` | Use shared ENU frame; reject multiple priors per rig frame |
| `scene/database_sqlite.cc`, `scene/database_test.cc` | Guarded additive scalar heading columns and migration; named-column reads; obsolete physical columns ignored |
| `sfm/global_mapper.*` | Fail-closed position/gravity/heading engagement and truthful stable summaries |
| `src/pycolmap/**` | No branch diff |

## 6. Test contract

Tests must prove behavior, not merely getters/defaults. Prune the current large
test additions while preserving these cases.

### 6.1 Unit and component tests

- `geometry/pose_prior_io_test`
  - Valid STD archive and valid full-covariance archive.
  - Archive with neither optional group, gravity only, and gravity plus heading
    round-trips. Heading without gravity, partial schema groups, partial-null
    rows, invalid heading ranges, and incorrect/extra
    conditional metadata are rejected.
  - JSON `null` produces an absent optional row group, never a parse error or a
    numeric/string value.
  - Arbitrary valid column order.
  - Every required metadata/schema/row failure described above.
  - Unknown key/column, duplicate name/column, unresolved name, non-finite
    value, invalid range, partial group, non-SPD covariance, and invalid
    gravity norm.
- `exe/pose_prior_importer_test`
  - Insert, `existing=error`, full replace, duplicate binding, unresolved
    binding, and atomic rollback after a later invalid/conflicting row.
- `geometry/pose_prior_test`
  - Valid/invalid covariance; zero/near-zero/valid gravity; wrapped/unwrapped,
    missing-half, and invalid-uncertainty heading.
- `scene/database_test`
  - Scalar heading present/absent round-trip, migration from an upstream
    database, and named-column reads from a pre-cleanup database that still has
    obsolete rotation BLOB columns.
- `geometry/gps_test`
  - ENU/ECEF rotations are inverses, orthonormal, right-handed, and correct at
    equator and near-pole fixtures.
- `math/geometric_median_test`
  - Every case listed in section 4.4.
- `scene/database_cache_test`
  - Shared origin determinism, WGS84 position conversion, local-to-shared ENU
    covariance rotation, per-row north/down rotation, and rejection of multiple
    priors for one frame.
- `controllers/pairing_test`
  - Every case listed in section 4.5.
- `estimators/alignment_test`
  - Exact Sim3 recovery, deterministic seeded inliers, a gross position
    outlier, and a covariance-anisotropy case whose Mahalanobis classification
    differs by axis without a separate horizontal/vertical estimator.
- `estimators/global_positioning_test`
  - `off` preserves upstream output.
  - `optimize` recovers metric scale and rejects a gross outlier.
  - High-uncertainty data has less influence than low-uncertainty data.
  - Fewer than three/non-degenerate priors and unusable solver summaries fail.
  - The dedicated weighted path accepts all-valid covariance and rejects any
    selected invalid covariance without a fallback.
  - The branch fallback-stddev knob is absent. Separate upstream tests prove
    `AlignReconstructionToPosePriors` and
    `prior_position_fallback_stddev` retain their original behavior (§4.6a).
- gravity robust-loss sufficiency (settles whether admission machinery is needed)
  - Converge a synthetic scene twice, identical except that one gravity reading
    is rotated to about 127 degrees, mirroring the largest residual observed in
    the baseline. Assert the two converged solutions differ by less than the
    3.0-degree delivery gate, and that the outlier's own residual stays
    visible in the report.
  - This is the evidence for *not* implementing hard admission caps or
    consensus rejection. If it ever fails, the fixed Cauchy radius is not
    sufficient on its own and admission must be reconsidered -- do not silently
    widen the delivery gate instead.
- `estimators/cost_functions/pose_prior_test`
  - Gravity yaw invariance, small-angle response, antipodal nonzero response,
    normalization contract, and expected optimum.
  - Heading zero/north, `+90 deg`/east, `-90 deg` wrap, and opposite/`180 deg`
    residual signs and magnitudes match the equations in §4.7a.
  - At the true pose, first-order response to pure yaw is correct and response
    to pure roll/pitch is zero or numerical noise. A near-vertical forward axis
    is rejected at the fixed projection threshold.
  - Heading uncertainty scales the standardized residual exactly: a `5 deg`
    error under `5 deg` sigma has magnitude one before the robust loss.
  - Rig and single-camera pose composition produce the same residual.
- `estimators/bundle_adjustment_ceres_test`
  - Covariance-weighted position changes the solution in the expected
    direction.
  - Gravity reduces roll/pitch while leaving yaw unconstrained.
  - Requested gravity adds the exact expected residual count.
  - Heading reduces yaw while leaving the gravity-determined roll/pitch
    unchanged; low-uncertainty heading has more influence than high-uncertainty
    heading, and the exact expected residual count is added.
- `sfm/global_mapper_test`
  - End-to-end metric scale remains fixed through later BA.
  - A deliberately tilted synthetic reconstruction improves substantially
    with gravity enabled, does not move yaw materially, and does not regress
    registration/reprojection.
  - Requested-but-unengaged position, gravity, or heading returns failure;
    requested heading with gravity disabled also fails.
- `exe/model_georeference_test` and `exe/model_test`
  - Complete versioned JSON parses.
  - Required support/residual percentile fields exist in every report.
  - Fixed warning and hard-gate boundaries flip on either side of each value.
  - Gross outliers remain in CSV and are marked non-inliers.
  - ENU and LichtFeld transforms and inverses match exact basis fixtures.
  - WGS84 to ECEF to `geometry_from_ecef` round-trips a known camera position
    to its written geometry coordinate.
  - `targets`, GLTF, and report-level fields are absent.
  - Heading fields are present with zero support, and the report shape is
    identical whether or not heading priors exist.
  - A synthetic true-north heading produces the expected signed residual after
    transforming through the row's local ENU frame.
  - Output write/reopen/sidecar failures return nonzero.
  - Same input and seed produces byte-identical JSON except explicitly
    nondeterministic provenance such as creation time.

### 6.2 Static removal and comment gates

Run these against branch-added tracked source/documentation lines. There are no
comment exemptions. The helper distinguishes ripgrep's expected `1` (no
matches) from execution failure:

```powershell
$Base = "7e7b86ec"
$Added = git diff --unified=0 "$Base..HEAD" -- src doc
if ($LASTEXITCODE -ne 0) { throw "git diff failed" }

function Assert-NoMatch([string]$Label, [string]$Pattern) {
    $Matches = $Added | rg -n -i -- $Pattern
    $Code = $LASTEXITCODE
    if ($Code -eq 0) { $Matches; throw "$Label found forbidden added lines" }
    if ($Code -ne 1) { throw "$Label scan failed with exit code $Code" }
}

Assert-NoMatch "private/run references" '^\+.*(\bgopro\b|\bstreet\b|\bcolmap_test[0-9_]*\b|\bmont_et_mare\b|\bmapping_grade\b|[A-Z]:\\splat|street_geolocated)'
Assert-NoMatch "planning references" '^\+.*(starter.prompt|planning.document|workstream|\bplan step\b|\bphase [0-9]+\b|\bstep [0-9]+ of\b|journal|[A-Za-z0-9_-]+\.md)'
Assert-NoMatch "deleted surface" 'PosePriorGravityBAMode|GLTF_Y_UP|frame_contract.*targets|GeoreferenceReportLevel|PosePriorRotationMode|rotation_covariance|ROT_COV_|QW.*QX.*QY.*QZ|use_pose_prior_orientation|orientation_max_error_deg|pose_prior_gravity_max_angle_deg|gravity_admitted|gravity_rejected_(absolute|consensus)'
Assert-NoMatch "deleted branch knobs" 'pose_prior_position_fallback_stddev|pose_prior_position_ransac_max_error|pose_prior_gravity_loss_scale|alignment_max_horizontal_error|alignment_max_vertical_error|pose_prior_cartesian_frame|enu_origin_(lat|lon|alt)'

rg -q "prior_position_fallback_stddev" src/colmap/estimators/alignment.h
if ($LASTEXITCODE -ne 0) { throw "upstream fallback contract was removed" }

git diff --quiet "$Base..HEAD" -- src/pycolmap
if ($LASTEXITCODE -eq 1) { throw "forbidden pycolmap branch diff" }
if ($LASTEXITCODE -ne 0) { throw "pycolmap diff check failed" }

$UpstreamOnly = @(
    'src/colmap/controllers/incremental_pipeline.h',
    'src/colmap/sfm/incremental_mapper.cc',
    'src/colmap/sfm/incremental_mapper.h',
    'src/colmap/sfm/incremental_mapper_test.cc'
)
git diff --quiet "$Base..HEAD" -- @UpstreamOnly
if ($LASTEXITCODE -eq 1) { throw "forbidden incremental-mapper branch diff" }
if ($LASTEXITCODE -ne 0) { throw "incremental-mapper diff check failed" }

$StagedPlans = git diff --cached --name-only --diff-filter=A | rg -n '^[^/\\]+\.md$'
$Code = $LASTEXITCODE
if ($Code -eq 0) { $StagedPlans; throw "planning document is staged" }
if ($Code -ne 1) { throw "staged-file scan failed" }

git diff --check "$Base..HEAD"
if ($LASTEXITCODE -ne 0) { throw "git diff --check failed" }
bash ./scripts/format/c++.sh
if ($LASTEXITCODE -ne 0) { throw "C++ formatting failed" }
git diff --check "$Base..HEAD"
if ($LASTEXITCODE -ne 0) { throw "post-format diff check failed" }
```

The scans intentionally inspect only added lines so upstream documentation is
not rewritten to satisfy this fork's hygiene rule. Formatting must not rewrite
unrelated upstream files.

## 7. Documentation and source-comment standard

`doc/pose_priors.rst` is for users and coding agents. Do not enforce an
arbitrary line count. Keep every fact needed to implement or operate the
retained feature, but state each fact once.

Its final structure is exactly:

1. Purpose and supported workflow.
2. Version-1 archive JSON and validation table.
3. Coordinate frames, units, covariance transform, gravity convention, and
   scalar true-north heading convention.
4. `pose_prior_importer` command.
5. `sequential_matcher` GPS gate.
6. `global_mapper` weighted position/gravity/heading behavior and failure
   semantics.
7. `model_aligner` command and the complete report/CSV schema.
8. LichtFeld and phone transform composition.
9. Diagnostics and fixed pass/fail thresholds.

Delete historical run results, dataset/camera names, experiment narratives,
alternative modes that no longer exist, PR justification, and duplicated math.
The baseline numbers belong only in this uncommitted handoff and final
validation evidence.

Every branch-added source comment must satisfy all of these rules:

- It explains a local invariant, coordinate direction, unit, failure semantic,
  or non-obvious mathematical reason needed to maintain the code.
- It is understandable without a planning document, commit history, issue, or
  private dataset.
- It does not mention a run name, site, capture device, workstream, phase,
  numbered plan step, journal, starter prompt, or planning-document filename.
- It does not narrate code that is already obvious from names.
- It does not describe deleted alternatives or call something "legacy" when
  no released compatibility contract exists.
- External normative references are allowed only when they identify the exact
  mathematical or downstream coordinate contract; they are not substitutes
  for stating the invariant locally.

Do not commit any root-level planning `.md` file, including this file. Preserve
the user's untracked files locally; exclude them from staging rather than
deleting them as part of source cleanup.

## 8. Production runner cleanup

Replace the experiment-heavy runner with
`E:\street\scripts\run_street_geolocation_pipeline.ps1`. Keep the old script
read-only until the new full run passes, then archive or delete it.

The production script has no experiment modes. It must:

- Use `E:\street\gopro_images` and an explicit, sorted image-name list.
- Generate the strict version-1 archive and assert its row count equals the
  image-list count.
- Read `GoPro:GravityVector` as a camera-frame unit direction, verify/normalize
  it with the archive rule, and do not compare it to `9.80665`. Assert the
  frozen production input yields 965 complete position rows and 965 gravity
  rows. Emit no heading columns or heading metadata until a trusted,
  declination-corrected camera heading source is actually supplied.
- Run GPU SIFT with `max_image_size=4096`, `first_octave=0`, and the established
  feature cap. Do not offer `first_octave=-1` for this 8 GB GPU at full image
  size.
- Import with `--existing replace` and assert imported/resolved count equals
  the image-list count.
- Run sequential matching with overlap `10`, quadratic overlap enabled, and
  `max_prior_distance=30` metres; retain spatial matching with the proven
  settings.
- Run `global_mapper` with exactly these telemetry options:

```text
--GlobalMapper.pose_prior_position_mode optimize
--GlobalMapper.pose_prior_use_gravity 1
--GlobalMapper.pose_prior_gravity_stddev_deg 5.0
--GlobalMapper.pose_prior_use_heading 0
--GlobalMapper.ra_use_gravity 0
--GlobalMapper.gp_use_gpu 1
--GlobalMapper.random_seed 0
```

**Compass-enabled variant (future capture):** keep the production path above as
the no-heading baseline. To enable weighted compass yaw, the runner must perform
these steps and no implicit substitutions:

1. Time-align each heading to its image; correct magnetic north to true north;
   transform body heading to the camera `+Z` optical-axis heading.
2. Write `HEADING_DEG` and `HEADING_STD_DEG` plus the exact heading metadata in
   §4.1. A sensor assessed at about five-degree one-sigma accuracy writes `5.0`
   on each applicable row. Rows without a reliable observation use JSON `null`.
3. Require the same row to contain valid `GX/GY/GZ`; reject the archive before
   import if a heading row lacks gravity.
4. Set `--GlobalMapper.pose_prior_use_heading 1`; do not change position,
   gravity, rotation averaging, or solver settings.
5. Assert `requested=true`, `engaged=true`, and the exact available/usable
   counts from the heading summary. Save heading residual columns and run the
   heading A/B gate in §10 before accepting the output.

- Assert the exact position summary says `requested=optimize, engaged=true` and
  reports the expected usable-prior count.
- Assert the exact gravity summary says `requested=true, engaged=true`, reports
  exactly 965 usable residuals at each expected BA stage. A generic search for
  `engaged=true` is insufficient.
- Assert the exact heading summary says `requested=false, engaged=false`,
  `available=0`, and `residuals=0`. This prevents accidental ingestion of an
  untrusted yaw source.
- Assert that the accepted global-positioning solve terminated successfully,
  whether on GPU or the one logged CPU retry.
- Run report-mode `model_aligner` with database, deterministic seed `0`, scene
  ID, JSON, CSV, and `LICHTFELD_COLMAP`. Remove all deleted options, including
  the old ten-metre isotropic gate and material-realignment switch.
- Do not precreate the aligned-model output directory if the command requires a
  non-existing publication target.
- Parse the JSON, verify `schema_version`, all finite transforms, inverse
  identities, hard-gate results, and expected support before continuing.
- Run `image_undistorter` only after those checks pass.
- Copy `georeference.json` beside every trained/exported asset and record its
  SHA-256. Never infer georeferencing from filenames.
- Use a configuration fingerprint for safe resume. A resumed stage is valid
  only when its inputs, executable hash, arguments, and output hashes match.
- Keep comments about current parameter meaning and safety only. Remove all
  references to previous runs, other datasets, planning documents, evidence
  labels, and historical fixes.

Remove unused ALIKED/LightGlue/transitive/ROI experiment switches,
`PositionMode`, position fallback/inflation knobs, manual GPU/CPU selection,
and threshold-acceptance switches from the production runner. The one supported
configuration is explicit and reproducible.

Kilometre-scale GPS rows are not a preflight failure. After alignment, sort the
CSV by `position_fit_inlier`, then `residual_3d_m` descending and save the
rejected rows as `gps_outlier_candidates.csv` for cleanup. This does not block
training when all hard report gates pass.

## 9. GPU-only rebuild

Use the builder repository, but do not modify or clean its current checkout.
That checkout is intentionally preserved: it is ahead of `origin/master`,
behind it, has modified submodules/files, and contains untracked releases. Make
the cleanup source a local commit first so the installed binary can identify an
exact tree. Do not stage this handoff or any other root planning artifact.

Create an isolated builder worktree from the refreshed upstream builder:

```powershell
$BuilderRepo = 'C:\splat\pipeline\build_gpu_colmap'
$CleanupRepo = 'C:\splat\pipeline\colmap'
$CleanupCommit = (git -C $CleanupRepo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve cleanup commit' }
$CleanupShort = $CleanupCommit.Substring(0, 8)

$Index = 0
while ($true) {
    $Suffix = if ($Index -eq 0) { '' } else { "_$Index" }
    $BuilderWork = "C:\splat\pipeline\build_gpu_colmap_cleanup_${CleanupShort}${Suffix}"
    $BuilderBranch = "codex/telemetry-cleanup-${CleanupShort}${Suffix}"
    git -C $BuilderRepo show-ref --verify --quiet "refs/heads/$BuilderBranch"
    $BranchProbeCode = $LASTEXITCODE
    $BranchExists = $BranchProbeCode -eq 0
    if ($BranchProbeCode -gt 1) { throw 'Cannot inspect builder branches' }
    if (-not (Test-Path -LiteralPath $BuilderWork) -and -not $BranchExists) { break }
    $Index++
}
git -C $BuilderRepo fetch origin
if ($LASTEXITCODE -ne 0) { throw 'Builder fetch failed' }
git -C $BuilderRepo worktree add -b $BuilderBranch $BuilderWork origin/master
if ($LASTEXITCODE -ne 0) { throw 'Builder worktree creation failed' }
git -C $BuilderWork submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { throw 'Builder submodule initialization failed' }

git -C $CleanupRepo diff --quiet HEAD --
if ($LASTEXITCODE -ne 0) { throw 'Cleanup source has unstaged tracked changes' }
git -C $CleanupRepo diff --cached --quiet HEAD --
if ($LASTEXITCODE -ne 0) { throw 'Cleanup source has staged uncommitted changes' }

$ColmapSub = Join-Path $BuilderWork 'third_party\colmap'
$SubStatus = git -C $ColmapSub status --porcelain
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect fresh COLMAP submodule' }
if ($SubStatus) { throw 'Fresh COLMAP submodule is dirty' }
git -C $ColmapSub fetch $CleanupRepo $CleanupCommit
if ($LASTEXITCODE -ne 0) { throw 'Fetching cleanup source into submodule failed' }
git -C $ColmapSub checkout --detach $CleanupCommit
if ($LASTEXITCODE -ne 0) { throw 'Checking out cleanup source failed' }
if ((git -C $ColmapSub rev-parse HEAD).Trim() -ne $CleanupCommit) {
    throw 'COLMAP submodule does not match cleanup commit'
}

$CeresCommit = '8a566fcc156322160b96f8ca5f0ff755241c2d33'
$CeresSub = Join-Path $BuilderWork 'third_party\ceres-solver'
git -C $CeresSub fetch origin $CeresCommit
if ($LASTEXITCODE -ne 0) { throw 'Fetching required Ceres commit failed' }
git -C $CeresSub checkout --detach $CeresCommit
if ($LASTEXITCODE -ne 0) { throw 'Checking out required Ceres commit failed' }
git -C $BuilderWork add third_party/colmap third_party/ceres-solver
```

Never delete, reset, clean, stash, or switch the original dirty checkout as part
of this task. The loop chooses a new explicit worktree and branch name when an
older cleanup attempt exists; it never overwrites or silently reuses one.

### 9.1 Make the builder fail closed

Update the Windows builder before using it:

- Add `-Tests` and pass `-DTESTS_ENABLED=ON` to the COLMAP external project.
- Add `-RequireCudss`; when set, missing cuDSS headers, import library, or DLL is
  a configuration failure rather than an optional warning.
- Keep `CUDA_ENABLED=ON`, `CASPAR_ENABLED=ON`, GUI off, and
  `CMAKE_CUDA_ARCHITECTURES=86`.
  Note that the currently accepted release was built with
  `CASPAR_ENABLED:BOOL=OFF` and `TESTS_ENABLED:BOOL=OFF`; both flip to `ON`
  here, so the candidate build configuration deliberately differs from the
  last one. Expect a full rebuild, not an incremental one.
- Remove the stale CMake comment claiming the single-target build supports
  architectures 75/80/89/90/120.
- Do not pass `-NoCuda` or `-NoCaspar` and do not run a CPU build.
- Fetch the exact `$CleanupCommit` from the local COLMAP repository into
  `$BuilderWork\third_party\colmap`, check it out detached, and stage that
  submodule pointer. Refuse a dirty COLMAP submodule before and after checkout.
- Set the Ceres submodule to exact commit
  `8a566fcc156322160b96f8ca5f0ff755241c2d33`; this is the required cuDSS 0.8
  support commit. Stage the pointer.
- Port the still-required changes from local builder commit `478fd373`
  semantically onto `origin/master`: exact `sm_86`, `nvJitLink` and cuDNN
  runtime discovery/copy, and dependency additions needed by tests. Replace its
  hard-coded Anaconda exclusion with a generic optional `-IgnorePrefixPath`
  builder parameter that is forwarded to both Ceres and COLMAP CMake; pass the
  local Anaconda library path explicitly in the invocation below. Point
  `CUDNN_ROOT` at the existing read-only package under the original builder and
  hash the copied DLLs; do not copy that untracked directory into the clean
  worktree. Do **not** cherry-pick `478fd373` wholesale: it also pins an obsolete
  COLMAP source revision and predates eight upstream builder commits.
- Add path-safe `-Clean`: resolve `$BuildDir`, require its parent to equal the
  resolved `$BuilderWork`, require its leaf name to equal `build`, and delete
  only that directory.
- Commit the builder/submodule changes on `$BuilderBranch` before building and
  require `git status --porcelain` to be empty.

The selected Ceres tree reports two different version strings, and BUILD_INFO.json
must record both. `git describe` says `2.2.0-163-g8a566fcc` because 2.3.0 is not
yet tagged; `include/ceres/version.h` declares `MAJOR 2, MINOR 3, REVISION 0`,
so anything compiling against `CERES_VERSION` sees **2.3.0**. The tree is
post-2.2.0 development that self-identifies as 2.3.0. Its commit
adds support for **cuDSS 0.8**. These are different version domains:
Ceres is based on 2.2.0; 0.8 is the NVIDIA cuDSS API/runtime, not Ceres 0.8.

The exact build invocation after the builder switches exist is:

```powershell
$CudnnRoot = 'C:\splat\pipeline\build_gpu_colmap\cudnn\cudnn-windows-x86_64-9.10.2.21_cuda12-archive'
if (-not (Test-Path -LiteralPath (Join-Path $CudnnRoot 'bin'))) {
    throw 'Pinned cuDNN runtime directory is missing'
}
$env:CUDNN_ROOT = $CudnnRoot
Set-Location $BuilderWork
& .\scripts_windows\build_colmap.ps1 `
    -Configuration Release `
    -Clean `
    -Tests `
    -RequireCudss `
    -IgnorePrefixPath 'C:\ProgramData\anaconda3\Library'
if ($LASTEXITCODE -ne 0) { throw "GPU COLMAP build failed" }
```

`-Clean` must delete only the validated `$BuilderWork\build`. It must not touch
the original builder checkout, either repository, submodules, or releases.

### 9.2 Build provenance

Write `BUILD_INFO.json` into the accepted release with:

- Exact COLMAP cleanup commit and `git status --porcelain` result.
- Builder `origin/master` base, exact local builder commit, and clean status.
- Ceres describe string/commit.
- CUDA toolkit/compiler version and `sm_86`.
- cuDSS DLL file version and SHA-256.
- cuDNN package version and every copied cuDNN DLL SHA-256.
- `CUDA_ENABLED`, `CASPAR_ENABLED`, `TESTS_ENABLED`, GUI, and cuDSS status.
- UTC build time and `colmap.exe` SHA-256.

Both the builder worktree and its COLMAP/Ceres submodules must be clean at build
start. The two recorded submodule gitlinks must equal the exact commits above;
an informational version banner is not acceptable provenance.

### 9.3 Binary validation

All of these are required:

```powershell
$Build = Join-Path $BuilderWork 'build'
$Bin = "$Build\install\colmap\bin\colmap.exe"

function Require-CacheValue([string]$Path, [string]$Pattern) {
    if (-not (Select-String -LiteralPath $Path -Pattern $Pattern -Quiet)) {
        throw "Missing required cache value '$Pattern' in $Path"
    }
}
Require-CacheValue "$Build\CMakeCache.txt" '^CMAKE_CUDA_ARCHITECTURES:STRING=86$'
Require-CacheValue "$Build\CMakeCache.txt" '^CASPAR_ENABLED:BOOL=ON$'
Require-CacheValue "$Build\colmap\CMakeCache.txt" '^CUDA_ENABLED:BOOL=ON$'
Require-CacheValue "$Build\colmap\CMakeCache.txt" '^CASPAR_ENABLED:BOOL=ON$'
Require-CacheValue "$Build\colmap\CMakeCache.txt" '^TESTS_ENABLED:BOOL=ON$'
Require-CacheValue "$Build\colmap\CMakeCache.txt" '^GUI_ENABLED:BOOL=OFF$'

$CudssDlls = Get-ChildItem "$Build\install\colmap\bin\cudss*.dll" -ErrorAction Stop
if (-not $CudssDlls) { throw 'Packaged cuDSS runtime is missing' }
$CudnnDlls = Get-ChildItem "$Build\install\colmap\bin\cudnn*.dll" -ErrorAction Stop
if (-not $CudnnDlls) { throw 'Packaged cuDNN runtime is missing' }

ctest --test-dir "$Build\colmap" --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'CTest failed' }
python "$BuilderWork\scripts\validate_caspar_sample.py" --colmap $Bin
if ($LASTEXITCODE -ne 0) { throw 'Caspar validation failed' }
& $Bin -h
if ($LASTEXITCODE -ne 0) { throw 'Packaged COLMAP does not start' }
```

Also inspect the configure/build logs and require explicit `CUDA`, `sm_86`,
`Caspar`, and `cuDSS` success lines. File presence alone is not proof a backend
was compiled and exercised.

`validate_caspar_sample.py` must run on a camera model Caspar supports. It is a
check that the backend compiled and solves, **not** a check that the street
pipeline uses it — that pipeline is `OPENCV_FISHEYE` and will keep selecting
`SPARSE_SCHUR / CUDA_SPARSE` (§1). Do not treat Caspar being unused by the
production solve as a build failure, and do not switch the production camera
model to exercise it.

Run a three-image SIFT smoke test under
`E:\street\scratch\sift_smoke_<cleanup-short-sha>` using the production
`4096/first_octave=0` settings. Require:

- GPU SIFT is selected in the log.
- All three images complete and write features.
- Runtime is finite and comparable to the known fast configuration, not the
  non-finishing `first_octave=-1` behavior.
- Dedicated VRAM does not remain pinned after process exit.

Do not use a CPU feature-extraction smoke test.

## 10. Full end-to-end validation

Run the new production script into a new directory named
`E:\street\colmap_cleanup_<cleanup-short-sha>`. Do not overwrite the known-good
baseline before the candidate passes.

The agent must continue diagnosing and fixing code, tests, build configuration,
or runner assertions until every hard gate below passes:

| Gate | Required result |
|---|---|
| Archive/import | One complete position prior per selected image; optional groups obey all-or-none row rules; no unresolved rows |
| Feature extraction | GPU SIFT completes all images with `first_octave=0` |
| Sequential GPS gate | Veto count greater than zero; base continuity pairs retained |
| Reconstruction | 965/965 registered on the frozen full dataset |
| Sparse quality | At least 203,296 points minus 5%; mean reprojection no worse than 0.9317 px |
| Position engagement | `optimize` requested and engaged with 965 usable priors |
| Gravity engagement | requested and engaged with 965 residuals at every later BA stage |
| Heading state | production run: requested false, engaged false, available/residual count zero |
| Metric scale | `abs(metres_per_input_unit - 1) <= 1e-4` |
| Position quality | **both** must hold: absolute ceiling 3D RMSE <= 2.0 m (product requirement), and regression check <= 1.824 m (10% above the 1.6582 m baseline). The regression check is the binding number today; the ceiling is what the product needs even if the baseline itself later moves. |
| Gravity quality | median over all 965 readings <= 3.0 degrees and no regression beyond 0.25 degrees from 1.698; mean/P90/maximum recorded so outliers stay visible; the robust-loss sufficiency test of section 6.1 passes |
| Final support | position inlier ratio >= 0.8; all 965 rows present in CSV |
| Final correction | <= 0.5 degrees, <= 1.0 m, and scale delta <= 0.01 |
| Report | complete schema-1 JSON, horizontal/vertical/3D percentiles, finite inverse transforms |
| Output frame | `LICHTFELD_COLMAP`, raw up `-Y`, displayed up `+Y`, metres |
| Consumer math | known WGS84 camera fixtures map through `geometry_from_ecef` to written coordinates within 1e-6 m numerical tolerance |
| Undistortion | all 965 images and the aligned sparse model are exported |

The fixed-seed cleanup run should normally remain close to the exact baseline.
If a result falls outside a tolerance, do not loosen the tolerance first.
Identify whether the difference comes from intended weighting cleanup, solver
backend, source mismatch, or a regression, then fix or document the causal
measurement.

### Gravity usefulness proof

The retained synthetic tilted-reconstruction test is the deterministic proof
that gravity improves roll/pitch. Also run one real-data A/B from the same
database/matches and fixed seeds with only `pose_prior_use_gravity` changed:

- Gravity-on must not reduce registration count.
- Reprojection may not worsen by more than `0.02 px`.
- Position RMSE may not worsen by more than `0.10 m`.
- Gravity median must improve measurably or already be below `2.0 degrees`.

If the real scene is already visually gravity-aligned and shows no measurable
change, the synthetic perturbation test plus no-regression real A/B is
sufficient. Do not delete the gravity feature merely because the visual solve
was already near its optimum.

### Heading usefulness proof

The cleanup must pass the deterministic heading tests in §6 even though the
frozen production archive contains no trusted heading. Those tests prove the
feature is a one-DoF weighted yaw constraint, handles wrap/180 degrees, respects
per-row uncertainty, does not inject roll/pitch, and works through rig poses.

Whenever the compass-enabled runner variant is used, add a same-input,
same-matches, fixed-seed A/B with only `pose_prior_use_heading` changed:

- Heading-on must not reduce registration count.
- Reprojection may not worsen by more than `0.02 px` and position RMSE may not
  worsen by more than `0.10 m`.
- Gravity median may not worsen by more than `0.25 deg`; this checks that a yaw
  observation did not leak into pitch/roll.
- Heading disagreement must improve measurably or already be at/below the
  declared sensor sigma. Inspect mean/P90/maximum for gross timestamp,
  declination, sign, or body-to-camera errors before accepting.
- The report and CSV counts must equal the preflight count of usable heading
  rows. Any mismatch is a failure, not a warning.

### Downstream handoff proof

Goal-mode completion must not wait for a person to click through a GUI. Use the
LichtFeld source at `C:\splat\pipeline\LichtFeld-Studio`; record its exact commit
and tracked status. Resolve an existing matching executable, or build that
source using its own documented GPU build instructions if none exists.

Before declaring success, run these automated gates:

1. Run LichtFeld's COLMAP loader/layout tests, including the binary sparse-model
   and scene-transform/write-back cases. Any skipped test relevant to the
   installed CUDA GPU must be treated as a failure and investigated.
2. Run a bounded headless training smoke on the newly undistorted dataset with
   explicit `--headless --data-path <dataset> --output-path <scratch>
   --centralize off --iter 50 --output-name trained_smoke`. Require exit zero,
   the expected camera/image count in the log, and a nonempty final PLY. Use a
   process timeout long enough for initialization plus 50 steps; a timeout is a
   failure to diagnose, never an excuse to skip the gate.
3. Assert the log contains no point-cloud or camera centralization action. Load
   the emitted PLY through LichtFeld's noninteractive `convert` command and
   require successful round-trip output with finite positions and unchanged
   bounding-box order of magnitude. This verifies load/train/export without a
   GUI and catches implicit unit or axis-scale collapse.
4. Copy the exact `georeference.json` beside the smoke PLY and assert its
   SHA-256 equals the aligned-model sidecar recorded by the runner.
5. Use `C:\splat\pipeline\splat-transform` at a recorded clean commit. Run its
   tests/build, then convert the headless smoke PLY without any transform:
   `node bin/cli.mjs trained_smoke.ply street_smoke.sog`. The supported bundled
   suffix is `.sog`, not `.ssog`; `meta.json` plus WebP files is the unbundled
   form. After human editing later produces `trained_clean.ply`, the identical
   validated command writes the production `street.sog`; manual cleanup is not
   a blocker for validating this COLMAP cleanup.
6. Do not claim that the current converter embeds `georeference.json`: its
   source has no such input/metadata contract. Publish the byte-identical
   `georeference.json` beside the generated `.sog`, record both hashes in the asset
   manifest, and make the web app fetch and validate both. A future single-file
   extension is a separate splat-transform feature, not assumed by this plan.
7. For each phone fix compute
   `geometry_position = geometry_from_ecef * WGS84ToECEF(phone_fix)`. Put the
   SOG entity under the report's `visualizer_from_geometry` transform and place
   the phone marker at
   `visualizer_from_geometry * geometry_position`, so splat and marker receive
   the exact same raw-data-to-PlayCanvas basis change. Perform WGS84-to-ECEF and
   the ECEF-to-local Sim3 with JavaScript-number arrays in double precision;
   never store the large ECEF transform/translation in PlayCanvas `Mat4`
   float32 storage. Only the resulting local-metre position and the small pure
   basis rotation enter PlayCanvas objects.
8. Verify the known camera GPS fixtures land on their corresponding transformed
   camera centers within the tolerance in the report before enabling live
   tracking. Reject stale/mismatched sidecar hashes and non-finite transforms.
9. Serve the app over HTTPS. Port 80 may redirect to the HTTPS origin, but the
   phone must not depend on geolocation from a plain LAN HTTP page. Display GPS
   permission/error state, timestamp, reported accuracy, and an explicit
   out-of-scene state; do not silently clamp a fix onto the street. Treat a
   missing/unknown-datum phone altitude as a horizontal-only fix and label it;
   never substitute zero altitude into the ellipsoidal WGS84 transform. Any
   ground projection must be an explicit viewer policy with the raw fix kept
   visible for diagnostics.

Opening the result in LichtFeld for an upright visual inspection is useful
confirmation and may be recorded in the run manifest, but it is not a blocking
goal-mode gate. Automated numeric/load/export checks above are the authority.

## 11. Deleting old builds and publishing the accepted one

Do not delete the last known-good executable until the cleanup candidate passes
the complete full run. After it passes:

1. Package the accepted GPU binary and dependencies into
   `C:\splat\pipeline\build_gpu_colmap\releases\colmap-telemetry-cleanup-<cleanup-short-sha>`.
   Refuse to overwrite an existing directory.
2. Copy `BUILD_INFO.json`, full CTest output, Caspar validation output, SIFT
   smoke log, and end-to-end run manifest into that release.
3. Recompute and record hashes after packaging.
4. Point the production runner at the accepted immutable release path.
5. Re-run the binary/hash preflight from that packaged path.
6. Create an explicit deletion manifest containing the original builder
   `build`, the isolated worktree's `build`, and every superseded candidate or
   portable release directory. Print the paths and their sizes first. Do not use
   a wildcard as the deletion target, and exclude the newly accepted release.
7. Resolve every target with `[IO.Path]::GetFullPath`; require its leaf name not
   to be empty, require it to be a strict child of either the resolved original
   builder root or isolated builder-worktree root using an ordinal-ignore-case
   prefix check, and fail the whole cleanup before deleting anything if any
   target violates those rules. Then delete each exact manifest path with
   `Remove-Item -LiteralPath ... -Recurse -Force` and verify it no longer exists.
8. Delete the temporary builder worktree itself only through
   `git -C $BuilderRepo worktree remove $BuilderWork` after its accepted release
   and provenance are safely outside it and its branch commit is recorded.
9. Keep source repositories, the builder branch/commit, submodule commits, the
   accepted release, and the final
   validation manifest.

There is no CPU release artifact.

## 12. Final completion report and draft-PR evidence

The implementing agent is finished only after delivering:

- Cleanup commit SHA and upstream merge base.
- Final `git diff --stat` and a feature-removal list.
- Proof that no planning/run references remain in added tracked comments/docs.
- Exact passing CTest list.
- GPU/CUDA `sm_86`, Caspar, cuDSS, Ceres, and binary provenance.
- SIFT smoke timing/VRAM result.
- Full-run metrics beside the baseline table.
- Gravity synthetic proof and real A/B results.
- Heading archive/database/cost/BA synthetic proof, plus real heading A/B only
  when a trusted heading source is supplied.
- Parsed report/CSV checks and the WGS84-to-geometry fixture result.
- LichtFeld training/export smoke result.
- SOG conversion, sidecar/hash, double-precision phone-to-local, and HTTPS app
  integration results.
- Accepted release path and SHA-256.
- Any remaining branch-added surface, with one sentence connecting it directly
  to the product goal and naming the test or real run that proves it.

The eventual draft PR may be broad because it represents the full tool, but its
code must be coherent: strict WGS84 telemetry import, covariance-weighted metric
position constraints, yaw-free weighted gravity, optional uncertainty-weighted
true-north yaw, GPS-aware pairing, and a tested LichtFeld/phone georeference
contract. Nothing else belongs in this fork.
