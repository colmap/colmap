.. _pose-priors:

Pose Priors
===========

COLMAP can ingest external position/orientation measurements (for example,
GNSS, RTK, an inertial navigation solution, survey control, or motion
capture) as **pose priors** on a per-image basis, use them to speed up and
stabilize global SfM, and produce a
scene-to-Earth georeference report after reconstruction. This page documents
the full pipeline: importing an archive, the exact conventions each field
must satisfy, the ``pose_prior_mapper``/``global_mapper`` modes that consume
priors, and the ``model_aligner`` georeference report.

Pose priors are optional at every stage. Missing fields (no orientation, no
covariance, no altitude) are represented explicitly as absent rather than
guessed, and every stage that can be skipped without a prior reports whether
it was requested and whether it actually engaged.


Importing a pose-prior archive
-------------------------------

``colmap pose_prior_importer`` reads a JSON archive and writes
:ref:`PosePrior <database>` rows into the database, resolving each row by
image name::

    colmap pose_prior_importer \
        --database_path DATABASE \
        --pose_prior_path priors.json \
        --existing error

Archive format::

    {
      "coordinate_system": "WGS84",
      "sensor_type": "CAMERA",
      "translation_convention": "WORLD_FROM_CAM",
      "ellipsoid": "WGS84",
      "height_datum": "ELLIPSOIDAL",
      "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
      "data": [
        ["img001.jpg", 47.3769, 8.5417, 500.0, 2.0, 2.0, 4.0],
        ["img002.jpg", 47.3770, 8.5418, 501.0, 2.0, 2.0, 4.0]
      ]
    }

An archive has three parts:

- **Metadata**: the coordinate system (``WGS84`` or ``CARTESIAN``), sensor
  type, reference ellipsoid, and the conventions below. These apply to every
  row in the archive.
- **Schema**: an ordered list of column names describing each data row.
  Column order is arbitrary and unknown columns are parsed but discarded, so
  archives can carry extra source-specific columns without breaking import.
- **Data**: one row per image. ``NAME`` resolves to a database image by
  filename; rows that don't resolve are skipped with a warning, not an
  error, so a partial archive against a partial dataset still imports.

Column groups, all optional independently:

- **Translation**: ``LAT, LON, ALT`` (geographic) or ``TX, TY, TZ``
  (Cartesian). Absent entirely if only ``NAME`` plus other groups are given.
- **Translation uncertainty**: ``STD_TX, STD_TY, STD_TZ`` (diagonal, used to
  build a diagonal covariance) or ``COV_TXX, COV_TXY, COV_TXZ, COV_TYY,
  COV_TYZ, COV_TZZ`` (full covariance). ``STD_*`` and ``COV_*`` are mutually
  exclusive within one archive.
- **Gravity**: ``GX, GY, GZ`` — the down direction in sensor coordinates,
  independent of full orientation. This is cheaper to obtain reliably than a
  full quaternion (see below) and is enough to seed an upright-camera
  assumption.
- **Absolute orientation**: ``QW, QX, QY, QZ``, Hamilton convention, W
  first, representing ``sensor_from_world`` (``sensor_from_local_enu`` for
  WGS84 archives, ``sensor_from_cartesian_world``/``sensor_from_local`` for
  Cartesian archives, selected by ``rotation_world_frame``).
- **Rotation uncertainty**: ``ROT_COV_XX`` .. ``ROT_COV_ZZ``, rad², right-
  multiplicative in the same world basis as the quaternion.

**Absence is a first-class state.** ``PosePrior::HasPosition()``,
``HasGravity()``, ``HasRotation()``, ``HasPositionCov()``, and
``HasRotationCov()`` each test for all-finite fields independently; a row
that supplies position but not orientation is exactly as valid as one that
supplies both, and every downstream consumer (mapper modes, the georeference
report) checks these predicates rather than assuming a field is present.
Horizontal-only GPS (no reliable altitude) is represented the same way: the
altitude component is left non-finite rather than defaulted to zero, and it
stays non-finite through the WGS84→ENU conversion (only East/North are
finite; Up is ``NaN``) so a horizontal fix is never silently treated as a
3D fix.

``--existing`` controls what happens when an incoming row's image already
has a prior in the database:

- ``error``: abort the whole import if any incoming resolved image already
  has a prior.
- ``replace``: the incoming row's groups become the complete prior for that
  image — groups absent from the row become absent on the stored prior too
  (e.g. re-importing with only ``LAT/LON/ALT`` clears a previously-stored
  orientation for that image).
- ``merge``: only the groups present in the row are updated; every other
  group already on the stored prior is preserved untouched.

There is no separate "update" flag — ``--existing`` is required and is the
only control for this behavior.


Mapper modes
------------

Pose priors are consumed by ``pose_prior_mapper`` (incremental SfM) and by
``global_mapper``'s two independent gauge-fixing controls,
``--GlobalMapper.pose_prior_position_mode`` and
``--GlobalMapper.pose_prior_rotation_mode``::

    colmap global_mapper \
        --database_path DATABASE \
        --output_path MODEL \
        --GlobalMapper.pose_prior_position_mode initialize \
        --GlobalMapper.pose_prior_rotation_mode initialize

Position mode (``off`` | ``initialize`` | ``optimize``):

- ``off``: pose priors are not used for global positioning.
- ``initialize``: camera positions are seeded from pose priors (converted to
  a metric local frame) before global positioning runs, giving the solver a
  much better starting point without changing the objective it optimizes.
  **This is the recommended default** for a first experiment — it captures
  most of the benefit of GPS priors with the least risk of a bad prior
  destabilizing the solve, since a bad seed can still be corrected by
  positioning/bundle adjustment.
- ``optimize``: pose priors are added as weighted terms directly in the
  global positioning objective (covariance-weighted when a covariance is
  present, a fallback stddev otherwise), pulling the final solve toward the
  priors rather than only seeding it. Use this once ``initialize`` has been
  validated on the same dataset and the priors are known to be trustworthy.

Whenever position mode is not ``off``, the mapper logs the resolved trust
knobs once at solve start: ``--GlobalMapper.pose_prior_position_loss_scale``,
``--GlobalMapper.pose_prior_position_fallback_stddev``, and their product,
the Sim3 RANSAC gate used by ``optimize`` to admit correspondences into its
gauge fit (``max_error = loss_scale × fallback_stddev`` — the coupling is
intentional, not an oversight: it avoids introducing a third, unrelated
tuning constant). Configuring ``optimize``'s weighting therefore always
means choosing this gate too; read the logged line before trusting an
``optimize`` run.

Rotation mode (``off`` | ``initialize``): selects a single global rotation
gauge from full-orientation pose priors via robust consensus among frames
that supply one, then fixes the remaining rotation-averaging gauge freedom
to it. This only engages when at least one image has a usable orientation
prior (``HasRotation()``); the mapper logs
``Pose prior rotation gauge: requested=true, engaged=false`` with a stated
reason (no orientation priors present, consensus failed, etc.) when it
cannot engage, and ``engaged=true`` with the chosen frame otherwise — never
silently falling back.

A sufficiently varied position track can determine a horizontal world gauge
from geometry alone. Absolute orientation is an independent, optional source
of information; it is most useful when the position layout is weak, when a
specific local-world orientation must be preserved, or when orientation
residuals are themselves diagnostically important. A position-only archive
is fully supported and leaves ``pose_prior_rotation_mode`` unengaged by
design.

Note that :ref:`spatial/sequential/retrieval pairing <cli>` (used before
feature matching) uses pose priors only to *propose* candidate image pairs;
it never substitutes for geometric verification, which remains the sole
authority on whether a pair is actually a valid two-view match.


model_aligner: Earth alignment and the georeference report
------------------------------------------------------------

``model_aligner`` aligns a reconstruction to WGS84/ENU using pose priors
already stored in the database (``--alignment_type enu``), and can
additionally write a scene georeference report describing every
sfm-to-Earth transform plus per-camera residuals::

    colmap model_aligner \
        --input_path MODEL \
        --output_path MODEL_ALIGNED \
        --database_path DATABASE \
        --alignment_type enu \
        --alignment_max_error 5.0 \
        --georeference_json report.json \
        --camera_residuals_csv residuals.csv

The alignment is a robust (RANSAC) similarity (``Sim3d``) fit between the
reconstruction's camera centers and their position pose priors converted to
a local ENU frame, so it tolerates a minority of grossly wrong priors
without needing them removed by hand first. ``--alignment_max_error`` is the
position inlier threshold in metres and must be positive; choose it from the
expected measurement, association, and calibration error rather than treating
the example value as universal. ``--alignment_random_seed`` exposes the
underlying RANSAC's seed (default: unseeded, matching prior behavior
byte-for-byte); pass an explicit non-negative integer to make a run
reproducible, or sweep several seeds to check that the fit is not an
artifact of one particular random sample. The resolved seed is recorded in
the JSON report as ``alignment_random_seed``.

**Choosing the ENU origin.** By default the origin is derived automatically
as the geometric median (Weiszfeld's algorithm — robust to outliers, unlike
a mean) of the WGS84 reference points, using the median altitude rather than
an arbitrary first row. After the initial robust fit identifies inliers, the
origin is recomputed once from only the inlier points and the fit is redone
once against that refined origin — this two-pass scheme keeps a single gross
outlier from skewing the very origin used to judge inliers. Alternatively,
supply an explicit origin with ``--enu_origin_lat/--enu_origin_lon/
--enu_origin_alt`` (all three or none); this is recorded in the report as
``explicit`` rather than ``derived``.

For Cartesian priors, Earth output additionally requires an explicit geodetic
origin and ``--pose_prior_cartesian_frame=ENU``. This explicit assertion is
necessary because the database's Cartesian coordinate-system tag does not by
itself distinguish a local arbitrary frame from an Earth-oriented ENU frame.

With ``--use_pose_prior_orientation=1``, the robust position-only Sim3 remains
the starting point and its position inlier set remains authoritative. COLMAP
then refines the same single global Sim3 with covariance-weighted position and
absolute-orientation residuals under robust losses. Orientation outliers are
classified with ``--orientation_max_error_deg`` and removed before one final
refinement. If no usable orientation remains or the refinement is unusable,
COLMAP retains the valid position-only transform and reports
``orientation_requested=true`` and ``orientation_engaged=false``. Relative
camera geometry is never adjusted independently by this operation.

**The JSON report** (``--georeference_json``) contains: a scene identifier
(supplied via ``--scene_id`` or a generated UUID), COLMAP's build/source
commit for provenance, input/output paths and reconstruction counts, the
WGS84 ellipsoid and the ellipsoidal-height convention, the chosen ENU
origin and whether it was explicit or derived, all six transform directions
between scene, ENU, and ECEF (each verified to numerically round-trip its
declared inverse before the report is written), metres-per-input-unit, and
position and orientation support/inlier/residual diagnostics; horizontal and
3D extent conditioning; baseline-to-measured-uncertainty ratio; maximum scene
radius; and ellipsoid-to-tangent-plane departure. These values describe the
fit and its geometry, not independent positioning accuracy. The report
intentionally does **not** depend on COLMAP's
optional download/curl/crypto feature and does not compute file hashes —
geometry/report SHA256 values belong one layer downstream, in an exported
asset's own sidecar (see below), where the actual asset bytes exist.

**The frame contract.** The report's ``frame_contract`` object (schema
version 1) states the scene geometry's frame explicitly rather than leaving
it to convention: ``geometry_frame: ENU_LOCAL``, right-handed, Z-up, metres,
already applied to the written reconstruction (``geometry_already_transformed:
true`` — there is no separate un-transformed variant to apply this to), plus
the WGS84 ellipsoid/height-datum/origin also reported at the top level.
``targets`` lists named export conventions as an explicit
``matrix_row_major_target_from_geometry`` matrix, stored row-major, mapping
*from* the reconstruction's ENU geometry *to* the named target frame — the
direction is stated in the key so a loader never has to guess which way to
apply it. The one shipped target, ``GLTF_Y_UP``, maps East→+X, Up→+Y,
North→−Z, matching glTF 2.0's +Y-up right-handed convention. Before trusting
any target entry in a real loader, round-trip a known ENU basis vector (and
one camera pose) through it once; a cropped or otherwise unchanged-geometry
asset preserves the sidecar unchanged, and only a rebasing edit composes a
new transform.

**Post-alignment warnings.** The report always emits two diagnostics
alongside their pipeline-policy thresholds, so the threshold can be
re-derived later without re-running the alignment:
``diagnostics.horizontal_condition_ratio`` (the centered horizontal
position-support singular-value ratio s2/s1 — small values mean the camera
track is close to collinear, leaving rotation about the track axis weakly
constrained) and ``diagnostics.gravity_consistency_angle_deg`` (the angle
between the aligned up-axis and the mean gravity direction reported by every
registered image's pose-prior gravity vector, robustly averaged by
normalizing the mean of the per-image unit down vectors; ``null`` when no
prior in the database has gravity). Both land in the top-level ``warnings``
object as ``{value, threshold, fired}``; the shipped policy defaults fire at
``s2/s1 < 0.1`` and gravity angle ``> 3.0°``. A fired warning is
``LOG(WARNING)``-only — it never fails the command.

**The CSV report** (``--camera_residuals_csv``) has one row per database
image, sorted by name::

    image_name,registered,has_position_prior,position_fit_inlier,
    prior_e,prior_n,prior_u,solved_e,solved_n,solved_u,
    residual_e,residual_n,residual_u,residual_horizontal,residual_vertical,residual_3d,
    has_orientation_prior,orientation_fit_inlier,orientation_residual_deg

Absent numeric values are empty cells, never ``0`` or the literal text
``NaN``. Registered prior-bearing images receive solved coordinates and
residuals whether they were fit inliers or outliers, so the CSV preserves the
very disagreements that are most useful for diagnosing bad metadata,
timestamp association, or reconstruction failure.

**Report diagnostics have limitations worth knowing before trusting them at
face value**: residuals are computed after the same robust fit that used
those very points as (candidate) inliers, so they describe fit quality, not
an independent validation; the RANSAC inlier/outlier split is a hard
threshold on one robust fit, not a calibrated per-point confidence; measured
covariance describes the upstream measurement model and can itself be wrong;
and a low residual cannot establish absolute accuracy without independent
control observations.


Downstream asset composition
----------------------------

COLMAP reports coordinate transforms for its reconstruction. An external
exporter can carry the same georeference into a point cloud, mesh, radiance
field, tiled map, or another derived representation without making COLMAP
depend on that representation's file format.

Every downstream asset sidecar copies the report's ``scene_id`` and
contains: an ``asset_id``, an optional ``parent_asset_id``, the geometry
filename and its SHA256, a local axis-aligned bounding box and bounding
sphere, ``ecef_from_asset`` and ``asset_from_ecef``, ``metres_per_asset_unit``,
and the source georeference report's own SHA256 (so an asset can be traced
back to the exact report that georeferenced it).

Composition rules:

- A crop or delete that does not move any remaining point's coordinates
  copies the parent's transforms unchanged, but still gets a new
  ``asset_id`` and a freshly recomputed geometry hash/bounds — identity
  follows content, not just the transform.
- If an editor applies a rigid/similarity edit and supplies
  ``parent_from_child``, compose:
  ``ecef_from_child = ecef_from_parent * parent_from_child``.
- A non-rigid or non-uniform deformation (sculpting, non-uniform scale per
  axis, mesh deformation) cannot be represented by one ``Sim3`` — such an
  edit must either mark the direct georeference invalid on the child asset,
  or emit a richer mapping than a single similarity transform. Never force
  a non-rigid edit into a ``Sim3`` field.
- To place a real-world WGS84 point from any positioning client into asset
  space: convert WGS84 to ECEF, then apply ``asset_from_ecef``. To place an
  asset-space point on Earth: apply ``ecef_from_asset``.

Formats with no metadata channel, including ordinary PLY, can use an adjacent
sidecar. Formats with extensible metadata may embed the same fields. The
composition contract is format-independent.


Source-adapter responsibilities
----------------------------------

COLMAP's pose-prior archive is intentionally source-agnostic. Producing a
*correct* archive from embedded image metadata, a GNSS/INS receiver, a mobile
device, a robotics stack, motion capture, or surveyed control is the source
adapter's responsibility. Before emitting a measurement, an adapter must
establish:

- the timestamp association between the measurement and image exposure;
- the height datum and any geoid or barometric conversion needed before
  declaring ``height_datum: ELLIPSOIDAL``;
- sensor-to-camera extrinsics and axis/handedness conventions;
- quaternion direction (``sensor_from_world`` rather than its inverse);
- the world frame used by positions, orientations, and their covariances;
- covariance units, basis, and whether the values are actually calibrated;
- any heading-reference correction required by the upstream navigation
  system.

A gravity vector is a unit down direction in camera-sensor coordinates only
after the relevant axis and mounting transforms have been applied. A raw
accelerometer sample is not automatically a gravity prior. Likewise, the
presence of a heading sensor does not prove that a fused orientation is
accurate or expressed in the required frame.

Missing information stays missing. If altitude, yaw, covariance, or full
orientation cannot be established, omit that measurement group instead of
inserting zero, a guessed value, or a large-covariance placeholder. Absence is
handled explicitly throughout import, mapping, and reporting.
