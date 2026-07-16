.. _pose-priors:

Pose Priors
===========

COLMAP can ingest external position/orientation measurements (GPS, RTK,
GNSS/INS, or a flight controller's onboard estimate) as **pose priors** on a
per-image basis, use them to speed up and stabilize global SfM, and produce a
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
  archives can carry extra vendor-specific columns without breaking import.
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

Rotation mode (``off`` | ``initialize``): selects a single global rotation
gauge from full-orientation pose priors via robust consensus among frames
that supply one, then fixes the remaining rotation-averaging gauge freedom
to it. This only engages when at least one image has a usable orientation
prior (``HasRotation()``); the mapper logs
``Pose prior rotation gauge: requested=true, engaged=false`` with a stated
reason (no orientation priors present, consensus failed, etc.) when it
cannot engage, and ``engaged=true`` with the chosen frame otherwise — never
silently falling back.

**GPS trajectory normally supplies north**; a compass/magnetometer-derived
heading is an optional refinement layered on top, not a requirement. A
GPS-only archive (position, no orientation) is a fully supported and common
input — it drives ``pose_prior_position_mode`` but leaves
``pose_prior_rotation_mode`` unengaged, which is expected, not a
misconfiguration.

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
        --georeference_json report.json \
        --camera_residuals_csv residuals.csv

The alignment is a robust (RANSAC) similarity (``Sim3d``) fit between the
reconstruction's camera centers and their position pose priors converted to
a local ENU frame, so it tolerates a minority of grossly wrong priors
without needing them removed by hand first.

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

``--use_pose_prior_orientation`` is accepted for forward compatibility with
an orientation-assisted joint refinement, but **is not yet implemented** —
passing it logs a warning and the alignment remains position-only. The
report always reflects this honestly: ``orientation_requested`` reflects the
flag, ``orientation_engaged`` is always ``false``, and every orientation
column in both outputs is empty rather than a placeholder zero.

**The JSON report** (``--georeference_json``) contains: a scene identifier
(supplied via ``--scene_id`` or a generated UUID), COLMAP's build/source
commit for provenance, input/output paths and reconstruction counts, the
WGS84 ellipsoid and the ellipsoidal-height convention, the chosen ENU
origin and whether it was explicit or derived, all six transform directions
between scene, ENU, and ECEF (each verified to numerically round-trip its
declared inverse before the report is written), metres-per-input-unit, and
position support/inlier/residual diagnostics (mean/median/p90/max, in 3D,
horizontal, and vertical). It intentionally does **not** depend on COLMAP's
optional download/curl/crypto feature and does not compute file hashes —
geometry/report SHA256 values belong one layer downstream, in an exported
asset's own sidecar (see below), where the actual asset bytes exist.

**The CSV report** (``--camera_residuals_csv``) has one row per database
image, sorted by name::

    image_name,registered,has_position_prior,position_fit_inlier,
    prior_e,prior_n,prior_u,solved_e,solved_n,solved_u,
    residual_e,residual_n,residual_u,residual_horizontal,residual_vertical,residual_3d,
    has_orientation_prior,orientation_fit_inlier,orientation_residual_deg

Absent numeric values (e.g. an image with no position prior, or the
always-deferred orientation columns) are empty cells, never ``0`` or the
literal text ``NaN`` — a downstream CSV reader must be able to distinguish
"no data" from "measured zero."

**Report diagnostics have limitations worth knowing before trusting them at
face value**: residuals are computed after the same robust fit that used
those very points as (candidate) inliers, so they describe fit quality, not
an independent validation; the RANSAC inlier/outlier split is a hard
threshold on one robust fit, not a calibrated per-point confidence; and
because orientation is not yet engaged, the report cannot detect an image
whose *position* prior is accurate but whose *orientation* would reveal a
larger underlying error.


Downstream asset composition (PLY/SOG)
----------------------------------------

COLMAP does not parse or write PLY/SOG asset sidecars itself; this section
documents the contract so an external exporter can compose one consistently
from a georeference report, without tying COLMAP to either format.

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
- To place a real-world WGS84 point (e.g. from a phone's GPS) into asset
  space: convert WGS84 → ECEF, then apply ``asset_from_ecef``. To place an
  asset-space point on Earth: apply ``ecef_from_asset``.

PLY assets use an adjacent sidecar file for these fields. SOG assets may
either embed the identical fields or also use an adjacent sidecar — the
report supplies everything an exporter needs regardless of which the target
format prefers.


Source-adapter responsibilities
----------------------------------

COLMAP's pose-prior archive format is intentionally generic; producing a
*correct* archive from a specific capture device is the adapter's
responsibility, not COLMAP's. The following conventions must be proven true
for the specific device/firmware before an adapter emits full orientation:

- **GoPro (JPEG APP6/GPMF), extracted-frame GPMF**: the altitude reported by
  consumer GPS is not ellipsoidal by default (it is typically a device- or
  firmware-specific mix of barometric and MSL/geoid-relative estimates); an
  adapter must apply an explicit geoid or barometric correction before
  writing ``ALT`` under ``height_datum: ELLIPSOIDAL``, or omit ``ALT``
  entirely (horizontal-only) rather than guess.
- **Betaflight (ESKF/compass) flight-controller logs**: a fused
  orientation estimate is only as trustworthy as its inputs; a magnetometer
  reading present in the log does **not** by itself make the fused
  quaternion trustworthy — magnetic declination at the capture location,
  camera/IMU extrinsic calibration, timestamp alignment between the log and
  the image/frame it's attached to, and the quaternion's axis
  convention/handedness must each be verified for that specific rig before
  the orientation is emitted as a pose prior.
- In general, for any adapter: ``GravityVector`` is only a unit down vector
  in sensor coordinates once the sensor-to-camera axis calibration (which
  axis is "down" in the raw IMU frame, and any mounting rotation) is known
  and applied — an uncalibrated raw accelerometer reading is not a gravity
  prior. Full orientation should only be emitted once quaternion direction
  (``sensor_from_world`` vs. its inverse), axis order, handedness,
  camera/IMU extrinsics, magnetic declination (if used), and log/frame
  timestamp alignment are all verified for that device. If any one of these
  is unverified, prefer emitting a gravity-only or position-only prior.
- **Missing yaw stays missing.** If a device cannot supply a trustworthy
  heading (e.g. no magnetometer, or GPS speed too low to derive course over
  ground), the adapter must leave the orientation prior entirely absent
  (``HasRotation() == false``) rather than emit a guessed quaternion with an
  inflated covariance to "represent the uncertainty" — a large-covariance
  guess still biases a weighted solve toward a specific (wrong) heading,
  while an absent prior contributes nothing and is handled correctly by
  every consumer described above.
