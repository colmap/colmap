.. _pose-priors:

Pose Priors
===========

.. contents::
   :local:

Purpose and supported workflow
------------------------------

A *pose prior* is measured sensor data about where a camera was, imported
alongside the images and used to constrain reconstruction. The telemetry
workflow is:

1. A trusted adapter reads a capture's telemetry and writes a **pose prior
   archive**: one JSON file, one row per image, stating a WGS84 position with
   its uncertainty and optionally a gravity direction and a compass heading.
2. ``colmap pose_prior_importer`` validates that archive completely and writes
   it into the database in a single transaction.
3. ``colmap sequential_matcher`` can use the positions to skip image pairs that
   are too far apart to overlap.
4. ``colmap global_mapper`` solves the reconstruction with the positions as
   covariance-weighted constraints, so the result is metric and placed on the
   Earth rather than in an arbitrary frame at an arbitrary scale.
5. ``colmap model_aligner`` publishes the result together with a georeference
   report: the transforms a downstream consumer needs to convert the exported
   geometry back to ECEF or WGS84, and the diagnostics needed to decide whether
   to trust it.

The output is geometry a viewer can place on the Earth from the report alone --
no cameras, no database, no reconstruction. That is what makes it survive
editing and export into other tools.

Everything on this path is fail-closed. An archive that does not match the
contract below is rejected rather than partially interpreted, and a constraint
that is requested but cannot engage stops the run instead of quietly doing
nothing.

The archive
-----------

There is exactly one accepted format::

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
        ["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, 0.0, 1.0, 0.0, 92.0, 5.0],
        ["b.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, null, null, null, null, null]
      ]
    }

Columns
~~~~~~~

Column order is free. Each column may appear at most once. No column outside
this list is legal.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Column
     - Required
     - Meaning
   * - ``NAME``
     - yes
     - Image name, non-empty, unique, and resolvable in the database.
   * - ``LAT``, ``LON``, ``ALT``
     - yes
     - Degrees, degrees, and metres of **ellipsoidal** height.
   * - ``STD_TX``, ``STD_TY``, ``STD_TZ``
     - one form
     - One-sigma East, North, Up position uncertainty in metres.
   * - ``COV_TXX`` … ``COV_TZZ``
     - one form
     - Upper triangle of the position covariance in m², in the same local ENU
       frame. Six columns: ``XX, XY, XZ, YY, YZ, ZZ``.
   * - ``GX``, ``GY``, ``GZ``
     - optional
     - Measured **down** direction in camera coordinates, as a unit vector.
   * - ``HEADING_DEG``, ``HEADING_STD_DEG``
     - optional
     - Azimuth of the camera's horizontally-projected forward axis, clockwise
       from true north, and its own one-sigma uncertainty, in degrees.

Position uncertainty must be stated exactly once, as either the three
``STD_T*`` columns or the six ``COV_T*`` columns -- never both, never neither.
A covariance-weighted solve cannot weight a prior that declares no uncertainty.

``GX/GY/GZ`` and ``HEADING_DEG/HEADING_STD_DEG`` are the only optional groups.
A group is either wholly present in ``schema`` or wholly absent from it, and
within a row every member is either a number or JSON ``null``. The heading
group requires the gravity group.

Metadata
~~~~~~~~

Every key has exactly one supported value and is validated against it, so an
archive that a future producer means something different by fails instead of
being misread.

``schema_version``, ``coordinate_system``, ``sensor_type``, ``ellipsoid``,
``height_datum`` and ``position_covariance_frame`` are always required, with
the values shown above.

``gravity_frame`` and ``gravity_direction`` are required when the gravity
group is in ``schema``, and **forbidden** otherwise. ``heading_reference``,
``heading_axis`` and ``heading_rotation`` are required when the heading group
is in ``schema``, and forbidden otherwise. No other top-level key is legal.

Validation
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Rule
     - Why
   * - Unknown, missing, or conditionally-forbidden top-level key is rejected
     - An unrecognized name is a producer typo far more often than a deliberate
       extension, and skipping it would drop the measurement the operator
       believed they supplied.
   * - Unsupported ``schema_version`` is rejected
     - A version bump exists to say the meaning of these fields changed.
   * - At least one row, exactly one cell per schema column
     - A row of the wrong width has no unambiguous reading.
   * - ``NAME`` non-empty and unique; every name resolves to a database image
     - Two rows for one image leave the winner arbitrary. All unresolved names
       are reported in one error, before anything is written.
   * - ``LAT`` in [-90, 90], ``LON`` in [-180, 180], all of ``LAT/LON/ALT``
       finite
     - Out-of-range values are the usual sign of a swapped or mis-scaled field.
   * - Each standard deviation strictly positive
     - Zero uncertainty is an infinite weight; that row would pin the solve.
   * - Covariance symmetric and strictly positive definite
     - A zero, singular, or indefinite matrix states a direction of perfect
       certainty. Symmetry is by construction -- only the upper triangle is
       read.
   * - Optional group all-present or all-null per row; no partial group
     - Half a direction is not a direction.
   * - Gravity norm within 0.01 of one, then normalized; zero or larger error
       rejected
     - This column is a device-fused unit direction. It is **not** compared
       against 9.80665: a normalized direction has no acceleration magnitude
       left in it. An archive written from raw accelerometer counts is
       rejected here rather than silently mis-weighted.
   * - Heading requires gravity on the same row; ``HEADING_DEG`` in [0, 360);
       ``HEADING_STD_DEG`` in (0, 180]
     - The azimuth is measured in the plane the measured down vector
       establishes. Every heading row states its own accuracy; there is no CLI
       value that substitutes for it.
   * - A geographically distant but well-formed row is **kept**
     - Whether it is an outlier is a question about the capture, which robust
       fitting answers with the other rows in hand. The parser only knows
       whether the row is well-formed.

Frames, units, and conventions
------------------------------

Position
~~~~~~~~

Archive positions are WGS84 latitude, longitude, and ellipsoidal height. An
orthometric or geoid height differs from an ellipsoidal one by tens of metres,
which is why ``height_datum`` is mandatory and checked rather than assumed.

Internally COLMAP works in a local **ENU** tangent frame: East ``+X``, North
``+Y``, Up ``+Z``, metres. The frame's origin is the geometric median of the
priors' ECEF positions, converted back to WGS84, with the median of the
altitudes that rows actually declare. The median, not the mean or the first
row, so that one gross GPS fix -- a valid archive row, rejected later by robust
fitting rather than at import -- cannot move the frame every transform is
expressed against.

The mapper and the report derive this frame with the same code from the same
rows, sorted by a stable identifier, so two processes given the same priors
compute a bitwise-identical origin. Without that, geometry solved against one
origin and a report published against another are offset with nothing to
indicate why.

Covariance
~~~~~~~~~~

A row's covariance is declared in the local ENU frame at **that row's own**
latitude and longitude, because that is where the uncertainty was measured.
Converting into the shared frame rotates it::

    C_shared = R * C_local * R^T

where ``R`` takes the row's local ENU axes into the shared ones. Over a single
reconstruction ``R`` is near-identity, but it is not identity, and skipping it
quietly points anisotropic uncertainty in the wrong direction.

Gravity
~~~~~~~

``GX/GY/GZ`` is the measured **down** direction in camera coordinates. COLMAP
camera axes are ``+X`` right, ``+Y`` down, ``+Z`` forward.

The residual compares that measurement with the down direction the solved
camera rotation predicts. It constrains roll and pitch and is exactly invariant
to yaw: rotating the camera about its own predicted-down axis leaves the
prediction unchanged. It is a full chordal difference rather than a
tangent-plane projection, because the projected form is also zero for an
exactly inverted prediction, and a gravity residual with a minimum at the
antipode can attract an upside-down solution.

Heading
~~~~~~~

``HEADING_DEG`` is the azimuth of the camera's forward axis, projected into the
horizontal plane, measured clockwise from true north.

A magnetic reading must be declination-corrected before it is written, and a
device-body heading must be transformed through a known body-to-camera
extrinsic. The archive always describes the camera.

The residual is one signed angle, computed in the plane that the row's measured
down vector ``d`` establishes, for a measured heading ``h``::

    f     = (0, 0, 1)                      camera forward
    f_h   = normalize(f - d * dot(d, f))   forward, projected horizontal
    r_h   = normalize(cross(d, f_h))
    n_m   = cos(h) * f_h - sin(h) * r_h    measured north, in the camera frame
    n_p   = R_cam_from_world * north_world
    n_p_h = normalize(n_p - d * dot(d, n_p))
    residual = atan2(dot(d, cross(n_m, n_p_h)), dot(n_m, n_p_h))

The ``atan2`` form is signed, continuous across the 0/360 wrap, and returns
magnitude π -- not a false zero -- for an exactly opposite heading. A compass
180 degrees out is a real failure mode, and a residual blind to it would make
the wrong solution look optimal.

A heading is one number and is represented as one number. It is deliberately
not stored or solved as a quaternion: that would let the solver read roll and
pitch out of a measurement containing neither, and those fabricated degrees of
freedom would compete with the gravity residual, which does measure them.

When the camera's forward axis is within ``1e-3`` of the measured vertical, its
azimuth is undefined and the row's heading residual is skipped and counted.

Importing
---------

::

    colmap pose_prior_importer \
        --database_path DATABASE \
        --pose_prior_path ARCHIVE.json \
        --existing {error,replace}

The archive, every image binding, and the existing database state are all
validated before the first write; the writes then happen in one transaction.
On any error nothing is written.

``--existing=error``
    Fail if any resolved image already has a prior.

``--existing=replace``
    Write the row's complete prior over any existing one. A null gravity or
    heading group **clears** previously stored values for that image.

There is no merge mode. A prior assembled from two archives has no single
provenance and no way to tell which field came from where.

On success the importer logs one line containing the archive row count,
resolved image count, inserted and replaced counts, and how many rows carried
gravity and heading.

Matching
--------

``--SequentialMatching.max_prior_distance`` (metres) skips quadratic image
pairs whose priors are further apart than the threshold. A negative value
disables the gate.

The gate applies only to WGS84 priors, which are converted to ECEF before the
distance is taken. A prior with no declared coordinate system is not treated as
metres: images in different parts of a capture differ by whole degrees, and
comparing those against a metre threshold would veto long-range pairs --
including loop closures -- as if they were kilometres apart.

Mapping
-------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Option
     - Meaning
   * - ``--GlobalMapper.pose_prior_position_mode {off,optimize}``
     - ``optimize`` adds covariance-weighted position residuals and establishes
       the metric ENU gauge. Default ``off``.
   * - ``--GlobalMapper.pose_prior_use_gravity {0,1}``
     - Adds the yaw-free gravity residual. Default ``0``.
   * - ``--GlobalMapper.pose_prior_gravity_stddev_deg``
     - Sensor-class angular uncertainty applied to every gravity row, in
       degrees. Default ``5.0``.
   * - ``--GlobalMapper.pose_prior_use_heading {0,1}``
     - Adds the one-DoF heading residual. Default ``0``. Each row supplies its
       own uncertainty.

Failure semantics
~~~~~~~~~~~~~~~~~

Requesting a constraint that does not take effect is an error, not a warning.
A run that quietly drops a constraint still produces a plausible-looking
reconstruction, and nothing in the output says the measurement was ignored.

* ``pose_prior_position_mode=optimize`` that does not engage fails the run.
* ``pose_prior_use_gravity=1`` requires ``optimize``: gravity is expressed
  against the ENU frame that only the position solve establishes. The
  uncertainty must be in (0, 180] degrees.
* ``pose_prior_use_heading=1`` requires both ``optimize`` and
  ``pose_prior_use_gravity=1``, and at least one row carrying both a heading
  and the gravity it depends on. There is no heading-only configuration.

Valid gravity and heading data is imported and retained regardless of these
flags, so enabling one later does not require re-importing.

Once the gauge is established, the position, gravity, and heading residuals are
carried into every subsequent pose-prior bundle-adjustment stage, including the
refinement stages that run through the incremental mapper. A constraint that
stops applying partway through a solve is one that silently stops being true.

Each residual is whitened by its own uncertainty and passed through a fixed
robust loss whose radius is the 95% chi-square confidence radius for that
residual's degrees of freedom: 3 for position, 2 for gravity (the chordal
residual has three components but rank two at the solution), 1 for heading.
These radii are named constants, not options.

Upstream's ``--GlobalMapper.ra_use_gravity`` rotation-averaging reduction is
unchanged and independent. Do not enable it together with
``pose_prior_use_gravity``: the soft and hard paths would consume the same
readings twice.

Publishing
----------

::

    colmap model_aligner \
        --input_path SPARSE --output_path ALIGNED \
        --database_path DATABASE \
        --alignment_type enu \
        --alignment_max_error 10 \
        --alignment_random_seed 12345 \
        --scene_id SCENE \
        --georeference_json georeference.json \
        --camera_residuals_csv camera_residuals.csv \
        --output_coordinate_frame {ENU_Z_UP,LICHTFELD_COLMAP}

Requesting ``--georeference_json`` or ``--camera_residuals_csv`` selects the
report path. Without them, ``model_aligner`` behaves exactly as upstream.

The report path publishes a delivery, so its input must already be solved in
the metric ENU gauge. Its Sim3 verification uses the same per-row covariance
weighting as the mapper and must be a no-op; a material correction means the
mapper solve and delivery verification disagree. That fails the run.
**To align a reconstruction that was never aligned, run** ``model_aligner``
**without the report flags.**

The path requires WGS84 priors with valid covariance, at least
``--min_common_images`` registered correspondences, and an altitude somewhere
in the archive -- a report states an ellipsoidal height and cannot substitute a
placeholder. Cartesian priors are rejected: they carry no datum to georeference
against.

The reconstruction is written first and reopened to confirm it matches the
in-memory counts; only then are the JSON and CSV published through checked
temporary files and atomic same-directory renames. A sidecar claiming success
is never left behind by a failed run.

Output frames
~~~~~~~~~~~~~

``ENU_Z_UP``
    East ``+X``, North ``+Y``, Up ``+Z``. The canonical frame.

``LICHTFELD_COLMAP``
    Raw written data is East ``+X``, ``-Up`` ``+Y``, North ``+Z``. LichtFeld
    Studio's COLMAP loader then applies its own ``diag(1, -1, -1)`` boundary
    rotation, after which the scene displays East ``+X``, Up ``+Y``, North
    ``-Z``. The pre-composition is what makes the scene appear upright.

The chosen transform is applied to the reconstruction that is actually written,
not only to the report's metadata.

The report
----------

Every ``georeference.json`` has the same shape, under one top-level
``schema_version: 1``. There is no verbosity setting: a consumer cannot ask for
a field that a previous run chose not to write.

* **Provenance** -- ``scene_id``, COLMAP version and source commit, creation
  time, input and output paths.
* **CRS** -- ellipsoid, height datum, and the selected ENU origin.
* **Support** -- database priors, registered images, registered prior
  correspondences, position inliers and outliers, gravity and heading
  observation counts.
* **Alignment** -- ``enu_from_input_sfm`` and its inverse, metres per input
  unit, the standardized robust radius, and the deterministic random seed.
* **Frame contract** -- geometry frame, handedness, up axis, units, and
  ``geometry_from_enu`` / ``enu_from_geometry`` / ``ecef_from_geometry`` /
  ``geometry_from_ecef``. For ``LICHTFELD_COLMAP``, also the visualizer
  boundary transform and displayed up axis.
* **Diagnostics** -- position residuals in metres (3D, horizontal, vertical),
  and gravity and heading residuals in degrees; each with mean, median, P90,
  maximum, and support.
* **Quality** -- each check's measured value, its fixed threshold, and whether
  it fired.
* **Final realignment check** -- rotation, translation, scale delta, their
  thresholds, and ``is_material``.

Heading fields are present with zero support when no headings exist, so
adopting a compass later does not change the report's shape.

Residual CSV
~~~~~~~~~~~~

One row per registered image, ordered by image name::

    image_name,image_id,has_position_prior,position_fit_inlier,
    residual_east_m,residual_north_m,residual_up_m,
    residual_horizontal_m,residual_3d_m,
    has_gravity_prior,gravity_residual_deg,
    has_heading_prior,heading_stddev_deg,heading_residual_deg

An empty cell means the quantity does not exist for that row. This file is the
operator's outlier-cleanup input, so it includes every registered image --
including rows robust fitting rejected. A robust loss down-weights a
measurement; it does not make it disappear from the record.

Thresholds
~~~~~~~~~~

Fixed policy, recorded in every report alongside the value they judged. None is
a CLI option: a gate an operator can widen from the command line is not a gate,
and the recorded threshold would describe that invocation rather than the
contract.

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Check
     - Threshold
     - Effect
   * - Collinearity (second/first horizontal singular-value ratio)
     - ``0.1``
     - Warning only. A capture that follows a road or a facade is naturally
       elongated.
   * - Median gravity disagreement
     - ``3.0°``
     - Fails delivery.
   * - Position inlier ratio
     - ``0.8``
     - Fails delivery.
   * - Final realignment rotation
     - ``0.5°``
     - Fails delivery.
   * - Final realignment translation
     - ``1.0 m``
     - Fails delivery.
   * - Final realignment scale delta
     - ``0.01``
     - Fails delivery.

Using the report downstream
---------------------------

For a point in the serialized geometry frame::

    point_ecef = ecef_from_geometry * point_geometry

``ecef_from_geometry`` and its siblings are recorded in the report, so a viewer
needs nothing else -- no cameras, no database, no deleted geometry -- to place
the asset on the Earth. Copy the report alongside the exported asset rather
than leaving it in the original COLMAP output directory.

This survives a deletion-only edit exactly. If a downstream editor only removes
points or Gaussians, every surviving coordinate is unchanged, so the same
transform remains valid for whatever fragment is left.

It does **not** survive an edit that translates, rotates, scales, or recentres
a scene node before export. That bakes an additional transform into the asset
which COLMAP never observes::

    ecef_from_child = ecef_from_parent * parent_from_child

with ``parent_from_child`` the identity for a deletion-only edit. Composing a
non-identity node transform is the exporting tool's responsibility.

To locate a device in the scene, convert its WGS84 fix to ECEF and apply
``geometry_from_ecef``. Positional uncertainty carries through the same
transform, so a reported accuracy radius becomes a radius in scene units.
