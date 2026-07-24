// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/estimators/alignment.h"
#include "colmap/geometry/sim3.h"
#include "colmap/optim/ransac.h"

#include <filesystem>
#include <limits>
#include <string>

namespace colmap {

////////////////////////////////////////////////////////////////////////////
// Output coordinate frames
////////////////////////////////////////////////////////////////////////////

// The output frame model_aligner writes its reconstruction in.
// ENU_Z_UP: unchanged existing behavior (East=+X, North=+Y, Up=+Z).
// GLTF_Y_UP: East=+X, Up=+Y, North=-Z, matching generic glTF 2.0 world-space
//   convention -- applied to the actual in-memory Reconstruction (points,
//   camera centers, orientations, frames, rigs) via Reconstruction::Transform,
//   not a post-export script. This is NOT necessarily correct for a COLMAP
//   dataset consumed by a specific tool's COLMAP importer, which may apply
//   its own data-to-visualizer basis change at its own load boundary -- see
//   LICHTFELD_COLMAP below.
// LICHTFELD_COLMAP: raw COLMAP-convention data (East=+X, -Up=+Y, North=+Z)
//   precomposed so that after LichtFeld Studio's own documented
//   visualizer_from_colmap_data = diag(1,-1,-1) boundary transform (see
//   LichtFeld's coordinate_conventions.hpp, upstream commit 11118860db73),
//   the scene displays East=+X, Up=+Y, North=-Z. This is the empirically
//   validated output contract for LichtFeld Studio import -- do not replace
//   it with GLTF_Y_UP, ENU_Z_UP, or an inferred "cleaner" convention.
enum class OutputCoordinateFrame { ENU_Z_UP, GLTF_Y_UP, LICHTFELD_COLMAP };

bool OutputCoordinateFrameFromString(const std::string& value,
                                     OutputCoordinateFrame* frame);

// Returns the pure-rotation Sim3d that this exporter applies to go from the
// canonical ENU frame to the requested output frame's raw written data.
// Identity for ENU_Z_UP.
Sim3d GeometryFromEnu(OutputCoordinateFrame output_coordinate_frame);

// LichtFeld's own documented data-to-visualizer boundary transform (a pure
// axis-sign flip, not applied by this exporter -- informational only, for
// composing the full "what the user actually sees" chain in the
// georeference report and for the `consumer_profile` contract metadata).
Sim3d LichtfeldVisualizerFromColmapData();

////////////////////////////////////////////////////////////////////////////
// Georeference report quality-warning thresholds
////////////////////////////////////////////////////////////////////////////

// Thresholds for the report's informational quality warnings (collinearity,
// gravity disagreement, position inlier ratio). These do not gate alignment
// admission -- they only control which `warnings.*.fired` flags the report
// sets, so a later policy can be reevaluated from the recorded value +
// threshold without rerunning COLMAP. Defaults match the values hard-coded
// prior to this option's introduction.
struct GeoreferenceQualityThresholds {
  double collinearity_ratio_threshold = 0.1;
  double gravity_median_threshold_deg = 3.0;
  double min_position_inlier_ratio = 0.8;
};

////////////////////////////////////////////////////////////////////////////
// Material-realignment thresholds
////////////////////////////////////////////////////////////////////////////

// Thresholds distinguishing a no-op re-fit from a "material" one, used both
// to evaluate the report's always-present final_realignment_check
// diagnostic and, when --reject_material_realignment is set, to enforce it
// as a hard gate. Defined once so evaluation and serialization cannot drift
// apart. Defaults match the values hard-coded prior to this option's
// introduction: 0.5 deg is well below the gravity-warning threshold used
// elsewhere in this report, 1 m is at the low end of consumer-GPS position
// uncertainty, 1% scale is far tighter than the metric-gauge regression
// test's 1% tolerance would even notice.
struct MaterialRealignmentThresholds {
  double max_rotation_deg = 0.5;
  double max_translation_m = 1.0;
  double max_scale_ratio = 0.01;
};

////////////////////////////////////////////////////////////////////////////
// Georeference report verbosity
////////////////////////////////////////////////////////////////////////////

// Controls how much machine-diagnostic detail --georeference_json contains.
// Both levels contain everything needed to georeference the output geometry:
// schema/version, output geometry frame and units, WGS84/height-datum
// information, ENU origin, geometry_from_enu/enu_from_geometry,
// ecef_from_geometry/geometry_from_ecef, support counts, evaluated quality
// values/thresholds/fired state, the final realignment result, and
// provenance. `full` additionally contains detailed percentile,
// singular-value, baseline, and ellipsoid-tangent diagnostics used for
// experiment verification. Machine-readable reports are versioned artifacts,
// not console logs -- this option does not affect console log verbosity
// (see --log_level/--v for that).
enum class GeoreferenceReportLevel { SUMMARY, FULL };

bool GeoreferenceReportLevelFromString(const std::string& value,
                                       GeoreferenceReportLevel* level);

////////////////////////////////////////////////////////////////////////////
// Options
////////////////////////////////////////////////////////////////////////////

// Bundled options for a model_aligner report-mode run. Replaces a long
// positional-argument list across the model.cc / model_georeference.cc
// boundary.
struct ModelGeoreferenceOptions {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::filesystem::path database_path;

  // The three origin values are all-or-none; altitude is WGS84 ellipsoidal.
  bool has_explicit_origin = false;
  double enu_origin_lat = std::numeric_limits<double>::quiet_NaN();
  double enu_origin_lon = std::numeric_limits<double>::quiet_NaN();
  double enu_origin_alt = std::numeric_limits<double>::quiet_NaN();

  std::string pose_prior_cartesian_frame;
  int min_common_images = 3;
  RANSACOptions ransac_options;
  // See AnisotropicPositionGate: unset (default) preserves the isotropic
  // ransac_options.max_error gate path bit for bit.
  AnisotropicPositionGate anisotropic_position_gate;

  bool use_pose_prior_orientation = false;
  double orientation_max_error_deg = 10.0;

  std::string scene_id;
  std::filesystem::path georeference_json;
  GeoreferenceReportLevel report_level = GeoreferenceReportLevel::SUMMARY;
  std::filesystem::path camera_residuals_csv;

  OutputCoordinateFrame output_coordinate_frame =
      OutputCoordinateFrame::ENU_Z_UP;

  // Off by default (preserves existing behavior byte-for-byte); the
  // pre/post-correction diagnostic is always reported regardless of this
  // flag. See RunModelAlignerReport's implementation for the rationale.
  bool reject_material_realignment = false;

  GeoreferenceQualityThresholds quality_thresholds;
  MaterialRealignmentThresholds material_realignment_thresholds;
};

// Extends model_aligner with the scene georeference report path. Reads the
// database's pose priors, robustly aligns the input reconstruction to them,
// optionally writes --georeference_json / --camera_residuals_csv, applies
// the requested output coordinate frame, and writes the aligned
// reconstruction to options.output_path.
int RunModelAlignerReport(const ModelGeoreferenceOptions& options);

////////////////////////////////////////////////////////////////////////////
// Durable georeferencing
////////////////////////////////////////////////////////////////////////////
//
// For a point in the serialized geometry frame:
//
//   point_ecef = ecef_from_geometry * point_geometry
//
// If a downstream editor only deletes points/Gaussians, every surviving
// point's coordinates are unchanged, so the same `ecef_from_geometry` (and
// its sibling transforms `geometry_from_enu` / `enu_from_geometry` /
// `geometry_from_ecef`, all recorded in the georeference report) remains
// exactly valid for a tiny remaining segment. No cameras or deleted geometry
// are required to apply it, and it must be copied alongside the exported
// PLY/SOG, not left only in the original COLMAP export directory.
//
// This does NOT hold if the editor translates, rotates, scales, or recenters
// a scene node before export: that bakes an additional
// `parent_geometry_from_child_geometry` transform into the exported asset
// that COLMAP never observes, so the correct chain becomes
//
//   ecef_from_child_geometry =
//       ecef_from_parent_geometry * parent_geometry_from_child_geometry
//
// with `parent_geometry_from_child_geometry = identity` for a deletion-only
// edit. Composing that node transform (when non-identity) is the downstream
// exporter's responsibility -- COLMAP cannot see the later edit and does not
// claim to preserve georeferencing through an arbitrary external
// recentering.

}  // namespace colmap
