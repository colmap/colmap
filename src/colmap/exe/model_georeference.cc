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

#include "colmap/exe/model_georeference.h"

#include "colmap/estimators/alignment.h"
#include "colmap/geometry/gps.h"
#include "colmap/math/geometric_median.h"
#include "colmap/math/math.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/version.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <locale>
#include <sstream>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace colmap {
namespace {

////////////////////////////////////////////////////////////////////////////
// JSON/CSV serialization helpers
////////////////////////////////////////////////////////////////////////////

std::string JSONEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (const char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          out += StringPrintf("\\u%04x", static_cast<unsigned char>(c));
        } else {
          out += c;
        }
    }
  }
  return out;
}

// 17-digit round-trippable double formatting in the classic ("C") locale, or
// JSON `null` for a non-finite (absent) value.
std::string JSONNumber(double value) {
  if (!std::isfinite(value)) {
    return "null";
  }
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream.precision(17);
  stream << value;
  return stream.str();
}

std::string JSONSim3(const Sim3d& s) {
  return StringPrintf(
      "{\"scale\":%s,\"rotation_wxyz\":[%s,%s,%s,%s],\"translation_xyz\":[%s,"
      "%s,%s]}",
      JSONNumber(s.scale()).c_str(),
      JSONNumber(s.rotation().w()).c_str(),
      JSONNumber(s.rotation().x()).c_str(),
      JSONNumber(s.rotation().y()).c_str(),
      JSONNumber(s.rotation().z()).c_str(),
      JSONNumber(s.translation().x()).c_str(),
      JSONNumber(s.translation().y()).c_str(),
      JSONNumber(s.translation().z()).c_str());
}

// Row-major 3x3 rotation matrix as a JSON array-of-arrays, e.g.
// [[1,0,0],[0,0,-1],[0,1,0]].
std::string JSONRotationMatrix(const Eigen::Quaterniond& rotation) {
  const Eigen::Matrix3d m = rotation.toRotationMatrix();
  return StringPrintf("[[%s,%s,%s],[%s,%s,%s],[%s,%s,%s]]",
                      JSONNumber(m(0, 0)).c_str(),
                      JSONNumber(m(0, 1)).c_str(),
                      JSONNumber(m(0, 2)).c_str(),
                      JSONNumber(m(1, 0)).c_str(),
                      JSONNumber(m(1, 1)).c_str(),
                      JSONNumber(m(1, 2)).c_str(),
                      JSONNumber(m(2, 0)).c_str(),
                      JSONNumber(m(2, 1)).c_str(),
                      JSONNumber(m(2, 2)).c_str());
}

bool IsValidSceneId(const std::string& scene_id) {
  if (scene_id.empty()) {
    return false;
  }
  for (const char c : scene_id) {
    if (static_cast<unsigned char>(c) < 0x20) {
      return false;
    }
  }
  return true;
}

// Deterministic, inlier-refined WGS84 origin: median ECEF of full-position
// (lat/lon/alt all finite) reference points, never the first row. Returns
// false if there are no full-position points to derive an origin from.
bool DeriveWGS84Origin(const std::vector<Eigen::Vector3d>& lla_points,
                       double* lat,
                       double* lon,
                       double* alt) {
  if (lla_points.empty()) {
    return false;
  }
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);
  const std::vector<Eigen::Vector3d> ecef_points =
      gps_transform.EllipsoidToECEF(lla_points);
  const Eigen::Vector3d median_ecef = GeometricMedian(ecef_points);
  const Eigen::Vector3d median_lla =
      gps_transform.ECEFToEllipsoid({median_ecef})[0];
  *lat = median_lla.x();
  *lon = median_lla.y();
  std::vector<double> alts;
  alts.reserve(lla_points.size());
  for (const Eigen::Vector3d& p : lla_points) {
    alts.push_back(p.z());
  }
  std::sort(alts.begin(), alts.end());
  *alt = alts[alts.size() / 2];
  return true;
}

// Per-camera georeference diagnostics.
struct CameraPosePriorResidual {
  image_t image_id = kInvalidImageId;
  std::string image_name;
  bool registered = false;
  bool has_position_prior = false;
  bool position_fit_inlier = false;
  Eigen::Vector3d prior_enu =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d solved_enu =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Matrix3d position_covariance_enu =
      Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
  bool has_gravity_prior = false;
  // Gravity (down) in the sensor coordinate system, frame-invariant under
  // the report's ENU re-expression of the position prior. Measured value as
  // read from the pose prior archive.
  Eigen::Vector3d gravity_sensor =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  // Predicted down direction in the sensor frame, i.e.
  // CamFromWorld().rotation() * ENU-down, for a registered image with a
  // gravity prior. Directly comparable to gravity_sensor and exported for
  // per-image outlier diagnosis.
  Eigen::Vector3d predicted_gravity_sensor =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  // Angle between predicted_gravity_sensor and gravity_sensor, in degrees,
  // allowing an individual bad gravity reading to be identified in the CSV.
  double gravity_residual_deg = std::numeric_limits<double>::quiet_NaN();
};

double Percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const double rank = fraction * static_cast<double>(values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(rank));
  const size_t hi = static_cast<size_t>(std::ceil(rank));
  if (lo == hi) {
    return values[lo];
  }
  const double t = rank - static_cast<double>(lo);
  return values[lo] * (1.0 - t) + values[hi] * t;
}

struct ScalarStatistics {
  double mean = std::numeric_limits<double>::quiet_NaN();
  double median = std::numeric_limits<double>::quiet_NaN();
  double p90 = std::numeric_limits<double>::quiet_NaN();
  double max = std::numeric_limits<double>::quiet_NaN();
};

ScalarStatistics ComputeStatistics(const std::vector<double>& values) {
  ScalarStatistics stats;
  if (values.empty()) {
    return stats;
  }
  stats.mean = Mean(values);
  stats.median = Percentile(values, 0.5);
  stats.p90 = Percentile(values, 0.9);
  stats.max = *std::max_element(values.begin(), values.end());
  return stats;
}

template <int kDim>
Eigen::Matrix<double, kDim, 1> CenteredRmsSingularValues(
    const std::vector<Eigen::Vector3d>& points) {
  Eigen::Matrix<double, kDim, 1> values =
      Eigen::Matrix<double, kDim, 1>::Constant(
          std::numeric_limits<double>::quiet_NaN());
  if (points.size() < static_cast<size_t>(kDim)) {
    return values;
  }
  Eigen::Matrix<double, Eigen::Dynamic, kDim> centered(points.size(), kDim);
  Eigen::Matrix<double, kDim, 1> mean = Eigen::Matrix<double, kDim, 1>::Zero();
  for (const Eigen::Vector3d& point : points) {
    mean += point.head<kDim>();
  }
  mean /= static_cast<double>(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    centered.row(i) = (points[i].head<kDim>() - mean).transpose();
  }
  values = centered.jacobiSvd().singularValues() /
           std::sqrt(static_cast<double>(points.size()));
  return values;
}

////////////////////////////////////////////////////////////////////////////
// Output coordinate frame transforms
////////////////////////////////////////////////////////////////////////////

// East->+X (raw), Up->-Y (raw), North->+Z (raw). This is the exact
// inverse/transpose of GltfYUpFromEnu()'s rotation, i.e. it precomposes the
// data so that LichtFeld's own visualizer_from_colmap_data = diag(1,-1,-1)
// boundary transform (applied by LichtFeld itself at import/render time, not
// by this exporter) results in a displayed East=+X, Up=+Y, North=-Z scene.
// Determinant +1 (proper rotation, handedness preserved). This is the
// empirically validated LichtFeld Studio import contract; do not change
// this matrix without re-verifying upright display in an installed
// LichtFeld build.
Sim3d LichtfeldColmapFromEnu() {
  Eigen::Matrix3d rotation;
  // clang-format off
  rotation << 1, 0,  0,
              0, 0, -1,
              0, 1,  0;
  // clang-format on
  return Sim3d(1.0, Eigen::Quaterniond(rotation), Eigen::Vector3d::Zero());
}

}  // namespace

bool OutputCoordinateFrameFromString(const std::string& value,
                                     OutputCoordinateFrame* frame) {
  if (value == "ENU_Z_UP") {
    *frame = OutputCoordinateFrame::ENU_Z_UP;
    return true;
  }
  if (value == "LICHTFELD_COLMAP") {
    *frame = OutputCoordinateFrame::LICHTFELD_COLMAP;
    return true;
  }
  return false;
}

Sim3d GeometryFromEnu(OutputCoordinateFrame output_coordinate_frame) {
  switch (output_coordinate_frame) {
    case OutputCoordinateFrame::ENU_Z_UP:
      return Sim3d();
    case OutputCoordinateFrame::LICHTFELD_COLMAP:
      return LichtfeldColmapFromEnu();
  }
  return Sim3d();
}

Sim3d LichtfeldVisualizerFromColmapData() {
  Eigen::Matrix3d rotation;
  // clang-format off
  rotation << 1, 0,  0,
              0, -1, 0,
              0, 0, -1;
  // clang-format on
  return Sim3d(1.0, Eigen::Quaterniond(rotation), Eigen::Vector3d::Zero());
}

bool GeoreferenceReportLevelFromString(const std::string& value,
                                       GeoreferenceReportLevel* level) {
  if (value == "summary") {
    *level = GeoreferenceReportLevel::SUMMARY;
    return true;
  }
  if (value == "full") {
    *level = GeoreferenceReportLevel::FULL;
    return true;
  }
  return false;
}

namespace {

////////////////////////////////////////////////////////////////////////////
// Scene georeference report (model_aligner --georeference_json/
// --camera_residuals_csv).
////////////////////////////////////////////////////////////////////////////

void WriteGeoreferenceReportJSON(
    const std::filesystem::path& path,
    const std::string& scene_id,
    const std::string& source_commit,
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    const Reconstruction& reconstruction,
    double origin_lat,
    double origin_lon,
    double origin_alt,
    bool origin_is_explicit,
    const Sim3d& enu_from_sfm,
    const std::vector<CameraPosePriorResidual>& residuals,
    int num_database_pose_priors,
    double max_ellipsoid_tangent_departure_m,
    double position_ransac_threshold,
    int alignment_random_seed,
    OutputCoordinateFrame output_coordinate_frame,
    bool reject_material_realignment_requested,
    const GeoreferenceQualityThresholds& quality_thresholds,
    const MaterialRealignmentThresholds& material_realignment_thresholds,
    GeoreferenceReportLevel report_level) {
  const bool full_report = report_level == GeoreferenceReportLevel::FULL;
  const Sim3d sfm_from_enu = Inverse(enu_from_sfm);
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);
  const Eigen::Matrix3d ecef_from_enu_rotation =
      GPSTransform::ECEFFromENU(origin_lat, origin_lon);
  const Eigen::Vector3d origin_ecef = gps_transform.EllipsoidToECEF(
      {Eigen::Vector3d(origin_lat, origin_lon, origin_alt)})[0];
  const Sim3d ecef_from_enu(
      1.0, Eigen::Quaterniond(ecef_from_enu_rotation), origin_ecef);
  const Sim3d enu_from_ecef = Inverse(ecef_from_enu);
  const Sim3d ecef_from_sfm = ecef_from_enu * enu_from_sfm;
  const Sim3d sfm_from_ecef = Inverse(ecef_from_sfm);

  // Verify the numerical inverse of every transform pair before publication.
  const auto verify_inverse = [](const Sim3d& a, const Sim3d& b) {
    const Eigen::Vector3d probe(1.0, 2.0, 3.0);
    const Eigen::Vector3d round_trip = b * (a * probe);
    THROW_CHECK_LT((round_trip - probe).norm(), 1e-6);
  };
  verify_inverse(enu_from_sfm, sfm_from_enu);
  verify_inverse(ecef_from_enu, enu_from_ecef);
  verify_inverse(ecef_from_sfm, sfm_from_ecef);

  std::vector<double> horizontal_residuals;
  std::vector<double> vertical_residuals;
  std::vector<double> full_residuals;
  std::vector<double> horizontal_sigmas;
  int num_position_inliers = 0;
  int num_registered_correspondences = 0;
  double max_horizontal_radius = 0.0;
  double max_3d_radius = 0.0;
  double max_horizontal_baseline = 0.0;
  std::vector<Eigen::Vector3d> inlier_prior_enu;
  for (const CameraPosePriorResidual& r : residuals) {
    if (r.registered && r.prior_enu.allFinite()) {
      ++num_registered_correspondences;
    }
    if (r.position_fit_inlier && r.prior_enu.allFinite() &&
        r.solved_enu.allFinite()) {
      ++num_position_inliers;
      const Eigen::Vector3d diff = r.solved_enu - r.prior_enu;
      horizontal_residuals.push_back(std::hypot(diff.x(), diff.y()));
      vertical_residuals.push_back(std::abs(diff.z()));
      full_residuals.push_back(diff.norm());
      max_horizontal_radius = std::max(
          max_horizontal_radius, std::hypot(r.prior_enu.x(), r.prior_enu.y()));
      max_3d_radius = std::max(max_3d_radius, r.prior_enu.norm());
      inlier_prior_enu.push_back(r.prior_enu);
      if (r.position_covariance_enu.allFinite()) {
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(
            r.position_covariance_enu.topLeftCorner<2, 2>());
        if (eig.info() == Eigen::Success && eig.eigenvalues().allFinite() &&
            eig.eigenvalues().maxCoeff() >= 0.0) {
          horizontal_sigmas.push_back(std::sqrt(eig.eigenvalues().maxCoeff()));
        }
      }
    }
  }
  for (size_t i = 0; i < inlier_prior_enu.size(); ++i) {
    for (size_t j = i + 1; j < inlier_prior_enu.size(); ++j) {
      const double d =
          (inlier_prior_enu[i].head<2>() - inlier_prior_enu[j].head<2>())
              .norm();
      max_horizontal_baseline = std::max(max_horizontal_baseline, d);
    }
  }

  const ScalarStatistics horizontal_stats =
      ComputeStatistics(horizontal_residuals);
  const ScalarStatistics vertical_stats = ComputeStatistics(vertical_residuals);
  const ScalarStatistics full_stats = ComputeStatistics(full_residuals);
  const double sigma_h = Percentile(horizontal_sigmas, 0.5);
  const double baseline_to_sigma =
      std::isfinite(sigma_h) && sigma_h > 0.0
          ? max_horizontal_baseline / sigma_h
          : std::numeric_limits<double>::quiet_NaN();
  const Eigen::Vector2d horizontal_rms_singular_values =
      CenteredRmsSingularValues<2>(inlier_prior_enu);
  const Eigen::Vector3d full_rms_singular_values =
      CenteredRmsSingularValues<3>(inlier_prior_enu);
  const double horizontal_condition_ratio =
      horizontal_rms_singular_values.allFinite() &&
              horizontal_rms_singular_values.x() > 0.0
          ? horizontal_rms_singular_values.y() /
                horizontal_rms_singular_values.x()
          : std::numeric_limits<double>::quiet_NaN();
  const double full_condition_ratio =
      full_rms_singular_values.allFinite() && full_rms_singular_values.x() > 0.0
          ? full_rms_singular_values.z() / full_rms_singular_values.x()
          : std::numeric_limits<double>::quiet_NaN();

  // Per-image gravity residual: for every registered image with a gravity
  // prior, rotate the sensor-frame down vector into the (already ENU-
  // aligned) world frame and compare it against ENU-down individually.
  // Deliberately not a single angle of the normalized mean vector -- that
  // construction lets images with opposite-signed errors cancel in the sum,
  // reading as a falsely small angle even when individual images disagree
  // substantially. median/p90/max/support are reported like every other
  // residual stat in this report (see position_3d_residual_m etc. above),
  // and the legacy single-number `gravity_consistency_angle_deg` field (and
  // the warning gate that uses it) is now defined as the robust median of
  // these per-image values, not the old mean-vector angle.
  std::vector<double> gravity_residuals_deg;
  for (const CameraPosePriorResidual& r : residuals) {
    if (!r.registered || !r.has_gravity_prior ||
        !std::isfinite(r.gravity_residual_deg)) {
      continue;
    }
    gravity_residuals_deg.push_back(r.gravity_residual_deg);
  }
  const int num_gravity_priors = static_cast<int>(gravity_residuals_deg.size());
  const ScalarStatistics gravity_stats =
      ComputeStatistics(gravity_residuals_deg);
  const double gravity_consistency_angle_deg = gravity_stats.median;

  // Pipeline-policy warning thresholds (see the ground-truth "Post-alignment
  // warnings" section); values and thresholds are recorded in the report so
  // policy can evolve without re-running.
  const double kCollinearityRatioThreshold =
      quality_thresholds.collinearity_ratio_threshold;
  const double kGravityAngleThresholdDeg =
      quality_thresholds.gravity_median_threshold_deg;
  const double kPositionInlierRatioThreshold =
      quality_thresholds.min_position_inlier_ratio;
  const bool collinearity_warning_fired =
      std::isfinite(horizontal_condition_ratio) &&
      horizontal_condition_ratio < kCollinearityRatioThreshold;
  const bool gravity_warning_fired =
      std::isfinite(gravity_consistency_angle_deg) &&
      gravity_consistency_angle_deg > kGravityAngleThresholdDeg;
  const double position_inlier_ratio =
      num_registered_correspondences > 0
          ? static_cast<double>(num_position_inliers) /
                num_registered_correspondences
          : std::numeric_limits<double>::quiet_NaN();
  const bool position_inlier_ratio_warning_fired =
      std::isfinite(position_inlier_ratio) &&
      position_inlier_ratio < kPositionInlierRatioThreshold;
  if (collinearity_warning_fired) {
    LOG(WARNING) << StringPrintf(
        "=> Near-collinear position support (horizontal singular-value "
        "ratio %.6f < %.2f) — rotation about the track axis is weakly "
        "constrained",
        horizontal_condition_ratio,
        kCollinearityRatioThreshold);
  }
  if (gravity_warning_fired) {
    LOG(WARNING) << StringPrintf(
        "=> Aligned up-axis disagrees with gravity priors (%.3f deg > %.2f "
        "deg)",
        gravity_consistency_angle_deg,
        kGravityAngleThresholdDeg);
  }
  if (position_inlier_ratio_warning_fired) {
    LOG(WARNING) << StringPrintf(
        "=> Large fraction of registered images disagree with the alignment "
        "(position inlier ratio %.6f < %.2f) — possible internal "
        "misregistration (repeated structure / false loop closures)",
        position_inlier_ratio,
        kPositionInlierRatioThreshold);
  }

  int num_registered = 0;
  int num_with_prior = 0;
  for (const CameraPosePriorResidual& r : residuals) {
    if (r.registered) {
      ++num_registered;
    }
    if (r.has_position_prior) {
      ++num_with_prior;
    }
  }

  std::ostringstream json;
  json.imbue(std::locale::classic());
  json << "{";
  json << "\"schema\":\"colmap_scene_georeference\",";
  json << "\"report_level\":\"" << (full_report ? "full" : "summary") << "\",";
  if (scene_id.empty()) {
    json << "\"scene_id\":null,";
  } else {
    json << "\"scene_id\":\"" << JSONEscape(scene_id) << "\",";
  }
  json << "\"source_commit\":\"" << JSONEscape(source_commit) << "\",";
  json << "\"input_path\":\"" << JSONEscape(input_path.string()) << "\",";
  json << "\"output_path\":\"" << JSONEscape(output_path.string()) << "\",";
  json << "\"num_registered_images\":" << reconstruction.NumRegImages() << ",";
  json << "\"num_points3D\":" << reconstruction.NumPoints3D() << ",";
  json << "\"ellipsoid\":\"WGS84\",";
  json << "\"height_datum\":\"ELLIPSOIDAL\",";
  json << "\"enu_origin\":{\"lat_deg\":" << JSONNumber(origin_lat)
       << ",\"lon_deg\":" << JSONNumber(origin_lon)
       << ",\"ellipsoidal_alt_m\":" << JSONNumber(origin_alt)
       << ",\"explicit\":" << (origin_is_explicit ? "true" : "false") << "},";
  json << "\"transforms\":{";
  json << "\"enu_from_sfm\":" << JSONSim3(enu_from_sfm) << ",";
  json << "\"sfm_from_enu\":" << JSONSim3(sfm_from_enu) << ",";
  json << "\"ecef_from_enu\":" << JSONSim3(ecef_from_enu) << ",";
  json << "\"enu_from_ecef\":" << JSONSim3(enu_from_ecef) << ",";
  json << "\"ecef_from_sfm\":" << JSONSim3(ecef_from_sfm) << ",";
  json << "\"sfm_from_ecef\":" << JSONSim3(sfm_from_ecef);
  json << "},";
  json << "\"metres_per_sfm_unit\":" << JSONNumber(enu_from_sfm.scale()) << ",";

  // Full transform chain relative to the geometry actually written to
  // output_path -- not just an informal inverse-rotation note. A consumer
  // must be able to recover ENU and ECEF/WGS84 from any delivered artifact,
  // and the frame choice changes the serialized world coordinates, so it
  // must compose into these transforms rather than being described only by
  // `targets`. These four transforms remain valid for any deletion-only
  // subset of the geometry (cropping/deleting points or Gaussians does not
  // change the transform that recovers ECEF/ENU for the surviving points),
  // so they must never be dropped from the report.
  const Sim3d geometry_from_enu = GeometryFromEnu(output_coordinate_frame);
  const Sim3d enu_from_geometry = Inverse(geometry_from_enu);
  const Sim3d ecef_from_geometry = ecef_from_enu * enu_from_geometry;
  const Sim3d geometry_from_ecef = Inverse(ecef_from_geometry);
  verify_inverse(geometry_from_enu, enu_from_geometry);
  verify_inverse(ecef_from_geometry, geometry_from_ecef);

  std::string geometry_frame_name;
  std::string up_axis;
  switch (output_coordinate_frame) {
    case OutputCoordinateFrame::ENU_Z_UP:
      geometry_frame_name = "ENU_LOCAL";
      up_axis = "Z";
      break;
    case OutputCoordinateFrame::LICHTFELD_COLMAP:
      geometry_frame_name = "LICHTFELD_COLMAP";
      // The bytes written to output_path are ordinary COLMAP convention
      // (Y increases downward); only after LichtFeld's own boundary
      // transform does the *displayed* scene have Y pointing up -- see
      // consumer_profile.visualizer_up_axis below. Do not read this as "the
      // file itself is Y-up".
      up_axis = "-Y";
      break;
  }

  // geometry_frame/up_axis describe the frame actually written to
  // output_path by this command (see the Transform applied just before
  // Reconstruction::Write in the caller) -- never claim ENU bytes while
  // Y-up or LichtFeld-precomposed bytes were written, or vice versa.
  json << "\"frame_contract\":{";
  json << "\"schema_version\":2,";
  json << "\"geometry_frame\":\"" << geometry_frame_name << "\",";
  json << "\"geometry_already_transformed\":true,";
  json << "\"handedness\":\"RIGHT\",";
  json << "\"up_axis\":\"" << up_axis << "\",";
  json << "\"units\":\"METRE\",";
  json << "\"crs\":{";
  json << "\"ellipsoid\":\"WGS84\",";
  json << "\"height_datum\":\"ELLIPSOIDAL\",";
  json << "\"geoid_model\":null,";
  json << "\"origin\":{\"lat_deg\":" << JSONNumber(origin_lat)
       << ",\"lon_deg\":" << JSONNumber(origin_lon)
       << ",\"ellipsoidal_alt_m\":" << JSONNumber(origin_alt) << "}";
  json << "},";
  json << "\"transforms\":{";
  json << "\"geometry_from_enu\":" << JSONSim3(geometry_from_enu) << ",";
  json << "\"enu_from_geometry\":" << JSONSim3(enu_from_geometry) << ",";
  json << "\"ecef_from_geometry\":" << JSONSim3(ecef_from_geometry) << ",";
  json << "\"geometry_from_ecef\":" << JSONSim3(geometry_from_ecef);
  json << "}";
  if (output_coordinate_frame == OutputCoordinateFrame::LICHTFELD_COLMAP) {
    const Sim3d visualizer_from_geometry = LichtfeldVisualizerFromColmapData();
    json << ",\"consumer_profile\":{";
    json << "\"name\":\"LICHTFELD_COLMAP\",";
    // Factual, versionable contract metadata describing the transform this
    // exporter applied -- not a runtime claim about which GUI build a
    // future user has installed (this binary cannot observe that). A
    // pipeline manifest or user-run acceptance record may separately state
    // that a particular installed build was visually verified.
    json << "\"contract_version\":1,";
    json << "\"boundary\":\"DATA_TO_VISUALIZER_WORLD_AXES\",";
    json << "\"source_reference\":"
            "\"https://github.com/MrNeRF/LichtFeld-Studio/blob/"
            "11118860db73ecf372bd9bc7448a1e250c8f3572/src/rendering/include/"
            "rendering/coordinate_conventions.hpp\",";
    json << "\"visualizer_from_geometry\":"
         << JSONSim3(visualizer_from_geometry) << ",";
    json << "\"visualizer_up_axis\":\"Y\"";
    json << "}";
  }
  json << "},";
  json << "\"support\":{";
  json << "\"num_database_pose_priors\":" << num_database_pose_priors << ",";
  json << "\"num_registered\":" << num_registered << ",";
  json << "\"num_with_position_prior\":" << num_with_prior << ",";
  json << "\"num_registered_position_correspondences\":"
       << num_registered_correspondences << ",";
  json << "\"num_position_inliers\":" << num_position_inliers;
  json << "},";
  // gravity_consistency_angle_deg is the one scalar diagnostic the
  // gravity_disagreement warning is evaluated against, so it is reported at
  // both levels; the detailed per-image stat breakdowns, singular-value/
  // baseline/radius/ellipsoid-tangent diagnostics below are `full`-only
  // experiment-verification detail.
  json << "\"diagnostics\":{";
  json << "\"gravity_consistency_angle_deg\":"
       << JSONNumber(gravity_consistency_angle_deg);
  if (full_report) {
    json << ",";
    json << "\"position_3d_residual_m\":{"
         << "\"mean\":" << JSONNumber(full_stats.mean)
         << ",\"median\":" << JSONNumber(full_stats.median)
         << ",\"p90\":" << JSONNumber(full_stats.p90)
         << ",\"max\":" << JSONNumber(full_stats.max) << "},";
    json << "\"position_horizontal_residual_m\":{"
         << "\"mean\":" << JSONNumber(horizontal_stats.mean)
         << ",\"median\":" << JSONNumber(horizontal_stats.median)
         << ",\"p90\":" << JSONNumber(horizontal_stats.p90)
         << ",\"max\":" << JSONNumber(horizontal_stats.max) << "},";
    json << "\"position_vertical_residual_m\":{"
         << "\"mean\":" << JSONNumber(vertical_stats.mean)
         << ",\"median\":" << JSONNumber(vertical_stats.median)
         << ",\"p90\":" << JSONNumber(vertical_stats.p90)
         << ",\"max\":" << JSONNumber(vertical_stats.max) << "},";
    json << "\"max_horizontal_baseline_m\":"
         << JSONNumber(max_horizontal_baseline) << ",";
    json << "\"horizontal_prior_sigma_median_m\":" << JSONNumber(sigma_h)
         << ",";
    json << "\"horizontal_baseline_to_sigma_ratio\":"
         << JSONNumber(baseline_to_sigma) << ",";
    json << "\"horizontal_centered_rms_singular_values_m\":["
         << JSONNumber(horizontal_rms_singular_values.x()) << ","
         << JSONNumber(horizontal_rms_singular_values.y()) << "],";
    json << "\"horizontal_condition_ratio\":"
         << JSONNumber(horizontal_condition_ratio) << ",";
    json << "\"centered_rms_singular_values_3d_m\":["
         << JSONNumber(full_rms_singular_values.x()) << ","
         << JSONNumber(full_rms_singular_values.y()) << ","
         << JSONNumber(full_rms_singular_values.z()) << "],";
    json << "\"condition_ratio_3d\":" << JSONNumber(full_condition_ratio)
         << ",";
    json << "\"max_horizontal_radius_m\":" << JSONNumber(max_horizontal_radius)
         << ",";
    json << "\"max_3d_radius_m\":" << JSONNumber(max_3d_radius) << ",";
    json << "\"max_ellipsoid_tangent_departure_m\":"
         << JSONNumber(max_ellipsoid_tangent_departure_m) << ",";
    json << "\"gravity_residual_deg\":{"
         << "\"mean\":" << JSONNumber(gravity_stats.mean)
         << ",\"median\":" << JSONNumber(gravity_stats.median)
         << ",\"p90\":" << JSONNumber(gravity_stats.p90)
         << ",\"max\":" << JSONNumber(gravity_stats.max)
         << ",\"num_support\":" << num_gravity_priors << "}";
  }
  json << "},";
  json << "\"warnings\":{";
  json << "\"collinearity\":{\"value\":"
       << JSONNumber(horizontal_condition_ratio)
       << ",\"threshold\":" << JSONNumber(kCollinearityRatioThreshold)
       << ",\"fired\":" << (collinearity_warning_fired ? "true" : "false")
       << "},";
  json << "\"gravity_disagreement\":{\"value\":"
       << JSONNumber(gravity_consistency_angle_deg)
       << ",\"threshold\":" << JSONNumber(kGravityAngleThresholdDeg)
       << ",\"fired\":" << (gravity_warning_fired ? "true" : "false") << "},";
  json << "\"position_inlier_ratio\":{\"value\":"
       << JSONNumber(position_inlier_ratio)
       << ",\"threshold\":" << JSONNumber(kPositionInlierRatioThreshold)
       << ",\"fired\":"
       << (position_inlier_ratio_warning_fired ? "true" : "false") << "}";
  json << "},";
  json << "\"position_ransac_threshold_m\":"
       << JSONNumber(position_ransac_threshold) << ",";
  json << "\"alignment_random_seed\":" << alignment_random_seed << ",";
  // How large a correction this report's own equal-weight robust Sim3 fit
  // (enu_from_sfm above) applied on top of whatever frame the input
  // reconstruction was already in. Always reported; only enforced as a hard
  // gate (RunModelAlignerReport returning EXIT_FAILURE before this file is
  // even written) when --reject_material_realignment is set -- see that
  // flag's registration for the threshold rationale.
  {
    const double kMaterialRealignmentRotationDegThreshold =
        material_realignment_thresholds.max_rotation_deg;
    const double kMaterialRealignmentTranslationMThreshold =
        material_realignment_thresholds.max_translation_m;
    const double kMaterialRealignmentScaleRatioThreshold =
        material_realignment_thresholds.max_scale_ratio;
    const double rotation_deg =
        RadToDeg(Eigen::AngleAxisd(enu_from_sfm.rotation()).angle());
    const double translation_m = enu_from_sfm.translation().norm();
    const double scale_ratio = std::abs(enu_from_sfm.scale() - 1.0);
    const bool is_material =
        rotation_deg > kMaterialRealignmentRotationDegThreshold ||
        translation_m > kMaterialRealignmentTranslationMThreshold ||
        scale_ratio > kMaterialRealignmentScaleRatioThreshold;
    json << "\"final_realignment_check\":{";
    json << "\"rotation_deg\":" << JSONNumber(rotation_deg) << ",";
    json << "\"translation_m\":" << JSONNumber(translation_m) << ",";
    json << "\"scale_ratio\":" << JSONNumber(scale_ratio) << ",";
    json << "\"rotation_deg_threshold\":"
         << JSONNumber(kMaterialRealignmentRotationDegThreshold) << ",";
    json << "\"translation_m_threshold\":"
         << JSONNumber(kMaterialRealignmentTranslationMThreshold) << ",";
    json << "\"scale_ratio_threshold\":"
         << JSONNumber(kMaterialRealignmentScaleRatioThreshold) << ",";
    json << "\"is_material\":" << (is_material ? "true" : "false") << ",";
    json << "\"enforced_as_hard_gate\":"
         << (reject_material_realignment_requested ? "true" : "false");
    json << "}";
  }
  json << "}";

  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  file.imbue(std::locale::classic());
  file << json.str();
}

std::string CSVField(const std::string& value) {
  bool needs_quoting = false;
  for (const char c : value) {
    if (c == ',' || c == '"' || c == '\n' || c == '\r') {
      needs_quoting = true;
      break;
    }
  }
  if (!needs_quoting) {
    return value;
  }
  std::string escaped = "\"";
  for (const char c : value) {
    if (c == '"') {
      escaped += "\"\"";
    } else {
      escaped += c;
    }
  }
  escaped += "\"";
  return escaped;
}

std::string CSVNumber(double value) {
  if (!std::isfinite(value)) {
    return "";
  }
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream.precision(17);
  stream << value;
  return stream.str();
}

void WriteCameraResidualsCSV(
    const std::filesystem::path& path,
    const std::vector<CameraPosePriorResidual>& residuals) {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  file.imbue(std::locale::classic());
  file.precision(17);

  std::vector<CameraPosePriorResidual> sorted_residuals = residuals;
  std::sort(
      sorted_residuals.begin(),
      sorted_residuals.end(),
      [](const CameraPosePriorResidual& a, const CameraPosePriorResidual& b) {
        return a.image_name < b.image_name;
      });

  file << "image_name,registered,has_position_prior,position_fit_inlier,"
          "prior_e,prior_n,prior_u,solved_e,solved_n,solved_u,"
          "residual_e,residual_n,residual_u,residual_horizontal,"
          "residual_vertical,residual_3d,"
          "has_gravity_prior,gravity_measured_x,gravity_measured_y,"
          "gravity_measured_z,gravity_predicted_x,gravity_predicted_y,"
          "gravity_predicted_z,gravity_residual_deg\n";
  for (const CameraPosePriorResidual& r : sorted_residuals) {
    Eigen::Vector3d residual =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    double residual_horizontal = std::numeric_limits<double>::quiet_NaN();
    double residual_3d = std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(r.prior_enu.x()) && std::isfinite(r.prior_enu.y()) &&
        std::isfinite(r.solved_enu.x()) && std::isfinite(r.solved_enu.y())) {
      residual.x() = r.solved_enu.x() - r.prior_enu.x();
      residual.y() = r.solved_enu.y() - r.prior_enu.y();
      residual_horizontal = std::hypot(residual.x(), residual.y());
    }
    if (std::isfinite(r.prior_enu.z()) && std::isfinite(r.solved_enu.z())) {
      residual.z() = r.solved_enu.z() - r.prior_enu.z();
    }
    if (residual.allFinite()) {
      residual_3d = residual.norm();
    }
    file << CSVField(r.image_name) << ',' << (r.registered ? 1 : 0) << ','
         << (r.has_position_prior ? 1 : 0) << ','
         << (r.position_fit_inlier ? 1 : 0) << ',' << CSVNumber(r.prior_enu.x())
         << ',' << CSVNumber(r.prior_enu.y()) << ','
         << CSVNumber(r.prior_enu.z()) << ',' << CSVNumber(r.solved_enu.x())
         << ',' << CSVNumber(r.solved_enu.y()) << ','
         << CSVNumber(r.solved_enu.z()) << ',' << CSVNumber(residual.x()) << ','
         << CSVNumber(residual.y()) << ',' << CSVNumber(residual.z()) << ','
         << CSVNumber(residual_horizontal) << ','
         << CSVNumber(std::abs(residual.z())) << ',' << CSVNumber(residual_3d)
         << ','
         << (r.has_gravity_prior ? 1 : 0) << ','
         << CSVNumber(r.gravity_sensor.x()) << ','
         << CSVNumber(r.gravity_sensor.y()) << ','
         << CSVNumber(r.gravity_sensor.z()) << ','
         << CSVNumber(r.predicted_gravity_sensor.x()) << ','
         << CSVNumber(r.predicted_gravity_sensor.y()) << ','
         << CSVNumber(r.predicted_gravity_sensor.z()) << ','
         << CSVNumber(r.gravity_residual_deg) << '\n';
  }
}

// Converts every camera-type pose prior's position to the shared ENU frame
// defined by (origin_lat, origin_lon, origin_alt), preserving all other
// prior fields. A horizontal-only prior (finite lat/lon, absent altitude)
// uses the origin altitude only to compute East/North, then resets Up back
// to NaN so no altitude is fabricated (mirrors DatabaseCache's WGS84
// conversion).
std::vector<PosePrior> ConvertPosePriorsToReportENU(
    const std::vector<PosePrior>& pose_priors,
    double origin_lat,
    double origin_lon,
    double origin_alt) {
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);
  std::vector<PosePrior> converted;
  converted.reserve(pose_priors.size());
  for (const PosePrior& prior : pose_priors) {
    PosePrior out = prior;
    if (prior.corr_data_id.sensor_id.type == SensorType::CAMERA &&
        std::isfinite(prior.position.x()) &&
        std::isfinite(prior.position.y())) {
      const bool has_full_altitude = std::isfinite(prior.position.z());
      const double alt = has_full_altitude ? prior.position.z() : origin_alt;
      Eigen::Vector3d enu = gps_transform.EllipsoidToENU(
          {Eigen::Vector3d(prior.position.x(), prior.position.y(), alt)},
          origin_lat,
          origin_lon,
          origin_alt)[0];
      if (!has_full_altitude) {
        enu.z() = std::numeric_limits<double>::quiet_NaN();
      }
      out.position = enu;

      const Eigen::Matrix3d shared_from_local =
          GPSTransform::ENUFromECEF(origin_lat, origin_lon) *
          GPSTransform::ECEFFromENU(prior.position.x(), prior.position.y());
      if (prior.HasPositionCov()) {
        out.position_covariance = shared_from_local *
                                  prior.position_covariance *
                                  shared_from_local.transpose();
      }
      // Heading is an azimuth from true north and needs no basis change when
      // the ENU origin moves: translating the tangent frame does not rotate
      // its north axis.
    }
    converted.push_back(out);
  }
  return converted;
}

}  // namespace

int RunModelAlignerReport(const ModelGeoreferenceOptions& o) {
  auto database = Database::Open(o.database_path);
  const std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  bool any_wgs84 = false;
  bool any_cartesian = false;
  for (const PosePrior& prior : pose_priors) {
    if (prior.coordinate_system == PosePrior::CoordinateSystem::WGS84) {
      any_wgs84 = true;
    } else if (prior.coordinate_system ==
               PosePrior::CoordinateSystem::CARTESIAN) {
      any_cartesian = true;
    }
  }
  if (any_wgs84 == any_cartesian) {
    LOG(ERROR) << "A report run requires the database's position priors to "
                  "be entirely WGS84 or entirely Cartesian ENU, not a mix "
                  "or neither";
    return EXIT_FAILURE;
  }
  if (any_cartesian && !o.has_explicit_origin) {
    LOG(ERROR) << "Cartesian ENU archives require an explicit "
                  "--enu_origin_lat/--enu_origin_lon/--enu_origin_alt for a "
                  "report run";
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(o.input_path);

  NodeHashMap<image_t, PosePrior> priors_by_image;
  for (const PosePrior& prior : pose_priors) {
    if (prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
      priors_by_image.emplace(static_cast<image_t>(prior.corr_data_id.id),
                              prior);
    }
  }

  double origin_lat = o.enu_origin_lat;
  double origin_lon = o.enu_origin_lon;
  double origin_alt = o.enu_origin_alt;
  const bool origin_is_explicit = o.has_explicit_origin;

  if (any_wgs84 && !o.has_explicit_origin) {
    std::vector<Eigen::Vector3d> full_position_lla;
    for (const PosePrior& prior : pose_priors) {
      if (prior.corr_data_id.sensor_id.type == SensorType::CAMERA &&
          prior.HasPosition()) {
        full_position_lla.push_back(prior.position);
      }
    }
    if (!DeriveWGS84Origin(
            full_position_lla, &origin_lat, &origin_lon, &origin_alt)) {
      LOG(ERROR) << "No full-position (lat/lon/alt) WGS84 priors to derive "
                    "an ENU origin from; supply --enu_origin_lat/"
                    "--enu_origin_lon/--enu_origin_alt";
      return EXIT_FAILURE;
    }
  }

  std::vector<PosePrior> enu_priors =
      any_wgs84 ? ConvertPosePriorsToReportENU(
                      pose_priors, origin_lat, origin_lon, origin_alt)
                : pose_priors;

  PosePriorAlignmentResult result =
      AlignReconstructionToPosePriorsRobust(
          reconstruction, enu_priors, o.ransac_options);
  if (!result.success) {
    LOG(ERROR) << "=> Alignment failed";
    return EXIT_FAILURE;
  }

  // Recompute the origin once from the position-fit inlier WGS84 points,
  // reconvert, and refit once. Never uses the first row by policy.
  if (any_wgs84 && !o.has_explicit_origin) {
    std::vector<Eigen::Vector3d> inlier_full_position_lla;
    for (size_t i = 0; i < result.correspondence_image_ids.size(); ++i) {
      if (!result.inlier_mask[i]) {
        continue;
      }
      const auto it = priors_by_image.find(result.correspondence_image_ids[i]);
      if (it != priors_by_image.end() && it->second.HasPosition()) {
        inlier_full_position_lla.push_back(it->second.position);
      }
    }
    if (!inlier_full_position_lla.empty()) {
      DeriveWGS84Origin(
          inlier_full_position_lla, &origin_lat, &origin_lon, &origin_alt);
      enu_priors = ConvertPosePriorsToReportENU(
          pose_priors, origin_lat, origin_lon, origin_alt);
      result =
          AlignReconstructionToPosePriorsRobust(
              reconstruction, enu_priors, o.ransac_options);
      if (!result.success) {
        LOG(ERROR) << "=> Alignment failed after origin refinement";
        return EXIT_FAILURE;
      }
    }
  }

  const int num_inliers = static_cast<int>(
      std::count(result.inlier_mask.begin(), result.inlier_mask.end(), 1));
  if (num_inliers < o.min_common_images) {
    LOG(ERROR) << "=> Too few position-prior inliers: " << num_inliers << " < "
               << o.min_common_images;
    return EXIT_FAILURE;
  }
  if (any_cartesian && o.pose_prior_cartesian_frame != "ENU") {
    LOG(ERROR) << "Cartesian pose priors require the explicit assertion "
                  "--pose_prior_cartesian_frame=ENU before an Earth report "
                  "can be emitted";
    return EXIT_FAILURE;
  }

  // This robust Sim3 fit is equal-weight among RANSAC inliers and ignores
  // per-row GPS covariance.
  // If the input reconstruction already came from an upstream
  // pose_prior_position_mode=optimize solve (which *did* use
  // covariance-weighted BA throughout), this correction should be small; a
  // material one means either that solve didn't hold for this input or this
  // step is silently overriding it with a less-informed fit. Thresholds are
  // a physically-motivated "this should be a no-op" bar, not tuned against
  // any specific dataset -- see MaterialRealignmentThresholds' defaults for
  // the rationale. The same struct instance is used here (enforcement) and
  // in WriteGeoreferenceReportJSON (the always-present diagnostic), so
  // evaluation and serialization cannot drift apart.
  const double kMaterialRealignmentRotationDegThreshold =
      o.material_realignment_thresholds.max_rotation_deg;
  const double kMaterialRealignmentTranslationMThreshold =
      o.material_realignment_thresholds.max_translation_m;
  const double kMaterialRealignmentScaleRatioThreshold =
      o.material_realignment_thresholds.max_scale_ratio;
  const double realignment_rotation_deg =
      RadToDeg(Eigen::AngleAxisd(result.tgt_from_src.rotation()).angle());
  const double realignment_translation_m =
      result.tgt_from_src.translation().norm();
  const double realignment_scale_ratio =
      std::abs(result.tgt_from_src.scale() - 1.0);
  const bool realignment_is_material =
      realignment_rotation_deg > kMaterialRealignmentRotationDegThreshold ||
      realignment_translation_m > kMaterialRealignmentTranslationMThreshold ||
      realignment_scale_ratio > kMaterialRealignmentScaleRatioThreshold;
  if (o.reject_material_realignment && realignment_is_material) {
    LOG(ERROR) << StringPrintf(
        "=> --reject_material_realignment: the final robust Sim3 fit found a "
        "material correction (rotation=%.4f deg > %.2f, translation=%.4f m "
        "> %.2f, scale_ratio=%.6f > %.4f) on an input declared "
        "already-metric-ENU-optimized. This equal-weight refit ignores "
        "per-row GPS covariance; a correction this large means either the "
        "upstream optimize solve did not actually hold for this input, or "
        "this step is silently overriding it with a less-informed fit. "
        "Investigate before deploying this result.",
        realignment_rotation_deg,
        kMaterialRealignmentRotationDegThreshold,
        realignment_translation_m,
        kMaterialRealignmentTranslationMThreshold,
        realignment_scale_ratio,
        kMaterialRealignmentScaleRatioThreshold);
    return EXIT_FAILURE;
  }

  reconstruction.Transform(result.tgt_from_src);

  FlatHashSet<image_t> inlier_image_ids;
  for (size_t i = 0; i < result.correspondence_image_ids.size(); ++i) {
    if (result.inlier_mask[i]) {
      inlier_image_ids.insert(result.correspondence_image_ids[i]);
    }
  }
  NodeHashMap<image_t, const PosePrior*> enu_prior_by_image;
  for (const PosePrior& prior : enu_priors) {
    if (prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
      const image_t image_id = static_cast<image_t>(prior.corr_data_id.id);
      enu_prior_by_image.emplace(image_id, &prior);
    }
  }

  std::vector<CameraPosePriorResidual> residuals;
  const std::vector<Image> database_images = database->ReadAllImages();
  residuals.reserve(database_images.size());
  for (const Image& database_image : database_images) {
    CameraPosePriorResidual r;
    r.image_id = database_image.ImageId();
    r.image_name = database_image.Name();
    r.registered = reconstruction.ExistsImage(r.image_id) &&
                   reconstruction.Image(r.image_id).HasPose();
    const auto prior_it = priors_by_image.find(r.image_id);
    if (prior_it != priors_by_image.end()) {
      r.has_position_prior = std::isfinite(prior_it->second.position.x()) &&
                             std::isfinite(prior_it->second.position.y());
      r.has_gravity_prior = prior_it->second.HasGravity();
      r.gravity_sensor = prior_it->second.gravity;
    }
    const auto enu_it = enu_prior_by_image.find(r.image_id);
    if (enu_it != enu_prior_by_image.end()) {
      r.prior_enu = enu_it->second->position;
      r.position_covariance_enu = enu_it->second->position_covariance;
    }
    r.position_fit_inlier = inlier_image_ids.count(r.image_id) > 0;
    if (r.registered) {
      const Image& image = reconstruction.Image(r.image_id);
      r.solved_enu = image.ProjectionCenter();
      if (r.has_gravity_prior) {
        // world_down_ convention: East=+X, North=+Y, Up=+Z (see
        // PosePriorBundleAdjuster::world_down_ in bundle_adjustment_ceres.cc
        // and this file's enu_down usage below) -- valid here because this
        // report only runs after the reconstruction has been aligned to ENU.
        r.predicted_gravity_sensor =
            image.CamFromWorld().rotation() * Eigen::Vector3d(0.0, 0.0, -1.0);
        const Eigen::Vector3d measured_down_sensor =
            r.gravity_sensor.normalized();
        const double cos_angle = std::clamp(
            r.predicted_gravity_sensor.normalized().dot(measured_down_sensor),
            -1.0,
            1.0);
        r.gravity_residual_deg = RadToDeg(std::acos(cos_angle));
      }
    }
    residuals.push_back(r);
  }

  double max_ellipsoid_tangent_departure_m =
      std::numeric_limits<double>::quiet_NaN();
  if (any_wgs84) {
    max_ellipsoid_tangent_departure_m = 0.0;
    for (const image_t image_id : inlier_image_ids) {
      const auto prior_it = priors_by_image.find(image_id);
      if (prior_it == priors_by_image.end() ||
          !std::isfinite(prior_it->second.position.x()) ||
          !std::isfinite(prior_it->second.position.y())) {
        continue;
      }
      const Eigen::Vector3d equal_altitude_enu =
          GPSTransform(GPSTransform::Ellipsoid::WGS84)
              .EllipsoidToENU({Eigen::Vector3d(prior_it->second.position.x(),
                                               prior_it->second.position.y(),
                                               origin_alt)},
                              origin_lat,
                              origin_lon,
                              origin_alt)[0];
      max_ellipsoid_tangent_departure_m = std::max(
          max_ellipsoid_tangent_departure_m, std::abs(equal_altitude_enu.z()));
    }
  }

  // A `scene_id` is only an identity label associating a report with a
  // scene -- it is never fabricated. If the caller supplies one it is
  // serialized verbatim (after validation); if not, the field is omitted
  // (JSON `null`) rather than filled with a random UUID, so that otherwise
  // identical reports remain deterministic.
  const std::string& scene_id = o.scene_id;
  if (!scene_id.empty() && !IsValidSceneId(scene_id)) {
    LOG(ERROR) << "=> --scene_id must be non-empty and free of control "
                  "characters";
    return EXIT_FAILURE;
  }
  if (!o.georeference_json.empty()) {
    WriteGeoreferenceReportJSON(o.georeference_json,
                                scene_id,
                                GetBuildInfo(),
                                o.input_path,
                                o.output_path,
                                reconstruction,
                                origin_lat,
                                origin_lon,
                                origin_alt,
                                origin_is_explicit,
                                result.tgt_from_src,
                                residuals,
                                static_cast<int>(pose_priors.size()),
                                max_ellipsoid_tangent_departure_m,
                                o.ransac_options.max_error,
                                o.ransac_options.random_seed,
                                o.output_coordinate_frame,
                                o.reject_material_realignment,
                                o.quality_thresholds,
                                o.material_realignment_thresholds,
                                o.report_level);
  }
  if (!o.camera_residuals_csv.empty()) {
    WriteCameraResidualsCSV(o.camera_residuals_csv, residuals);
  }

  LOG(INFO) << StringPrintf(
      "=> Alignment succeeded: %d position-prior inliers, ENU origin "
      "(lat=%.8f, lon=%.8f, alt=%.3f, explicit=%s)",
      num_inliers,
      origin_lat,
      origin_lon,
      origin_alt,
      origin_is_explicit ? "true" : "false");
  // Applied last, after every diagnostic/report computation above (which
  // reads camera rotations/positions and assumes the ENU convention -- e.g.
  // the gravity_consistency_angle_deg comparison against enu_down), so only
  // the written bytes change frame, never the report's own math.
  if (o.output_coordinate_frame != OutputCoordinateFrame::ENU_Z_UP) {
    reconstruction.Transform(GeometryFromEnu(o.output_coordinate_frame));
  }
  reconstruction.Write(o.output_path);
  return EXIT_SUCCESS;
}

}  // namespace colmap
