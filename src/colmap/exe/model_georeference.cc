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
#include "colmap/geometry/pose_prior_transform.h"
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
#include <ctime>
#include <fstream>
#include <iomanip>
#include <locale>
#include <optional>
#include <sstream>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <boost/property_tree/json_parser.hpp>

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

std::string CurrentUtcTimestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
#ifdef _WIN32
  gmtime_s(&utc, &now);
#else
  gmtime_r(&now, &utc);
#endif
  std::ostringstream stream;
  stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
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
  bool has_heading_prior = false;
  double heading_stddev_deg = std::numeric_limits<double>::quiet_NaN();
  double heading_residual_deg = std::numeric_limits<double>::quiet_NaN();
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
// LichtFeld Studio import contract; changes require an upright display
// verification against the referenced consumer boundary transform.
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

namespace {

////////////////////////////////////////////////////////////////////////////
// Scene georeference report (model_aligner --georeference_json/
// --camera_residuals_csv).
////////////////////////////////////////////////////////////////////////////

std::string SerializeGeoreferenceReportJSON(
    const std::string& scene_id,
    const std::string& source_commit,
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    const Reconstruction& reconstruction,
    double origin_lat,
    double origin_lon,
    double origin_alt,
    const Sim3d& enu_from_sfm,
    const std::vector<CameraPosePriorResidual>& residuals,
    int num_database_pose_priors,
    double max_ellipsoid_tangent_departure_m,
    double position_ransac_threshold,
    int alignment_random_seed,
    OutputCoordinateFrame output_coordinate_frame,
    const GeoreferenceQualityThresholds& quality_thresholds,
    const MaterialRealignmentThresholds& material_realignment_thresholds) {
  const Sim3d sfm_from_enu = Inverse(enu_from_sfm);
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);
  const Eigen::Matrix3d ecef_from_enu_rotation =
      GPSTransform::ECEFFromENU(origin_lat, origin_lon);
  const Eigen::Vector3d origin_ecef = gps_transform.EllipsoidToECEF(
      {Eigen::Vector3d(origin_lat, origin_lon, origin_alt)})[0];
  const Sim3d ecef_from_enu(
      1.0, Eigen::Quaterniond(ecef_from_enu_rotation), origin_ecef);
  const Sim3d enu_from_ecef = Inverse(ecef_from_enu);

  // Verify the numerical inverse of every transform pair before publication.
  const auto verify_inverse = [](const Sim3d& a, const Sim3d& b) {
    const Eigen::Vector3d probe(1.0, 2.0, 3.0);
    const Eigen::Vector3d round_trip = b * (a * probe);
    THROW_CHECK_LT((round_trip - probe).norm(), 1e-6);
  };
  verify_inverse(enu_from_sfm, sfm_from_enu);
  verify_inverse(ecef_from_enu, enu_from_ecef);

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
    if (r.registered && r.prior_enu.allFinite() && r.solved_enu.allFinite()) {
      ++num_registered_correspondences;
      const Eigen::Vector3d diff = r.solved_enu - r.prior_enu;
      horizontal_residuals.push_back(std::hypot(diff.x(), diff.y()));
      vertical_residuals.push_back(std::abs(diff.z()));
      full_residuals.push_back(diff.norm());
    }
    if (r.position_fit_inlier && r.prior_enu.allFinite() &&
        r.solved_enu.allFinite()) {
      ++num_position_inliers;
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

  std::vector<double> gravity_residuals_deg;
  std::vector<double> heading_residuals_deg;
  for (const CameraPosePriorResidual& r : residuals) {
    if (!r.registered || !r.has_gravity_prior ||
        !std::isfinite(r.gravity_residual_deg)) {
      continue;
    }
    gravity_residuals_deg.push_back(r.gravity_residual_deg);
    if (r.has_heading_prior && std::isfinite(r.heading_residual_deg)) {
      heading_residuals_deg.push_back(std::abs(r.heading_residual_deg));
    }
  }
  const int num_gravity_priors = static_cast<int>(gravity_residuals_deg.size());
  const ScalarStatistics gravity_stats =
      ComputeStatistics(gravity_residuals_deg);
  const double gravity_consistency_angle_deg = gravity_stats.median;
  const int num_heading_priors = static_cast<int>(heading_residuals_deg.size());
  const ScalarStatistics heading_stats =
      ComputeStatistics(heading_residuals_deg);

  const double kCollinearityRatioThreshold =
      quality_thresholds.collinearity_ratio_threshold;
  const double kGravityAngleThresholdDeg =
      quality_thresholds.gravity_median_threshold_deg;
  const double kPositionInlierRatioThreshold =
      quality_thresholds.min_position_inlier_ratio;
  const bool collinearity_warning_fired =
      std::isfinite(horizontal_condition_ratio) &&
      horizontal_condition_ratio < kCollinearityRatioThreshold;
  const bool gravity_failure_fired =
      std::isfinite(gravity_consistency_angle_deg) &&
      gravity_consistency_angle_deg > kGravityAngleThresholdDeg;
  const double position_inlier_ratio =
      num_registered_correspondences > 0
          ? static_cast<double>(num_position_inliers) /
                num_registered_correspondences
          : std::numeric_limits<double>::quiet_NaN();
  const bool position_inlier_ratio_failure_fired =
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
  if (gravity_failure_fired) {
    LOG(ERROR) << StringPrintf(
        "=> Aligned up-axis disagrees with gravity priors (%.3f deg > %.2f "
        "deg)",
        gravity_consistency_angle_deg,
        kGravityAngleThresholdDeg);
  }
  if (position_inlier_ratio_failure_fired) {
    LOG(ERROR) << StringPrintf(
        "=> Large fraction of registered images disagree with the alignment "
        "(position inlier ratio %.6f < %.2f) — possible internal "
        "misregistration (repeated structure / false loop closures)",
        position_inlier_ratio,
        kPositionInlierRatioThreshold);
  }
  THROW_CHECK(!gravity_failure_fired)
      << "Gravity quality gate rejected the georeference report";
  THROW_CHECK(!position_inlier_ratio_failure_fired)
      << "Position-support gate rejected the georeference report";

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
  json << "\"schema_version\":1,";
  json << "\"provenance\":{";
  json << "\"scene_id\":\"" << JSONEscape(scene_id) << "\",";
  json << "\"colmap_version\":\"" << JSONEscape(GetVersionInfo()) << "\",";
  json << "\"colmap_build\":\"" << JSONEscape(source_commit) << "\",";
  json << "\"creation_utc\":\"" << CurrentUtcTimestamp() << "\",";
  json << "\"input_path\":\"" << JSONEscape(input_path.string()) << "\",";
  json << "\"output_path\":\"" << JSONEscape(output_path.string()) << "\"},";
  json << "\"crs\":{\"ellipsoid\":\"WGS84\",";
  json << "\"height_datum\":\"ELLIPSOIDAL\",";
  json << "\"origin\":{\"lat_deg\":" << JSONNumber(origin_lat)
       << ",\"lon_deg\":" << JSONNumber(origin_lon)
       << ",\"ellipsoidal_alt_m\":" << JSONNumber(origin_alt) << "}},";
  json << "\"alignment\":{";
  json << "\"enu_from_input_sfm\":" << JSONSim3(enu_from_sfm) << ",";
  json << "\"input_sfm_from_enu\":" << JSONSim3(sfm_from_enu) << ",";
  json << "\"metres_per_input_unit\":" << JSONNumber(enu_from_sfm.scale())
       << ",";
  json << "\"standardized_ransac_radius\":"
       << JSONNumber(position_ransac_threshold) << ",";
  json << "\"random_seed\":" << alignment_random_seed << "},";

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
  json << "\"num_position_inliers\":" << num_position_inliers << ",";
  json << "\"num_position_outliers\":"
       << num_registered_correspondences - num_position_inliers << ",";
  json << "\"num_gravity_observations\":" << num_gravity_priors << ",";
  json << "\"num_heading_observations\":" << num_heading_priors;
  json << "},";
  json << "\"diagnostics\":{";
  json << "\"position_3d_residual_m\":{"
       << "\"mean\":" << JSONNumber(full_stats.mean)
       << ",\"median\":" << JSONNumber(full_stats.median)
       << ",\"p90\":" << JSONNumber(full_stats.p90)
       << ",\"max\":" << JSONNumber(full_stats.max)
       << ",\"num_support\":" << full_residuals.size() << "},";
  json << "\"position_horizontal_residual_m\":{"
       << "\"mean\":" << JSONNumber(horizontal_stats.mean)
       << ",\"median\":" << JSONNumber(horizontal_stats.median)
       << ",\"p90\":" << JSONNumber(horizontal_stats.p90)
       << ",\"max\":" << JSONNumber(horizontal_stats.max)
       << ",\"num_support\":" << horizontal_residuals.size() << "},";
  json << "\"position_vertical_residual_m\":{"
       << "\"mean\":" << JSONNumber(vertical_stats.mean)
       << ",\"median\":" << JSONNumber(vertical_stats.median)
       << ",\"p90\":" << JSONNumber(vertical_stats.p90)
       << ",\"max\":" << JSONNumber(vertical_stats.max)
       << ",\"num_support\":" << vertical_residuals.size() << "},";
  json << "\"max_horizontal_baseline_m\":"
       << JSONNumber(max_horizontal_baseline) << ",";
  json << "\"horizontal_prior_sigma_median_m\":" << JSONNumber(sigma_h) << ",";
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
  json << "\"condition_ratio_3d\":" << JSONNumber(full_condition_ratio) << ",";
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
       << ",\"num_support\":" << num_gravity_priors << "},";
  json << "\"heading_residual_deg\":{"
       << "\"mean\":" << JSONNumber(heading_stats.mean)
       << ",\"median\":" << JSONNumber(heading_stats.median)
       << ",\"p90\":" << JSONNumber(heading_stats.p90)
       << ",\"max\":" << JSONNumber(heading_stats.max)
       << ",\"num_support\":" << num_heading_priors << "}";
  json << "},";
  json << "\"quality\":{";
  json << "\"collinearity\":{\"value\":"
       << JSONNumber(horizontal_condition_ratio)
       << ",\"threshold\":" << JSONNumber(kCollinearityRatioThreshold)
       << ",\"severity\":\"WARNING\""
       << ",\"fired\":" << (collinearity_warning_fired ? "true" : "false")
       << "},";
  json << "\"gravity_disagreement\":{\"value\":"
       << JSONNumber(gravity_consistency_angle_deg)
       << ",\"threshold\":" << JSONNumber(kGravityAngleThresholdDeg)
       << ",\"severity\":\"FAILURE\""
       << ",\"fired\":" << (gravity_failure_fired ? "true" : "false") << "},";
  json << "\"position_inlier_ratio\":{\"value\":"
       << JSONNumber(position_inlier_ratio)
       << ",\"threshold\":" << JSONNumber(kPositionInlierRatioThreshold)
       << ",\"severity\":\"FAILURE\""
       << ",\"fired\":"
       << (position_inlier_ratio_failure_fired ? "true" : "false") << "}";
  json << "},";
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
    json << "\"is_material\":" << (is_material ? "true" : "false");
    json << "}";
  }
  json << "}";

  std::istringstream input(json.str());
  boost::property_tree::ptree parsed;
  boost::property_tree::read_json(input, parsed);
  THROW_CHECK_EQ(parsed.get<int>("schema_version"), 1);
  return json.str();
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

std::string SerializeCameraResidualsCSV(
    const std::vector<CameraPosePriorResidual>& residuals) {
  std::vector<CameraPosePriorResidual> sorted_residuals = residuals;
  std::sort(
      sorted_residuals.begin(),
      sorted_residuals.end(),
      [](const CameraPosePriorResidual& a, const CameraPosePriorResidual& b) {
        return a.image_name < b.image_name;
      });

  std::ostringstream csv;
  csv.imbue(std::locale::classic());
  csv.precision(17);
  csv << "image_name,image_id,has_position_prior,position_fit_inlier,"
         "residual_east_m,residual_north_m,residual_up_m,"
         "residual_horizontal_m,residual_3d_m,"
         "has_gravity_prior,gravity_residual_deg,"
         "has_heading_prior,heading_stddev_deg,heading_residual_deg\n";
  for (const CameraPosePriorResidual& r : sorted_residuals) {
    if (!r.registered) {
      continue;
    }
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
    csv << CSVField(r.image_name) << ',' << r.image_id << ','
        << (r.has_position_prior ? 1 : 0) << ','
        << (r.position_fit_inlier ? 1 : 0) << ',' << CSVNumber(residual.x())
        << ',' << CSVNumber(residual.y()) << ',' << CSVNumber(residual.z())
        << ',' << CSVNumber(residual_horizontal) << ','
        << CSVNumber(residual_3d) << ',' << (r.has_gravity_prior ? 1 : 0) << ','
        << CSVNumber(r.gravity_residual_deg) << ','
        << (r.has_heading_prior ? 1 : 0) << ','
        << CSVNumber(r.heading_stddev_deg) << ','
        << CSVNumber(r.heading_residual_deg) << '\n';
  }
  return csv.str();
}

void ValidateCSV(const std::string& csv,
                 const size_t expected_columns,
                 const size_t expected_rows) {
  size_t columns = 1;
  size_t rows = 0;
  bool quoted = false;
  for (size_t i = 0; i < csv.size(); ++i) {
    const char c = csv[i];
    if (c == '"') {
      if (quoted && i + 1 < csv.size() && csv[i + 1] == '"') {
        ++i;
      } else {
        quoted = !quoted;
      }
    } else if (!quoted && c == ',') {
      ++columns;
    } else if (!quoted && c == '\n') {
      THROW_CHECK_EQ(columns, expected_columns);
      columns = 1;
      ++rows;
    }
  }
  THROW_CHECK(!quoted);
  THROW_CHECK_EQ(rows, expected_rows);
}

void PublishFileAtomically(const std::filesystem::path& path,
                           const std::string& contents) {
  THROW_CHECK(!ExistsFile(path) && !ExistsDir(path))
      << "Refusing to overwrite " << path;
  const std::filesystem::path temporary = path.string() + ".tmp";
  THROW_CHECK(!ExistsFile(temporary) && !ExistsDir(temporary))
      << "Stale temporary publication target: " << temporary;
  {
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    THROW_CHECK_FILE_OPEN(file, temporary);
    file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    file.flush();
    THROW_CHECK(file.good()) << "Failed writing " << temporary;
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  THROW_CHECK(!error) << "Failed publishing " << path << ": "
                      << error.message();
}

// Converts every camera-type pose prior into `enu_frame`, preserving all other
// prior fields. Rows the frame cannot place (no usable WGS84 position) pass
// through untouched, so the caller's per-image bookkeeping still sees them.
std::vector<PosePrior> ConvertPosePriorsToReportENU(
    const std::vector<PosePrior>& pose_priors,
    const PosePriorEnuFrame& enu_frame) {
  std::vector<PosePrior> converted;
  converted.reserve(pose_priors.size());
  for (const PosePrior& prior : pose_priors) {
    PosePrior out = prior;
    if (prior.corr_data_id.sensor_id.type == SensorType::CAMERA &&
        PosePriorEnuFrame::IsUsable(prior)) {
      // Covariance first: it is read from the prior's own latitude/longitude,
      // which the position assignment overwrites.
      if (prior.HasPositionCov()) {
        out.position_covariance = enu_frame.CovarianceInEnu(prior);
      }
      out.position = enu_frame.PositionInEnu(prior);
    }
    converted.push_back(out);
  }
  return converted;
}

}  // namespace

int RunModelAlignerReport(const ModelGeoreferenceOptions& o) {
  auto database = Database::Open(o.database_path);
  const std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  // A report states where the scene is on the Earth, so every position prior
  // must carry an Earth position. A Cartesian prior is already in somebody
  // else's local frame with no recorded datum; there is nothing to resolve it
  // against, and guessing one would publish a confident wrong answer.
  bool any_cartesian = false;
  for (const PosePrior& prior : pose_priors) {
    if (prior.coordinate_system == PosePrior::CoordinateSystem::CARTESIAN) {
      any_cartesian = true;
    }
  }
  if (any_cartesian) {
    LOG(ERROR) << "A report run requires WGS84 position priors; this database "
                  "contains Cartesian priors, which carry no datum to "
                  "georeference against";
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

  // The same frame the mapper used, derived by the same rule from the same
  // rows. This report's transforms are published against it, so if it were
  // derived differently -- from the inliers of this fit, say, or from an
  // operator-supplied origin -- the geometry the mapper solved and the
  // geometry this file describes would be offset with nothing to say why.
  // GeometricMedian is robust, so the origin does not need an outlier pass to
  // defend it.
  const std::optional<PosePriorEnuFrame> enu_frame =
      PosePriorEnuFrame::Derive(pose_priors);
  if (!enu_frame.has_value()) {
    LOG(ERROR) << "No WGS84 position priors to derive an ENU origin from";
    return EXIT_FAILURE;
  }
  if (!enu_frame->HasRealAltitude()) {
    LOG(ERROR) << "No position prior carries an altitude; a report states an "
                  "ellipsoidal height and cannot substitute a placeholder";
    return EXIT_FAILURE;
  }
  const double origin_lat = enu_frame->OriginWgs84().x();
  const double origin_lon = enu_frame->OriginWgs84().y();
  const double origin_alt = enu_frame->OriginWgs84().z();

  const std::vector<PosePrior> enu_priors =
      ConvertPosePriorsToReportENU(pose_priors, *enu_frame);

  PosePriorAlignmentResult result = AlignReconstructionToPosePriorsWeighted(
      reconstruction, enu_priors, o.ransac_options.random_seed);
  if (!result.success) {
    LOG(ERROR) << "=> Alignment failed";
    return EXIT_FAILURE;
  }

  const int num_inliers = static_cast<int>(
      std::count(result.inlier_mask.begin(), result.inlier_mask.end(), 1));
  if (num_inliers < o.min_common_images) {
    LOG(ERROR) << "=> Too few position-prior inliers: " << num_inliers << " < "
               << o.min_common_images;
    return EXIT_FAILURE;
  }
  // The report path publishes a delivery whose input was already solved in the
  // metric ENU gauge, so this covariance-weighted fit must be a no-op.
  //
  // This is also what makes the gate a check on the *input*: a reconstruction
  // that was never aligned produces a large tgt_from_src and is rejected here,
  // because a first alignment is not a delivery. Upstream `model_aligner`
  // without --georeference_json/--camera_residuals_csv still performs first
  // alignments and does not reach this path.
  //
  // The same struct instance is used here (enforcement) and in
  // WriteGeoreferenceReportJSON (the recorded diagnostic), so evaluation and
  // serialization cannot drift apart.
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
  if (realignment_is_material) {
    LOG(ERROR) << StringPrintf(
        "=> The final robust Sim3 fit found a material correction "
        "(rotation=%.4f deg > %.2f, translation=%.4f m > %.2f, "
        "scale_ratio=%.6f > %.4f). A report run publishes a delivery, whose "
        "input must already be solved in the metric ENU gauge, so this fit "
        "must be a no-op. Either the mapper's optimize solve did not hold for "
        "this input, or the final weighted fit disagrees with it. Investigate "
        "before deploying this result. To align an unaligned "
        "reconstruction, run model_aligner without --georeference_json/"
        "--camera_residuals_csv.",
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
      r.has_heading_prior = prior_it->second.HasHeading();
      if (r.has_heading_prior) {
        r.heading_stddev_deg = RadToDeg(prior_it->second.heading_stddev_rad);
      }
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
        const Eigen::Matrix3d shared_from_local =
            enu_frame->SharedFromLocalEnu(prior_it->second.position);
        const Eigen::Vector3d down_world =
            shared_from_local * Eigen::Vector3d(0.0, 0.0, -1.0);
        r.predicted_gravity_sensor =
            image.CamFromWorld().rotation() * down_world;
        const Eigen::Vector3d measured_down_sensor =
            r.gravity_sensor.normalized();
        const double cos_angle = std::clamp(
            r.predicted_gravity_sensor.normalized().dot(measured_down_sensor),
            -1.0,
            1.0);
        r.gravity_residual_deg = RadToDeg(std::acos(cos_angle));

        if (r.has_heading_prior) {
          const Eigen::Vector3d forward(0.0, 0.0, 1.0);
          const Eigen::Vector3d forward_horizontal =
              forward -
              measured_down_sensor * measured_down_sensor.dot(forward);
          if (forward_horizontal.norm() >= 1e-3) {
            const Eigen::Vector3d forward_unit =
                forward_horizontal.normalized();
            const Eigen::Vector3d right_unit =
                measured_down_sensor.cross(forward_unit).normalized();
            const Eigen::Vector3d measured_north_sensor =
                std::cos(prior_it->second.heading_rad) * forward_unit -
                std::sin(prior_it->second.heading_rad) * right_unit;
            const Eigen::Vector3d north_world =
                shared_from_local * Eigen::Vector3d(0.0, 1.0, 0.0);
            Eigen::Vector3d predicted_north_sensor =
                image.CamFromWorld().rotation() * north_world;
            predicted_north_sensor -=
                measured_down_sensor *
                measured_down_sensor.dot(predicted_north_sensor);
            if (predicted_north_sensor.norm() >= 1e-12) {
              predicted_north_sensor.normalize();
              r.heading_residual_deg = RadToDeg(std::atan2(
                  measured_down_sensor.dot(
                      measured_north_sensor.cross(predicted_north_sensor)),
                  measured_north_sensor.dot(predicted_north_sensor)));
            }
          }
        }
      }
    }
    residuals.push_back(r);
  }

  // Every prior here is WGS84; a Cartesian archive was rejected above.
  double max_ellipsoid_tangent_departure_m =
      std::numeric_limits<double>::quiet_NaN();
  {
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

  const std::string& scene_id = o.scene_id;
  if (!IsValidSceneId(scene_id)) {
    LOG(ERROR) << "=> --scene_id must be non-empty and free of control "
                  "characters";
    return EXIT_FAILURE;
  }
  const std::string report =
      SerializeGeoreferenceReportJSON(scene_id,
                                      GetBuildInfo(),
                                      o.input_path,
                                      o.output_path,
                                      reconstruction,
                                      origin_lat,
                                      origin_lon,
                                      origin_alt,
                                      result.tgt_from_src,
                                      residuals,
                                      static_cast<int>(pose_priors.size()),
                                      max_ellipsoid_tangent_departure_m,
                                      kPosePriorPositionRobustRadius,
                                      o.ransac_options.random_seed,
                                      o.output_coordinate_frame,
                                      o.quality_thresholds,
                                      o.material_realignment_thresholds);
  const std::string csv = SerializeCameraResidualsCSV(residuals);
  ValidateCSV(csv, 14, reconstruction.NumRegImages() + 1);

  if (ExistsDir(o.output_path) || ExistsFile(o.output_path)) {
    LOG(ERROR) << "=> Refusing to overwrite output path " << o.output_path;
    return EXIT_FAILURE;
  }
  for (const std::filesystem::path& sidecar :
       {o.camera_residuals_csv, o.georeference_json}) {
    const std::filesystem::path temporary = sidecar.string() + ".tmp";
    if (ExistsFile(sidecar) || ExistsDir(sidecar) || ExistsFile(temporary) ||
        ExistsDir(temporary)) {
      LOG(ERROR) << "=> Refusing existing sidecar publication target "
                 << sidecar;
      return EXIT_FAILURE;
    }
  }

  // Applied last, after every diagnostic/report computation above (which
  // reads camera rotations/positions and assumes the ENU convention -- e.g.
  // the gravity_consistency_angle_deg comparison against enu_down), so only
  // the written bytes change frame, never the report's own math.
  if (o.output_coordinate_frame != OutputCoordinateFrame::ENU_Z_UP) {
    reconstruction.Transform(GeometryFromEnu(o.output_coordinate_frame));
  }
  CreateDirIfNotExists(o.output_path, /*recursive=*/true);
  reconstruction.Write(o.output_path);

  Reconstruction written;
  written.Read(o.output_path);
  THROW_CHECK_EQ(written.NumCameras(), reconstruction.NumCameras());
  THROW_CHECK_EQ(written.NumImages(), reconstruction.NumImages());
  THROW_CHECK_EQ(written.NumRegImages(), reconstruction.NumRegImages());
  THROW_CHECK_EQ(written.NumPoints3D(), reconstruction.NumPoints3D());

  // The JSON is the success sidecar, so publish it last.
  PublishFileAtomically(o.camera_residuals_csv, csv);
  PublishFileAtomically(o.georeference_json, report);
  LOG(INFO) << StringPrintf(
      "=> Alignment succeeded: %d position-prior inliers, ENU origin "
      "(lat=%.8f, lon=%.8f, alt=%.3f)",
      num_inliers,
      origin_lat,
      origin_lon,
      origin_alt);
  return EXIT_SUCCESS;
}

}  // namespace colmap
