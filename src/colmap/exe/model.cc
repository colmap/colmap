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

#include "colmap/exe/model.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/controllers/reconstruction_clustering.h"
#include "colmap/estimators/alignment.h"
#include "colmap/estimators/coordinate_frame.h"
#include "colmap/geometry/bbox.h"
#include "colmap/geometry/gps.h"
#include "colmap/math/geometric_median.h"
#include "colmap/math/math.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_io.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"
#include "colmap/util/version.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <locale>
#include <random>
#include <sstream>

namespace colmap {
namespace {

Eigen::Vector3d TransformLatLonAltToModelCoords(const Sim3d& tform,
                                                const double lat,
                                                const double lon,
                                                const double alt) {
  // Since this is intended for use in ENU aligned models we want to define the
  // altitude along the ENU frame z axis and not the Earth's radius. Thus, we
  // set the altitude to 0 when converting from LLA to ECEF and then we use the
  // altitude at the end, after scaling, to set it as the z coordinate in the
  // ENU frame.
  Eigen::Vector3d xyz =
      tform * GPSTransform(GPSTransform::Ellipsoid::WGS84)
                  .EllipsoidToECEF({Eigen::Vector3d(lat, lon, 0.0)})[0];
  xyz(2) = tform.scale() * alt;
  return xyz;
}

void WriteBoundingBox(const std::filesystem::path& reconstruction_path,
                      const Eigen::AlignedBox3d& bbox,
                      const std::string& suffix = "") {
  const Eigen::Vector3d extent = bbox.diagonal();
  // write axis-aligned bounding box
  {
    const auto path = reconstruction_path / ("bbox_aligned" + suffix + ".txt");
    std::ofstream file(path, std::ios::trunc);
    THROW_CHECK_FILE_OPEN(file, path);

    // Ensure that we don't lose any precision by storing in text.
    file.imbue(std::locale::classic());
    file.precision(17);
    file << bbox.min().transpose() << '\n';
    file << bbox.max().transpose() << '\n';
  }
  // write oriented bounding box
  {
    const auto path = reconstruction_path / ("bbox_oriented" + suffix + ".txt");
    std::ofstream file(path, std::ios::trunc);
    THROW_CHECK_FILE_OPEN(file, path);

    // Ensure that we don't lose any precision by storing in text.
    file.imbue(std::locale::classic());
    file.precision(17);
    const Eigen::Vector3d center = (bbox.min() + bbox.max()) * 0.5;
    file << center.transpose() << "\n\n";
    file << "1 0 0\n0 1 0\n0 0 1\n\n";
    file << extent.transpose() << '\n';
  }
}

std::vector<Eigen::Vector3d> ConvertCameraLocations(
    const bool ref_is_gps,
    const std::string& alignment_type,
    const std::vector<Eigen::Vector3d>& ref_locations) {
  if (ref_is_gps) {
    const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);
    if (alignment_type != "enu") {
      LOG(INFO) << "Converting Alignment Coordinates from GPS (lat/lon/alt) "
                   "to ECEF.";
      return gps_transform.EllipsoidToECEF(ref_locations);
    } else {
      THROW_CHECK(!ref_locations.empty());
      LOG(INFO) << "Converting Alignment Coordinates from GPS (lat/lon/alt) "
                   "to ENU. Using the first GPS coordinate as the ENU origin: "
                << ref_locations[0].transpose();
      return gps_transform.EllipsoidToENU(ref_locations,
                                          ref_locations[0](0),
                                          ref_locations[0](1),
                                          ref_locations[0](2));
    }
  } else {
    LOG(INFO) << "Cartesian Alignment Coordinates extracted (MUST NOT BE "
                 "GPS coords!).";
    return ref_locations;
  }
}

void ReadFileCameraLocations(const std::filesystem::path& ref_images_path,
                             const bool ref_is_gps,
                             const std::string& alignment_type,
                             std::vector<std::string>* ref_image_names,
                             std::vector<Eigen::Vector3d>* ref_locations) {
  for (const auto& line : ReadTextFileLines(ref_images_path)) {
    std::stringstream line_parser(line);
    line_parser.imbue(std::locale::classic());
    std::string image_name;
    Eigen::Vector3d camera_position;
    THROW_CHECK(line_parser >> image_name >> camera_position[0] >>
                camera_position[1] >> camera_position[2]);
    ref_image_names->push_back(image_name);
    ref_locations->push_back(camera_position);
  }

  *ref_locations =
      ConvertCameraLocations(ref_is_gps, alignment_type, *ref_locations);
}

void ReadDatabaseCameraLocations(const std::filesystem::path& database_path,
                                 const bool ref_is_gps,
                                 const std::string& alignment_type,
                                 std::vector<std::string>* ref_image_names,
                                 std::vector<Eigen::Vector3d>* ref_locations) {
  auto database = Database::Open(database_path);

  // Index pose priors by their associated data ID.
  NodeHashMap<data_t, PosePrior> pose_priors_by_data_id;
  for (const auto& pose_prior : database->ReadAllPosePriors()) {
    pose_priors_by_data_id.emplace(pose_prior.corr_data_id, pose_prior);
  }

  for (const auto& image : database->ReadAllImages()) {
    const auto it = pose_priors_by_data_id.find(image.DataId());
    if (it != pose_priors_by_data_id.end()) {
      ref_image_names->push_back(image.Name());
      const auto& pose_prior = it->second;
      if (ref_is_gps) {
        THROW_CHECK_EQ(static_cast<int>(pose_prior.coordinate_system),
                       static_cast<int>(PosePrior::CoordinateSystem::WGS84));
      }
      ref_locations->push_back(pose_prior.position);
    }
  }

  *ref_locations =
      ConvertCameraLocations(ref_is_gps, alignment_type, *ref_locations);
}

void WriteComparisonErrorsCSV(const std::filesystem::path& path,
                              const std::vector<ImageAlignmentError>& errors) {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file.imbue(std::locale::classic());
  file.precision(17);
  file << "# Model comparison pose errors: one entry per common image\n";
  file << "# <rotation error (deg)>, <proj center error>\n";
  for (size_t i = 0; i < errors.size(); ++i) {
    file << errors[i].rotation_error_deg << ", " << errors[i].proj_center_error
         << '\n';
  }
}

void PrintErrorStats(std::ostream& out,
                     const AlignmentErrorSummary::Statistics& stats) {
  out << "Min:    " << stats.min << '\n';
  out << "Max:    " << stats.max << '\n';
  out << "Mean:   " << stats.mean << '\n';
  out << "Median: " << stats.median << '\n';
  out << "P90:    " << stats.p90 << '\n';
  out << "P99:    " << stats.p99 << '\n';
}

void PrintComparisonSummary(std::ostream& out,
                            const std::vector<ImageAlignmentError>& errors) {
  if (errors.empty()) {
    out << "Cannot extract error statistics from empty input\n";
    return;
  }
  AlignmentErrorSummary summary = AlignmentErrorSummary::Compute(errors);
  out << "\nRotation errors (degrees)\n";
  PrintErrorStats(out, summary.rotation_errors_deg);
  out << "\nProjection center errors\n";
  PrintErrorStats(out, summary.proj_center_errors);
}

////////////////////////////////////////////////////////////////////////////
// Scene georeference report (model_aligner --georeference_json/
// --camera_residuals_csv).
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

// RFC 4122 version-4 UUID.
std::string GenerateUUIDv4() {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dist;
  uint64_t hi = dist(gen);
  uint64_t lo = dist(gen);
  hi = (hi & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
  lo = (lo & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;
  return StringPrintf("%08x-%04x-%04x-%04x-%04x%08x",
                      static_cast<unsigned int>(hi >> 32),
                      static_cast<unsigned int>((hi >> 16) & 0xFFFFu),
                      static_cast<unsigned int>(hi & 0xFFFFu),
                      static_cast<unsigned int>(lo >> 48),
                      static_cast<unsigned int>((lo >> 32) & 0xFFFFu),
                      static_cast<unsigned int>(lo & 0xFFFFFFFFu));
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
struct CameraResidual {
  image_t image_id = kInvalidImageId;
  std::string image_name;
  bool registered = false;
  bool has_position_prior = false;
  bool position_fit_inlier = false;
  Eigen::Vector3d prior_enu =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d solved_enu =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
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

void WriteGeoreferenceReportJSON(const std::filesystem::path& path,
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
                                 const std::vector<CameraResidual>& residuals,
                                 double position_ransac_threshold,
                                 double orientation_max_error_deg,
                                 bool orientation_requested,
                                 bool orientation_engaged) {
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
  int num_position_inliers = 0;
  double max_horizontal_radius = 0.0;
  double max_3d_radius = 0.0;
  double max_horizontal_baseline = 0.0;
  std::vector<Eigen::Vector3d> inlier_solved_enu;
  for (const CameraResidual& r : residuals) {
    if (r.position_fit_inlier) {
      ++num_position_inliers;
      const Eigen::Vector3d diff = r.solved_enu - r.prior_enu;
      horizontal_residuals.push_back(std::hypot(diff.x(), diff.y()));
      vertical_residuals.push_back(std::abs(diff.z()));
      full_residuals.push_back(diff.norm());
      max_horizontal_radius =
          std::max(max_horizontal_radius,
                   std::hypot(r.solved_enu.x(), r.solved_enu.y()));
      max_3d_radius = std::max(max_3d_radius, r.solved_enu.norm());
      inlier_solved_enu.push_back(r.solved_enu);
    }
  }
  for (size_t i = 0; i < inlier_solved_enu.size(); ++i) {
    for (size_t j = i + 1; j < inlier_solved_enu.size(); ++j) {
      const double d =
          (inlier_solved_enu[i].head<2>() - inlier_solved_enu[j].head<2>())
              .norm();
      max_horizontal_baseline = std::max(max_horizontal_baseline, d);
    }
  }

  int num_registered = 0;
  int num_with_prior = 0;
  int num_orientation_candidates = 0;
  for (const CameraResidual& r : residuals) {
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
  json << "\"scene_id\":\"" << JSONEscape(scene_id) << "\",";
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
  json << "\"support\":{";
  json << "\"num_registered\":" << num_registered << ",";
  json << "\"num_with_position_prior\":" << num_with_prior << ",";
  json << "\"num_position_inliers\":" << num_position_inliers << ",";
  json << "\"num_orientation_candidates\":" << num_orientation_candidates
       << ",";
  json << "\"num_orientation_inliers\":0";
  json << "},";
  json << "\"diagnostics\":{";
  json << "\"position_3d_residual_m\":{"
       << "\"mean\":" << JSONNumber(Mean(full_residuals))
       << ",\"median\":" << JSONNumber(Percentile(full_residuals, 0.5))
       << ",\"p90\":" << JSONNumber(Percentile(full_residuals, 0.9))
       << ",\"max\":"
       << JSONNumber(full_residuals.empty()
                         ? std::numeric_limits<double>::quiet_NaN()
                         : *std::max_element(full_residuals.begin(),
                                             full_residuals.end()))
       << "},";
  json << "\"position_horizontal_residual_m\":{"
       << "\"mean\":" << JSONNumber(Mean(horizontal_residuals))
       << ",\"median\":" << JSONNumber(Percentile(horizontal_residuals, 0.5))
       << "},";
  json << "\"position_vertical_residual_m\":{"
       << "\"mean\":" << JSONNumber(Mean(vertical_residuals))
       << ",\"median\":" << JSONNumber(Percentile(vertical_residuals, 0.5))
       << "},";
  json << "\"max_horizontal_baseline_m\":"
       << JSONNumber(max_horizontal_baseline) << ",";
  json << "\"max_horizontal_radius_m\":" << JSONNumber(max_horizontal_radius)
       << ",";
  json << "\"max_3d_radius_m\":" << JSONNumber(max_3d_radius);
  json << "},";
  json << "\"position_ransac_threshold_m\":"
       << JSONNumber(position_ransac_threshold) << ",";
  json << "\"orientation_max_error_deg\":"
       << JSONNumber(orientation_max_error_deg) << ",";
  json << "\"orientation_requested\":"
       << (orientation_requested ? "true" : "false") << ",";
  json << "\"orientation_engaged\":"
       << (orientation_engaged ? "true" : "false");
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

void WriteCameraResidualsCSV(const std::filesystem::path& path,
                             const std::vector<CameraResidual>& residuals) {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  file.imbue(std::locale::classic());
  file.precision(17);

  std::vector<CameraResidual> sorted_residuals = residuals;
  std::sort(sorted_residuals.begin(),
            sorted_residuals.end(),
            [](const CameraResidual& a, const CameraResidual& b) {
              return a.image_name < b.image_name;
            });

  file << "image_name,registered,has_position_prior,position_fit_inlier,"
          "prior_e,prior_n,prior_u,solved_e,solved_n,solved_u,"
          "residual_e,residual_n,residual_u,residual_horizontal,"
          "residual_vertical,residual_3d,"
          "has_orientation_prior,orientation_fit_inlier,"
          "orientation_residual_deg\n";
  for (const CameraResidual& r : sorted_residuals) {
    Eigen::Vector3d residual =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    double residual_horizontal = std::numeric_limits<double>::quiet_NaN();
    double residual_3d = std::numeric_limits<double>::quiet_NaN();
    if (r.position_fit_inlier) {
      residual = r.solved_enu - r.prior_enu;
      residual_horizontal = std::hypot(residual.x(), residual.y());
      residual_3d = residual.norm();
    }
    // Orientation columns are always absent: orientation-assisted refinement
    // is not implemented, so reporting a prior's mere presence here without
    // ever fitting it would be misleading.
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
         << ",0,0,\n";
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
    }
    converted.push_back(out);
  }
  return converted;
}

// Extends model_aligner with the scene georeference report path. Position-
// only: orientation-assisted joint Sim3 refinement is not yet implemented
// (see --use_pose_prior_orientation below).
int RunModelAlignerReport(const std::filesystem::path& input_path,
                          const std::filesystem::path& output_path,
                          const std::filesystem::path& database_path,
                          bool has_explicit_origin,
                          double enu_origin_lat,
                          double enu_origin_lon,
                          double enu_origin_alt,
                          int min_common_images,
                          const RANSACOptions& ransac_options,
                          double orientation_max_error_deg,
                          const std::string& scene_id_option,
                          const std::filesystem::path& georeference_json,
                          const std::filesystem::path& camera_residuals_csv) {
  auto database = Database::Open(database_path);
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
  if (any_cartesian && !has_explicit_origin) {
    LOG(ERROR) << "Cartesian ENU archives require an explicit "
                  "--enu_origin_lat/--enu_origin_lon/--enu_origin_alt for a "
                  "report run";
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  NodeHashMap<image_t, PosePrior> priors_by_image;
  for (const PosePrior& prior : pose_priors) {
    if (prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
      priors_by_image.emplace(static_cast<image_t>(prior.corr_data_id.id),
                              prior);
    }
  }

  double origin_lat = enu_origin_lat;
  double origin_lon = enu_origin_lon;
  double origin_alt = enu_origin_alt;
  const bool origin_is_explicit = has_explicit_origin;

  if (any_wgs84 && !has_explicit_origin) {
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

  PosePriorAlignmentResult result = AlignReconstructionToPosePriorsRobust(
      reconstruction, enu_priors, ransac_options);
  if (!result.success) {
    LOG(ERROR) << "=> Alignment failed";
    return EXIT_FAILURE;
  }

  // Recompute the origin once from the position-fit inlier WGS84 points,
  // reconvert, and refit once. Never uses the first row by policy.
  if (any_wgs84 && !has_explicit_origin) {
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
      result = AlignReconstructionToPosePriorsRobust(
          reconstruction, enu_priors, ransac_options);
      if (!result.success) {
        LOG(ERROR) << "=> Alignment failed after origin refinement";
        return EXIT_FAILURE;
      }
    }
  }

  const int num_inliers = static_cast<int>(
      std::count(result.inlier_mask.begin(), result.inlier_mask.end(), 1));
  if (num_inliers < min_common_images) {
    LOG(ERROR) << "=> Too few position-prior inliers: " << num_inliers << " < "
               << min_common_images;
    return EXIT_FAILURE;
  }

  reconstruction.Transform(result.tgt_from_src);

  FlatHashSet<image_t> inlier_image_ids;
  for (size_t i = 0; i < result.correspondence_image_ids.size(); ++i) {
    if (result.inlier_mask[i]) {
      inlier_image_ids.insert(result.correspondence_image_ids[i]);
    }
  }
  NodeHashMap<image_t, Eigen::Vector3d> enu_prior_position_by_image;
  for (const PosePrior& prior : enu_priors) {
    if (prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
      enu_prior_position_by_image.emplace(
          static_cast<image_t>(prior.corr_data_id.id), prior.position);
    }
  }

  std::vector<CameraResidual> residuals;
  residuals.reserve(reconstruction.NumImages());
  for (const auto& [image_id, image] : reconstruction.Images()) {
    CameraResidual r;
    r.image_id = image_id;
    r.image_name = image.Name();
    r.registered = image.HasPose();
    const auto prior_it = priors_by_image.find(image_id);
    if (prior_it != priors_by_image.end()) {
      r.has_position_prior = std::isfinite(prior_it->second.position.x()) &&
                             std::isfinite(prior_it->second.position.y());
    }
    if (r.registered && inlier_image_ids.count(image_id)) {
      r.position_fit_inlier = true;
      r.solved_enu = image.ProjectionCenter();
      const auto enu_it = enu_prior_position_by_image.find(image_id);
      if (enu_it != enu_prior_position_by_image.end()) {
        r.prior_enu = enu_it->second;
      }
    }
    residuals.push_back(r);
  }

  std::string scene_id = scene_id_option;
  if (!georeference_json.empty()) {
    if (!scene_id.empty() && !IsValidSceneId(scene_id)) {
      LOG(ERROR) << "=> --scene_id must be non-empty and free of control "
                    "characters";
      return EXIT_FAILURE;
    }
    if (scene_id.empty()) {
      scene_id = GenerateUUIDv4();
    }
    WriteGeoreferenceReportJSON(georeference_json,
                                scene_id,
                                GetBuildInfo(),
                                input_path,
                                output_path,
                                reconstruction,
                                origin_lat,
                                origin_lon,
                                origin_alt,
                                origin_is_explicit,
                                result.tgt_from_src,
                                residuals,
                                ransac_options.max_error,
                                orientation_max_error_deg,
                                /*orientation_requested=*/false,
                                /*orientation_engaged=*/false);
  }
  if (!camera_residuals_csv.empty()) {
    WriteCameraResidualsCSV(camera_residuals_csv, residuals);
  }

  LOG(INFO) << StringPrintf(
      "=> Alignment succeeded: %d position-prior inliers, ENU origin "
      "(lat=%.8f, lon=%.8f, alt=%.3f, explicit=%s)",
      num_inliers,
      origin_lat,
      origin_lon,
      origin_alt,
      origin_is_explicit ? "true" : "false");
  reconstruction.Write(output_path);
  return EXIT_SUCCESS;
}

}  // namespace

// Align given reconstruction with user provided cameras positions
// (can be used for geo-registration for instance).
// The cameras positions to be used for aligning the reconstruction
// model must be provided either by a txt file (with each line being: img_name x
// y z) or through a colmap database file containing a prior position for the
// registered images.
//
// Required Options:
// - input_path: path to initial reconstruction model
// - output_path: path to store the aligned reconstruction model
//
// Additional Options:
// - database_path: path to database file with prior positions for
// reconstruction images
// - ref_images_path: path to txt file with prior positions for reconstruction
// images (WARNING: provide only one of the above)
// - ref_is_gps: if true the prior positions are converted from GPS
// (lat/lon/alt) to ECEF or ENU
// - merge_image_and_ref_origins: if true the reconstruction will be shifted so
// that the first prior position is used for its camera position
// - transform_path: path to store the Sim3 transformation used for the
// alignment
// - alignment_type:
//    > plane: align with reconstruction principal plane
//    > ecef: align with ecef coords. (requires gps coords. or user provided
//    ecef coords.)
//    > enu: align with enu coords. (requires gps coords. or user provided enu
//    coords.)
//    > enu-plane: align to ecef and then to enu plane (requires gps
//    coords. or user provided ecef coords.)
//    > enu-plane-unscaled: same as above but do not apply the computed
//    scale when aligning the reconstruction
//    > custom: align to provided coords.
// - min_common_images: minimum number of images with prior positions to perform
// the estimate an alignment
// - estimate_scale: if true apply the computed scale when aligning the
// reconstruction
// - alignment_max_error: ransac error to use
int RunModelAligner(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::filesystem::path database_path;
  std::filesystem::path ref_model_path;
  std::filesystem::path ref_images_path;
  bool ref_is_gps = true;
  bool merge_origins = false;
  std::filesystem::path transform_path;
  std::string alignment_type = "custom";
  int min_common_images = 3;
  RANSACOptions ransac_options;

  // Scene georeference report options. The three origin values are
  // all-or-none; altitude is WGS84 ellipsoidal.
  double enu_origin_lat = std::numeric_limits<double>::quiet_NaN();
  double enu_origin_lon = std::numeric_limits<double>::quiet_NaN();
  double enu_origin_alt = std::numeric_limits<double>::quiet_NaN();
  bool use_pose_prior_orientation = false;
  double orientation_max_error_deg = 10.0;
  std::string scene_id;
  std::filesystem::path georeference_json;
  std::filesystem::path camera_residuals_csv;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("database_path", &database_path);
  options.AddDefaultOption("ref_model_path", &ref_model_path);
  options.AddDefaultOption("ref_images_path", &ref_images_path);
  options.AddDefaultOption("ref_is_gps", &ref_is_gps);
  options.AddDefaultOption("merge_image_and_ref_origins", &merge_origins);
  options.AddDefaultOption("transform_path", &transform_path);
  options.AddDefaultOption(
      "alignment_type",
      &alignment_type,
      "{plane, ecef, enu, enu-plane, enu-plane-unscaled, custom}");
  options.AddDefaultOption("min_common_images", &min_common_images);
  options.AddDefaultOption("alignment_max_error", &ransac_options.max_error);
  options.AddDefaultOption("enu_origin_lat", &enu_origin_lat);
  options.AddDefaultOption("enu_origin_lon", &enu_origin_lon);
  options.AddDefaultOption("enu_origin_alt", &enu_origin_alt);
  options.AddDefaultOption("use_pose_prior_orientation",
                           &use_pose_prior_orientation);
  options.AddDefaultOption("orientation_max_error_deg",
                           &orientation_max_error_deg);
  options.AddDefaultOption("scene_id", &scene_id);
  options.AddDefaultOption("georeference_json", &georeference_json);
  options.AddDefaultOption("camera_residuals_csv", &camera_residuals_csv);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  const bool report_requested =
      !georeference_json.empty() || !camera_residuals_csv.empty();
  if (report_requested) {
    if (alignment_type != "custom" && alignment_type != "enu") {
      LOG(ERROR) << "=> A report run requires --alignment_type=enu (the "
                    "default `custom` is also accepted before the type "
                    "check below overrides it)";
    }
    alignment_type = "enu";
    if (database_path.empty()) {
      LOG(ERROR) << "=> A report run requires --database_path";
      return EXIT_FAILURE;
    }
    if (!ref_model_path.empty() || !ref_images_path.empty()) {
      LOG(ERROR) << "=> A report run requires neither --ref_model_path nor "
                    "--ref_images_path (the database's own position priors "
                    "are used)";
      return EXIT_FAILURE;
    }
    if (merge_origins) {
      LOG(ERROR) << "=> --merge_image_and_ref_origins is not supported for "
                    "a report run; the robust final ENU origin already "
                    "defines the output origin";
      return EXIT_FAILURE;
    }
    if (!scene_id.empty() && georeference_json.empty()) {
      LOG(ERROR) << "=> --scene_id requires --georeference_json (the CSV "
                    "has no scene-identity field)";
      return EXIT_FAILURE;
    }
    const bool any_origin_component = !std::isnan(enu_origin_lat) ||
                                      !std::isnan(enu_origin_lon) ||
                                      !std::isnan(enu_origin_alt);
    const bool all_origin_components = !std::isnan(enu_origin_lat) &&
                                       !std::isnan(enu_origin_lon) &&
                                       !std::isnan(enu_origin_alt);
    if (any_origin_component && !all_origin_components) {
      LOG(ERROR) << "=> --enu_origin_lat/--enu_origin_lon/--enu_origin_alt "
                    "must be supplied all together or not at all";
      return EXIT_FAILURE;
    }
    if (ransac_options.max_error <= 0) {
      LOG(ERROR) << "You must provide a maximum alignment error > 0";
      return EXIT_FAILURE;
    }
    if (use_pose_prior_orientation) {
      LOG(WARNING) << "=> --use_pose_prior_orientation is accepted but not "
                      "yet implemented in this build; the report always "
                      "reports orientation_requested=false, "
                      "orientation_engaged=false and every CSV orientation "
                      "column stays empty";
    }
    return RunModelAlignerReport(input_path,
                                 output_path,
                                 database_path,
                                 all_origin_components,
                                 enu_origin_lat,
                                 enu_origin_lon,
                                 enu_origin_alt,
                                 min_common_images,
                                 ransac_options,
                                 orientation_max_error_deg,
                                 scene_id,
                                 georeference_json,
                                 camera_residuals_csv);
  }

  StringToLower(&alignment_type);
  const FlatHashSet<std::string> alignment_options{
      "plane", "ecef", "enu", "enu-plane", "enu-plane-unscaled", "custom"};
  if (alignment_options.count(alignment_type) == 0) {
    LOG(ERROR) << "Invalid `alignment_type` - supported values are "
                  "{'plane', 'ecef', 'enu', 'enu-plane', 'enu-plane-unscaled', "
                  "'custom'}";
    return EXIT_FAILURE;
  }

  if (ransac_options.max_error <= 0) {
    LOG(ERROR) << "You must provide a maximum alignment error > 0";
    return EXIT_FAILURE;
  }

  if (ref_model_path.empty() && database_path.empty() &&
      ref_images_path.empty() && alignment_type != "plane") {
    LOG(ERROR) << "One of the following arguments must be specified: "
                  "--ref_model_path, --database_path, "
                  "--ref_images_path, --alignment_type=plane";
    return EXIT_FAILURE;
  }

  std::vector<std::string> ref_image_names;
  std::vector<Eigen::Vector3d> ref_locations;
  if (!ref_model_path.empty() && database_path.empty()) {
    Reconstruction reconstruction;
    reconstruction.Read(ref_model_path);
    for (const auto& image : reconstruction.Images()) {
      ref_image_names.push_back(image.second.Name());
      ref_locations.push_back(image.second.ProjectionCenter());
    }
  } else if (!ref_images_path.empty() && database_path.empty()) {
    ReadFileCameraLocations(ref_images_path,
                            ref_is_gps,
                            alignment_type,
                            &ref_image_names,
                            &ref_locations);
  } else if (!database_path.empty() && ref_images_path.empty()) {
    ReadDatabaseCameraLocations(database_path,
                                ref_is_gps,
                                alignment_type,
                                &ref_image_names,
                                &ref_locations);
  } else if (alignment_type != "plane") {
    LOG(ERROR) << "Use location file or database, not both";
    return EXIT_FAILURE;
  }

  if (alignment_type != "plane" &&
      static_cast<int>(ref_locations.size()) < min_common_images) {
    LOG(ERROR) << "Cannot align with insufficient reference locations.";
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);
  Sim3d tform;

  if (alignment_type == "plane") {
    LOG_HEADING2("Aligning reconstruction to principal plane");
    AlignToPrincipalPlane(&reconstruction, &tform);
  } else {
    LOG_HEADING2("Aligning reconstruction to " + alignment_type);
    LOG(INFO) << StringPrintf("=> Using %d reference images",
                              ref_image_names.size());

    const bool alignment_success =
        AlignReconstructionToLocations(reconstruction,
                                       ref_image_names,
                                       ref_locations,
                                       min_common_images,
                                       ransac_options,
                                       &tform);

    if (!alignment_success) {
      LOG(ERROR) << "=> Alignment failed";
      return EXIT_FAILURE;
    }

    reconstruction.Transform(tform);

    std::vector<double> errors;
    errors.reserve(ref_image_names.size());

    for (size_t i = 0; i < ref_image_names.size(); ++i) {
      const Image* image = reconstruction.FindImageWithName(ref_image_names[i]);
      if (image != nullptr) {
        errors.push_back((image->ProjectionCenter() - ref_locations[i]).norm());
      }
    }
    LOG(INFO) << StringPrintf("=> Alignment error: %f (mean), %f (median)",
                              Mean(errors),
                              Median(errors));

    if (alignment_success && StringStartsWith(alignment_type, "enu-plane")) {
      LOG_HEADING2("Aligning ECEF aligned reconstruction to ENU plane");
      AlignToENUPlane(
          &reconstruction, &tform, alignment_type == "enu-plane-unscaled");
    }
  }

  if (merge_origins) {
    for (size_t i = 0; i < ref_image_names.size(); i++) {
      const Image* first_image =
          reconstruction.FindImageWithName(ref_image_names[i]);

      if (first_image != nullptr) {
        const Eigen::Vector3d& first_img_position = ref_locations[i];
        const Eigen::Vector3d trans_align =
            first_img_position - first_image->ProjectionCenter();
        const Sim3d origin_align(
            1.0, Eigen::Quaterniond::Identity(), trans_align);

        LOG(INFO) << "\n Aligning reconstruction's origin with ref origin: "
                  << first_img_position.transpose() << '\n';

        reconstruction.Transform(origin_align);

        // Update the Sim3 transformation in case it is stored next.
        tform = Sim3d(
            tform.scale(), tform.rotation(), tform.translation() + trans_align);

        break;
      }
    }
  }

  LOG(INFO) << "=> Alignment succeeded";
  reconstruction.Write(output_path);
  if (!transform_path.empty()) {
    tform.ToFile(transform_path);
  }

  return EXIT_SUCCESS;
}

int RunModelAnalyzer(int argc, char** argv) {
  std::filesystem::path path;
  bool verbose = false;

  OptionManager options;
  options.AddRequiredOption("path", &path);
  options.AddDefaultOption("verbose", &verbose);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(path);

  LOG(INFO) << StringPrintf("Rigs: %d", reconstruction.NumRigs());
  LOG(INFO) << StringPrintf("Cameras: %d", reconstruction.NumCameras());
  LOG(INFO) << StringPrintf("Frames: %d", reconstruction.NumFrames());
  LOG(INFO) << StringPrintf("Registered frames: %d",
                            reconstruction.NumRegFrames());
  LOG(INFO) << StringPrintf("Images: %d", reconstruction.NumImages());
  LOG(INFO) << StringPrintf("Registered images: %d",
                            reconstruction.NumRegImages());
  LOG(INFO) << StringPrintf("Points: %d", reconstruction.NumPoints3D());
  LOG(INFO) << StringPrintf("Observations: %d",
                            reconstruction.ComputeNumObservations());
  LOG(INFO) << StringPrintf("Mean track length: %f",
                            reconstruction.ComputeMeanTrackLength());
  LOG(INFO) << StringPrintf(
      "Mean observations per image: %f",
      reconstruction.ComputeMeanObservationsPerRegImage());
  LOG(INFO) << StringPrintf("Mean reprojection error: %fpx",
                            reconstruction.ComputeMeanReprojectionError());

  // verbose information
  if (verbose) {
    LOG_HEADING2("Cameras");
    for (const auto& camera : reconstruction.Cameras()) {
      LOG(INFO) << StringPrintf(" - Camera Id: %d, Model Name: %s, Params: %s",
                                camera.first,
                                camera.second.ModelName().c_str(),
                                camera.second.ParamsToString().c_str());
    }

    LOG_HEADING2("Images");
    for (const auto& image_id : reconstruction.RegImageIds()) {
      LOG(INFO) << StringPrintf(" - Registered Image Id: %d, Name: %s",
                                image_id,
                                reconstruction.Image(image_id).Name().c_str());
    }
  }

  return EXIT_SUCCESS;
}

int RunModelClusterer(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddReconstructionClustererOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory";
    return EXIT_FAILURE;
  }

  LOG_HEADING1("Loading model");
  auto reconstruction = std::make_shared<Reconstruction>();
  reconstruction->Read(input_path);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  ReconstructionClustererController controller(
      *options.reconstruction_clusterer,
      reconstruction,
      reconstruction_manager);
  controller.Run();

  LOG_HEADING1("Writing clustered model(s)");
  reconstruction_manager->Write(output_path);

  return EXIT_SUCCESS;
}

int RunModelComparer(int argc, char** argv) {
  std::filesystem::path input_path1;
  std::filesystem::path input_path2;
  std::filesystem::path output_path;
  std::string alignment_error = "reprojection";
  double min_inlier_observations = 0.3;
  double max_reproj_error = 8.0;
  double max_proj_center_error = 0.1;

  OptionManager options;
  options.AddRequiredOption("input_path1", &input_path1);
  options.AddRequiredOption("input_path2", &input_path2);
  options.AddDefaultOption("output_path", &output_path);
  options.AddDefaultOption(
      "alignment_error", &alignment_error, "{reprojection, proj_center}");
  options.AddDefaultOption("min_inlier_observations", &min_inlier_observations);
  options.AddDefaultOption("max_reproj_error", &max_reproj_error);
  options.AddDefaultOption("max_proj_center_error", &max_proj_center_error);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!output_path.empty() && !ExistsDir(output_path)) {
    LOG(ERROR) << "Provided output path is not a valid directory";
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction1;
  reconstruction1.Read(input_path1);
  Reconstruction reconstruction2;
  reconstruction2.Read(input_path2);
  std::vector<ImageAlignmentError> errors;
  Sim3d rec2_from_rec1;
  bool success = CompareModels(reconstruction1,
                               reconstruction2,
                               alignment_error,
                               min_inlier_observations,
                               max_reproj_error,
                               max_proj_center_error,
                               errors,
                               rec2_from_rec1);
  if (!success) {
    return EXIT_FAILURE;
  }
  if (!output_path.empty()) {
    const auto errors_path = output_path / "errors.csv";
    WriteComparisonErrorsCSV(errors_path, errors);
    const auto summary_path = output_path / "errors_summary.txt";
    std::ofstream file(summary_path, std::ios::trunc);
    THROW_CHECK_FILE_OPEN(file, summary_path);
    PrintComparisonSummary(file, errors);
  }
  return EXIT_SUCCESS;
}

bool CompareModels(const Reconstruction& reconstruction1,
                   const Reconstruction& reconstruction2,
                   const std::string& alignment_error,
                   const double min_inlier_observations,
                   const double max_reproj_error,
                   const double max_proj_center_error,
                   std::vector<ImageAlignmentError>& errors,
                   Sim3d& rec2_from_rec1) {
  LOG_HEADING1("Reconstruction 1");
  LOG(INFO) << StringPrintf("Frames: %d", reconstruction1.NumRegFrames());
  LOG(INFO) << StringPrintf("Images: %d", reconstruction1.NumRegImages());
  LOG(INFO) << StringPrintf("Points: %d", reconstruction1.NumPoints3D());

  LOG_HEADING1("Reconstruction 2");
  LOG(INFO) << StringPrintf("Frames: %d", reconstruction2.NumRegFrames());
  LOG(INFO) << StringPrintf("Images: %d", reconstruction2.NumRegImages());
  LOG(INFO) << StringPrintf("Points: %d", reconstruction2.NumPoints3D());

  LOG_HEADING1("Comparing reconstructed image poses");
  const std::vector<std::pair<image_t, image_t>> common_image_ids =
      reconstruction1.FindCommonRegImageIds(reconstruction2);
  LOG(INFO) << StringPrintf("Common images: %d", common_image_ids.size());

  bool success = false;
  if (alignment_error == "reprojection") {
    success = AlignReconstructionsViaReprojections(
        reconstruction1,
        reconstruction2,
        /*min_inlier_observations=*/min_inlier_observations,
        /*max_reproj_error=*/max_reproj_error,
        &rec2_from_rec1);
  } else if (alignment_error == "proj_center") {
    success = AlignReconstructionsViaProjCenters(
        reconstruction1,
        reconstruction2,
        /*max_proj_center_error=*/max_proj_center_error,
        &rec2_from_rec1);
  } else {
    LOG(ERROR) << "Invalid alignment_error specified.";
    return false;
  }

  if (!success) {
    LOG(INFO) << "=> Reconstruction alignment failed";
    return false;
  }

  LOG(INFO) << "Computed alignment transform:\n" << rec2_from_rec1.ToMatrix();

  errors = ComputeImageAlignmentError(
      reconstruction1, reconstruction2, rec2_from_rec1);

  LOG_HEADING2("Image alignment error summary");
  PrintComparisonSummary(std::cout, errors);

  return true;
}

int RunModelConverter(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::string output_type;
  bool skip_distortion = false;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("output_type",
                            &output_type,
                            "{BIN, TXT, NVM, Bundler, VRML, PLY, R3D, CAM}");
  options.AddDefaultOption("skip_distortion", &skip_distortion);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  StringToLower(&output_type);
  if (output_type == "bin") {
    reconstruction.WriteBinary(output_path);
  } else if (output_type == "txt") {
    reconstruction.WriteText(output_path);
  } else if (output_type == "nvm") {
    ExportNVM(reconstruction, output_path, skip_distortion);
  } else if (output_type == "bundler") {
    ExportBundler(reconstruction,
                  AddFileExtension(output_path, ".bundle.out"),
                  AddFileExtension(output_path, ".list.txt"),
                  skip_distortion);
  } else if (output_type == "r3d") {
    ExportRecon3D(reconstruction, output_path, skip_distortion);
  } else if (output_type == "cam") {
    ExportCam(reconstruction, output_path, skip_distortion);
  } else if (output_type == "ply") {
    ExportPLY(reconstruction, output_path);
  } else if (output_type == "vrml") {
    const auto base_path = output_path.parent_path() / output_path.stem();
    ExportVRML(reconstruction,
               AddFileExtension(base_path, ".images.wrl"),
               AddFileExtension(base_path, ".points3D.wrl"),
               1,
               Eigen::Vector3d(1, 0, 0));
  } else {
    LOG(ERROR) << "Invalid `output_type`";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int RunModelCropper(int argc, char** argv) {
  Timer timer;
  timer.Start();

  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::string boundary;
  std::filesystem::path gps_transform_path;
  bool is_gps = false;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("boundary", &boundary);
  options.AddDefaultOption("gps_transform_path", &gps_transform_path);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory";
    return EXIT_FAILURE;
  }

  std::vector<double> boundary_elements = CSVToVector<double>(boundary);
  if (boundary_elements.size() != 2 && boundary_elements.size() != 6) {
    LOG(ERROR) << "Invalid `boundary` - supported values are "
                  "'x1,y1,z1,x2,y2,z2' or 'p1,p2'.";
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  LOG_HEADING2("Calculating boundary coordinates");
  Eigen::AlignedBox3d bounding_box;
  if (boundary_elements.size() == 6) {
    Sim3d tform;
    if (!gps_transform_path.empty()) {
      LOG_HEADING2("Reading model to ECEF transform");
      is_gps = true;
      tform = Inverse(Sim3d::FromFile(gps_transform_path));
    }
    bounding_box.min() =
        is_gps ? TransformLatLonAltToModelCoords(tform,
                                                 boundary_elements[0],
                                                 boundary_elements[1],
                                                 boundary_elements[2])
               : Eigen::Vector3d(boundary_elements[0],
                                 boundary_elements[1],
                                 boundary_elements[2]);
    bounding_box.max() =
        is_gps ? TransformLatLonAltToModelCoords(tform,
                                                 boundary_elements[3],
                                                 boundary_elements[4],
                                                 boundary_elements[5])
               : Eigen::Vector3d(boundary_elements[3],
                                 boundary_elements[4],
                                 boundary_elements[5]);
  } else {
    bounding_box = reconstruction.ComputeBoundingBox(boundary_elements[0],
                                                     boundary_elements[1]);
  }

  LOG_HEADING2("Cropping reconstruction");
  reconstruction.Crop(bounding_box).Write(output_path);
  WriteBoundingBox(output_path, bounding_box);

  LOG(INFO) << "=> Cropping succeeded";
  timer.PrintMinutes();
  return EXIT_SUCCESS;
}

int RunModelMerger(int argc, char** argv) {
  std::filesystem::path input_path1;
  std::filesystem::path input_path2;
  std::filesystem::path output_path;
  double max_reproj_error = 64.0;

  OptionManager options;
  options.AddRequiredOption("input_path1", &input_path1);
  options.AddRequiredOption("input_path2", &input_path2);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("max_reproj_error", &max_reproj_error);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction1;
  reconstruction1.Read(input_path1);
  LOG_HEADING2("Reconstruction 1");
  LOG(INFO) << StringPrintf("Images: %d", reconstruction1.NumRegImages());
  LOG(INFO) << StringPrintf("Points: %d", reconstruction1.NumPoints3D());

  Reconstruction reconstruction2;
  reconstruction2.Read(input_path2);
  LOG_HEADING2("Reconstruction 2");
  LOG(INFO) << StringPrintf("Images: %d", reconstruction2.NumRegImages());
  LOG(INFO) << StringPrintf("Points: %d", reconstruction2.NumPoints3D());

  LOG_HEADING2("Merging reconstructions");
  if (MergeAndFilterReconstructions(
          max_reproj_error, reconstruction1, reconstruction2)) {
    LOG(INFO) << "=> Merge succeeded";
    LOG_HEADING2("Merged reconstruction");
    LOG(INFO) << StringPrintf("Images: %d", reconstruction2.NumRegImages());
    LOG(INFO) << StringPrintf("Points: %d", reconstruction2.NumPoints3D());
  } else {
    LOG(INFO) << "=> Merge failed";
  }

  reconstruction2.Write(output_path);

  return EXIT_SUCCESS;
}

int RunModelOrientationAligner(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
#ifdef COLMAP_LSD_ENABLED
  std::string method = "MANHATTAN-WORLD";
#else
  std::string method = "IMAGE-ORIENTATION";
#endif

  ManhattanWorldFrameEstimationOptions frame_estimation_options;

  OptionManager options;
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
#ifdef COLMAP_LSD_ENABLED
  options.AddDefaultOption(
      "method", &method, "{MANHATTAN-WORLD, IMAGE-ORIENTATION}");
#else
  options.AddDefaultOption("method", &method, "{IMAGE-ORIENTATION}");
#endif
  options.AddDefaultOption("max_image_size",
                           &frame_estimation_options.max_image_size);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  StringToLower(&method);
#ifdef COLMAP_LSD_ENABLED
  if (method != "manhattan-world" && method != "image-orientation") {
    LOG(ERROR) << "Invalid `method` - supported values are "
                  "'MANHATTAN-WORLD' or 'IMAGE-ORIENTATION'.";
    return EXIT_FAILURE;
  }
#else
  if (method != "image-orientation") {
    LOG(ERROR) << "Invalid `method` - supported values are "
                  "'IMAGE-ORIENTATION'.";
    return EXIT_FAILURE;
  }
#endif

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  LOG_HEADING1("Aligning Reconstruction");

  Sim3d new_from_old_world;

#ifdef COLMAP_LSD_ENABLED
  if (method == "manhattan-world") {
    const Eigen::Matrix3d frame = EstimateManhattanWorldFrame(
        frame_estimation_options, reconstruction, options.image_path->string());

    if (frame.col(0).lpNorm<1>() == 0) {
      LOG(INFO) << "Only aligning vertical axis";
      new_from_old_world.rotation() = Eigen::Quaterniond::FromTwoVectors(
          frame.col(1), Eigen::Vector3d(0, 1, 0));
    } else if (frame.col(1).lpNorm<1>() == 0) {
      new_from_old_world.rotation() = Eigen::Quaterniond::FromTwoVectors(
          frame.col(0), Eigen::Vector3d(1, 0, 0));
      LOG(INFO) << "Only aligning horizontal axis";
    } else {
      new_from_old_world.rotation() = Eigen::Quaterniond(frame.transpose());
      LOG(INFO) << "Aligning horizontal and vertical axes";
    }
  } else if (method == "image-orientation") {
#else
  if (method == "image-orientation") {
#endif
    const Eigen::Vector3d gravity_axis =
        EstimateGravityVectorFromImageOrientation(reconstruction);
    new_from_old_world.rotation() = Eigen::Quaterniond::FromTwoVectors(
        gravity_axis, Eigen::Vector3d(0, 1, 0));

  } else {
    LOG(FATAL_THROW) << "Alignment method not supported";
  }

  LOG(INFO) << "Using the rotation matrix:";
  LOG(INFO) << new_from_old_world.rotation().toRotationMatrix();

  reconstruction.Transform(new_from_old_world);

  LOG(INFO) << "Writing aligned reconstruction...";
  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunModelSplitter(int argc, char** argv) {
  Timer timer;
  timer.Start();

  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::string split_type;
  std::string split_params;
  std::filesystem::path gps_transform_path;
  int num_threads = -1;
  int min_reg_images = 10;
  int min_num_points = 100;
  double overlap_ratio = 0.0;
  double min_area_ratio = 0.0;
  bool is_gps = false;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption(
      "split_type", &split_type, "{tiles, extent, parts}");
  options.AddRequiredOption("split_params", &split_params);
  options.AddDefaultOption("gps_transform_path", &gps_transform_path);
  options.AddDefaultOption("num_threads", &num_threads);
  options.AddDefaultOption("min_reg_images", &min_reg_images);
  options.AddDefaultOption("min_num_points", &min_num_points);
  options.AddDefaultOption("overlap_ratio", &overlap_ratio);
  options.AddDefaultOption("min_area_ratio", &min_area_ratio);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory";
    return EXIT_FAILURE;
  }

  if (overlap_ratio < 0) {
    LOG(WARNING) << "Invalid `overlap_ratio`; resetting to 0";
    overlap_ratio = 0.0;
  }

  LOG_HEADING1("Splitting sparse model");
  LOG(INFO) << StringPrintf("=> Using \"%s\" split type", split_type.c_str());

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  Sim3d tform;
  if (!gps_transform_path.empty()) {
    LOG_HEADING2("Reading model to ECEF transform");
    is_gps = true;
    tform = Inverse(Sim3d::FromFile(gps_transform_path));
  }

  // Create the necessary number of reconstructions based on the split method
  // and get the bounding boxes for each sub-reconstruction
  LOG_HEADING2("Computing bounding boxes");
  std::vector<std::string> tile_keys;
  std::vector<Eigen::AlignedBox3d> exact_bboxes;
  StringToLower(&split_type);
  if (split_type == "tiles") {
    std::ifstream file(split_params);
    THROW_CHECK_FILE_OPEN(file, split_params);
    file.imbue(std::locale::classic());

    double x1, y1, z1, x2, y2, z2;
    std::string tile_key;
    std::vector<Eigen::AlignedBox3d> bounds;
    tile_keys.clear();
    file >> tile_key >> x1 >> y1 >> z1 >> x2 >> y2 >> z2;
    while (!file.fail()) {
      tile_keys.push_back(tile_key);
      if (is_gps) {
        exact_bboxes.emplace_back(
            TransformLatLonAltToModelCoords(tform, x1, y1, z1),
            TransformLatLonAltToModelCoords(tform, x2, y2, z2));
      } else {
        exact_bboxes.emplace_back(Eigen::Vector3d(x1, y1, z1),
                                  Eigen::Vector3d(x2, y2, z2));
      }
      file >> tile_key >> x1 >> y1 >> z1 >> x2 >> y2 >> z2;
    }
  } else if (split_type == "extent") {
    std::vector<double> parts = CSVToVector<double>(split_params);
    Eigen::Vector3d extent(std::numeric_limits<double>::max(),
                           std::numeric_limits<double>::max(),
                           std::numeric_limits<double>::max());
    for (size_t i = 0; i < parts.size(); ++i) {
      extent(i) = parts[i] * tform.scale();
    }

    const Eigen::AlignedBox3d bbox = reconstruction.ComputeBoundingBox();
    const Eigen::Vector3d full_bbox = bbox.diagonal();
    const Eigen::Vector3i split(static_cast<int>(full_bbox(0) / extent(0)) + 1,
                                static_cast<int>(full_bbox(1) / extent(1)) + 1,
                                static_cast<int>(full_bbox(2) / extent(2)) + 1);

    exact_bboxes = ComputeEqualPartsBboxes(bbox, split);
  } else if (split_type == "parts") {
    auto parts = CSVToVector<int>(split_params);
    Eigen::Vector3i split(1, 1, 1);
    for (size_t i = 0; i < parts.size(); ++i) {
      split(i) = parts[i];
      if (split(i) < 1) {
        LOG(ERROR) << "Cannot split in less than 1 parts for dim " << i;
        return EXIT_FAILURE;
      }
    }
    exact_bboxes =
        ComputeEqualPartsBboxes(reconstruction.ComputeBoundingBox(), split);
  } else {
    LOG(ERROR) << "Invalid split type: " << split_type;
    return EXIT_FAILURE;
  }

  std::vector<Eigen::AlignedBox3d> padded_bboxes;
  for (const auto& bbox : exact_bboxes) {
    const Eigen::Vector3d padding = overlap_ratio * bbox.diagonal();
    padded_bboxes.emplace_back(bbox.min() - padding, bbox.max() + padding);
  }

  LOG_HEADING2("Applying split and writing reconstructions");
  const size_t num_parts = padded_bboxes.size();
  LOG(INFO) << StringPrintf("=> Splitting to %d parts", num_parts);

  const bool use_tile_keys = split_type == "tiles";

  auto SplitReconstruction = [&](const int idx) {
    Reconstruction tile_recon = reconstruction.Crop(padded_bboxes[idx]);
    // calculate area covered by model as proportion of box area
    const Eigen::Vector3d bbox_extent = padded_bboxes[idx].diagonal();
    const Eigen::AlignedBox3d model_bbox = tile_recon.ComputeBoundingBox();
    const Eigen::Vector3d model_extent = model_bbox.diagonal();
    const double area_ratio =
        (model_extent(0) * model_extent(1)) / (bbox_extent(0) * bbox_extent(1));
    const int tile_num_points = tile_recon.NumPoints3D();

    const std::string name =
        use_tile_keys ? tile_keys[idx] : std::to_string(idx);
    const bool include_tile =
        area_ratio >= min_area_ratio &&       //
        tile_num_points >= min_num_points &&  //
        tile_recon.NumRegImages() >= static_cast<size_t>(min_reg_images);

    if (include_tile) {
      LOG(INFO) << StringPrintf(
          "Writing reconstruction %s with %d images, %d points, "
          "and %.2f%% area coverage",
          name.c_str(),
          tile_recon.NumRegImages(),
          tile_num_points,
          100.0 * area_ratio);
      const auto reconstruction_path = output_path / name;
      CreateDirIfNotExists(reconstruction_path);
      tile_recon.Write(reconstruction_path);
      WriteBoundingBox(reconstruction_path, padded_bboxes[idx]);
      WriteBoundingBox(reconstruction_path, exact_bboxes[idx], "_exact");
    } else {
      LOG(INFO) << StringPrintf(
          "Skipping reconstruction %s with %d images, %d points, "
          "and %.2f%% area coverage",
          name.c_str(),
          tile_recon.NumRegImages(),
          tile_num_points,
          100.0 * area_ratio);
    }
  };

  ThreadPool thread_pool(GetEffectiveNumThreads(num_threads));
  for (size_t idx = 0; idx < num_parts; ++idx) {
    thread_pool.AddTask(SplitReconstruction, idx);
  }
  thread_pool.Wait();

  timer.PrintMinutes();
  return EXIT_SUCCESS;
}

int RunModelTransformer(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::filesystem::path transform_path;
  bool is_inverse = false;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("transform_path", &transform_path);
  options.AddDefaultOption("is_inverse", &is_inverse);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  LOG(INFO) << "Reading points input: " << input_path;
  Reconstruction recon;
  bool is_dense = false;
  if (HasFileExtension(input_path, ".ply")) {
    is_dense = true;
    recon.ImportPLY(input_path);
  } else if (ExistsDir(input_path)) {
    recon.Read(input_path);
  } else {
    LOG(ERROR)
        << "Invalid model input; not a PLY file or sparse reconstruction "
           "directory.";
    return EXIT_FAILURE;
  }

  LOG(INFO) << "Reading transform input: " << transform_path;
  Sim3d tform = Sim3d::FromFile(transform_path);
  if (is_inverse) {
    tform = Inverse(tform);
  }

  LOG(INFO) << "Applying transform to recon with " << recon.NumPoints3D()
            << " points";
  recon.Transform(tform);

  LOG(INFO) << "Writing output: " << output_path;
  if (is_dense) {
    ExportPLY(recon, output_path);
  } else {
    recon.Write(output_path);
  }

  return EXIT_SUCCESS;
}

}  // namespace colmap
