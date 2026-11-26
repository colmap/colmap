// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/apriltag_alignment.h"

#include "colmap/estimators/solvers/similarity_transform.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/math/math.h"
#include "colmap/scene/projection.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/string.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <utility>

#include <Eigen/SVD>

namespace colmap {
namespace {

struct RayObservation {
  image_t image_id = kInvalidImageId;
  const Camera* camera = nullptr;
  Eigen::Matrix3x4d cam_from_world_mat;
  Eigen::Vector3d proj_center;
  Eigen::Vector3d cam_ray;
  Eigen::Vector3d world_ray;
  Eigen::Vector2d img_point;
};

bool HasPositiveCheirality(const RayObservation& obs,
                           const Eigen::Vector3d& point3D) {
  if (obs.camera->IsPerspective()) {
    return HasPointPositiveDepth(obs.cam_from_world_mat, point3D);
  }
  return (obs.cam_from_world_mat * point3D.homogeneous()).dot(obs.cam_ray) >
         0.0;
}

bool TriangulateRays(const std::vector<RayObservation>& observations,
                     const std::vector<size_t>& indices,
                     Eigen::Vector3d* point3D) {
  THROW_CHECK_NOTNULL(point3D);
  if (indices.size() < 2) {
    return false;
  }

  Eigen::Vector3d origin_mean = Eigen::Vector3d::Zero();
  for (const size_t idx : indices) {
    origin_mean += observations[idx].proj_center;
  }
  origin_mean /= static_cast<double>(indices.size());

  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  for (const size_t idx : indices) {
    const Eigen::Vector3d& d = observations[idx].world_ray;
    const Eigen::Matrix3d projector =
        Eigen::Matrix3d::Identity() - d * d.transpose();
    A += projector;
    b += projector * (observations[idx].proj_center - origin_mean);
  }

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d svals = svd.singularValues();
  if (!svals.allFinite() || svals(0) <= 0.0 || svals(2) <= 1e-8 * svals(0)) {
    return false;
  }

  const Eigen::Vector3d candidate = origin_mean + svd.solve(b);
  if (!candidate.allFinite()) {
    return false;
  }
  for (const size_t idx : indices) {
    if (!HasPositiveCheirality(observations[idx], candidate)) {
      return false;
    }
  }
  *point3D = candidate;
  return true;
}

double ComputeReprojectionErrorPx(const RayObservation& obs,
                                  const Eigen::Vector3d& point3D) {
  const double sq_err = CalculateSquaredReprojectionError(
      obs.img_point, point3D, obs.cam_from_world_mat, *obs.camera);
  if (sq_err >= std::numeric_limits<double>::max() * 0.5) {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(sq_err);
}

std::vector<size_t> SelectHypothesisIndices(const size_t total_views,
                                            const size_t max_views) {
  std::vector<size_t> indices;
  if (total_views <= max_views) {
    indices.resize(total_views);
    for (size_t i = 0; i < total_views; ++i) {
      indices[i] = i;
    }
    return indices;
  }
  indices.reserve(max_views);
  for (size_t k = 0; k < max_views; ++k) {
    const size_t idx = (k * (total_views - 1)) / (max_views - 1);
    if (indices.empty() || idx != indices.back()) {
      indices.push_back(idx);
    }
  }
  return indices;
}

void TriangulateSingleTag(const int tag_id,
                          const std::vector<RayObservation>& observations,
                          const AprilTagTriangulationOptions& options,
                          std::vector<TriangulatedAprilTag>* triangulated_tags,
                          std::vector<RejectedAprilTag>* rejected_tags) {
  auto reject = [&](const std::string& reason) {
    if (rejected_tags != nullptr) {
      RejectedAprilTag rejected;
      rejected.tag_id = tag_id;
      rejected.num_observations = observations.size();
      rejected.reason = reason;
      rejected_tags->push_back(std::move(rejected));
    }
  };

  const size_t min_views = static_cast<size_t>(options.min_views);
  if (observations.size() < min_views) {
    reject("insufficient_views");
    return;
  }

  const double min_parallax_rad = DegToRad(options.min_parallax_angle_deg);
  const std::vector<size_t> precheck_indices =
      SelectHypothesisIndices(observations.size(), /*max_views=*/256);
  double max_ray_angle_rad = 0.0;
  std::pair<size_t, size_t> max_angle_pair{0, 1};
  for (size_t a = 0; a < precheck_indices.size(); ++a) {
    const size_t i = precheck_indices[a];
    for (size_t b = a + 1; b < precheck_indices.size(); ++b) {
      const size_t j = precheck_indices[b];
      const double angle = CalculateAngleBetweenVectors(
          observations[i].world_ray, observations[j].world_ray);
      if (angle > max_ray_angle_rad) {
        max_ray_angle_rad = angle;
        max_angle_pair = {i, j};
      }
    }
  }

  if (max_ray_angle_rad < min_parallax_rad) {
    reject("insufficient_parallax");
    return;
  }

  std::vector<size_t> active;
  if (observations.size() == 2) {
    active = {0, 1};
  } else {
    constexpr size_t kMaxHypothesisViews = 64;
    const std::vector<size_t> hyp_indices =
        SelectHypothesisIndices(observations.size(), kMaxHypothesisViews);
    std::vector<std::pair<size_t, size_t>> candidate_pairs;
    for (size_t a = 0; a < hyp_indices.size(); ++a) {
      for (size_t b = a + 1; b < hyp_indices.size(); ++b) {
        candidate_pairs.emplace_back(hyp_indices[a], hyp_indices[b]);
      }
    }
    if (observations.size() > kMaxHypothesisViews) {
      candidate_pairs.push_back(max_angle_pair);
    }

    const double consensus_thresh_px = 1.5 * options.max_reprojection_error_px;
    size_t best_inlier_count = 0;
    double best_error_sum = std::numeric_limits<double>::infinity();
    std::vector<size_t> best_inliers;

    std::vector<size_t> current_inliers;
    current_inliers.reserve(observations.size());
    for (const auto& [i, j] : candidate_pairs) {
      const double ray_angle = CalculateAngleBetweenVectors(
          observations[i].world_ray, observations[j].world_ray);
      if (ray_angle < min_parallax_rad) {
        continue;
      }

      Eigen::Vector3d candidate_xyz;
      if (!TriangulateRays(observations, {i, j}, &candidate_xyz)) {
        continue;
      }

      current_inliers.clear();
      double error_sum = 0.0;
      for (size_t k = 0; k < observations.size(); ++k) {
        if (!HasPositiveCheirality(observations[k], candidate_xyz)) {
          continue;
        }
        const double err =
            ComputeReprojectionErrorPx(observations[k], candidate_xyz);
        if (err <= consensus_thresh_px) {
          current_inliers.push_back(k);
          error_sum += err;
        }
      }

      if (current_inliers.size() > best_inlier_count ||
          (current_inliers.size() == best_inlier_count &&
           error_sum < best_error_sum)) {
        best_inlier_count = current_inliers.size();
        best_error_sum = error_sum;
        best_inliers = current_inliers;
      }
    }

    if (best_inliers.size() < min_views) {
      reject("consensus_failed");
      return;
    }
    active = std::move(best_inliers);
  }

  Eigen::Vector3d point3D = Eigen::Vector3d::Zero();
  while (active.size() >= min_views) {
    if (!TriangulateRays(observations, active, &point3D)) {
      reject("triangulation_degenerate");
      return;
    }

    size_t worst_pos = 0;
    double worst_err = -1.0;
    bool has_invalid = false;
    for (size_t pos = 0; pos < active.size(); ++pos) {
      const RayObservation& obs = observations[active[pos]];
      if (!HasPositiveCheirality(obs, point3D)) {
        worst_pos = pos;
        worst_err = std::numeric_limits<double>::infinity();
        has_invalid = true;
        break;
      }
      const double err = ComputeReprojectionErrorPx(obs, point3D);
      if (!std::isfinite(err)) {
        worst_pos = pos;
        worst_err = std::numeric_limits<double>::infinity();
        has_invalid = true;
        break;
      }
      if (err > worst_err) {
        worst_err = err;
        worst_pos = pos;
      }
    }

    if (!has_invalid && worst_err <= options.max_reprojection_error_px) {
      break;
    }
    active.erase(active.begin() + static_cast<std::ptrdiff_t>(worst_pos));
  }

  if (active.size() < min_views) {
    reject("too_few_inliers_after_pruning");
    return;
  }

  double max_inlier_parallax_rad = 0.0;
  for (size_t a = 0; a < active.size(); ++a) {
    for (size_t b = a + 1; b < active.size(); ++b) {
      const double angle = CalculateTriangulationAngle(
          observations[active[a]].proj_center,
          observations[active[b]].proj_center,
          point3D);
      max_inlier_parallax_rad = std::max(max_inlier_parallax_rad, angle);
    }
  }
  const double max_inlier_parallax_deg = RadToDeg(max_inlier_parallax_rad);
  if (max_inlier_parallax_deg < options.min_parallax_angle_deg) {
    reject("insufficient_inlier_parallax");
    return;
  }

  double err_sum = 0.0;
  double max_err = 0.0;
  for (const size_t idx : active) {
    const double err = ComputeReprojectionErrorPx(observations[idx], point3D);
    err_sum += err;
    max_err = std::max(max_err, err);
  }

  TriangulatedAprilTag result;
  result.tag_id = tag_id;
  result.xyz = point3D;
  result.num_observations = observations.size();
  result.num_inliers = active.size();
  result.mean_reprojection_error_px =
      err_sum / static_cast<double>(active.size());
  result.max_reprojection_error_px = max_err;
  result.max_parallax_deg = max_inlier_parallax_deg;
  triangulated_tags->push_back(result);
}

struct DistanceCandidate {
  double measured_distance = 0.0;
  double target_distance = 0.0;
  double scale = 0.0;
  double radius = 0.0;
};

double ComputeInlierScore(const std::vector<DistanceCandidate>& candidates,
                          const double scale,
                          std::vector<size_t>* inlier_indices,
                          double* rel_error_sum) {
  inlier_indices->clear();
  *rel_error_sum = 0.0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    const DistanceCandidate& c = candidates[i];
    if (std::abs(scale - c.scale) <= c.radius + 1e-12) {
      inlier_indices->push_back(i);
      const double rel_err =
          std::abs(scale * c.measured_distance - c.target_distance) /
          c.target_distance;
      *rel_error_sum += rel_err;
    }
  }
  return static_cast<double>(inlier_indices->size());
}

bool SolveWeightedLeastSquaresScale(
    const std::vector<DistanceCandidate>& candidates,
    const std::vector<size_t>& indices,
    double* scale) {
  if (indices.empty()) {
    return false;
  }
  double num = 0.0;
  double den = 0.0;
  for (const size_t idx : indices) {
    const double w =
        candidates[idx].measured_distance / candidates[idx].target_distance;
    num += w;
    den += w * w;
  }
  if (!(den > 0.0) || !std::isfinite(num / den) || (num / den) <= 0.0) {
    return false;
  }
  *scale = num / den;
  return true;
}

std::string EscapeJsonString(const std::string& input) {
  std::ostringstream out;
  for (const unsigned char ch : input) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (ch < 0x20) {
          out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(ch) << std::dec;
        } else {
          out << static_cast<char>(ch);
        }
        break;
    }
  }
  return out.str();
}

void StripInlineComment(std::string* line) {
  const size_t hash_pos = line->find('#');
  if (hash_pos != std::string::npos) {
    line->erase(hash_pos);
  }
  StringTrim(line);
}

}  // namespace

bool AprilTagTriangulationOptions::Check() const {
  CHECK_OPTION_GE(min_views, 2);
  CHECK_OPTION_GE(min_parallax_angle_deg, 0.0);
  CHECK_OPTION_LT(min_parallax_angle_deg, 180.0);
  CHECK_OPTION_GT(max_reprojection_error_px, 0.0);
  return true;
}

bool AprilTagScaleAlignmentOptions::Check() const {
  CHECK_OPTION_GE(min_inliers, 1);
  CHECK_OPTION_GE(min_inlier_ratio, 0.0);
  CHECK_OPTION_LE(min_inlier_ratio, 1.0);
  CHECK_OPTION_GE(max_relative_error, 0.0);
  CHECK_OPTION_GE(max_absolute_error, 0.0);
  CHECK_OPTION(max_relative_error > 0.0 || max_absolute_error > 0.0);
  return true;
}

bool AprilTagPositionAlignmentOptions::Check() const {
  CHECK_OPTION_GE(min_common_tags, 3);
  CHECK_OPTION_GE(min_inlier_ratio, 0.0);
  CHECK_OPTION_LE(min_inlier_ratio, 1.0);
  CHECK_OPTION_GT(ransac_options.max_error, 0.0);
  ransac_options.Check();
  return true;
}

std::vector<TagDistanceConstraint> ReadTagDistanceConstraints(
    const std::filesystem::path& path) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);

  std::vector<TagDistanceConstraint> constraints;
  std::string line;
  size_t line_num = 0;
  while (std::getline(file, line)) {
    ++line_num;
    StripInlineComment(&line);
    if (line.empty()) {
      continue;
    }

    std::istringstream parser(line);
    SetFullPrecTextStream(parser);
    TagDistanceConstraint constraint;
    THROW_CHECK(parser >> constraint.tag_id_a >> constraint.tag_id_b >>
                constraint.target_distance)
        << "Invalid tag distance constraint at line " << line_num << ": "
        << line;
    std::string trailing;
    THROW_CHECK(!(parser >> trailing))
        << "Unexpected trailing token '" << trailing << "' at line "
        << line_num;
    THROW_CHECK_NE(constraint.tag_id_a, constraint.tag_id_b)
        << "Distance constraint requires distinct tag IDs at line " << line_num;
    THROW_CHECK(std::isfinite(constraint.target_distance) &&
                constraint.target_distance > 0.0)
        << "Target distance must be finite and positive at line " << line_num;
    constraints.push_back(constraint);
  }
  return constraints;
}

std::vector<TagPositionConstraint> ReadTagPositionConstraints(
    const std::filesystem::path& path) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);

  std::vector<TagPositionConstraint> constraints;
  FlatHashSet<int> seen_tag_ids;
  std::string line;
  size_t line_num = 0;
  while (std::getline(file, line)) {
    ++line_num;
    StripInlineComment(&line);
    if (line.empty()) {
      continue;
    }

    std::istringstream parser(line);
    SetFullPrecTextStream(parser);
    TagPositionConstraint constraint;
    THROW_CHECK(parser >> constraint.tag_id >> constraint.position.x() >>
                constraint.position.y() >> constraint.position.z())
        << "Invalid tag position constraint at line " << line_num << ": "
        << line;
    std::string trailing;
    THROW_CHECK(!(parser >> trailing))
        << "Unexpected trailing token '" << trailing << "' at line "
        << line_num;
    THROW_CHECK(constraint.position.allFinite())
        << "Tag position must be finite at line " << line_num;
    THROW_CHECK(seen_tag_ids.insert(constraint.tag_id).second)
        << "Duplicate tag ID " << constraint.tag_id << " at line " << line_num;
    constraints.push_back(constraint);
  }
  return constraints;
}

std::vector<AprilTagImageObservation> DetectAprilTagsInReconstruction(
    const Reconstruction& reconstruction,
    const std::filesystem::path& image_path,
    const AprilTagDetectionOptions& options,
    size_t* num_images_processed,
    size_t* num_images_missing) {
  THROW_CHECK(options.Check());

  std::vector<image_t> reg_image_ids = reconstruction.RegImageIds();
  std::sort(reg_image_ids.begin(), reg_image_ids.end());

  if (num_images_processed != nullptr) {
    *num_images_processed = 0;
  }
  if (num_images_missing != nullptr) {
    *num_images_missing = 0;
  }
  if (reg_image_ids.empty()) {
    return {};
  }

  AprilTagDetectionOptions worker_options = options;
  worker_options.num_threads = 1;

  ThreadPool thread_pool(GetEffectiveNumThreads(options.num_threads));
  std::vector<std::unique_ptr<AprilTagDetector>> detectors(
      thread_pool.NumThreads());

  struct PerImageResult {
    bool read_ok = false;
    std::vector<AprilTagImageObservation> observations;
  };
  std::vector<PerImageResult> per_image_results(reg_image_ids.size());

  for (size_t idx = 0; idx < reg_image_ids.size(); ++idx) {
    thread_pool.AddTask([&, idx]() {
      const int thread_idx = thread_pool.GetThreadIndex();
      std::unique_ptr<AprilTagDetector>& detector = detectors.at(thread_idx);
      if (detector == nullptr) {
        detector = std::make_unique<AprilTagDetector>(worker_options);
      }

      const image_t image_id = reg_image_ids[idx];
      const Image& image = reconstruction.Image(image_id);
      const Camera& camera = *image.CameraPtr();

      Bitmap bitmap;
      if (!bitmap.Read(image_path / image.Name(), /*as_rgb=*/false)) {
        LOG(WARNING) << "Failed to read image for AprilTag detection: "
                     << (image_path / image.Name());
        return;
      }

      PerImageResult& res = per_image_results[idx];
      res.read_ok = true;

      if (bitmap.Width() <= 0 || bitmap.Height() <= 0 || camera.width == 0 ||
          camera.height == 0) {
        return;
      }

      const std::vector<AprilTagDetection> detections =
          detector->Detect(bitmap);
      if (detections.empty()) {
        return;
      }

      const double scale_x = static_cast<double>(camera.width) /
                             static_cast<double>(bitmap.Width());
      const double scale_y = static_cast<double>(camera.height) /
                             static_cast<double>(bitmap.Height());

      // Keep only the strongest detection per tag_id within each image.
      FlatHashMap<int, AprilTagDetection> best_per_tag;
      for (const AprilTagDetection& det : detections) {
        auto [it, inserted] = best_per_tag.emplace(det.id, det);
        if (!inserted) {
          const bool better =
              (det.decision_margin > it->second.decision_margin) ||
              (det.decision_margin == it->second.decision_margin &&
               det.hamming < it->second.hamming);
          if (better) {
            it->second = det;
          }
        }
      }

      res.observations.reserve(best_per_tag.size());
      for (const auto& [tag_id, det] : best_per_tag) {
        AprilTagImageObservation obs;
        obs.image_id = image_id;
        obs.image_name = image.Name();
        obs.tag_id = tag_id;
        obs.center_px =
            Eigen::Vector2d(det.center.x() * scale_x, det.center.y() * scale_y);
        obs.decision_margin = det.decision_margin;
        obs.hamming = det.hamming;
        res.observations.push_back(std::move(obs));
      }
    });
  }
  thread_pool.Wait();

  std::vector<AprilTagImageObservation> all_observations;
  size_t processed = 0;
  size_t missing = 0;
  for (PerImageResult& res : per_image_results) {
    if (res.read_ok) {
      ++processed;
    } else {
      ++missing;
    }
    all_observations.insert(all_observations.end(),
                            std::make_move_iterator(res.observations.begin()),
                            std::make_move_iterator(res.observations.end()));
  }

  std::sort(all_observations.begin(),
            all_observations.end(),
            [](const AprilTagImageObservation& a,
               const AprilTagImageObservation& b) {
              if (a.tag_id != b.tag_id) {
                return a.tag_id < b.tag_id;
              }
              return a.image_id < b.image_id;
            });

  if (num_images_processed != nullptr) {
    *num_images_processed = processed;
  }
  if (num_images_missing != nullptr) {
    *num_images_missing = missing;
  }

  return all_observations;
}

void TriangulateAprilTags(
    const Reconstruction& reconstruction,
    const std::vector<AprilTagImageObservation>& observations,
    const AprilTagTriangulationOptions& options,
    std::vector<TriangulatedAprilTag>* triangulated_tags,
    std::vector<RejectedAprilTag>* rejected_tags) {
  THROW_CHECK(options.Check());
  THROW_CHECK_NOTNULL(triangulated_tags);
  triangulated_tags->clear();
  if (rejected_tags != nullptr) {
    rejected_tags->clear();
  }

  // Group observations deterministically by tag_id.
  std::map<int, std::vector<RayObservation>> grouped;
  for (const AprilTagImageObservation& obs : observations) {
    if (!reconstruction.ExistsImage(obs.image_id)) {
      continue;
    }
    const Image& image = reconstruction.Image(obs.image_id);
    if (!image.HasPose()) {
      continue;
    }
    const Camera* camera = image.CameraPtr();
    if (camera == nullptr) {
      continue;
    }
    const std::optional<Eigen::Vector3d> cam_ray =
        camera->CamRayFromImg(obs.center_px);
    if (!cam_ray.has_value()) {
      continue;
    }

    RayObservation ray_obs;
    ray_obs.image_id = obs.image_id;
    ray_obs.camera = camera;
    ray_obs.cam_from_world_mat = image.CamFromWorld().ToMatrix();
    ray_obs.proj_center = image.ProjectionCenter();
    ray_obs.cam_ray = *cam_ray;
    ray_obs.world_ray =
        (image.CamFromWorld().rotation().inverse() * (*cam_ray)).normalized();
    ray_obs.img_point = obs.center_px;
    grouped[obs.tag_id].push_back(ray_obs);
  }

  for (auto& [tag_id, tag_obs] : grouped) {
    std::sort(tag_obs.begin(),
              tag_obs.end(),
              [](const RayObservation& a, const RayObservation& b) {
                return a.image_id < b.image_id;
              });
    TriangulateSingleTag(
        tag_id, tag_obs, options, triangulated_tags, rejected_tags);
  }
}

bool EstimateAprilTagScale(
    const std::vector<TriangulatedAprilTag>& triangulated_tags,
    const std::vector<TagDistanceConstraint>& constraints,
    const AprilTagScaleAlignmentOptions& options,
    Sim3d* tgt_from_src,
    AprilTagAlignmentErrorMetrics* metrics,
    std::vector<TagDistanceResidual>* residuals) {
  THROW_CHECK(options.Check());
  THROW_CHECK_NOTNULL(tgt_from_src);

  if (residuals != nullptr) {
    residuals->clear();
  }
  if (metrics != nullptr) {
    *metrics = AprilTagAlignmentErrorMetrics{};
    metrics->num_constraints = constraints.size();
  }

  FlatHashMap<int, Eigen::Vector3d> tag_positions;
  tag_positions.reserve(triangulated_tags.size());
  for (const TriangulatedAprilTag& tag : triangulated_tags) {
    tag_positions.emplace(tag.tag_id, tag.xyz);
  }

  constexpr double kMinMeasuredDistance = 1e-12;
  std::vector<DistanceCandidate> candidates;
  std::vector<size_t> candidate_to_constraint;
  candidates.reserve(constraints.size());
  candidate_to_constraint.reserve(constraints.size());

  for (size_t i = 0; i < constraints.size(); ++i) {
    const TagDistanceConstraint& c = constraints[i];
    if (c.target_distance <= 0.0 || !std::isfinite(c.target_distance)) {
      continue;
    }
    const auto it_a = tag_positions.find(c.tag_id_a);
    const auto it_b = tag_positions.find(c.tag_id_b);
    if (it_a == tag_positions.end() || it_b == tag_positions.end()) {
      continue;
    }
    const double measured_dist = (it_b->second - it_a->second).norm();
    if (!std::isfinite(measured_dist) ||
        measured_dist <= kMinMeasuredDistance) {
      continue;
    }

    DistanceCandidate cand;
    cand.measured_distance = measured_dist;
    cand.target_distance = c.target_distance;
    cand.scale = c.target_distance / measured_dist;
    cand.radius = std::max(options.max_relative_error * cand.scale,
                           options.max_absolute_error / measured_dist);
    candidates.push_back(cand);
    candidate_to_constraint.push_back(i);
  }

  if (metrics != nullptr) {
    metrics->num_evaluable = candidates.size();
  }
  if (candidates.size() < static_cast<size_t>(options.min_inliers)) {
    return false;
  }

  // Exact 1D interval sweep-line to find the scale region with maximum overlap.
  struct SweepEvent {
    double x = 0.0;
    int delta = 0;
  };
  std::vector<SweepEvent> events;
  events.reserve(candidates.size() * 2);
  std::vector<double> candidate_scales;
  candidate_scales.reserve(candidates.size());

  for (const DistanceCandidate& c : candidates) {
    const double left = std::max(1e-12, c.scale - c.radius);
    const double right = c.scale + c.radius;
    events.push_back({left, 1});
    events.push_back({right, -1});
    candidate_scales.push_back(c.scale);
  }

  std::sort(events.begin(),
            events.end(),
            [](const SweepEvent& a, const SweepEvent& b) {
              if (a.x != b.x) {
                return a.x < b.x;
              }
              return a.delta > b.delta;
            });

  struct Segment {
    double left = 0.0;
    double right = 0.0;
  };
  int active = 0;
  int best_overlap = 0;
  bool in_best_segment = false;
  double current_left = 0.0;
  std::vector<Segment> best_segments;

  for (size_t i = 0; i < events.size();) {
    const double x = events[i].x;
    while (i < events.size() && events[i].x == x && events[i].delta > 0) {
      active += events[i].delta;
      ++i;
    }

    if (active > best_overlap) {
      best_overlap = active;
      best_segments.clear();
      current_left = x;
      in_best_segment = true;
    } else if (active == best_overlap && active > 0 && !in_best_segment) {
      current_left = x;
      in_best_segment = true;
    }

    while (i < events.size() && events[i].x == x && events[i].delta < 0) {
      active += events[i].delta;
      ++i;
    }

    if (in_best_segment && active < best_overlap) {
      best_segments.push_back({current_left, x});
      in_best_segment = false;
    }
  }

  if (best_overlap < options.min_inliers || best_segments.empty()) {
    return false;
  }

  const double median_scale = Median(candidate_scales);
  double best_mid_scale = 0.0;
  double best_median_dist = std::numeric_limits<double>::infinity();
  double best_mid_rel_err_sum = std::numeric_limits<double>::infinity();
  std::vector<size_t> inliers;

  for (const Segment& seg : best_segments) {
    const double mid = 0.5 * (seg.left + seg.right);
    double rel_err_sum = 0.0;
    ComputeInlierScore(candidates, mid, &inliers, &rel_err_sum);
    const double dist_to_median = std::abs(mid - median_scale);
    if (dist_to_median < best_median_dist ||
        (dist_to_median == best_median_dist &&
         rel_err_sum < best_mid_rel_err_sum)) {
      best_median_dist = dist_to_median;
      best_mid_rel_err_sum = rel_err_sum;
      best_mid_scale = mid;
    }
  }

  // Two-pass weighted least squares refinement minimizing sum ((s*d - D)/D)^2.
  double refined_scale = best_mid_scale;
  double unused_err_sum = 0.0;
  ComputeInlierScore(candidates, refined_scale, &inliers, &unused_err_sum);
  const size_t initial_inlier_count = inliers.size();

  double pass1_scale = refined_scale;
  if (SolveWeightedLeastSquaresScale(candidates, inliers, &pass1_scale)) {
    std::vector<size_t> pass1_inliers;
    ComputeInlierScore(
        candidates, pass1_scale, &pass1_inliers, &unused_err_sum);
    if (pass1_inliers.size() >= initial_inlier_count) {
      refined_scale = pass1_scale;
      inliers = std::move(pass1_inliers);

      double pass2_scale = refined_scale;
      if (SolveWeightedLeastSquaresScale(candidates, inliers, &pass2_scale)) {
        std::vector<size_t> pass2_inliers;
        ComputeInlierScore(
            candidates, pass2_scale, &pass2_inliers, &unused_err_sum);
        if (pass2_inliers.size() >= inliers.size()) {
          refined_scale = pass2_scale;
          inliers = std::move(pass2_inliers);
        }
      }
    }
  }

  const size_t num_inliers = inliers.size();
  const double inlier_ratio =
      static_cast<double>(num_inliers) / static_cast<double>(candidates.size());
  if (num_inliers < static_cast<size_t>(options.min_inliers) ||
      inlier_ratio < options.min_inlier_ratio) {
    return false;
  }

  *tgt_from_src = Sim3d(
      refined_scale, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());

  FlatHashSet<size_t> inlier_set(inliers.begin(), inliers.end());
  double sq_err_sum = 0.0;
  double abs_err_sum = 0.0;
  double max_err = 0.0;

  if (residuals != nullptr) {
    residuals->reserve(candidates.size());
  }
  for (size_t i = 0; i < candidates.size(); ++i) {
    const DistanceCandidate& c = candidates[i];
    const TagDistanceConstraint& src_c =
        constraints[candidate_to_constraint[i]];
    const double scaled_dist = refined_scale * c.measured_distance;
    const double abs_err = std::abs(scaled_dist - c.target_distance);
    const double rel_err = abs_err / c.target_distance;
    const bool is_inlier = inlier_set.count(i) > 0;
    if (is_inlier) {
      sq_err_sum += abs_err * abs_err;
      abs_err_sum += abs_err;
      max_err = std::max(max_err, abs_err);
    }
    if (residuals != nullptr) {
      TagDistanceResidual r;
      r.tag_id_a = src_c.tag_id_a;
      r.tag_id_b = src_c.tag_id_b;
      r.target_distance = c.target_distance;
      r.measured_distance = c.measured_distance;
      r.scaled_distance = scaled_dist;
      r.error = abs_err;
      r.relative_error = rel_err;
      r.inlier = is_inlier;
      residuals->push_back(r);
    }
  }

  if (metrics != nullptr) {
    metrics->num_inliers = num_inliers;
    metrics->inlier_ratio = inlier_ratio;
    metrics->rmse = std::sqrt(sq_err_sum / static_cast<double>(num_inliers));
    metrics->mae = abs_err_sum / static_cast<double>(num_inliers);
    metrics->max_error = max_err;
  }

  return true;
}

bool EstimateAprilTagPositionAlignment(
    const std::vector<TriangulatedAprilTag>& triangulated_tags,
    const std::vector<TagPositionConstraint>& constraints,
    const AprilTagPositionAlignmentOptions& options,
    Sim3d* tgt_from_src,
    AprilTagAlignmentErrorMetrics* metrics,
    std::vector<TagPositionResidual>* residuals) {
  THROW_CHECK(options.Check());
  THROW_CHECK_NOTNULL(tgt_from_src);

  if (residuals != nullptr) {
    residuals->clear();
  }
  if (metrics != nullptr) {
    *metrics = AprilTagAlignmentErrorMetrics{};
    metrics->num_constraints = constraints.size();
  }

  FlatHashMap<int, Eigen::Vector3d> tag_positions;
  tag_positions.reserve(triangulated_tags.size());
  for (const TriangulatedAprilTag& tag : triangulated_tags) {
    tag_positions.emplace(tag.tag_id, tag.xyz);
  }

  std::vector<int> matched_tag_ids;
  std::vector<Eigen::Vector3d> src_xyz;
  std::vector<Eigen::Vector3d> tgt_xyz;
  matched_tag_ids.reserve(constraints.size());
  src_xyz.reserve(constraints.size());
  tgt_xyz.reserve(constraints.size());

  for (const TagPositionConstraint& c : constraints) {
    const auto it = tag_positions.find(c.tag_id);
    if (it == tag_positions.end()) {
      continue;
    }
    matched_tag_ids.push_back(c.tag_id);
    src_xyz.push_back(it->second);
    tgt_xyz.push_back(c.position);
  }

  if (metrics != nullptr) {
    metrics->num_evaluable = src_xyz.size();
  }
  if (src_xyz.size() < static_cast<size_t>(options.min_common_tags)) {
    return false;
  }

  Sim3d est_tgt_from_src;
  const auto report = EstimateSim3dRobust(
      src_xyz, tgt_xyz, options.ransac_options, est_tgt_from_src);
  if (!report.success) {
    return false;
  }

  const size_t num_inliers = report.support.num_inliers;
  const double inlier_ratio =
      static_cast<double>(num_inliers) / static_cast<double>(src_xyz.size());
  if (num_inliers < static_cast<size_t>(options.min_common_tags) ||
      inlier_ratio < options.min_inlier_ratio) {
    return false;
  }

  *tgt_from_src = est_tgt_from_src;

  double sq_err_sum = 0.0;
  double abs_err_sum = 0.0;
  double max_err = 0.0;
  if (residuals != nullptr) {
    residuals->reserve(src_xyz.size());
  }
  for (size_t i = 0; i < src_xyz.size(); ++i) {
    const Eigen::Vector3d aligned = est_tgt_from_src * src_xyz[i];
    const double err = (aligned - tgt_xyz[i]).norm();
    const bool is_inlier = report.inlier_mask[i] != 0;
    if (is_inlier) {
      sq_err_sum += err * err;
      abs_err_sum += err;
      max_err = std::max(max_err, err);
    }
    if (residuals != nullptr) {
      TagPositionResidual r;
      r.tag_id = matched_tag_ids[i];
      r.source_position = src_xyz[i];
      r.aligned_position = aligned;
      r.target_position = tgt_xyz[i];
      r.error = err;
      r.inlier = is_inlier;
      residuals->push_back(r);
    }
  }

  if (metrics != nullptr) {
    metrics->num_inliers = num_inliers;
    metrics->inlier_ratio = inlier_ratio;
    metrics->rmse = std::sqrt(sq_err_sum / static_cast<double>(num_inliers));
    metrics->mae = abs_err_sum / static_cast<double>(num_inliers);
    metrics->max_error = max_err;
  }

  return true;
}

bool AlignReconstructionWithAprilTagDistances(
    const std::filesystem::path& image_path,
    const std::vector<TagDistanceConstraint>& constraints,
    const AprilTagDetectionOptions& detection_options,
    const AprilTagTriangulationOptions& triangulation_options,
    const AprilTagScaleAlignmentOptions& scale_options,
    Reconstruction* reconstruction,
    AprilTagAlignmentReport* report) {
  THROW_CHECK_NOTNULL(reconstruction);

  AprilTagAlignmentReport local_report;
  local_report.mode = "distances";

  const std::vector<AprilTagImageObservation> observations =
      DetectAprilTagsInReconstruction(*reconstruction,
                                      image_path,
                                      detection_options,
                                      &local_report.num_images_processed,
                                      &local_report.num_images_missing);
  local_report.num_raw_detections = observations.size();

  TriangulateAprilTags(*reconstruction,
                       observations,
                       triangulation_options,
                       &local_report.triangulated_tags,
                       &local_report.rejected_tags);

  Sim3d tgt_from_src;
  const bool ok = EstimateAprilTagScale(local_report.triangulated_tags,
                                        constraints,
                                        scale_options,
                                        &tgt_from_src,
                                        &local_report.metrics,
                                        &local_report.distance_residuals);
  local_report.success = ok;
  local_report.tgt_from_src = tgt_from_src;
  if (ok) {
    reconstruction->Transform(tgt_from_src);
  }

  if (report != nullptr) {
    *report = std::move(local_report);
  }
  return ok;
}

bool AlignReconstructionWithAprilTagPositions(
    const std::filesystem::path& image_path,
    const std::vector<TagPositionConstraint>& constraints,
    const AprilTagDetectionOptions& detection_options,
    const AprilTagTriangulationOptions& triangulation_options,
    const AprilTagPositionAlignmentOptions& position_options,
    Reconstruction* reconstruction,
    AprilTagAlignmentReport* report) {
  THROW_CHECK_NOTNULL(reconstruction);

  AprilTagAlignmentReport local_report;
  local_report.mode = "positions";

  const std::vector<AprilTagImageObservation> observations =
      DetectAprilTagsInReconstruction(*reconstruction,
                                      image_path,
                                      detection_options,
                                      &local_report.num_images_processed,
                                      &local_report.num_images_missing);
  local_report.num_raw_detections = observations.size();

  TriangulateAprilTags(*reconstruction,
                       observations,
                       triangulation_options,
                       &local_report.triangulated_tags,
                       &local_report.rejected_tags);

  Sim3d tgt_from_src;
  const bool ok =
      EstimateAprilTagPositionAlignment(local_report.triangulated_tags,
                                        constraints,
                                        position_options,
                                        &tgt_from_src,
                                        &local_report.metrics,
                                        &local_report.position_residuals);
  local_report.success = ok;
  local_report.tgt_from_src = tgt_from_src;
  if (ok) {
    reconstruction->Transform(tgt_from_src);
  }

  if (report != nullptr) {
    *report = std::move(local_report);
  }
  return ok;
}

void WriteAprilTagAlignmentReportJson(const std::filesystem::path& path,
                                      const AprilTagAlignmentReport& report) {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  SetFullPrecTextStream(file);

  const Eigen::Matrix3x4d mat = report.tgt_from_src.ToMatrix();
  file << "{\n";
  file << "  \"success\": " << (report.success ? "true" : "false") << ",\n";
  file << "  \"mode\": \"" << EscapeJsonString(report.mode) << "\",\n";
  file << "  \"num_images_processed\": " << report.num_images_processed
       << ",\n";
  file << "  \"num_images_missing\": " << report.num_images_missing << ",\n";
  file << "  \"num_raw_detections\": " << report.num_raw_detections << ",\n";
  file << "  \"transform\": {\n";
  file << "    \"scale\": " << report.tgt_from_src.scale() << ",\n";
  file << "    \"rotation_xyzw\": [" << report.tgt_from_src.rotation().x()
       << ", " << report.tgt_from_src.rotation().y() << ", "
       << report.tgt_from_src.rotation().z() << ", "
       << report.tgt_from_src.rotation().w() << "],\n";
  file << "    \"translation\": [" << report.tgt_from_src.translation().x()
       << ", " << report.tgt_from_src.translation().y() << ", "
       << report.tgt_from_src.translation().z() << "],\n";
  file << "    \"matrix_3x4\": [\n";
  for (int r = 0; r < 3; ++r) {
    file << "      [" << mat(r, 0) << ", " << mat(r, 1) << ", " << mat(r, 2)
         << ", " << mat(r, 3) << "]" << (r < 2 ? "," : "") << "\n";
  }
  file << "    ]\n";
  file << "  },\n";
  file << "  \"metrics\": {\n";
  file << "    \"num_constraints\": " << report.metrics.num_constraints
       << ",\n";
  file << "    \"num_evaluable\": " << report.metrics.num_evaluable << ",\n";
  file << "    \"num_inliers\": " << report.metrics.num_inliers << ",\n";
  file << "    \"inlier_ratio\": " << report.metrics.inlier_ratio << ",\n";
  file << "    \"rmse\": " << report.metrics.rmse << ",\n";
  file << "    \"mae\": " << report.metrics.mae << ",\n";
  file << "    \"max_error\": " << report.metrics.max_error << "\n";
  file << "  },\n";

  file << "  \"triangulated_tags\": [\n";
  for (size_t i = 0; i < report.triangulated_tags.size(); ++i) {
    const TriangulatedAprilTag& t = report.triangulated_tags[i];
    file << "    {\n";
    file << "      \"tag_id\": " << t.tag_id << ",\n";
    file << "      \"xyz\": [" << t.xyz.x() << ", " << t.xyz.y() << ", "
         << t.xyz.z() << "],\n";
    file << "      \"num_observations\": " << t.num_observations << ",\n";
    file << "      \"num_inliers\": " << t.num_inliers << ",\n";
    file << "      \"mean_reprojection_error_px\": "
         << t.mean_reprojection_error_px << ",\n";
    file << "      \"max_reprojection_error_px\": "
         << t.max_reprojection_error_px << ",\n";
    file << "      \"max_parallax_deg\": " << t.max_parallax_deg << "\n";
    file << "    }" << (i + 1 < report.triangulated_tags.size() ? "," : "")
         << "\n";
  }
  file << "  ],\n";

  file << "  \"rejected_tags\": [\n";
  for (size_t i = 0; i < report.rejected_tags.size(); ++i) {
    const RejectedAprilTag& r = report.rejected_tags[i];
    file << "    {\n";
    file << "      \"tag_id\": " << r.tag_id << ",\n";
    file << "      \"num_observations\": " << r.num_observations << ",\n";
    file << "      \"reason\": \"" << EscapeJsonString(r.reason) << "\"\n";
    file << "    }" << (i + 1 < report.rejected_tags.size() ? "," : "")
         << "\n";
  }
  file << "  ],\n";

  file << "  \"distance_residuals\": [\n";
  for (size_t i = 0; i < report.distance_residuals.size(); ++i) {
    const TagDistanceResidual& d = report.distance_residuals[i];
    file << "    {\n";
    file << "      \"tag_id_a\": " << d.tag_id_a << ",\n";
    file << "      \"tag_id_b\": " << d.tag_id_b << ",\n";
    file << "      \"target_distance\": " << d.target_distance << ",\n";
    file << "      \"measured_distance\": " << d.measured_distance << ",\n";
    file << "      \"scaled_distance\": " << d.scaled_distance << ",\n";
    file << "      \"error\": " << d.error << ",\n";
    file << "      \"relative_error\": " << d.relative_error << ",\n";
    file << "      \"inlier\": " << (d.inlier ? "true" : "false") << "\n";
    file << "    }" << (i + 1 < report.distance_residuals.size() ? "," : "")
         << "\n";
  }
  file << "  ],\n";

  file << "  \"position_residuals\": [\n";
  for (size_t i = 0; i < report.position_residuals.size(); ++i) {
    const TagPositionResidual& p = report.position_residuals[i];
    file << "    {\n";
    file << "      \"tag_id\": " << p.tag_id << ",\n";
    file << "      \"source_position\": [" << p.source_position.x() << ", "
         << p.source_position.y() << ", " << p.source_position.z() << "],\n";
    file << "      \"aligned_position\": [" << p.aligned_position.x() << ", "
         << p.aligned_position.y() << ", " << p.aligned_position.z() << "],\n";
    file << "      \"target_position\": [" << p.target_position.x() << ", "
         << p.target_position.y() << ", " << p.target_position.z() << "],\n";
    file << "      \"error\": " << p.error << ",\n";
    file << "      \"inlier\": " << (p.inlier ? "true" : "false") << "\n";
    file << "    }" << (i + 1 < report.position_residuals.size() ? "," : "")
         << "\n";
  }
  file << "  ]\n";
  file << "}\n";
}

}  // namespace colmap
