// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/photometric_bundle_adjustment.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/photometric_error.h"
#include "colmap/estimators/cost_functions/reprojection_error.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

#include <ceres/ceres.h>

namespace colmap {

std::vector<int> PhotometricBundleAdjustmentOptions::ParsePyramidLevels()
    const {
  return CSVToVector<int>(pyramid_levels);
}

bool PhotometricBundleAdjustmentOptions::Check() const {
  CHECK_OPTION_GE(patch_size, 2);
  CHECK_OPTION(max_image_size == -1 || max_image_size >= patch_size);
  const std::vector<int> levels = ParsePyramidLevels();
  CHECK_OPTION(!levels.empty());
  for (const int lvl : levels) {
    CHECK_OPTION_GE(lvl, 0);
    CHECK_OPTION_LE(lvl, 16);
  }
  CHECK_OPTION_GE(structure_warmup_max_num_iterations, 0);
  CHECK_OPTION(std::isfinite(min_patch_std));
  CHECK_OPTION_GE(min_patch_std, 0.0);
  CHECK_OPTION_GE(max_num_observations_per_point, 0);
  CHECK_OPTION(
      IsValidCeresLossFunction(loss_function_type, loss_function_scale, 1.0));
  CHECK_OPTION(std::isfinite(geometric_reprojection_weight));
  CHECK_OPTION_GE(geometric_reprojection_weight, 0.0);
  return true;
}

namespace {

enum class LandmarkFilterStatus : uint8_t {
  kOk = 0,
  kInvalid = 1,
  kLowTexture = 2,
};

struct SourceSelectionResult {
  LandmarkFilterStatus status = LandmarkFilterStatus::kInvalid;
  image_t image_id = kInvalidImageId;
};

// Unproject a pixel using Camera::CamFromImg for perspective models or
// Camera::CamRayFromImg for spherical models.
std::optional<Eigen::Vector3d> CameraPixelToRay(const Camera& camera,
                                                const Eigen::Vector2d& xy) {
  if (CameraModelIsPerspective(camera.model_id)) {
    const auto ray2 = camera.CamFromImg(xy);
    if (!ray2.has_value()) {
      return std::nullopt;
    }
    return Eigen::Vector3d(ray2->x(), ray2->y(), 1.0);
  }
  const auto ray3 = camera.CamRayFromImg(xy);
  if (!ray3.has_value()) {
    return std::nullopt;
  }
  return *ray3;
}

// Compute mean horizontal + vertical neighbor spacing (in full-res pixels)
// across a P x P projected patch grid.
double ComputeMeanPatchSpacing(const std::vector<double>& xs,
                               const std::vector<double>& ys,
                               const int patch_size,
                               const double fallback) {
  double spacing_sum = 0.0;
  int spacing_cnt = 0;
  for (int r = 0; r < patch_size; ++r) {
    for (int c = 0; c + 1 < patch_size; ++c) {
      const int i0 = r * patch_size + c;
      const int i1 = i0 + 1;
      spacing_sum += std::hypot(xs[i1] - xs[i0], ys[i1] - ys[i0]);
      ++spacing_cnt;
    }
  }
  for (int r = 0; r + 1 < patch_size; ++r) {
    for (int c = 0; c < patch_size; ++c) {
      const int i0 = r * patch_size + c;
      const int i1 = i0 + patch_size;
      spacing_sum += std::hypot(xs[i1] - xs[i0], ys[i1] - ys[i0]);
      ++spacing_cnt;
    }
  }
  return spacing_cnt > 0 ? (spacing_sum / spacing_cnt) : fallback;
}

// Construct a plane vector n in source camera frame passing through X_in_src
// with unit world normal n_world (pointing from surface toward cameras).
std::optional<Eigen::Vector3d> InitPlaneInSourceCamera(
    const Rigid3d& src_cam_from_world,
    const Eigen::Vector3d& X_world,
    const Eigen::Vector3d& n_world_unit) {
  const Eigen::Vector3d X_src = src_cam_from_world * X_world;
  const double dist = X_src.norm();
  if (!(dist > 1e-8) || !X_src.allFinite()) {
    return std::nullopt;
  }
  const Eigen::Vector3d ray_unit = X_src / dist;
  // Plane normal pointing away from source camera so n^T X_src > 0.
  Eigen::Vector3d n_src_dir = -(src_cam_from_world.rotation() * n_world_unit);
  const double n_norm = n_src_dir.norm();
  if (n_norm < 1e-8) {
    n_src_dir = ray_unit;
  } else {
    n_src_dir /= n_norm;
  }

  // Clamp overly grazing angles (> ~75 deg from viewing ray) toward
  // fronto-parallel so 1 / (n^T X_src) remains well-conditioned.
  constexpr double kMinCosAngle = 0.25;
  if (n_src_dir.dot(ray_unit) < kMinCosAngle) {
    n_src_dir = ray_unit;
  }

  const double denom = n_src_dir.dot(X_src);
  if (!(denom > 1e-8)) {
    return std::nullopt;
  }
  return n_src_dir / denom;
}

// Compute unit average viewing direction (pointing from 3D point toward
// observing cameras) in world frame.
Eigen::Vector3d ComputeAverageViewingDirection(
    const Eigen::Vector3d& X_world,
    const std::vector<std::pair<image_t, point2D_t>>& obs,
    const FlatHashMap<image_t, Eigen::Vector3d>& cam_centers) {
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  for (const auto& [image_id, p2d_idx] : obs) {
    const auto it = cam_centers.find(image_id);
    if (it == cam_centers.end()) {
      continue;
    }
    const Eigen::Vector3d dir = it->second - X_world;
    const double norm = dir.norm();
    if (norm > 1e-12) {
      sum += dir / norm;
    }
  }
  const double sum_norm = sum.norm();
  if (sum_norm > 1e-8) {
    return sum / sum_norm;
  }
  // Degenerate antipodal fallback: use first camera's viewing direction.
  for (const auto& [image_id, p2d_idx] : obs) {
    const auto it = cam_centers.find(image_id);
    if (it != cam_centers.end()) {
      const Eigen::Vector3d dir = it->second - X_world;
      if (dir.norm() > 1e-12) {
        return dir.normalized();
      }
    }
  }
  return Eigen::Vector3d::UnitZ();
}

// Orthonormal tangent basis (u, v) perpendicular to unit vector n.
void BuildTangentBasis(const Eigen::Vector3d& n_unit,
                       Eigen::Vector3d* u,
                       Eigen::Vector3d* v) {
  const Eigen::Vector3d ref = (std::abs(n_unit.x()) < 0.9)
                                  ? Eigen::Vector3d::UnitX()
                                  : Eigen::Vector3d::UnitY();
  *u = n_unit.cross(ref).normalized();
  *v = n_unit.cross(*u).normalized();
}

// Normalize a patch intensity vector to zero mean and unit L2 norm.
// Returns false if ||vals - mean||_2 < min_norm.
bool NormalizePatchVector(const std::vector<double>& vals,
                          const double min_norm,
                          std::vector<double>* out_norm,
                          double* out_l2_norm = nullptr) {
  const Eigen::Index n = static_cast<Eigen::Index>(vals.size());
  out_norm->resize(vals.size());
  const Eigen::Map<const Eigen::VectorXd> vals_vec(vals.data(), n);
  Eigen::Map<Eigen::VectorXd> out_vec(out_norm->data(), n);
  out_vec = vals_vec.array() - vals_vec.mean();
  const double l2 = out_vec.norm();
  if (out_l2_norm != nullptr) {
    *out_l2_norm = l2;
  }
  if (!(l2 > 0.0 && l2 >= min_norm)) {
    return false;
  }
  out_vec /= l2;
  return true;
}

// Select the source image I_k for a 3D point following LSPBA §3.4:
// sample a P x P 3D tangent-plane patch into all observing images, compute
// the IRLS robust mean of normalized patches, and return the view closest to
// the robust mean (avoiding specularities/occlusions).
SourceSelectionResult SelectRobustSourceImage(
    const Reconstruction& reconstruction,
    const FlatHashMap<image_t, GrayscaleImagePyramid>& pyramids,
    const Eigen::Vector3d& X_world,
    const Eigen::Vector3d& n_world_unit,
    const std::vector<std::pair<image_t, point2D_t>>& obs,
    const int patch_size,
    const int req_level,
    const double min_norm,
    const double robust_scale) {
  // Estimate median physical 3D step corresponding to 1 pixel at each view's
  // clamped pyramid level (accounting for any max_image_size base scaling).
  std::vector<double> steps_3d;
  steps_3d.reserve(obs.size());
  for (const auto& [image_id, p2d_idx] : obs) {
    const auto pyr_it = pyramids.find(image_id);
    if (pyr_it == pyramids.end() || pyr_it->second.IsEmpty()) {
      continue;
    }
    const GrayscaleImageLevel& view_lvl = pyr_it->second.Level(
        std::min(req_level, pyr_it->second.NumLevels() - 1));
    const double view_pixel_step =
        0.5 * (1.0 / view_lvl.scale_x + 1.0 / view_lvl.scale_y);

    const Image& img = reconstruction.Image(image_id);
    const Camera& cam = *img.CameraPtr();
    const Eigen::Vector3d X_cam = img.CamFromWorld() * X_world;
    const double depth =
        CameraModelIsPerspective(cam.model_id) ? X_cam.z() : X_cam.norm();
    if (!(depth > 1e-6)) {
      continue;
    }
    const double focal = CameraModelIsPerspective(cam.model_id)
                             ? cam.MeanFocalLength()
                             : static_cast<double>(cam.width) / (2.0 * M_PI);
    if (focal > 1e-6) {
      steps_3d.push_back((depth / focal) * view_pixel_step);
    }
  }
  if (steps_3d.size() < 2) {
    return {LandmarkFilterStatus::kInvalid, kInvalidImageId};
  }
  std::nth_element(
      steps_3d.begin(), steps_3d.begin() + steps_3d.size() / 2, steps_3d.end());
  const double step_3d = steps_3d[steps_3d.size() / 2];

  Eigen::Vector3d u_axis, v_axis;
  BuildTangentBasis(n_world_unit, &u_axis, &v_axis);

  const int M = patch_size * patch_size;
  const double half = 0.5 * (patch_size - 1);
  std::vector<Eigen::Vector3d> plane_pts(M);
  for (int dy = 0; dy < patch_size; ++dy) {
    for (int dx = 0; dx < patch_size; ++dx) {
      plane_pts[dy * patch_size + dx] =
          X_world + ((dx - half) * u_axis + (dy - half) * v_axis) * step_3d;
    }
  }

  struct ViewCandidate {
    image_t image_id = kInvalidImageId;
    std::vector<double> norm_patch;
    double l2_norm = 0.0;
  };
  std::vector<ViewCandidate> candidates;
  candidates.reserve(obs.size());

  std::vector<double> xs(M), ys(M), vals(M);
  int num_interpolated = 0;
  for (const auto& [image_id, p2d_idx] : obs) {
    const auto pyr_it = pyramids.find(image_id);
    if (pyr_it == pyramids.end() || pyr_it->second.IsEmpty()) {
      continue;
    }
    const Image& img = reconstruction.Image(image_id);
    const Camera& cam = *img.CameraPtr();
    const Rigid3d cam_from_world = img.CamFromWorld();

    bool proj_ok = true;
    for (int m = 0; m < M; ++m) {
      const Eigen::Vector3d X_cam = cam_from_world * plane_pts[m];
      const auto xy = cam.ImgFromCam(X_cam);
      if (!xy.has_value()) {
        proj_ok = false;
        break;
      }
      xs[m] = xy->x();
      ys[m] = xy->y();
    }
    if (!proj_ok) {
      continue;
    }

    const GrayscaleImageLevel& clamped_lvl = pyr_it->second.Level(
        std::min(req_level, pyr_it->second.NumLevels() - 1));
    const double fallback_step =
        0.5 * (1.0 / clamped_lvl.scale_x + 1.0 / clamped_lvl.scale_y);
    const double mean_spacing =
        ComputeMeanPatchSpacing(xs, ys, patch_size, fallback_step);
    const int lvl = pyr_it->second.SelectLevelForSpacing(mean_spacing);
    const GrayscaleImageLevel& img_lvl = pyr_it->second.Level(lvl);

    bool interp_ok = true;
    for (int m = 0; m < M; ++m) {
      const auto val = img_lvl.InterpolateBilinear(xs[m], ys[m]);
      if (!val.has_value()) {
        interp_ok = false;
        break;
      }
      vals[m] = *val;
    }
    if (!interp_ok) {
      continue;
    }
    ++num_interpolated;

    ViewCandidate cand;
    cand.image_id = image_id;
    if (NormalizePatchVector(vals, min_norm, &cand.norm_patch, &cand.l2_norm)) {
      candidates.push_back(std::move(cand));
    }
  }

  if (candidates.empty()) {
    const LandmarkFilterStatus reason = num_interpolated > 0
                                            ? LandmarkFilterStatus::kLowTexture
                                            : LandmarkFilterStatus::kInvalid;
    return {reason, kInvalidImageId};
  }
  if (candidates.size() <= 2) {
    // With 1 or 2 valid views, all pairwise distances to the mean are equal;
    // prefer the view with higher patch texture contrast.
    auto best =
        std::max_element(candidates.begin(),
                         candidates.end(),
                         [](const ViewCandidate& a, const ViewCandidate& b) {
                           return a.l2_norm < b.l2_norm;
                         });
    return {LandmarkFilterStatus::kOk, best->image_id};
  }

  // IRLS robust mean with Geman-McClure weights.
  std::vector<double> mean_patch(M, 0.0);
  for (const auto& c : candidates) {
    for (int m = 0; m < M; ++m) mean_patch[m] += c.norm_patch[m];
  }
  double sq = 0.0;
  for (int m = 0; m < M; ++m) sq += mean_patch[m] * mean_patch[m];
  if (sq > 1e-12) {
    const double inv = 1.0 / std::sqrt(sq);
    for (double& v : mean_patch) v *= inv;
  }

  const double c2 = std::max(1e-6, robust_scale * robust_scale);
  constexpr int kIrlsIters = 5;
  for (int iter = 0; iter < kIrlsIters; ++iter) {
    std::vector<double> next_mean(M, 0.0);
    for (const auto& c : candidates) {
      double dist_sq = 0.0;
      for (int m = 0; m < M; ++m) {
        const double d = c.norm_patch[m] - mean_patch[m];
        dist_sq += d * d;
      }
      const double denom = c2 + dist_sq;
      const double w = (c2 * c2) / (denom * denom);
      for (int m = 0; m < M; ++m) {
        next_mean[m] += w * c.norm_patch[m];
      }
    }
    double next_sq = 0.0;
    for (int m = 0; m < M; ++m) next_sq += next_mean[m] * next_mean[m];
    if (next_sq > 1e-12) {
      const double inv = 1.0 / std::sqrt(next_sq);
      for (int m = 0; m < M; ++m) mean_patch[m] = next_mean[m] * inv;
    }
  }

  image_t best_image = candidates[0].image_id;
  double best_dist_sq = std::numeric_limits<double>::infinity();
  for (const auto& c : candidates) {
    double dist_sq = 0.0;
    for (int m = 0; m < M; ++m) {
      const double d = c.norm_patch[m] - mean_patch[m];
      dist_sq += d * d;
    }
    if (dist_sq < best_dist_sq) {
      best_dist_sq = dist_sq;
      best_image = c.image_id;
    }
  }
  return {LandmarkFilterStatus::kOk, best_image};
}

// Persistent per-track state carried across pyramid levels.
struct TrackState {
  point3D_t point3D_id = kInvalidPoint3DId;
  image_t src_image_id = kInvalidImageId;
  Eigen::Vector3d world_normal = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d initial_xyz_in_src = Eigen::Vector3d::Zero();
  bool is_constant = false;
  int last_updated_stage = -1;
  std::vector<std::pair<image_t, point2D_t>> obs;  // (image_id, point2D_idx)
};

// Active per-landmark data for a single pyramid level solve.
struct ActiveLandmark {
  size_t track_idx = 0;
  Eigen::Vector3d plane = Eigen::Vector3d::UnitZ();
  PhotometricPatchData patch;
  std::vector<std::pair<image_t, const GrayscaleImageLevel*>> targets;
};

enum class LevelStatus : uint8_t {
  kContinue = 0,
  kStoppedBeforeJoint = 1,
  kStoppedInJoint = 2,
  kFailed = 3,
};

struct LevelFilterStats {
  int kept = 0;
  int filtered_invalid = 0;
  int filtered_low_texture = 0;
  size_t total_target_obs = 0;
};

struct LevelProblemData {
  std::set<image_t> parameterized_images;
  std::set<camera_t> parameterized_cameras;
  std::vector<double*> plane_blocks;
  std::vector<double*> geo_point_blocks;
  std::vector<double*> camera_side_blocks;
};

// Robust median squared 2D geometric reprojection error [px^2] across all
// track observations, clamped so implied sigma_geo lies in [0.1, 2.0] px.
double ComputeGeometricMedianSqError(const Reconstruction& reconstruction,
                                     const std::vector<TrackState>& tracks) {
  std::vector<double> sq_errors;
  size_t total_obs = 0;
  for (const TrackState& tr : tracks) {
    total_obs += tr.obs.size();
  }
  sq_errors.reserve(total_obs);
  for (const TrackState& tr : tracks) {
    const Eigen::Vector3d& X_w = reconstruction.Point3D(tr.point3D_id).xyz;
    for (const auto& [image_id, p2d_idx] : tr.obs) {
      const Image& img = reconstruction.Image(image_id);
      const Camera& cam = *img.CameraPtr();
      const auto proj = cam.ImgFromCam(img.CamFromWorld() * X_w);
      if (!proj.has_value()) {
        continue;
      }
      double dx = proj->x() - img.Point2D(p2d_idx).xy.x();
      double dy = proj->y() - img.Point2D(p2d_idx).xy.y();
      if (cam.model_id == CameraModelId::kEquirectangular) {
        const double w = cam.params[0];
        dx -= w * std::floor(dx / w + 0.5);
      }
      const double sq = dx * dx + dy * dy;
      if (std::isfinite(sq)) {
        sq_errors.push_back(sq);
      }
    }
  }
  const double kChi2Median2 = 2.0 * std::log(2.0);
  if (sq_errors.empty()) {
    return kChi2Median2;
  }
  std::nth_element(sq_errors.begin(),
                   sq_errors.begin() + sq_errors.size() / 2,
                   sq_errors.end());
  const double median_sq = sq_errors[sq_errors.size() / 2];
  return std::clamp(median_sq, 0.01 * kChi2Median2, 4.0 * kChi2Median2);
}

// Robust median squared LS-NCC block error ||r_photo||^2 across all raw
// (un-robustified, non-barrier) photometric patch blocks.
double ComputePhotometricMedianSqError(ceres::Problem* problem,
                                       const int patch_num_samples,
                                       const int num_threads) {
  ceres::Problem::EvaluateOptions eval_opts;
  eval_opts.apply_loss_function = false;
  eval_opts.num_threads = num_threads;
  std::vector<double> residuals;
  if (!problem->Evaluate(eval_opts, nullptr, &residuals, nullptr, nullptr) ||
      residuals.empty()) {
    return 1e-3;
  }
  const size_t M = static_cast<size_t>(patch_num_samples);
  const size_t num_blocks = residuals.size() / M;
  std::vector<double> sq_norms;
  sq_norms.reserve(num_blocks);
  for (size_t b = 0; b < num_blocks; ++b) {
    double sq = 0.0;
    const double* r = residuals.data() + b * M;
    for (size_t m = 0; m < M; ++m) {
      sq += r[m] * r[m];
    }
    // Exclude OOB/low-contrast penalty barrier blocks (||r||^2 == 4.0).
    if (std::isfinite(sq) && sq < 3.999) {
      sq_norms.push_back(sq);
    }
  }
  if (sq_norms.empty()) {
    return 1e-3;
  }
  std::nth_element(
      sq_norms.begin(), sq_norms.begin() + sq_norms.size() / 2, sq_norms.end());
  return std::max(1e-5, sq_norms[sq_norms.size() / 2]);
}

class PhotometricBundleAdjusterImpl : public BundleAdjuster {
 public:
  PhotometricBundleAdjusterImpl(
      const BundleAdjustmentOptions& ba_options,
      const PhotometricBundleAdjustmentOptions& photometric_options,
      const BundleAdjustmentConfig& config,
      std::filesystem::path image_path,
      std::shared_ptr<const FlatHashMap<image_t, GrayscaleImagePyramid>>
          preloaded_pyramids,
      Reconstruction& reconstruction)
      : BundleAdjuster(ba_options, config),
        photometric_options_(photometric_options),
        image_path_(std::move(image_path)),
        preloaded_pyramids_(std::move(preloaded_pyramids)),
        reconstruction_(reconstruction) {
    THROW_CHECK(options_.Check());
    THROW_CHECK(photometric_options_.Check());
    THROW_CHECK(reconstruction_.IsValid());
  }

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    auto summary = std::make_shared<PhotometricBundleAdjustmentSummary>();
    summary->termination_type = BundleAdjustmentTerminationType::CONVERGENCE;

    const std::vector<int> pyramid_levels =
        photometric_options_.ParsePyramidLevels();
    const int max_requested_level =
        *std::max_element(pyramid_levels.begin(), pyramid_levels.end());
    // Extra pyramid levels above max_requested_level allow mipmap anti-aliasing
    // when target views are farther away than the source view.
    const int num_pyramid_levels = max_requested_level + 3;
    const int num_threads =
        GetEffectiveNumThreads(photometric_options_.num_threads);

    std::set<image_t> active_image_ids(config_.Images().begin(),
                                       config_.Images().end());
    if (active_image_ids.empty()) {
      LOG(WARNING) << "Photometric BA: no images in BundleAdjustmentConfig.";
      return summary;
    }

    // Validate track/config contract before loading images or mutating state.
    std::vector<TrackState> tracks;
    int filtered_short = 0;
    CollectTracks(active_image_ids, &tracks, &filtered_short);

    // 1. Load or reference grayscale image pyramids.
    Timer load_timer;
    load_timer.Start();
    std::shared_ptr<const FlatHashMap<image_t, GrayscaleImagePyramid>>
        pyramids = preloaded_pyramids_;
    if (!pyramids) {
      pyramids =
          LoadImagePyramids(active_image_ids, num_pyramid_levels, num_threads);
    }
    summary->image_load_seconds = load_timer.ElapsedSeconds();
    LOG(INFO) << StringPrintf(
        "Photometric BA: prepared %zu image pyramids (up to %d levels) in "
        "%.2fs",
        pyramids->size(),
        num_pyramid_levels,
        summary->image_load_seconds);

    if (options_.check_if_stopped && options_.check_if_stopped()) {
      summary->termination_type = BundleAdjustmentTerminationType::USER_SUCCESS;
      return summary;
    }

    Timer opt_timer;
    opt_timer.Start();

    // 2. Initialize surface normals from each track's average viewing
    // direction, and compute geometric variance factor + scratch 3D points if
    // VCE geometric regularization is enabled.
    InitializeTrackNormals(active_image_ids, num_threads, &tracks);
    const int total_tracks = static_cast<int>(tracks.size()) + filtered_short;

    const double med_sq_geo =
        photometric_options_.geometric_reprojection_weight > 0.0
            ? ComputeGeometricMedianSqError(reconstruction_, tracks)
            : 1.0;
    std::vector<Eigen::Vector3d> geo_points3D;
    if (photometric_options_.geometric_reprojection_weight > 0.0) {
      geo_points3D.resize(tracks.size());
      for (size_t i = 0; i < tracks.size(); ++i) {
        geo_points3D[i] = reconstruction_.Point3D(tracks[i].point3D_id).xyz;
      }
    }

    // 3. Coarse-to-fine pyramid optimization loop.
    const int M =
        photometric_options_.patch_size * photometric_options_.patch_size;
    const double min_patch_norm =
        photometric_options_.min_patch_std * std::sqrt(static_cast<double>(M));
    int last_num_residuals = 0;
    int last_updated_stage = -1;
    bool stopped_early = false;
    bool failed = false;
    bool geo_regularization_ran = false;

    for (size_t stage_idx = 0; stage_idx < pyramid_levels.size(); ++stage_idx) {
      if (options_.check_if_stopped && options_.check_if_stopped()) {
        summary->termination_type =
            BundleAdjustmentTerminationType::USER_SUCCESS;
        stopped_early = true;
        break;
      }

      const int req_level = pyramid_levels[stage_idx];
      std::vector<ActiveLandmark> active_landmarks;
      LevelFilterStats stats;
      BuildActiveLandmarksForLevel(req_level,
                                   *pyramids,
                                   min_patch_norm,
                                   num_threads,
                                   &tracks,
                                   &active_landmarks,
                                   &stats);

      const int total_filtered = total_tracks - stats.kept;
      const double filtered_pct =
          total_tracks > 0
              ? (100.0 * static_cast<double>(total_filtered) / total_tracks)
              : 0.0;
      LOG(INFO) << StringPrintf(
          "Photometric BA level %d (scale 1/%d): kept %d / %d tracks "
          "(%.1f%% filtered: %d short, %d invalid/oob, %d low-texture "
          "[std < %.2f]), %zu patch residuals across %zu images.",
          req_level,
          1 << req_level,
          stats.kept,
          total_tracks,
          filtered_pct,
          filtered_short,
          stats.filtered_invalid,
          stats.filtered_low_texture,
          photometric_options_.min_patch_std,
          stats.total_target_obs,
          active_image_ids.size());

      summary->num_tracks_total = total_tracks;
      summary->num_tracks_kept = stats.kept;
      summary->num_tracks_filtered_short = filtered_short;
      summary->num_tracks_filtered_invalid = stats.filtered_invalid;
      summary->num_tracks_filtered_low_texture = stats.filtered_low_texture;

      if (active_landmarks.empty()) {
        LOG(WARNING) << "Photometric BA level " << req_level
                     << ": no active landmarks, skipping level.";
        continue;
      }

      bool level_used_geo = false;
      const LevelStatus level_status = RunLevelOptimization(req_level,
                                                            min_patch_norm,
                                                            med_sq_geo,
                                                            num_threads,
                                                            tracks,
                                                            &active_landmarks,
                                                            &geo_points3D,
                                                            summary.get(),
                                                            &last_num_residuals,
                                                            &level_used_geo);
      if (level_status == LevelStatus::kFailed) {
        failed = true;
        break;
      }
      if (level_status == LevelStatus::kStoppedBeforeJoint) {
        stopped_early = true;
        break;
      }
      geo_regularization_ran = geo_regularization_ran || level_used_geo;
      const int stage_num = static_cast<int>(stage_idx);
      UpdatePointsFromPlanes(stage_num, req_level, active_landmarks, &tracks);
      if (level_used_geo && geo_points3D.size() == tracks.size()) {
        // Keep photometrically inactive tracks consistent with the updated
        // camera poses and intrinsics at every stage boundary so tracks
        // recovered at finer levels initialize from up-to-date 3D positions.
        int updated_geo_only = 0;
        for (size_t idx = 0; idx < tracks.size(); ++idx) {
          const TrackState& tr = tracks[idx];
          if (tr.last_updated_stage != stage_num && !tr.is_constant &&
              geo_points3D[idx].allFinite()) {
            reconstruction_.Point3D(tr.point3D_id).xyz = geo_points3D[idx];
            ++updated_geo_only;
          }
        }
        if (updated_geo_only > 0) {
          VLOG(1) << "Photometric BA level " << req_level << ": updated "
                  << updated_geo_only
                  << " photometrically inactive tracks from geometric BA.";
        }
      }
      last_updated_stage = stage_num;
      if (level_status == LevelStatus::kStoppedInJoint) {
        stopped_early = true;
        break;
      }
    }

    if (!failed && !stopped_early && !geo_regularization_ran &&
        last_updated_stage >= 0 && pyramid_levels.size() > 1) {
      RevertCoarseOnlyTracks(tracks, last_updated_stage);
    }

    summary->num_residuals = last_num_residuals;
    summary->optimization_seconds = opt_timer.ElapsedSeconds();
    return summary;
  }

 private:
  bool HasVariablePose(const image_t image_id) const {
    const Image& img = reconstruction_.Image(image_id);
    const bool rfw_var = options_.refine_rig_from_world &&
                         !config_.HasConstantRigFromWorldPose(img.FrameId());
    const bool sfr_var =
        !img.IsRefInFrame() && options_.refine_sensor_from_rig &&
        !config_.HasConstantSensorFromRigPose(img.CameraPtr()->SensorId());
    return rfw_var || sfr_var;
  }

  bool HasVariableIntrinsics(const camera_t camera_id) const {
    if (config_.HasConstantCamIntrinsics(camera_id)) {
      return false;
    }
    const Camera& cam = reconstruction_.Camera(camera_id);
    return (options_.refine_focal_length && !cam.FocalLengthIdxs().empty()) ||
           (options_.refine_principal_point &&
            !cam.PrincipalPointIdxs().empty()) ||
           (options_.refine_extra_params && !cam.ExtraParamsIdxs().empty());
  }

  bool HasAnyVariableCameraSide(const std::vector<TrackState>& tracks) const {
    FlatHashSet<image_t> checked_images;
    FlatHashSet<camera_t> checked_cameras;
    for (const TrackState& tr : tracks) {
      for (const auto& [image_id, p2d_idx] : tr.obs) {
        if (checked_images.insert(image_id).second &&
            HasVariablePose(image_id)) {
          return true;
        }
        const camera_t cam_id = reconstruction_.Image(image_id).CameraId();
        if (checked_cameras.insert(cam_id).second &&
            HasVariableIntrinsics(cam_id)) {
          return true;
        }
      }
    }
    return false;
  }

  std::vector<double*> CollectCameraSideBlocks(
      const std::set<camera_t>& cameras,
      const std::set<image_t>& images,
      const ceres::Problem& problem) {
    std::vector<double*> blocks;
    FlatHashSet<double*> seen;
    for (const camera_t cam_id : cameras) {
      double* p = reconstruction_.Camera(cam_id).params.data();
      if (problem.HasParameterBlock(p) && seen.insert(p).second) {
        blocks.push_back(p);
      }
    }
    for (const image_t img_id : images) {
      Image& img = reconstruction_.Image(img_id);
      double* rfw = img.FramePtr()->RigFromWorld().params.data();
      if (problem.HasParameterBlock(rfw) && seen.insert(rfw).second) {
        blocks.push_back(rfw);
      }
      if (!img.IsRefInFrame()) {
        double* sfr = img.FramePtr()
                          ->RigPtr()
                          ->SensorFromRig(img.CameraPtr()->SensorId())
                          .params.data();
        if (problem.HasParameterBlock(sfr) && seen.insert(sfr).second) {
          blocks.push_back(sfr);
        }
      }
    }
    return blocks;
  }

  void CollectTracks(const std::set<image_t>& active_image_ids,
                     std::vector<TrackState>* tracks,
                     int* filtered_short) const {
    FlatHashMap<point3D_t, TrackState> track_map;
    const int min_track_len = std::max(2, options_.min_track_length);
    for (const image_t image_id : active_image_ids) {
      const Image& image = reconstruction_.Image(image_id);
      const auto& points2D = image.Points2D();
      for (point2D_t p2d_idx = 0; p2d_idx < points2D.size(); ++p2d_idx) {
        const Point2D& p2d = points2D[p2d_idx];
        if (!p2d.HasPoint3D()) {
          continue;
        }
        const point3D_t p3d_id = p2d.point3D_id;
        if (config_.IsIgnoredPoint(p3d_id)) {
          continue;
        }
        auto& st = track_map[p3d_id];
        st.point3D_id = p3d_id;
        st.obs.emplace_back(image_id, p2d_idx);
      }
    }

    for (const point3D_t p3d_id : config_.VariablePoints()) {
      THROW_CHECK(track_map.count(p3d_id) > 0)
          << "Photometric BA does not support VariablePoints() outside "
             "config.Images() (point3D_id="
          << p3d_id << ").";
    }
    for (const point3D_t p3d_id : config_.ConstantPoints()) {
      auto it = track_map.find(p3d_id);
      THROW_CHECK(it != track_map.end())
          << "Photometric BA does not support ConstantPoints() outside "
             "config.Images() (point3D_id="
          << p3d_id << ").";
      it->second.is_constant = true;
    }

    tracks->clear();
    tracks->reserve(track_map.size());
    *filtered_short = 0;
    for (auto& [p3d_id, st] : track_map) {
      // Deduplicate multiple observations of the same 3D point in the same
      // image.
      std::sort(st.obs.begin(), st.obs.end());
      st.obs.erase(std::unique(st.obs.begin(),
                               st.obs.end(),
                               [](const auto& a, const auto& b) {
                                 return a.first == b.first;
                               }),
                   st.obs.end());
      if (static_cast<int>(st.obs.size()) < min_track_len) {
        ++(*filtered_short);
        continue;
      }
      if (!options_.refine_points3D) {
        st.is_constant = true;
      }
      if (st.is_constant) {
        // In LSPBA's source-anchored plane parameterization, holding n_k
        // constant fixes the point in its source camera frame, not world space.
        // Fail loudly before mutating Reconstruction if any observing view of a
        // constant point has a variable pose.
        for (const auto& [obs_img_id, p2d_idx] : st.obs) {
          THROW_CHECK(!HasVariablePose(obs_img_id))
              << "Photometric BA holds constant 3D points fixed in their "
                 "source camera frame; world-fixed constant points with "
                 "variable poses are not supported.";
        }
      }
      tracks->push_back(std::move(st));
    }

    std::sort(tracks->begin(),
              tracks->end(),
              [](const TrackState& a, const TrackState& b) {
                return a.point3D_id < b.point3D_id;
              });
  }

  void InitializeTrackNormals(const std::set<image_t>& active_image_ids,
                              const int num_threads,
                              std::vector<TrackState>* tracks) const {
    FlatHashMap<image_t, Eigen::Vector3d> cam_centers;
    cam_centers.reserve(active_image_ids.size());
    for (const image_t image_id : active_image_ids) {
      cam_centers.emplace(image_id,
                          reconstruction_.Image(image_id).ProjectionCenter());
    }
    ThreadPool pool(num_threads);
    const size_t n_tracks = tracks->size();
    const size_t chunk = std::max<size_t>(
        1, (n_tracks + num_threads * 4 - 1) / (num_threads * 4));
    for (size_t start = 0; start < n_tracks; start += chunk) {
      const size_t end = std::min(n_tracks, start + chunk);
      pool.AddTask([&, start, end]() {
        for (size_t idx = start; idx < end; ++idx) {
          TrackState& tr = (*tracks)[idx];
          const Eigen::Vector3d& X_w =
              reconstruction_.Point3D(tr.point3D_id).xyz;
          tr.world_normal =
              ComputeAverageViewingDirection(X_w, tr.obs, cam_centers);
        }
      });
    }
    pool.Wait();
  }

  LandmarkFilterStatus BuildLandmarkForTrack(
      const size_t idx,
      const int req_level,
      const FlatHashMap<image_t, GrayscaleImagePyramid>& pyramids,
      const double min_patch_norm,
      const double irls_scale,
      TrackState* tr,
      std::vector<double>* src_vals,
      std::vector<Eigen::Vector3d>* pts_tgt,
      std::vector<double>* tgt_xs,
      std::vector<double>* tgt_ys,
      ActiveLandmark* out_lm) const {
    const int P = photometric_options_.patch_size;
    const int M = P * P;
    const double half = 0.5 * (P - 1);
    const Eigen::Vector3d& X_w = reconstruction_.Point3D(tr->point3D_id).xyz;

    // Select (or retry) robust source frame at this level if not yet assigned,
    // so tracks whose coarse patches were OOB or blurred recover at finer
    // levels.
    if (tr->src_image_id == kInvalidImageId) {
      const SourceSelectionResult sel =
          SelectRobustSourceImage(reconstruction_,
                                  pyramids,
                                  X_w,
                                  tr->world_normal,
                                  tr->obs,
                                  P,
                                  req_level,
                                  min_patch_norm,
                                  irls_scale);
      if (sel.status != LandmarkFilterStatus::kOk) {
        return sel.status;
      }
      tr->src_image_id = sel.image_id;
      tr->initial_xyz_in_src =
          reconstruction_.Image(tr->src_image_id).CamFromWorld() * X_w;
    }

    const auto src_pyr_it = pyramids.find(tr->src_image_id);
    if (src_pyr_it == pyramids.end() || src_pyr_it->second.IsEmpty()) {
      return LandmarkFilterStatus::kInvalid;
    }
    const GrayscaleImagePyramid& src_pyr = src_pyr_it->second;
    const int src_lvl_idx = std::min(req_level, src_pyr.NumLevels() - 1);
    const GrayscaleImageLevel& src_lvl = src_pyr.Level(src_lvl_idx);

    const Image& src_img = reconstruction_.Image(tr->src_image_id);
    const Camera& src_cam = *src_img.CameraPtr();
    const Rigid3d src_cfw = src_img.CamFromWorld();

    const Eigen::Vector3d X_src = src_cfw * X_w;
    const auto anchor_opt = src_cam.ImgFromCam(X_src);
    if (!anchor_opt.has_value()) {
      return LandmarkFilterStatus::kInvalid;
    }

    const auto plane_opt =
        InitPlaneInSourceCamera(src_cfw, X_w, tr->world_normal);
    if (!plane_opt.has_value()) {
      return LandmarkFilterStatus::kInvalid;
    }
    const Eigen::Vector3d plane = *plane_opt;

    // Patch step in full-res pixels matches 1 pixel at src_lvl.
    const double step_x = 1.0 / src_lvl.scale_x;
    const double step_y = 1.0 / src_lvl.scale_y;
    const double fallback_step = 0.5 * (step_x + step_y);

    PhotometricPatchData patch;
    patch.anchor_xy = *anchor_opt;
    patch.src_cam_params_init = src_cam.params;
    patch.src_xs.resize(M);
    patch.src_ys.resize(M);
    patch.src_rays_init.resize(M);

    for (int dy = 0; dy < P; ++dy) {
      for (int dx = 0; dx < P; ++dx) {
        const int m = dy * P + dx;
        const double px = patch.anchor_xy.x() + (dx - half) * step_x;
        const double py = patch.anchor_xy.y() + (dy - half) * step_y;
        const auto val = src_lvl.InterpolateBilinear(px, py);
        if (!val.has_value()) {
          return LandmarkFilterStatus::kInvalid;
        }
        (*src_vals)[m] = *val;
        patch.src_xs[m] = px;
        patch.src_ys[m] = py;
        const auto ray_opt = CameraPixelToRay(src_cam, Eigen::Vector2d(px, py));
        if (!ray_opt.has_value() || !(plane.dot(*ray_opt) > 1e-8)) {
          return LandmarkFilterStatus::kInvalid;
        }
        patch.src_rays_init[m] = *ray_opt;
      }
    }

    if (!NormalizePatchVector(*src_vals, min_patch_norm, &patch.src_norm)) {
      return LandmarkFilterStatus::kLowTexture;
    }

    // Collect valid target views and choose each view's mipmap level.
    ActiveLandmark lm;
    lm.track_idx = idx;
    lm.plane = plane;
    lm.patch = std::move(patch);
    lm.targets.reserve(tr->obs.size());

    const int max_obs = photometric_options_.max_num_observations_per_point;
    for (const auto& [tgt_img_id, p2d_idx] : tr->obs) {
      if (tgt_img_id == tr->src_image_id) {
        continue;
      }
      const auto tgt_pyr_it = pyramids.find(tgt_img_id);
      if (tgt_pyr_it == pyramids.end() || tgt_pyr_it->second.IsEmpty()) {
        continue;
      }
      const Image& tgt_img = reconstruction_.Image(tgt_img_id);
      const Camera& tgt_cam = *tgt_img.CameraPtr();
      const Rigid3d tgt_from_src = tgt_img.CamFromWorld() * Inverse(src_cfw);
      const Eigen::Matrix3d H_ji =
          tgt_from_src.rotation().toRotationMatrix() +
          tgt_from_src.translation() * plane.transpose();
      Eigen::Map<Eigen::Matrix3Xd>(pts_tgt->data()->data(), 3, M).noalias() =
          H_ji * Eigen::Map<const Eigen::Matrix3Xd>(
                     lm.patch.src_rays_init.data()->data(), 3, M);
      if (!photometric_internal::ProjectBatch(
              tgt_cam.model_id,
              span<const double>(tgt_cam.params.data(), tgt_cam.params.size()),
              span<const Eigen::Vector3d>(pts_tgt->data(), M),
              span<double>(tgt_xs->data(), M),
              span<double>(tgt_ys->data(), M),
              span<double>(nullptr, 0),
              span<double>(nullptr, 0))) {
        continue;
      }

      int tgt_lvl_idx = std::min(req_level, tgt_pyr_it->second.NumLevels() - 1);
      if (photometric_options_.use_target_mipmap) {
        const double mean_spacing =
            ComputeMeanPatchSpacing(*tgt_xs, *tgt_ys, P, fallback_step);
        tgt_lvl_idx = tgt_pyr_it->second.SelectLevelForSpacing(mean_spacing);
      }
      const GrayscaleImageLevel& tgt_lvl =
          tgt_pyr_it->second.Level(tgt_lvl_idx);

      bool in_bounds = true;
      for (int m = 0; m < M; ++m) {
        const double u = (*tgt_xs)[m] * tgt_lvl.scale_x - 0.5;
        const double v = (*tgt_ys)[m] * tgt_lvl.scale_y - 0.5;
        if (!(u >= 0.0 && u < static_cast<double>(tgt_lvl.bitmap.Width() - 1) &&
              v >= 0.0 &&
              v < static_cast<double>(tgt_lvl.bitmap.Height() - 1))) {
          in_bounds = false;
          break;
        }
      }
      if (!in_bounds) {
        continue;
      }

      lm.targets.emplace_back(tgt_img_id, &tgt_lvl);
      if (max_obs > 0 && static_cast<int>(lm.targets.size()) >= max_obs) {
        break;
      }
    }

    if (lm.targets.empty()) {
      return LandmarkFilterStatus::kInvalid;
    }
    *out_lm = std::move(lm);
    return LandmarkFilterStatus::kOk;
  }

  void BuildActiveLandmarksForLevel(
      const int req_level,
      const FlatHashMap<image_t, GrayscaleImagePyramid>& pyramids,
      const double min_patch_norm,
      const int num_threads,
      std::vector<TrackState>* tracks,
      std::vector<ActiveLandmark>* active_landmarks,
      LevelFilterStats* stats) const {
    const int M =
        photometric_options_.patch_size * photometric_options_.patch_size;
    // Decouple IRLS Geman-McClure source-selection scale from non-GM Ceres
    // losses (which may legitimately have loss_function_scale == 0).
    const double irls_scale = (photometric_options_.loss_function_type ==
                               CeresLossFunctionType::GEMAN_MCCLURE)
                                  ? photometric_options_.loss_function_scale
                                  : 0.5;

    std::vector<ActiveLandmark> landmarks(tracks->size());
    std::vector<LandmarkFilterStatus> landmark_status(
        tracks->size(), LandmarkFilterStatus::kInvalid);

    {
      ThreadPool pool(num_threads);
      const size_t n_tracks = tracks->size();
      const size_t chunk = std::max<size_t>(
          1, (n_tracks + num_threads * 4 - 1) / (num_threads * 4));
      for (size_t start = 0; start < n_tracks; start += chunk) {
        const size_t end = std::min(n_tracks, start + chunk);
        pool.AddTask([&, start, end]() {
          std::vector<double> src_vals(M);
          std::vector<Eigen::Vector3d> pts_tgt(M);
          std::vector<double> tgt_xs(M), tgt_ys(M);
          for (size_t idx = start; idx < end; ++idx) {
            landmark_status[idx] = BuildLandmarkForTrack(idx,
                                                         req_level,
                                                         pyramids,
                                                         min_patch_norm,
                                                         irls_scale,
                                                         &(*tracks)[idx],
                                                         &src_vals,
                                                         &pts_tgt,
                                                         &tgt_xs,
                                                         &tgt_ys,
                                                         &landmarks[idx]);
          }
        });
      }
      pool.Wait();
    }

    active_landmarks->clear();
    active_landmarks->reserve(tracks->size());
    *stats = LevelFilterStats{};
    for (size_t idx = 0; idx < tracks->size(); ++idx) {
      if (landmark_status[idx] == LandmarkFilterStatus::kOk) {
        stats->total_target_obs += landmarks[idx].targets.size();
        active_landmarks->push_back(std::move(landmarks[idx]));
      } else if (landmark_status[idx] == LandmarkFilterStatus::kInvalid) {
        ++stats->filtered_invalid;
      } else {
        ++stats->filtered_low_texture;
      }
    }
    stats->kept = static_cast<int>(active_landmarks->size());
  }

  ceres::Solver::Options ConfigureSolverOptions(
      const ceres::Problem& problem) const {
    ceres::Solver::Options solver_opts =
        options_.ceres->CreateSolverOptions(config_, problem);
    if (solver_opts.function_tolerance == 0.0) {
      solver_opts.function_tolerance = 1e-6;
    }
    if (solver_opts.parameter_tolerance == 0.0) {
      solver_opts.parameter_tolerance = 1e-8;
    }
    if (photometric_options_.num_threads > 0) {
      solver_opts.num_threads =
          (problem.NumResiduals() >=
           options_.ceres->min_num_residuals_for_cpu_multi_threading)
              ? photometric_options_.num_threads
              : 1;
    }
    return solver_opts;
  }

  LevelProblemData BuildLevelProblem(
      const double min_patch_norm,
      const std::vector<TrackState>& tracks,
      std::vector<ActiveLandmark>* active_landmarks,
      ceres::LossFunction* photo_loss,
      ceres::Problem* problem) {
    LevelProblemData data;
    data.plane_blocks.reserve(active_landmarks->size());
    const double min_target_norm = std::max(1e-3, 0.25 * min_patch_norm);

    for (ActiveLandmark& lm : *active_landmarks) {
      const TrackState& tr = tracks[lm.track_idx];
      Image& src_img = reconstruction_.Image(tr.src_image_id);
      Camera& src_cam = *src_img.CameraPtr();
      Frame& src_frame = *src_img.FramePtr();
      const sensor_t src_sensor_id = src_cam.SensorId();
      const bool src_is_ref = src_img.IsRefInFrame();
      double* src_sfr_ptr =
          src_is_ref
              ? nullptr
              : src_frame.RigPtr()->SensorFromRig(src_sensor_id).params.data();

      data.parameterized_images.insert(tr.src_image_id);
      data.parameterized_cameras.insert(src_img.CameraId());

      for (const auto& [tgt_img_id, tgt_lvl] : lm.targets) {
        Image& tgt_img = reconstruction_.Image(tgt_img_id);
        Camera& tgt_cam = *tgt_img.CameraPtr();
        Frame& tgt_frame = *tgt_img.FramePtr();
        const sensor_t tgt_sensor_id = tgt_cam.SensorId();
        const bool tgt_is_ref = tgt_img.IsRefInFrame();
        double* tgt_sfr_ptr = tgt_is_ref ? nullptr
                                         : tgt_frame.RigPtr()
                                               ->SensorFromRig(tgt_sensor_id)
                                               .params.data();

        data.parameterized_images.insert(tgt_img_id);
        data.parameterized_cameras.insert(tgt_img.CameraId());

        const auto layout =
            photometric_internal::SourceTargetBlockLayout::Create(
                src_cam.model_id,
                tgt_cam.model_id,
                src_img.FrameId(),
                tgt_img.FrameId(),
                src_sensor_id,
                tgt_sensor_id,
                src_is_ref,
                tgt_is_ref,
                src_img.CameraId(),
                tgt_img.CameraId());

        std::vector<double*> param_ptrs = layout.BuildParameterPointers(
            lm.plane.data(),
            src_frame.RigFromWorld().params.data(),
            tgt_frame.RigFromWorld().params.data(),
            src_sfr_ptr,
            tgt_sfr_ptr,
            src_cam.params.data(),
            tgt_cam.params.data());

        problem->AddResidualBlock(
            new PhotometricNCCCostFunction(
                lm.patch, *tgt_lvl, layout, min_target_norm),
            photo_loss,
            param_ptrs);
      }

      if (tr.is_constant) {
        problem->SetParameterBlockConstant(lm.plane.data());
      } else {
        data.plane_blocks.push_back(lm.plane.data());
      }
    }

    data.camera_side_blocks = CollectCameraSideBlocks(
        data.parameterized_cameras, data.parameterized_images, *problem);

    ceres_ba_internal::ParameterizeCameras(options_,
                                           config_,
                                           data.parameterized_cameras,
                                           reconstruction_,
                                           *problem);
    ceres_ba_internal::ParameterizeRigsAndFrames(options_,
                                                 config_,
                                                 data.parameterized_images,
                                                 reconstruction_,
                                                 *problem);
    return data;
  }

  // Add full-track geometric reprojection residuals (using COLMAP's standard
  // ReprojErrorCostFunctor / RigReprojErrorCostFunctor on decoupled scratch 3D
  // points) to regularize camera poses and intrinsics across all tracks
  // (including photometrically filtered-out tracks).
  void AddGeometricRegularization(const std::vector<TrackState>& tracks,
                                  std::vector<Eigen::Vector3d>* geo_points3D,
                                  ceres::LossFunction* geo_loss,
                                  LevelProblemData* data,
                                  ceres::Problem* problem) {
    data->geo_point_blocks.reserve(tracks.size());

    for (size_t idx = 0; idx < tracks.size(); ++idx) {
      const TrackState& tr = tracks[idx];
      double* xyz_ptr = (*geo_points3D)[idx].data();
      for (const auto& [image_id, p2d_idx] : tr.obs) {
        Image& img = reconstruction_.Image(image_id);
        Camera& cam = *img.CameraPtr();
        Frame& frame = *img.FramePtr();
        const Eigen::Vector2d& xy = img.Point2D(p2d_idx).xy;

        if (img.IsRefInFrame()) {
          problem->AddResidualBlock(
              CreateCameraCostFunction<ReprojErrorCostFunctor>(cam.model_id,
                                                               xy),
              geo_loss,
              xyz_ptr,
              frame.RigFromWorld().params.data(),
              cam.params.data());
        } else {
          Rigid3d& sensor_from_rig =
              frame.RigPtr()->SensorFromRig(cam.SensorId());
          problem->AddResidualBlock(
              CreateCameraCostFunction<RigReprojErrorCostFunctor>(cam.model_id,
                                                                  xy),
              geo_loss,
              xyz_ptr,
              sensor_from_rig.params.data(),
              frame.RigFromWorld().params.data(),
              cam.params.data());
        }

        data->parameterized_images.insert(image_id);
        data->parameterized_cameras.insert(img.CameraId());
      }

      if (tr.is_constant) {
        problem->SetParameterBlockConstant(xyz_ptr);
      } else {
        data->geo_point_blocks.push_back(xyz_ptr);
      }
    }

    // Re-collect and re-parameterize the full camera/frame/rig sets so
    // newly introduced rig_from_world blocks (e.g. intra-frame-only frames)
    // and rigs whose reference sensor enters via geometric tracks receive
    // consistent manifolds, constant flags, and Schur ordering.
    data->camera_side_blocks = CollectCameraSideBlocks(
        data->parameterized_cameras, data->parameterized_images, *problem);
    for (double* ptr : data->camera_side_blocks) {
      problem->SetParameterBlockVariable(ptr);
    }
    ceres_ba_internal::ParameterizeCameras(options_,
                                           config_,
                                           data->parameterized_cameras,
                                           reconstruction_,
                                           *problem);
    ceres_ba_internal::ParameterizeRigsAndFrames(options_,
                                                 config_,
                                                 data->parameterized_images,
                                                 reconstruction_,
                                                 *problem);
  }

  LevelStatus RunStructureWarmup(
      const int req_level,
      const LevelProblemData& data,
      ceres::Problem* problem,
      PhotometricBundleAdjustmentSummary* summary) const {
    if (!photometric_options_.refine_structure_first ||
        !options_.refine_points3D ||
        photometric_options_.structure_warmup_max_num_iterations <= 0 ||
        data.plane_blocks.empty()) {
      return LevelStatus::kContinue;
    }

    std::vector<double*> temporarily_frozen;
    temporarily_frozen.reserve(data.camera_side_blocks.size());
    for (double* ptr : data.camera_side_blocks) {
      if (!problem->IsParameterBlockConstant(ptr)) {
        problem->SetParameterBlockConstant(ptr);
        temporarily_frozen.push_back(ptr);
      }
    }
    if (temporarily_frozen.empty()) {
      return LevelStatus::kContinue;
    }

    ceres::Solver::Options warmup_opts = ConfigureSolverOptions(*problem);
    warmup_opts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    warmup_opts.max_num_iterations =
        photometric_options_.structure_warmup_max_num_iterations;
    ceres_ba_internal::CancellationCallback cancel_cb(
        options_.check_if_stopped);
    if (options_.check_if_stopped) {
      warmup_opts.callbacks.push_back(&cancel_cb);
    }

    ceres::Solver::Summary warmup_summary;
    ceres::Solve(warmup_opts, problem, &warmup_summary);
    LOG(INFO) << StringPrintf(
        "Photometric BA level %d warmup: %d iters, cost %.4f -> %.4f (%.2fs)",
        req_level,
        static_cast<int>(warmup_summary.iterations.size()) - 1,
        warmup_summary.initial_cost,
        warmup_summary.final_cost,
        warmup_summary.total_time_in_seconds);

    for (double* ptr : temporarily_frozen) {
      problem->SetParameterBlockVariable(ptr);
    }

    if (warmup_summary.termination_type == ceres::USER_SUCCESS ||
        (options_.check_if_stopped && options_.check_if_stopped())) {
      summary->termination_type = BundleAdjustmentTerminationType::USER_SUCCESS;
      return LevelStatus::kStoppedBeforeJoint;
    }
    if (!warmup_summary.IsSolutionUsable()) {
      LOG(ERROR) << "Photometric BA level " << req_level
                 << " warmup failed: " << warmup_summary.message;
      summary->termination_type =
          ceres_ba_internal::CeresTerminationTypeToTerminationType(
              warmup_summary.termination_type);
      return LevelStatus::kFailed;
    }
    return LevelStatus::kContinue;
  }

  LevelStatus RunLevelOptimization(
      const int req_level,
      const double min_patch_norm,
      const double med_sq_geo,
      const int num_threads,
      const std::vector<TrackState>& tracks,
      std::vector<ActiveLandmark>* active_landmarks,
      std::vector<Eigen::Vector3d>* geo_points3D,
      PhotometricBundleAdjustmentSummary* summary,
      int* last_num_residuals,
      bool* level_used_geo) {
    *level_used_geo = false;
    auto photo_loss =
        CreateCeresLossFunction(photometric_options_.loss_function_type,
                                photometric_options_.loss_function_scale,
                                1.0);
    std::unique_ptr<ceres::LossFunction> geo_loss;

    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);

    LevelProblemData data = BuildLevelProblem(
        min_patch_norm, tracks, active_landmarks, photo_loss.get(), &problem);

    const LevelStatus warmup_status =
        RunStructureWarmup(req_level, data, &problem, summary);
    if (warmup_status != LevelStatus::kContinue) {
      return warmup_status;
    }

    // Add VCE-weighted full-track geometric reprojection regularization after
    // structure warmup whenever any track observes a variable pose or camera.
    if (photometric_options_.geometric_reprojection_weight > 0.0 &&
        HasAnyVariableCameraSide(tracks)) {
      const int M =
          photometric_options_.patch_size * photometric_options_.patch_size;
      const double med_sq_photo =
          ComputePhotometricMedianSqError(&problem, M, num_threads);
      const double effective_geo_weight =
          photometric_options_.geometric_reprojection_weight *
          (med_sq_photo / med_sq_geo);
      LOG(INFO) << StringPrintf(
          "Photometric BA level %d VCE: med_geo=%.3f px, med_znssd=%.4f, "
          "geo_weight=%.5f (lambda=%.3f, %zu tracks)",
          req_level,
          std::sqrt(med_sq_geo),
          med_sq_photo,
          effective_geo_weight,
          photometric_options_.geometric_reprojection_weight,
          tracks.size());
      geo_loss = CreateCeresLossFunction(options_.ceres->loss_function_type,
                                         options_.ceres->loss_function_scale,
                                         effective_geo_weight);
      AddGeometricRegularization(
          tracks, geo_points3D, geo_loss.get(), &data, &problem);
      *level_used_geo = true;
    }

    int num_var_structure = 0;
    for (double* p : data.plane_blocks) {
      if (!problem.IsParameterBlockConstant(p)) ++num_var_structure;
    }
    for (double* p : data.geo_point_blocks) {
      if (!problem.IsParameterBlockConstant(p)) ++num_var_structure;
    }
    int num_var_cam_side = 0;
    for (double* p : data.camera_side_blocks) {
      if (!problem.IsParameterBlockConstant(p)) ++num_var_cam_side;
    }
    if (num_var_structure == 0 && num_var_cam_side == 0) {
      return LevelStatus::kContinue;
    }

    ceres::Solver::Options solver_opts = ConfigureSolverOptions(problem);
    if (num_var_structure > 0 && num_var_cam_side > 0) {
      // Explicit Schur ordering: eliminate independent 3x3 planes and geometric
      // 3D points first.
      auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
      for (double* p : data.plane_blocks) {
        ordering->AddElementToGroup(p, 0);
      }
      for (double* p : data.geo_point_blocks) {
        ordering->AddElementToGroup(p, 0);
      }
      for (double* p : data.camera_side_blocks) {
        ordering->AddElementToGroup(p, 1);
      }
      solver_opts.linear_solver_ordering = ordering;
    } else {
      // Structure-only (3x3 block-diagonal) or cameras-only: direct sparse
      // Cholesky avoids Schur overhead.
      solver_opts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }

    ceres_ba_internal::CancellationCallback cancel_cb(
        options_.check_if_stopped);
    if (options_.check_if_stopped) {
      solver_opts.callbacks.push_back(&cancel_cb);
    }

    ceres::Solver::Summary level_summary;
    ceres::Solve(solver_opts, &problem, &level_summary);
    *last_num_residuals = level_summary.num_residuals_reduced;
    summary->termination_type =
        ceres_ba_internal::CeresTerminationTypeToTerminationType(
            level_summary.termination_type);

    if (options_.print_summary) {
      const char* cost_unit = *level_used_geo ? "RMS weighted" : "RMS ZNSSD";
      LOG(INFO) << StringPrintf(
          "Photometric BA report (level %d)\n"
          "    Residuals : %d\n"
          "   Parameters : %d\n"
          "   Iterations : %d\n"
          "         Time : %.5f [s]\n"
          " Initial cost : %.6f [%s]\n"
          "   Final cost : %.6f [%s]\n"
          "  Termination : %s\n",
          req_level,
          level_summary.num_residuals_reduced,
          level_summary.num_effective_parameters_reduced,
          static_cast<int>(level_summary.iterations.size()) - 1,
          level_summary.total_time_in_seconds,
          std::sqrt(2.0 * level_summary.initial_cost /
                    std::max(1, level_summary.num_residuals_reduced)),
          cost_unit,
          std::sqrt(2.0 * level_summary.final_cost /
                    std::max(1, level_summary.num_residuals_reduced)),
          cost_unit,
          ceres::TerminationTypeToString(level_summary.termination_type));
    } else {
      LOG(INFO) << StringPrintf(
          "Photometric BA level %d joint: %d iters, cost %.4f -> %.4f (%.2fs)",
          req_level,
          static_cast<int>(level_summary.iterations.size()) - 1,
          level_summary.initial_cost,
          level_summary.final_cost,
          level_summary.total_time_in_seconds);
    }

    if (!level_summary.IsSolutionUsable()) {
      LOG(ERROR) << "Photometric BA level " << req_level
                 << " failed: " << level_summary.message;
      return LevelStatus::kFailed;
    }
    if (level_summary.termination_type == ceres::USER_SUCCESS) {
      return LevelStatus::kStoppedInJoint;
    }
    return LevelStatus::kContinue;
  }

  void UpdatePointsFromPlanes(
      const int stage_num,
      const int req_level,
      const std::vector<ActiveLandmark>& active_landmarks,
      std::vector<TrackState>* tracks) {
    int updated_points = 0;
    for (const ActiveLandmark& lm : active_landmarks) {
      TrackState& tr = (*tracks)[lm.track_idx];
      const Image& src_img = reconstruction_.Image(tr.src_image_id);
      const Camera& src_cam = *src_img.CameraPtr();
      const Rigid3d src_cfw = src_img.CamFromWorld();

      const auto anchor_ray_opt = CameraPixelToRay(src_cam, lm.patch.anchor_xy);
      if (!anchor_ray_opt.has_value()) {
        continue;
      }
      const double alpha = lm.plane.dot(*anchor_ray_opt);
      if (!(alpha > 1e-8) || !std::isfinite(alpha)) {
        continue;
      }
      const Eigen::Vector3d X_src_new = (*anchor_ray_opt) / alpha;
      if (CameraModelIsPerspective(src_cam.model_id) &&
          !(X_src_new.z() > 1e-8)) {
        continue;
      }
      const Eigen::Vector3d X_w_new = Inverse(src_cfw) * X_src_new;
      if (!X_w_new.allFinite()) {
        continue;
      }
      if (!tr.is_constant) {
        reconstruction_.Point3D(tr.point3D_id).xyz = X_w_new;
        tr.last_updated_stage = stage_num;
        ++updated_points;
      }
      const double plane_norm = lm.plane.norm();
      if (plane_norm > 1e-8) {
        tr.world_normal =
            -(src_cfw.rotation().inverse() * (lm.plane / plane_norm))
                 .normalized();
      }
    }
    VLOG(1) << "Photometric BA level " << req_level << ": updated "
            << updated_points << " 3D points.";
  }

  void RevertCoarseOnlyTracks(const std::vector<TrackState>& tracks,
                              const int last_updated_stage) {
    // Do not leave tracks that were filtered out at the finest level stranded
    // at a blurry coarse-level depth; preserve their initial position relative
    // to their source camera.
    int reverted_coarse_only = 0;
    for (const TrackState& tr : tracks) {
      if (tr.last_updated_stage >= 0 &&
          tr.last_updated_stage < last_updated_stage && !tr.is_constant &&
          tr.src_image_id != kInvalidImageId) {
        const Rigid3d src_cfw =
            reconstruction_.Image(tr.src_image_id).CamFromWorld();
        reconstruction_.Point3D(tr.point3D_id).xyz =
            Inverse(src_cfw) * tr.initial_xyz_in_src;
        ++reverted_coarse_only;
      }
    }
    if (reverted_coarse_only > 0) {
      VLOG(1) << "Photometric BA: reverted " << reverted_coarse_only
              << " coarse-only tracks filtered at the finest level.";
    }
  }

  std::shared_ptr<const FlatHashMap<image_t, GrayscaleImagePyramid>>
  LoadImagePyramids(const std::set<image_t>& image_ids,
                    const int num_levels,
                    const int num_threads) const {
    std::vector<image_t> id_vec(image_ids.begin(), image_ids.end());
    std::vector<GrayscaleImagePyramid> built(id_vec.size());
    std::vector<std::string> errors(id_vec.size());

    {
      ThreadPool pool(num_threads);
      for (size_t i = 0; i < id_vec.size(); ++i) {
        pool.AddTask([&, i]() {
          const Image& image = reconstruction_.Image(id_vec[i]);
          const Camera& camera = *image.CameraPtr();
          const std::filesystem::path path = image_path_ / image.Name();
          Bitmap bitmap;
          if (!bitmap.Read(path, /*as_rgb=*/false)) {
            errors[i] = "Failed to read image: " + path.string();
            return;
          }
          if (static_cast<size_t>(bitmap.Width()) != camera.width ||
              static_cast<size_t>(bitmap.Height()) != camera.height) {
            errors[i] = StringPrintf(
                "Image dimensions (%dx%d) do not match camera (%zux%zu) for %s",
                bitmap.Width(),
                bitmap.Height(),
                camera.width,
                camera.height,
                path.string().c_str());
            return;
          }
          built[i] = GrayscaleImagePyramid(
              bitmap, num_levels, photometric_options_.max_image_size);
        });
      }
      pool.Wait();
    }

    auto result =
        std::make_shared<FlatHashMap<image_t, GrayscaleImagePyramid>>();
    result->reserve(id_vec.size());
    for (size_t i = 0; i < id_vec.size(); ++i) {
      THROW_CHECK(errors[i].empty()) << errors[i];
      result->emplace(id_vec[i], std::move(built[i]));
    }
    return result;
  }

  PhotometricBundleAdjustmentOptions photometric_options_;
  std::filesystem::path image_path_;
  std::shared_ptr<const FlatHashMap<image_t, GrayscaleImagePyramid>>
      preloaded_pyramids_;
  Reconstruction& reconstruction_;
};

}  // namespace

std::unique_ptr<BundleAdjuster> CreatePhotometricBundleAdjuster(
    const BundleAdjustmentOptions& ba_options,
    const PhotometricBundleAdjustmentOptions& photometric_options,
    const BundleAdjustmentConfig& config,
    const std::filesystem::path& image_path,
    Reconstruction& reconstruction) {
  return std::make_unique<PhotometricBundleAdjusterImpl>(ba_options,
                                                         photometric_options,
                                                         config,
                                                         image_path,
                                                         nullptr,
                                                         reconstruction);
}

std::unique_ptr<BundleAdjuster> CreatePhotometricBundleAdjuster(
    const BundleAdjustmentOptions& ba_options,
    const PhotometricBundleAdjustmentOptions& photometric_options,
    const BundleAdjustmentConfig& config,
    std::shared_ptr<const FlatHashMap<image_t, GrayscaleImagePyramid>> pyramids,
    Reconstruction& reconstruction) {
  THROW_CHECK_NOTNULL(pyramids);
  return std::make_unique<PhotometricBundleAdjusterImpl>(
      ba_options,
      photometric_options,
      config,
      std::filesystem::path(),
      std::move(pyramids),
      reconstruction);
}

}  // namespace colmap
