// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/two_view_pose_salvage.h"

#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/cost_functions/tiny_manifold.h"
#include "colmap/estimators/cost_functions/tiny_sampson_error.h"
#include "colmap/estimators/rotation_averaging_statistics.h"
#include "colmap/estimators/solvers/essential_matrix.h"
#include "colmap/estimators/solvers/utils.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/optim/loransac.h"
#include "colmap/optim/support_measurement.h"
#include "colmap/optim/tiny_solver.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"
#include "colmap/util/timer.h"

#include <algorithm>

#include <Eigen/Cholesky>
#include <PoseLib/solvers/relpose_5pt.h>

namespace colmap {
namespace {

using RelativePoseManifold =
    ProductManifold<EigenQuaternionManifold, SphereManifold<3>>;

Eigen::Vector3d ComputeRightRotationError(
    const Eigen::Quaterniond& cand_cam2_from_cam1_rotation,
    const Eigen::Quaterniond& prior_cam2_from_cam1_rotation) {
  const Eigen::Quaterniond error =
      (cand_cam2_from_cam1_rotation.conjugate() * prior_cam2_from_cam1_rotation)
          .normalized();
  Eigen::Vector3d angle_axis;
  AngleAxisFromEigenQuaternion(error.coeffs().data(), angle_axis.data());
  return angle_axis;
}

double ComputeChiSquaredThreshold(const double significance,
                                  const int num_dofs) {
  double lo = 0.0;
  double hi = 100.0;
  while (ChiSquaredSurvival(hi, num_dofs) > significance && hi < 1e6) {
    hi *= 2.0;
  }
  for (int iter = 0; iter < 60; ++iter) {
    const double mid = 0.5 * (lo + hi);
    if (ChiSquaredSurvival(mid, num_dofs) >= significance) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

class RotationGatedEssentialMatrixEstimator {
 public:
  using X_t = CamRayWithJac;
  using Y_t = CamRayWithJac;
  using M_t = Rigid3d;

  static const int kMinNumSamples = 5;

  RotationGatedEssentialMatrixEstimator(
      const Eigen::Quaterniond& prior_cam2_from_cam1_rotation,
      const Eigen::Matrix3d& gate_info,
      const double gate_significance)
      : prior_cam2_from_cam1_rotation_(
            prior_cam2_from_cam1_rotation.normalized()),
        gate_info_(gate_info),
        max_gate_stat_(
            ComputeChiSquaredThreshold(gate_significance, /*num_dofs=*/3)) {}

  bool PassesRotationGate(
      const Eigen::Quaterniond& cand_cam2_from_cam1_rotation) const {
    const Eigen::Vector3d delta = ComputeRightRotationError(
        cand_cam2_from_cam1_rotation, prior_cam2_from_cam1_rotation_);
    const double stat = delta.transpose() * gate_info_ * delta;
    return std::isfinite(stat) && stat <= max_gate_stat_;
  }

  void Estimate(const std::vector<X_t>& cam_rays1_with_jac,
                const std::vector<Y_t>& cam_rays2_with_jac,
                std::vector<M_t>* models) const {
    const size_t num_samples = cam_rays1_with_jac.size();
    THROW_CHECK_EQ(num_samples, cam_rays2_with_jac.size());
    THROW_CHECK_GE(num_samples, kMinNumSamples);
    THROW_CHECK_NOTNULL(models)->clear();

    cam_rays1_.resize(num_samples);
    cam_rays2_.resize(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
      cam_rays1_[i] = cam_rays1_with_jac[i].ray;
      cam_rays2_[i] = cam_rays2_with_jac[i].ray;
    }

    candidate_Es_.clear();
    if (num_samples == kMinNumSamples) {
      poselib::relpose_5pt(cam_rays1_, cam_rays2_, &candidate_Es_);
    } else {
      EssentialMatrixFivePointEstimator::Estimate(
          cam_rays1_, cam_rays2_, &candidate_Es_);
    }

    Eigen::Matrix3d R1;
    Eigen::Matrix3d R2;
    Eigen::Vector3d t;
    for (const Eigen::Matrix3d& candidate_E : candidate_Es_) {
      DecomposeEssentialMatrix(candidate_E, &R1, &R2, &t);
      const std::array<const Eigen::Matrix3d*, 2> candidate_rot_mats = {&R1,
                                                                        &R2};
      for (const Eigen::Matrix3d* rot_mat : candidate_rot_mats) {
        const Eigen::Quaterniond rot(*rot_mat);
        if (!PassesRotationGate(rot)) {
          continue;
        }
        // Determine the only possible translation sign from the first sample
        // ray and verify that all sample rays have positive depth before
        // running full cheirality triangulation.
        const Eigen::Vector3d r0 = (*rot_mat) * cam_rays1_[0];
        const double a0 = -r0.dot(cam_rays2_[0]);
        const double b1_0 = -r0.dot(t);
        const double b2_0 = cam_rays2_[0].dot(t);
        const double sign = (b1_0 - a0 * b2_0 >= 0.0) ? 1.0 : -1.0;
        if (sign * (b1_0 - a0 * b2_0) <= 0.0 ||
            sign * (b2_0 - a0 * b1_0) <= 0.0) {
          continue;
        }
        bool all_positive_depth = true;
        for (size_t i = 1; i < num_samples; ++i) {
          const Eigen::Vector3d ri = (*rot_mat) * cam_rays1_[i];
          const double ai = -ri.dot(cam_rays2_[i]);
          const double b1_i = -ri.dot(t);
          const double b2_i = cam_rays2_[i].dot(t);
          if (sign * (b1_i - ai * b2_i) <= 0.0 ||
              sign * (b2_i - ai * b1_i) <= 0.0) {
            all_positive_depth = false;
            break;
          }
        }
        if (!all_positive_depth) {
          continue;
        }
        const Rigid3d candidate_pose(rot, sign * t);
        CheckCheirality(
            candidate_pose, cam_rays1_, cam_rays2_, &valid_indices_);
        if (valid_indices_.size() == num_samples) {
          models->push_back(candidate_pose);
        }
      }
    }
  }

  bool Refine(const std::vector<X_t>& cam_rays1_with_jac,
              const std::vector<Y_t>& cam_rays2_with_jac,
              M_t* cam2_from_cam1) const {
    THROW_CHECK_EQ(cam_rays1_with_jac.size(), cam_rays2_with_jac.size());
    THROW_CHECK_GE(cam_rays1_with_jac.size(), kMinNumSamples);
    THROW_CHECK_NOTNULL(cam2_from_cam1);

    TinyTangentSampsonErrorCostFunctor f(cam_rays1_with_jac,
                                         cam_rays2_with_jac);
    using Solver = TinySolver<decltype(f), RelativePoseManifold>;
    Solver solver;
    Solver::Options solver_options;
    solver_options.max_num_iterations = 25;

    RelPoseParams x = RelPoseParamsFromRigid3d(*cam2_from_cam1);
    if (solver.Solve(f, &x, solver_options).status ==
        Solver::NUMERICAL_FAILURE) {
      return false;
    }

    const Rigid3d refined = Rigid3dFromRelPoseParams(x.data());
    if (!PassesRotationGate(refined.rotation())) {
      return false;
    }

    *cam2_from_cam1 = refined;
    return true;
  }

  void Residuals(const std::vector<X_t>& cam_rays1_with_jac,
                 const std::vector<Y_t>& cam_rays2_with_jac,
                 const M_t& cam2_from_cam1,
                 std::vector<double>* residuals) const {
    const size_t num_rays = cam_rays1_with_jac.size();
    THROW_CHECK_EQ(num_rays, cam_rays2_with_jac.size());
    residuals->resize(num_rays);

    const Eigen::Matrix3d rot = cam2_from_cam1.rotation().toRotationMatrix();
    const Eigen::Vector3d& t = cam2_from_cam1.translation();
    const Eigen::Matrix3d E = CrossProductMatrix(t.normalized()) * rot;

    for (size_t i = 0; i < num_rays; ++i) {
      const Eigen::Vector3d ray1_in_cam2 = rot * cam_rays1_with_jac[i].ray;
      const double a = -ray1_in_cam2.dot(cam_rays2_with_jac[i].ray);
      const double b1 = -ray1_in_cam2.dot(t);
      const double b2 = cam_rays2_with_jac[i].ray.dot(t);
      if (b1 - a * b2 > 0.0 && b2 - a * b1 > 0.0) {
        (*residuals)[i] = ComputeSquaredTangentSampsonError(
            cam_rays1_with_jac[i], cam_rays2_with_jac[i], E);
      } else {
        (*residuals)[i] = std::numeric_limits<double>::max();
      }
    }
  }

 private:
  Eigen::Quaterniond prior_cam2_from_cam1_rotation_;
  Eigen::Matrix3d gate_info_;
  double max_gate_stat_;
  mutable std::vector<Eigen::Vector3d> cam_rays1_;
  mutable std::vector<Eigen::Vector3d> cam_rays2_;
  mutable std::vector<Eigen::Matrix3d> candidate_Es_;
  mutable std::vector<int> valid_indices_;
};

std::vector<Eigen::Vector2d> ExtractPoints2D(const Image& image) {
  std::vector<Eigen::Vector2d> points;
  points.reserve(image.NumPoints2D());
  for (const auto& point : image.Points2D()) {
    points.push_back(point.xy);
  }
  return points;
}

}  // namespace

bool TwoViewPoseSalvageOptions::Check() const {
  CHECK_OPTION_GT(max_epipolar_error_px, 0.0);
  CHECK_OPTION_GE(min_num_inliers, 5);
  CHECK_OPTION_GE(min_tri_angle_deg, 0.0);
  CHECK_OPTION_GT(max_num_trials, 0);
  CHECK_OPTION_GE(min_num_trials, 0);
  CHECK_OPTION_GT(confidence, 0.0);
  CHECK_OPTION_LE(confidence, 1.0);
  CHECK_OPTION_GE(sample_rotation_sigma_deg, 0.0);
  CHECK_OPTION_GT(gate_significance, 0.0);
  CHECK_OPTION_LT(gate_significance, 1.0);
  CHECK_OPTION_GT(posterior_significance, 0.0);
  CHECK_OPTION_LT(posterior_significance, 1.0);
  return true;
}

std::optional<SalvagedTwoViewPose> SalvageTwoViewPose(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches,
    const Eigen::Quaterniond& prior_cam2_from_cam1_rotation,
    const Eigen::Matrix3d& prior_cam2_from_cam1_rotation_cov,
    const TwoViewPoseSalvageOptions& options,
    const double variance_factor) {
  THROW_CHECK(options.Check());
  THROW_CHECK_GT(variance_factor, 0.0);

  const size_t min_num_inliers = static_cast<size_t>(options.min_num_inliers);
  if (matches.size() < min_num_inliers) {
    return std::nullopt;
  }

  std::vector<CamRayWithJac> matched_rays1;
  std::vector<CamRayWithJac> matched_rays2;
  std::vector<Eigen::Vector3d> matched_bearings1;
  std::vector<Eigen::Vector3d> matched_bearings2;
  FeatureMatches valid_matches;
  matched_rays1.reserve(matches.size());
  matched_rays2.reserve(matches.size());
  matched_bearings1.reserve(matches.size());
  matched_bearings2.reserve(matches.size());
  valid_matches.reserve(matches.size());

  for (const auto& match : matches) {
    if (match.point2D_idx1 >= points1.size() ||
        match.point2D_idx2 >= points2.size()) {
      continue;
    }
    const std::optional<CamRayWithJac> ray1 =
        camera1.CamRayFromImgWithJac(points1[match.point2D_idx1]);
    const std::optional<CamRayWithJac> ray2 =
        camera2.CamRayFromImgWithJac(points2[match.point2D_idx2]);
    if (!ray1.has_value() || !ray2.has_value()) {
      continue;
    }
    matched_rays1.push_back(*ray1);
    matched_rays2.push_back(*ray2);
    matched_bearings1.push_back(ray1->ray);
    matched_bearings2.push_back(ray2->ray);
    valid_matches.push_back(match);
  }

  if (valid_matches.size() < min_num_inliers) {
    return std::nullopt;
  }

  const double sample_sigma_rad = DegToRad(options.sample_rotation_sigma_deg);
  const Eigen::Matrix3d gate_cov =
      0.5 * (prior_cam2_from_cam1_rotation_cov +
             prior_cam2_from_cam1_rotation_cov.transpose()) +
      (sample_sigma_rad * sample_sigma_rad) * Eigen::Matrix3d::Identity();
  const Eigen::LLT<Eigen::Matrix3d> gate_llt(gate_cov);
  if (gate_llt.info() != Eigen::Success) {
    return std::nullopt;
  }
  const Eigen::Matrix3d gate_info = gate_llt.solve(Eigen::Matrix3d::Identity());

  RANSACOptions ransac_options;
  ransac_options.max_error = options.max_epipolar_error_px;
  ransac_options.min_inlier_ratio = 0.05;
  ransac_options.confidence = options.confidence;
  ransac_options.max_num_trials = static_cast<size_t>(options.max_num_trials);
  ransac_options.min_num_trials = static_cast<size_t>(options.min_num_trials);
  ransac_options.random_seed = options.random_seed;
  ransac_options.num_threads = options.num_threads;

  if (options.random_seed >= 0) {
    SetPRNGSeed(static_cast<unsigned>(options.random_seed));
  }

  RotationGatedEssentialMatrixEstimator estimator(
      prior_cam2_from_cam1_rotation, gate_info, options.gate_significance);
  LORANSAC<RotationGatedEssentialMatrixEstimator,
           RotationGatedEssentialMatrixEstimator,
           MEstimatorSupportMeasurer>
      ransac(ransac_options, estimator, estimator);

  const auto report = ransac.Estimate(matched_rays1, matched_rays2);
  if (!report.success || report.support.num_inliers < min_num_inliers) {
    return std::nullopt;
  }

  std::vector<Eigen::Vector3d> inlier_bearings1;
  std::vector<Eigen::Vector3d> inlier_bearings2;
  FeatureMatches inlier_matches;
  inlier_bearings1.reserve(report.support.num_inliers);
  inlier_bearings2.reserve(report.support.num_inliers);
  inlier_matches.reserve(report.support.num_inliers);
  for (size_t i = 0; i < valid_matches.size(); ++i) {
    if (report.inlier_mask[i]) {
      inlier_bearings1.push_back(matched_bearings1[i]);
      inlier_bearings2.push_back(matched_bearings2[i]);
      inlier_matches.push_back(valid_matches[i]);
    }
  }

  Rigid3d cam2_from_cam1 = report.model;
  const double trans_norm = cam2_from_cam1.translation().norm();
  if (trans_norm <= 1e-12) {
    return std::nullopt;
  }
  cam2_from_cam1.translation() /= trans_norm;

  std::vector<int> cheiral_indices;
  CheckCheirality(
      cam2_from_cam1, inlier_bearings1, inlier_bearings2, &cheiral_indices);
  if (cheiral_indices.size() < min_num_inliers) {
    return std::nullopt;
  }

  const Eigen::Quaterniond cam1_from_cam2_rotation =
      cam2_from_cam1.rotation().inverse();
  std::vector<double> tri_angles;
  tri_angles.reserve(cheiral_indices.size());
  for (const int idx : cheiral_indices) {
    const double angle = CalculateAngleBetweenVectors(
        inlier_bearings1[idx],
        cam1_from_cam2_rotation * inlier_bearings2[idx]);
    tri_angles.push_back(
        std::min(angle, static_cast<double>(EIGEN_PI) - angle));
  }
  const double median_tri_angle = Median(tri_angles);
  if (median_tri_angle < DegToRad(options.min_tri_angle_deg)) {
    return std::nullopt;
  }

  FeatureMatches cheiral_matches(cheiral_indices.size());
  for (size_t i = 0; i < cheiral_indices.size(); ++i) {
    cheiral_matches[i] = inlier_matches[cheiral_indices[i]];
  }

  TwoViewGeometry geometry;
  geometry.config = TwoViewGeometry::CALIBRATED;
  geometry.E = EssentialMatrixFromPose(cam2_from_cam1);
  geometry.cam2_from_cam1 = cam2_from_cam1;
  geometry.inlier_matches = std::move(cheiral_matches);
  geometry.tri_angle = median_tri_angle;

  const std::optional<TwoViewPoseCovariance> pose_cov =
      EstimateTwoViewPoseCovariance(camera1,
                                    points1,
                                    camera2,
                                    points2,
                                    geometry,
                                    options.covariance_options);
  if (!pose_cov.has_value() || !pose_cov->cov_rot.has_value()) {
    return std::nullopt;
  }

  const Eigen::Matrix3d posterior_cov =
      variance_factor * (*pose_cov->cov_rot) +
      0.5 * (prior_cam2_from_cam1_rotation_cov +
             prior_cam2_from_cam1_rotation_cov.transpose());
  const Eigen::LLT<Eigen::Matrix3d> posterior_llt(posterior_cov);
  if (posterior_llt.info() != Eigen::Success) {
    return std::nullopt;
  }

  const Eigen::Vector3d delta_post = ComputeRightRotationError(
      cam2_from_cam1.rotation(), prior_cam2_from_cam1_rotation);
  const double posterior_statistic =
      delta_post.dot(posterior_llt.solve(delta_post));
  if (!std::isfinite(posterior_statistic)) {
    return std::nullopt;
  }
  const double posterior_p_value =
      ChiSquaredSurvival(posterior_statistic, /*num_dofs=*/3);
  if (posterior_p_value < options.posterior_significance) {
    return std::nullopt;
  }

  SalvagedTwoViewPose result;
  result.geometry = std::move(geometry);
  result.cam2_from_cam1_rotation_cov = *pose_cov->cov_rot;
  result.posterior_statistic = posterior_statistic;
  result.posterior_p_value = posterior_p_value;
  return result;
}

size_t SalvageTwoViewPoses(const TwoViewPoseSalvageOptions& options,
                           const RotationEstimatorOptions& rotation_options,
                           const DatabaseCache& database_cache,
                           Reconstruction& reconstruction,
                           PoseGraph& pose_graph,
                           CorrespondenceGraph& correspondence_graph) {
  THROW_CHECK(options.Check());

  Timer timer;
  timer.Start();

  const size_t min_num_inliers = static_cast<size_t>(options.min_num_inliers);

  // Collect candidate pairs between registered images:
  // 1. Invalid edges in pose_graph.
  // 2. Unverified raw match pairs in database_cache.Matches() that do not have
  //    a valid edge in pose_graph.
  FlatHashSet<image_pair_t> candidate_pair_set;
  for (const auto& [pair_id, edge] : pose_graph.Edges()) {
    if (edge.valid) {
      continue;
    }
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (reconstruction.ExistsImage(image_id1) &&
        reconstruction.ExistsImage(image_id2) &&
        reconstruction.Image(image_id1).HasPose() &&
        reconstruction.Image(image_id2).HasPose() &&
        image_id1 != image_id2) {
      candidate_pair_set.insert(pair_id);
    }
  }

  for (const auto& [pair_id, matches] : database_cache.Matches()) {
    if (matches.size() < min_num_inliers || pose_graph.IsValid(pair_id)) {
      continue;
    }
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (reconstruction.ExistsImage(image_id1) &&
        reconstruction.ExistsImage(image_id2) &&
        reconstruction.Image(image_id1).HasPose() &&
        reconstruction.Image(image_id2).HasPose() &&
        image_id1 != image_id2) {
      candidate_pair_set.insert(pair_id);
    }
  }

  if (candidate_pair_set.empty()) {
    return 0;
  }

  struct CandidatePair {
    image_pair_t pair_id = kInvalidImagePairId;
    image_t image_id1 = kInvalidImageId;
    image_t image_id2 = kInvalidImageId;
    FeatureMatches matches;
    bool was_in_pose_graph = false;
  };

  std::vector<image_pair_t> sorted_pair_ids(candidate_pair_set.begin(),
                                            candidate_pair_set.end());
  std::sort(sorted_pair_ids.begin(), sorted_pair_ids.end());

  std::vector<CandidatePair> candidates;
  candidates.reserve(sorted_pair_ids.size());
  FlatHashSet<image_pair_t> query_pairs;
  query_pairs.reserve(sorted_pair_ids.size());

  for (const image_pair_t pair_id : sorted_pair_ids) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    FeatureMatches matches;
    const auto raw_it = database_cache.Matches().find(pair_id);
    if (raw_it != database_cache.Matches().end()) {
      matches = raw_it->second;
    } else {
      correspondence_graph.ExtractMatchesBetweenImages(
          image_id1, image_id2, matches);
    }
    if (matches.size() < min_num_inliers) {
      continue;
    }
    CandidatePair cand;
    cand.pair_id = pair_id;
    cand.image_id1 = image_id1;
    cand.image_id2 = image_id2;
    cand.matches = std::move(matches);
    cand.was_in_pose_graph = pose_graph.HasEdge(image_id1, image_id2);
    candidates.push_back(std::move(cand));
    query_pairs.insert(pair_id);
  }

  if (candidates.empty()) {
    return 0;
  }

  RotationEstimatorOptions stats_options = rotation_options;
  stats_options.filter_unregistered = true;
  const std::optional<RotationAveragingStatistics> stats =
      EstimateRotationAveragingStatistics(
          stats_options, pose_graph, reconstruction, query_pairs);
  if (!stats.has_value()) {
    LOG(WARNING) << "Failed to estimate rotation averaging statistics for "
                    "two-view pose salvage";
    return 0;
  }

  const double edge_cov_variance_factor =
      stats_options.reweighting == RotationAveragingReweighting::COVARIANCE
          ? stats->variance_factor
          : 1.0;

  FlatHashSet<image_t> needed_image_ids;
  needed_image_ids.reserve(2 * candidates.size());
  for (const auto& cand : candidates) {
    needed_image_ids.insert(cand.image_id1);
    needed_image_ids.insert(cand.image_id2);
  }
  FlatHashMap<image_t, std::vector<Eigen::Vector2d>> points_by_image;
  points_by_image.reserve(needed_image_ids.size());
  for (const image_t image_id : needed_image_ids) {
    points_by_image.emplace(
        image_id, ExtractPoints2D(database_cache.Image(image_id)));
  }

  std::vector<std::optional<SalvagedTwoViewPose>> results(candidates.size());
  ThreadPool thread_pool(GetEffectiveNumThreads(options.num_threads));
  for (size_t i = 0; i < candidates.size(); ++i) {
    thread_pool.AddTask([&, i]() {
      const CandidatePair& cand = candidates[i];
      const auto stat_it = stats->edges.find(cand.pair_id);
      if (stat_it == stats->edges.end()) {
        return;
      }
      const RelativeRotationStatistics& edge_stats = stat_it->second;
      const Image& image1 = database_cache.Image(cand.image_id1);
      const Image& image2 = database_cache.Image(cand.image_id2);
      const Camera& camera1 = database_cache.Camera(image1.CameraId());
      const Camera& camera2 = database_cache.Camera(image2.CameraId());

      TwoViewPoseSalvageOptions pair_options = options;
      pair_options.num_threads = 1;
      if (options.random_seed >= 0) {
        pair_options.random_seed = options.random_seed + static_cast<int>(i);
      }

      results[i] = SalvageTwoViewPose(camera1,
                                      points_by_image.at(cand.image_id1),
                                      camera2,
                                      points_by_image.at(cand.image_id2),
                                      cand.matches,
                                      edge_stats.cam2_from_cam1_rotation,
                                      edge_stats.cam2_from_cam1_rotation_cov,
                                      pair_options,
                                      edge_cov_variance_factor);
    });
  }
  thread_pool.Wait();

  size_t num_salvaged_edges = 0;
  size_t num_salvaged_new_pairs = 0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (!results[i].has_value()) {
      continue;
    }
    const CandidatePair& cand = candidates[i];
    SalvagedTwoViewPose& salvaged = *results[i];

    PoseGraph::Edge edge(salvaged.geometry.cam2_from_cam1.value());
    edge.cam2_from_cam1_rotation_cov = salvaged.cam2_from_cam1_rotation_cov;
    edge.num_matches = static_cast<int>(salvaged.geometry.inlier_matches.size());
    edge.valid = true;

    if (cand.was_in_pose_graph) {
      pose_graph.UpdateEdge(cand.image_id1, cand.image_id2, std::move(edge));
      ++num_salvaged_edges;
    } else {
      pose_graph.AddEdge(cand.image_id1, cand.image_id2, std::move(edge));
      ++num_salvaged_new_pairs;
    }

    correspondence_graph.AddOrUpdateTwoViewGeometry(
        cand.image_id1, cand.image_id2, std::move(salvaged.geometry));
  }

  const size_t total_salvaged = num_salvaged_edges + num_salvaged_new_pairs;
  LOG(INFO) << StringPrintf(
      "Salvaged %zu / %zu two-view poses (%zu rejected edges, %zu unverified "
      "pairs) in %.3fs",
      total_salvaged,
      candidates.size(),
      num_salvaged_edges,
      num_salvaged_new_pairs,
      timer.ElapsedSeconds());

  return total_salvaged;
}

}  // namespace colmap
