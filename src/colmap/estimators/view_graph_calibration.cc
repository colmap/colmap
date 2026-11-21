// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/view_graph_calibration.h"

#include "colmap/estimators/cost_functions/calibration.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"

#include <cmath>
#include <limits>
#include <mutex>

namespace colmap {
namespace {

// Bounds for focal length optimization to prevent numerical issues.
constexpr double kMinFocalLength = 1e-3;
constexpr double kMaxFocalLength = 1e8;

// Input for focal length calibration: an image pair with its F matrix.
struct FocalLengthCalibInput {
  image_pair_t pair_id = kInvalidImagePairId;
  camera_t camera_id1 = kInvalidCameraId;
  camera_t camera_id2 = kInvalidCameraId;
  Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
  size_t num_inliers = 0;
};

// Result of focal length calibration.
struct FocalLengthCalibResult {
  // Optimized focal lengths per camera.
  FlatHashMap<camera_t, double> focal_lengths;
  // Calibration error per image pair, i.e., the normalized distance of the
  // calibrated fundamental matrix to the essential manifold.
  FlatHashMap<image_pair_t, double> calibration_errors;
  // Whether the calibration succeeded.
  bool success = false;
};

// Cross-validate prior focal lengths by checking the ratio of calibrated vs
// uncalibrated pairs per camera. UNCALIBRATED pairs are converted to
// CALIBRATED if both cameras have reliable priors.
void CrossValidatePriorFocalLengths(
    double min_calibrated_pair_ratio,
    const NodeHashMap<image_t, const Camera*>& image_id_to_camera,
    std::vector<std::pair<image_pair_t, TwoViewGeometry>>& pairs) {
  // For each camera, count the number of calibrated vs uncalibrated pairs.
  // first: total count, second: calibrated count.
  NodeHashMap<camera_t, std::pair<int, int>> camera_counter;
  for (const auto& [pair_id, tvg] : pairs) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const Camera& camera1 = *image_id_to_camera.at(image_id1);
    const Camera& camera2 = *image_id_to_camera.at(image_id2);
    if (!camera1.HasPriorFocalLength() || !camera2.HasPriorFocalLength()) {
      continue;
    }

    auto& [total1, calibrated1] = camera_counter[camera1.camera_id];
    auto& [total2, calibrated2] = camera_counter[camera2.camera_id];
    ++total1;
    ++total2;
    if (tvg.config == TwoViewGeometry::CALIBRATED) {
      ++calibrated1;
      ++calibrated2;
    }
  }

  // Camera is valid if the ratio of calibrated pairs exceeds threshold.
  NodeHashMap<camera_t, bool> camera_validity;
  camera_validity.reserve(camera_counter.size());
  for (const auto& [camera_id, counter] : camera_counter) {
    const auto [total, calibrated] = counter;
    const double ratio = static_cast<double>(calibrated) / total;
    camera_validity[camera_id] = ratio > min_calibrated_pair_ratio;
  }

  // Convert UNCALIBRATED pairs to CALIBRATED if both cameras are valid.
  // Compute E from F using the prior camera calibrations.
  size_t num_upgraded_pairs = 0;
  for (auto& [pair_id, tvg] : pairs) {
    if (tvg.config != TwoViewGeometry::UNCALIBRATED) {
      continue;
    }

    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const Camera& camera1 = *image_id_to_camera.at(image_id1);
    const Camera& camera2 = *image_id_to_camera.at(image_id2);

    // Computing E from F assumes a pinhole projection for both cameras. See
    // CalibrateFocalLengths for why fisheye models are excluded.
    if (camera1.IsPerspectivePinhole() && camera2.IsPerspectivePinhole() &&
        camera_validity[camera1.camera_id] &&
        camera_validity[camera2.camera_id]) {
      THROW_CHECK(tvg.F.has_value())
          << "UNCALIBRATED two-view geometry must have F matrix";
      tvg.E = EssentialFromFundamentalMatrix(camera2.CalibrationMatrix(),
                                             tvg.F.value(),
                                             camera1.CalibrationMatrix());
      tvg.config = TwoViewGeometry::CALIBRATED;
      // E is now built from the K of the cross-validated focal priors, and
      // consumers calibrate the rays with camera1/camera2 whenever set. Clear
      // them so the rays use the same K as E, not the solver's own focal.
      tvg.camera1.reset();
      tvg.camera2.reset();
      ++num_upgraded_pairs;
    }
  }

  LOG(INFO) << "Upgraded " << num_upgraded_pairs << " / " << pairs.size()
            << " pairs to calibrated through cross-validation";
}

// Re-estimate relative poses for all pairs using calibrated cameras.
void ReestimateRelativePoses(
    const ViewGraphCalibrationOptions& options,
    std::vector<std::pair<image_pair_t, TwoViewGeometry>>& pairs,
    const NodeHashMap<image_t, const Camera*>& image_id_to_camera,
    Database* database) {
  LOG(INFO) << "Re-estimating relative poses for " << pairs.size() << " pairs";

  TwoViewGeometryOptions two_view_options;
  two_view_options.compute_relative_pose = true;
  two_view_options.ransac_options.max_error = options.relpose_max_error;
  two_view_options.min_num_inliers = options.relpose_min_num_inliers;
  two_view_options.min_inlier_ratio = options.relpose_min_inlier_ratio;
  two_view_options.ransac_options.random_seed = options.random_seed;

  // Pre-read all keypoints from database.
  FlatHashSet<image_t> image_ids;
  image_ids.reserve(2 * pairs.size());
  for (const auto& [pair_id, tvg] : pairs) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    image_ids.insert(image_id1);
    image_ids.insert(image_id2);
  }

  NodeHashMap<image_t, std::vector<Eigen::Vector2d>> image_points;
  image_points.reserve(image_ids.size());
  for (const image_t image_id : image_ids) {
    const FeatureKeypoints keypoints = database->ReadKeypoints(image_id);
    std::vector<Eigen::Vector2d> points(keypoints.size());
    for (size_t j = 0; j < keypoints.size(); ++j) {
      points[j] = Eigen::Vector2d(keypoints[j].x, keypoints[j].y);
    }
    image_points[image_id] = std::move(points);
  }

  // Parallel estimation with mutex-protected database access for matches.
  std::mutex database_mutex;
  ThreadPool thread_pool(options.solver_options.num_threads);

  for (size_t i = 0; i < pairs.size(); ++i) {
    thread_pool.AddTask([&, i]() {
      auto& [pair_id, tvg] = pairs[i];
      const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);

      FeatureMatches matches;
      {
        std::lock_guard<std::mutex> lock(database_mutex);
        matches = database->ReadMatches(image_id1, image_id2);
      }

      const Camera& camera1 = *image_id_to_camera.at(image_id1);
      const Camera& camera2 = *image_id_to_camera.at(image_id2);

      const std::vector<Eigen::Vector2d>& points1 = image_points.at(image_id1);
      const std::vector<Eigen::Vector2d>& points2 = image_points.at(image_id2);

      std::optional<Eigen::Matrix3d> orig_F = std::move(tvg.F);
      tvg = EstimateCalibratedTwoViewGeometry(
          camera1, points1, camera2, points2, matches, two_view_options);
      tvg.F = std::move(orig_F);
    });
  }
  thread_pool.Wait();
}

// Computes the closed-form focal length estimates of an image pair from its
// fundamental matrix (Bougnoux), assuming principal points at the origin.
// Returns the positive solutions as pairs of squared focal lengths. With
// omega_i = diag(f_i^2, f_i^2, 1) and e the epipole in the second image, the
// Kruppa equations F * omega_1 * F^T ~ [e]_x * omega_2 * [e]_x^T are
// restricted to the complement of e, where both sides are symmetric 2x2
// matrices. Their proportionality gives two bilinear equations in (f1^2, f2^2),
// whose intersections are the solutions.
std::vector<std::pair<double, double>> SquaredFocalLengthsFromFundamental(
    const Eigen::Matrix3d& F) {
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      F, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singular_values = svd.singularValues();
  singular_values(2) = 0;
  const Eigen::Matrix3d rank2_F =
      svd.matrixU() * singular_values.asDiagonal() * svd.matrixV().transpose();
  const Eigen::Vector3d epipole = svd.matrixU().col(2);
  const Eigen::Vector3d basis1 = svd.matrixU().col(0);
  const Eigen::Vector3d basis2 = svd.matrixU().col(1);

  // Restricts a symmetric 3x3 matrix to the complement of the epipole.
  const auto restrict = [&](const Eigen::Matrix3d& M) {
    return Eigen::Vector3d(basis1.dot(M * basis1),
                           basis2.dot(M * basis2),
                           std::sqrt(2.0) * basis1.dot(M * basis2));
  };
  const Eigen::Matrix3d D1 = Eigen::Vector3d(1, 1, 0).asDiagonal();
  const Eigen::Matrix3d D0 = Eigen::Vector3d(0, 0, 1).asDiagonal();
  const Eigen::Matrix3d epipole_cross = CrossProductMatrix(epipole);
  // Restricted left side x1 * A1 + A0 and right side x2 * B1 + B0 with
  // x_i = f_i^2.
  const Eigen::Vector3d A1 = restrict(rank2_F * D1 * rank2_F.transpose());
  const Eigen::Vector3d A0 = restrict(rank2_F * D0 * rank2_F.transpose());
  const Eigen::Vector3d B1 =
      restrict(epipole_cross * D1 * epipole_cross.transpose());
  const Eigen::Vector3d B0 =
      restrict(epipole_cross * D0 * epipole_cross.transpose());

  // The 2x2 minor (r, s) of the proportionality yields the bilinear equation
  // c11 * x1 * x2 + c10 * x1 + c01 * x2 + c00 = 0.
  const auto bilinear = [&](int r, int s) {
    const auto minor = [r, s](const Eigen::Vector3d& u,
                              const Eigen::Vector3d& v) {
      return u(r) * v(s) - u(s) * v(r);
    };
    return Eigen::Vector4d(
        minor(A1, B1), minor(A1, B0), minor(A0, B1), minor(A0, B0));
  };
  int pivot = 0;
  B1.cwiseAbs().maxCoeff(&pivot);
  const Eigen::Vector4d c = bilinear(pivot, (pivot + 1) % 3);
  const Eigen::Vector4d d = bilinear(pivot, (pivot + 2) % 3);

  // Eliminating x2 = -(c10 * x1 + c00) / (c11 * x1 + c01) yields a quadratic
  // alpha * x1^2 + beta * x1 + gamma = 0.
  const double alpha = c(1) * d(0) - d(1) * c(0);
  const double beta = c(1) * d(2) + c(3) * d(0) - d(1) * c(2) - d(3) * c(0);
  const double gamma = c(3) * d(2) - d(3) * c(2);
  constexpr double kEps = 1e-30;
  std::vector<double> x1_roots;
  if (std::abs(alpha) < kEps) {
    if (std::abs(beta) > kEps) {
      x1_roots.push_back(-gamma / beta);
    }
  } else {
    const double discriminant = beta * beta - 4 * alpha * gamma;
    if (discriminant < 0) {
      return {};
    }
    const double sqrt_discriminant = std::sqrt(discriminant);
    x1_roots.push_back((-beta + sqrt_discriminant) / (2 * alpha));
    x1_roots.push_back((-beta - sqrt_discriminant) / (2 * alpha));
  }

  std::vector<std::pair<double, double>> solutions;
  for (const double x1 : x1_roots) {
    const double denom = c(0) * x1 + c(2);
    if (std::abs(denom) < kEps) {
      continue;
    }
    const double x2 = -(c(1) * x1 + c(3)) / denom;
    if (x1 > 0 && x2 > 0) {
      solutions.emplace_back(x1, x2);
    }
  }
  return solutions;
}

// Initializes the focal lengths of cameras without prior focal length from a
// consensus of the closed-form two-view estimates of their image pairs. Each
// pair votes with its estimate that is within the plausible range relative to
// the prior and closest to the essential manifold. A camera with at least
// three votes of small spread is initialized to their median. Otherwise, it
// falls back to its prior focal length scaled by the median ratio of
// estimated to prior focal lengths over all votes. Cameras with prior focal
// length keep it.
FlatHashMap<camera_t, double> InitializeFocalLengths(
    const ViewGraphCalibrationOptions& options,
    const std::vector<FocalLengthCalibInput>& inputs,
    const NodeHashMap<camera_t, Camera>& cameras) {
  constexpr size_t kMinNumVotes = 3;
  constexpr double kMaxVoteSpread = 1.2;

  // Collect votes as ratios of estimated to prior focal length.
  FlatHashMap<camera_t, std::vector<double>> votes;
  for (const auto& input : inputs) {
    if (input.num_inliers < static_cast<size_t>(options.min_num_inliers)) {
      continue;
    }
    const Camera& camera1 = cameras.at(input.camera_id1);
    const Camera& camera2 = cameras.at(input.camera_id2);
    const Eigen::Matrix3d K1 = camera1.CalibrationMatrix();
    const Eigen::Matrix3d K2 = camera2.CalibrationMatrix();
    Eigen::Matrix3d normalized_F = K2.transpose() * input.F * K1;
    const double norm = normalized_F.norm();
    if (!std::isfinite(norm) || norm < 1e-30) {
      continue;
    }
    normalized_F /= norm;

    double min_error = std::numeric_limits<double>::max();
    std::pair<double, double> best_ratios;
    for (const auto& [ratio1_sq, ratio2_sq] :
         SquaredFocalLengthsFromFundamental(normalized_F)) {
      const double ratio1 = std::sqrt(ratio1_sq);
      const double ratio2 = std::sqrt(ratio2_sq);
      if (ratio1 < options.min_focal_length_ratio ||
          ratio1 > options.max_focal_length_ratio ||
          ratio2 < options.min_focal_length_ratio ||
          ratio2 > options.max_focal_length_ratio) {
        continue;
      }
      const double focal_length1 = ratio1 * camera1.MeanFocalLength();
      const double focal_length2 = ratio2 * camera2.MeanFocalLength();
      const double log_focal_length1 = std::log(focal_length1);
      const double log_focal_length2 = std::log(focal_length2);
      const EssentialManifoldFocalLengthCostFunctor cost_functor(
          input.F,
          camera1.PrincipalPoint(),
          camera2.PrincipalPoint(),
          focal_length1,
          focal_length2);
      Eigen::Vector2d residual;
      cost_functor(&log_focal_length1, &log_focal_length2, residual.data());
      if (residual.squaredNorm() < min_error) {
        min_error = residual.squaredNorm();
        best_ratios = {ratio1, ratio2};
      }
    }
    if (min_error == std::numeric_limits<double>::max()) {
      continue;
    }
    votes[input.camera_id1].push_back(best_ratios.first);
    votes[input.camera_id2].push_back(best_ratios.second);
  }

  std::vector<double> scene_ratios;
  for (const auto& [camera_id, camera_votes] : votes) {
    if (!cameras.at(camera_id).HasPriorFocalLength()) {
      scene_ratios.insert(
          scene_ratios.end(), camera_votes.begin(), camera_votes.end());
    }
  }
  const double scene_ratio =
      scene_ratios.empty() ? 1.0 : Median(std::move(scene_ratios));

  FlatHashMap<camera_t, double> focal_lengths;
  size_t num_consensus_cameras = 0;
  size_t num_fallback_cameras = 0;
  for (const auto& input : inputs) {
    for (const camera_t camera_id : {input.camera_id1, input.camera_id2}) {
      if (focal_lengths.count(camera_id)) {
        continue;
      }
      const Camera& camera = cameras.at(camera_id);
      if (camera.HasPriorFocalLength()) {
        focal_lengths[camera_id] = camera.MeanFocalLength();
        continue;
      }
      double ratio = scene_ratio;
      const auto votes_it = votes.find(camera_id);
      if (votes_it != votes.end() && votes_it->second.size() >= kMinNumVotes &&
          Percentile(votes_it->second, 75) <
              kMaxVoteSpread *
                  std::max(Percentile(votes_it->second, 25), 1e-12)) {
        ratio = Median(votes_it->second);
        ++num_consensus_cameras;
      } else {
        ++num_fallback_cameras;
      }
      focal_lengths[camera_id] = ratio * camera.MeanFocalLength();
    }
  }

  LOG(INFO) << "Initialized focal lengths of " << num_consensus_cameras
            << " cameras from consensus and " << num_fallback_cameras
            << " cameras from scaled priors (scene ratio: " << scene_ratio
            << ")";

  return focal_lengths;
}

// Calibrates the focal lengths of all cameras without prior focal length. The
// focal lengths are first initialized from a consensus of closed-form two-view
// estimates and then jointly refined by minimizing the robustified distances
// of the calibrated fundamental matrices to the essential manifold, first with
// a Cauchy loss and then with a redescending Tukey loss.
// See: "Smooth Essential-Manifold Residuals for View-Graph Calibration",
// 3DV 2027.
FocalLengthCalibResult CalibrateFocalLengths(
    const ViewGraphCalibrationOptions& options,
    const std::vector<FocalLengthCalibInput>& inputs,
    const NodeHashMap<camera_t, Camera>& cameras) {
  FocalLengthCalibResult result;

  if (inputs.empty()) {
    result.success = true;
    return result;
  }

  const FlatHashMap<camera_t, double> initial_focal_lengths =
      InitializeFocalLengths(options, inputs, cameras);

  // Parameterize the focal lengths by their logarithm to enforce positivity.
  // Ceres holds pointers to the values, so the container must be
  // reference-stable.
  NodeHashMap<camera_t, double> log_focal_lengths;
  log_focal_lengths.reserve(initial_focal_lengths.size());
  for (const auto& [camera_id, focal_length] : initial_focal_lengths) {
    log_focal_lengths[camera_id] = std::log(focal_length);
  }

  // The loss function is swapped between the two optimization stages, so it
  // must outlive and not be owned by the problem.
  ceres::LossFunctionWrapper loss_function(
      new ceres::CauchyLoss(options.warmup_loss_function_scale),
      ceres::TAKE_OWNERSHIP);
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  for (const auto& input : inputs) {
    if (input.num_inliers < static_cast<size_t>(options.min_num_inliers)) {
      continue;
    }
    if (input.camera_id1 == input.camera_id2) {
      problem.AddResidualBlock(
          EssentialManifoldFocalLengthSameCameraCostFunctor::Create(
              input.F,
              cameras.at(input.camera_id1).PrincipalPoint(),
              initial_focal_lengths.at(input.camera_id1)),
          &loss_function,
          &log_focal_lengths.at(input.camera_id1));
    } else {
      problem.AddResidualBlock(
          EssentialManifoldFocalLengthCostFunctor::Create(
              input.F,
              cameras.at(input.camera_id1).PrincipalPoint(),
              cameras.at(input.camera_id2).PrincipalPoint(),
              initial_focal_lengths.at(input.camera_id1),
              initial_focal_lengths.at(input.camera_id2)),
          &loss_function,
          &log_focal_lengths.at(input.camera_id1),
          &log_focal_lengths.at(input.camera_id2));
    }
  }

  // Fix cameras with prior focal length and bound the others.
  std::vector<camera_t> optimized_camera_ids;
  for (auto& [camera_id, log_focal_length] : log_focal_lengths) {
    if (!problem.HasParameterBlock(&log_focal_length)) {
      continue;
    }
    if (cameras.at(camera_id).HasPriorFocalLength()) {
      problem.SetParameterBlockConstant(&log_focal_length);
    } else {
      problem.SetParameterLowerBound(
          &log_focal_length, 0, std::log(kMinFocalLength));
      problem.SetParameterUpperBound(
          &log_focal_length, 0, std::log(kMaxFocalLength));
      optimized_camera_ids.push_back(camera_id);
    }
  }

  if (optimized_camera_ids.empty()) {
    LOG(INFO) << "No cameras to optimize";
  } else {
    ceres::Solver::Options solver_options = options.solver_options;
    if (cameras.size() < 50) {
      solver_options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    } else {
      solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
    solver_options.num_threads =
        GetEffectiveNumThreads(solver_options.num_threads);
    solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);

    // The Cauchy warm-up converges to the right basin from inaccurate
    // initializations, while the subsequent Tukey loss gives zero weight to
    // outlier pairs.
    for (const bool warmup : {true, false}) {
      if (!warmup) {
        loss_function.Reset(new ceres::TukeyLoss(options.loss_function_scale),
                            ceres::TAKE_OWNERSHIP);
      }
      ceres::Solver::Summary summary;
      ceres::Solve(solver_options, &problem, &summary);
      VLOG(2) << summary.FullReport();
      if (!summary.IsSolutionUsable()) {
        LOG(ERROR) << "Ceres solver failed";
        return result;
      }
    }

    // Reject degenerate focal lengths.
    size_t num_rejected_cameras = 0;
    for (const camera_t camera_id : optimized_camera_ids) {
      const double focal_length = std::exp(log_focal_lengths.at(camera_id));
      const double prior_focal_length = cameras.at(camera_id).MeanFocalLength();
      const double focal_length_ratio = focal_length / prior_focal_length;
      if (focal_length_ratio > options.max_focal_length_ratio ||
          focal_length_ratio < options.min_focal_length_ratio) {
        VLOG(2) << "Ignoring degenerate camera " << camera_id
                << " focal: " << focal_length
                << " original focal: " << prior_focal_length;
        ++num_rejected_cameras;
        continue;
      }
      result.focal_lengths[camera_id] = focal_length;
    }
    LOG(INFO) << num_rejected_cameras
              << " cameras rejected in view graph calibration";
  }

  // Evaluate the calibration errors of all pairs with the final focal lengths.
  const auto final_focal_length = [&](camera_t camera_id) {
    const auto it = result.focal_lengths.find(camera_id);
    return it != result.focal_lengths.end()
               ? it->second
               : cameras.at(camera_id).MeanFocalLength();
  };
  for (const auto& input : inputs) {
    const double focal_length1 = final_focal_length(input.camera_id1);
    const double focal_length2 = final_focal_length(input.camera_id2);
    const double log_focal_length1 = std::log(focal_length1);
    const double log_focal_length2 = std::log(focal_length2);
    const EssentialManifoldFocalLengthCostFunctor cost_functor(
        input.F,
        cameras.at(input.camera_id1).PrincipalPoint(),
        cameras.at(input.camera_id2).PrincipalPoint(),
        focal_length1,
        focal_length2);
    Eigen::Vector2d residual;
    cost_functor(&log_focal_length1, &log_focal_length2, residual.data());
    result.calibration_errors[input.pair_id] = residual.norm();
  }

  result.success = true;
  return result;
}

}  // namespace

bool CalibrateViewGraph(const ViewGraphCalibrationOptions& options,
                        Database* database) {
  THROW_CHECK_NOTNULL(database);

  // Read cameras, preferring the best calibration that is NOT VIEW_GRAPH,
  // so that re-running view graph calibration does not rely on its own previous
  // output.
  NodeHashMap<camera_t, Camera> cameras =
      database->ReadAllCamerasExcludingSources({CameraSource::VIEW_GRAPH});
  NodeHashMap<image_t, const Camera*> image_id_to_camera;
  for (const Image& image : database->ReadAllImages()) {
    image_id_to_camera[image.ImageId()] = &cameras.at(image.CameraId());
  }

  // Read UNCALIBRATED and CALIBRATED two-view geometries. A pair whose
  // intrinsics a two-view solver estimated is UNCALIBRATED too: its focal is
  // re-estimated from F rather than trusted, so any estimated intrinsics in
  // camera1/camera2 are ignored here and cleared on upgrade to CALIBRATED.
  // TODO: study whether seeding from the estimated camera1/camera2 intrinsics,
  // instead of ignoring them, improves the calibration initialization.
  std::vector<std::pair<image_pair_t, TwoViewGeometry>> pairs;
  for (auto& [pair_id, tvg] : database->ReadTwoViewGeometries()) {
    if (tvg.config != TwoViewGeometry::UNCALIBRATED &&
        tvg.config != TwoViewGeometry::CALIBRATED) {
      continue;
    }
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const Camera& camera1 = *image_id_to_camera.at(image_id1);
    const Camera& camera2 = *image_id_to_camera.at(image_id2);
    // A previous run of view graph calibration upgrades UNCALIBRATED pairs to
    // CALIBRATED and re-estimates cam2_from_cam1 with the VIEW_GRAPH cameras.
    // Since VIEW_GRAPH cameras are excluded above, downgrade any pair whose
    // cameras do not both have a non-VIEW_GRAPH prior back to UNCALIBRATED so
    // its pixel-space F is not overwritten using GUESS intrinsics below.
    if (tvg.config == TwoViewGeometry::CALIBRATED && !camera1.IsSpherical() &&
        !camera2.IsSpherical() &&
        (!camera1.HasPriorFocalLength() || !camera2.HasPriorFocalLength())) {
      tvg.config = TwoViewGeometry::UNCALIBRATED;
    }
    pairs.emplace_back(pair_id, std::move(tvg));
  }

  if (pairs.empty()) {
    LOG(WARNING) << "No image pairs to calibrate";
    return true;
  }

  LOG(INFO) << "Calibrating view graph with " << pairs.size() << " pairs";

  if (options.cross_validate_prior_focal_lengths) {
    CrossValidatePriorFocalLengths(
        options.min_calibrated_pair_ratio, image_id_to_camera, pairs);
  }

  // Recompute F from E for CALIBRATED pairs using current calibration. This
  // goes through the calibration matrices, so it is restricted to pinhole
  // models for the same reason as CalibrateFocalLengths below.
  for (auto& [pair_id, tvg] : pairs) {
    if (tvg.config != TwoViewGeometry::CALIBRATED ||
        !tvg.cam2_from_cam1.has_value()) {
      continue;
    }
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const Camera& camera1 = *image_id_to_camera.at(image_id1);
    const Camera& camera2 = *image_id_to_camera.at(image_id2);
    if (!camera1.IsPerspectivePinhole() || !camera2.IsPerspectivePinhole()) {
      continue;
    }
    tvg.F = FundamentalFromEssentialMatrix(
        camera2.CalibrationMatrix(),
        EssentialMatrixFromPose(*tvg.cam2_from_cam1),
        camera1.CalibrationMatrix());
  }

  // Prepare inputs and run Ceres optimization.
  std::vector<FocalLengthCalibInput> inputs;
  inputs.reserve(pairs.size());
  for (const auto& [pair_id, tvg] : pairs) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const Camera& camera1 = *image_id_to_camera.at(image_id1);
    const Camera& camera2 = *image_id_to_camera.at(image_id2);
    if (!camera1.IsPerspectivePinhole() || !camera2.IsPerspectivePinhole()) {
      continue;
    }
    THROW_CHECK(tvg.F.has_value())
        << "Two-view geometry must have F matrix for focal length calibration";
    inputs.push_back({pair_id,
                      camera1.camera_id,
                      camera2.camera_id,
                      tvg.F.value(),
                      tvg.inlier_matches.size()});
  }

  const FocalLengthCalibResult calib_result =
      CalibrateFocalLengths(options, inputs, cameras);
  if (!calib_result.success) {
    return false;
  }

  // Update cameras with estimated focal lengths.
  for (const auto& [camera_id, focal_length] : calib_result.focal_lengths) {
    Camera& camera = cameras.at(camera_id);
    camera.SetFocalLength(focal_length);
    camera.source = CameraSource::VIEW_GRAPH;
    database->UpdateCamera(camera);
  }

  // Process pairs: tag degenerate or compute E matrix.
  size_t invalid_counter = 0;
  std::vector<size_t> valid_pair_indices;

  for (size_t i = 0; i < pairs.size(); ++i) {
    auto& [pair_id, tvg] = pairs[i];
    if (tvg.config != TwoViewGeometry::CALIBRATED &&
        tvg.config != TwoViewGeometry::UNCALIBRATED)
      continue;

    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const auto error_it = calib_result.calibration_errors.find(pair_id);
    if (error_it == calib_result.calibration_errors.end()) {
      continue;
    }

    if (error_it->second > options.max_calibration_error) {
      invalid_counter++;
      tvg.config = TwoViewGeometry::DEGENERATE;
      database->UpdateTwoViewGeometry(image_id1, image_id2, tvg);
    } else {
      THROW_CHECK(tvg.F.has_value())
          << "Two-view geometry must have F matrix for E computation";
      const Camera& camera1 = *image_id_to_camera.at(image_id1);
      const Camera& camera2 = *image_id_to_camera.at(image_id2);
      tvg.E = EssentialFromFundamentalMatrix(camera2.CalibrationMatrix(),
                                             tvg.F.value(),
                                             camera1.CalibrationMatrix());
      tvg.config = TwoViewGeometry::CALIBRATED;
      // E is now built from the K of the calibrated focals, and consumers
      // calibrate the rays with camera1/camera2 whenever set. Clear them so
      // the rays use the same K as E, not the solver's own focal.
      tvg.camera1.reset();
      tvg.camera2.reset();
      valid_pair_indices.push_back(i);
    }
  }
  LOG(INFO) << "Invalid / total number of two-view geometry: "
            << invalid_counter << " / " << pairs.size();

  // Re-estimate relative poses for valid pairs.
  if (options.reestimate_relative_pose && !valid_pair_indices.empty()) {
    std::vector<std::pair<image_pair_t, TwoViewGeometry>> valid_pairs;
    valid_pairs.reserve(valid_pair_indices.size());
    for (size_t idx : valid_pair_indices) {
      valid_pairs.push_back(std::move(pairs[idx]));
    }
    ReestimateRelativePoses(options, valid_pairs, image_id_to_camera, database);
    for (const auto& [pair_id, tvg] : valid_pairs) {
      const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
      database->UpdateTwoViewGeometry(image_id1, image_id2, tvg);
    }
  } else {
    for (const size_t idx : valid_pair_indices) {
      const auto& [pair_id, tvg] = pairs[idx];
      const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
      database->UpdateTwoViewGeometry(image_id1, image_id2, tvg);
    }
  }

  return true;
}

}  // namespace colmap
