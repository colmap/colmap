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

#include "colmap/estimators/view_graph_calibration.h"

#include "colmap/estimators/cost_functions.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"

#include <unordered_set>

namespace colmap {
namespace {

// Input for focal length calibration: an image pair with its F matrix.
struct ImagePairFundamental {
  image_pair_t pair_id;
  Eigen::Matrix3d F;
  camera_t camera_id1;
  camera_t camera_id2;
};

// Result of focal length calibration.
struct FocalLengthCalibrationResult {
  // Optimized focal lengths per camera.
  std::unordered_map<camera_t, double> focal_lengths;
  // Squared calibration error per image pair.
  std::unordered_map<image_pair_t, double> calibration_errors_sq;
  // Whether the Ceres solver succeeded.
  bool success = false;
};

// Cross-validate prior focal lengths by checking the ratio of calibrated vs
// uncalibrated pairs per camera. UNCALIBRATED pairs are converted to
// CALIBRATED if both cameras have reliable priors (majority of pairs are
// calibrated).
void CrossValidatePriorFocalLengths(
    std::vector<std::pair<image_pair_t, TwoViewGeometry>>& pairs,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, camera_t>& image_to_camera) {
  // For each camera, count the number of calibrated vs uncalibrated pairs.
  // first: total count, second: calibrated count
  std::unordered_map<camera_t, std::pair<int, int>> camera_counter;

  for (const auto& [pair_id, tvg] : pairs) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const camera_t camera_id1 = image_to_camera.at(image_id1);
    const camera_t camera_id2 = image_to_camera.at(image_id2);

    const Camera& camera1 = cameras.at(camera_id1);
    const Camera& camera2 = cameras.at(camera_id2);
    if (!camera1.has_prior_focal_length || !camera2.has_prior_focal_length)
      continue;

    if (tvg.config == TwoViewGeometry::CALIBRATED) {
      camera_counter[camera_id1].first++;
      camera_counter[camera_id2].first++;
      camera_counter[camera_id1].second++;
      camera_counter[camera_id2].second++;
    } else if (tvg.config == TwoViewGeometry::UNCALIBRATED) {
      camera_counter[camera_id1].first++;
      camera_counter[camera_id2].first++;
    }
  }

  // Camera is valid if majority (>50%) of its pairs are calibrated.
  std::unordered_map<camera_t, bool> camera_validity;
  for (auto& [camera_id, counter] : camera_counter) {
    camera_validity[camera_id] = counter.second * 1. / counter.first > 0.5;
  }

  // Convert UNCALIBRATED pairs to CALIBRATED if both cameras are valid.
  for (auto& [pair_id, tvg] : pairs) {
    if (tvg.config != TwoViewGeometry::UNCALIBRATED) continue;

    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const camera_t camera_id1 = image_to_camera.at(image_id1);
    const camera_t camera_id2 = image_to_camera.at(image_id2);

    if (camera_validity[camera_id1] && camera_validity[camera_id2]) {
      tvg.config = TwoViewGeometry::CALIBRATED;
    }
  }
}

// Re-estimate relative poses for all pairs using calibrated cameras.
void ReestimateRelativePoses(
    const ViewGraphCalibrationOptions& options,
    std::vector<std::pair<image_pair_t, TwoViewGeometry>>& pairs,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, camera_t>& image_to_camera,
    Database* database) {
  LOG(INFO) << "Re-estimating relative poses for " << pairs.size() << " pairs";

  TwoViewGeometryOptions two_view_options;
  two_view_options.compute_relative_pose = true;
  two_view_options.ransac_options.max_error = options.relpose_max_error;
  two_view_options.min_num_inliers = options.relpose_min_num_inliers;
  two_view_options.min_inlier_ratio = options.relpose_min_inlier_ratio;
  if (options.random_seed >= 0) {
    two_view_options.ransac_options.random_seed = options.random_seed;
  }

  // Pre-read all keypoints and matches from database (SQLite is not thread-safe).
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>> image_points;
  std::vector<FeatureMatches> pair_matches(pairs.size());

  std::unordered_set<image_t> image_ids;
  for (const auto& [pair_id, tvg] : pairs) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    image_ids.insert(image_id1);
    image_ids.insert(image_id2);
  }

  for (const image_t image_id : image_ids) {
    const FeatureKeypoints keypoints = database->ReadKeypoints(image_id);
    std::vector<Eigen::Vector2d> points(keypoints.size());
    for (size_t j = 0; j < keypoints.size(); ++j) {
      points[j] = Eigen::Vector2d(keypoints[j].x, keypoints[j].y);
    }
    image_points[image_id] = std::move(points);
  }

  for (size_t i = 0; i < pairs.size(); ++i) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pairs[i].first);
    pair_matches[i] = database->ReadMatches(image_id1, image_id2);
  }

  // Parallel estimation.
  ThreadPool thread_pool(options.solver_options.num_threads);

  for (size_t i = 0; i < pairs.size(); ++i) {
    thread_pool.AddTask([&, i]() {
      auto& [pair_id, tvg] = pairs[i];
      const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);

      const camera_t camera_id1 = image_to_camera.at(image_id1);
      const camera_t camera_id2 = image_to_camera.at(image_id2);
      const Camera& camera1 = cameras.at(camera_id1);
      const Camera& camera2 = cameras.at(camera_id2);

      const std::vector<Eigen::Vector2d>& points1 = image_points.at(image_id1);
      const std::vector<Eigen::Vector2d>& points2 = image_points.at(image_id2);

      TwoViewGeometry new_tvg = EstimateCalibratedTwoViewGeometry(
          camera1, points1, camera2, points2, pair_matches[i], two_view_options);

      tvg = std::move(new_tvg);
    });
  }
  thread_pool.Wait();
}

// Core Ceres optimization for focal length calibration.
// This is a pure function with no I/O dependencies.
// See: "Stable Intrinsic Auto-Calibration from Fundamental Matrices of Devices
// with Uncorrelated Camera Parameters", Fetzer et al., WACV 2020.
FocalLengthCalibrationResult CalibrateFocalLengths(
    const ViewGraphCalibrationOptions& options,
    const std::vector<ImagePairFundamental>& inputs,
    const std::unordered_map<camera_t, Camera>& cameras) {
  FocalLengthCalibrationResult result;

  if (inputs.empty()) {
    result.success = true;
    return result;
  }

  // Initialize focal lengths from cameras.
  std::unordered_map<camera_t, double> focals;
  std::unordered_map<camera_t, double> original_focals;
  for (const auto& [camera_id, camera] : cameras) {
    focals[camera_id] = camera.MeanFocalLength();
    original_focals[camera_id] = camera.MeanFocalLength();
  }

  // Build Ceres problem.
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);
  auto loss_function = options.CreateLossFunction();

  for (const auto& input : inputs) {
    if (input.camera_id1 == input.camera_id2) {
      problem.AddResidualBlock(
          FetzerFocalLengthSameCameraCostFunctor::Create(
              input.F, cameras.at(input.camera_id1).PrincipalPoint()),
          loss_function.get(),
          &(focals[input.camera_id1]));
    } else {
      problem.AddResidualBlock(
          FetzerFocalLengthCostFunctor::Create(
              input.F,
              cameras.at(input.camera_id1).PrincipalPoint(),
              cameras.at(input.camera_id2).PrincipalPoint()),
          loss_function.get(),
          &(focals[input.camera_id1]),
          &(focals[input.camera_id2]));
    }
  }

  // Parameterize cameras (fix those with prior, set lower bound).
  size_t num_cameras = 0;
  for (const auto& [camera_id, camera] : cameras) {
    if (!problem.HasParameterBlock(&(focals[camera_id]))) continue;

    num_cameras++;
    problem.SetParameterLowerBound(&(focals[camera_id]), 0, 1e-3);
    if (camera.has_prior_focal_length) {
      problem.SetParameterBlockConstant(&(focals[camera_id]));
      num_cameras--;
    }
  }

  if (num_cameras == 0) {
    LOG(INFO) << "No cameras to optimize";
    result.focal_lengths = std::move(focals);
    result.success = true;
    return result;
  }

  // Set solver options.
  ceres::Solver::Options solver_options = options.solver_options;
  if (cameras.size() < 50) {
    solver_options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  } else {
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  }
  solver_options.num_threads = GetEffectiveNumThreads(solver_options.num_threads);
  solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);

  // Solve.
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  VLOG(2) << summary.FullReport();

  if (!summary.IsSolutionUsable()) {
    LOG(ERROR) << "Ceres solver failed";
    result.success = false;
    return result;
  }

  // Validate focal lengths and revert degenerate ones.
  size_t rejected_cameras = 0;
  for (const auto& [camera_id, camera] : cameras) {
    if (!problem.HasParameterBlock(&(focals[camera_id]))) continue;

    const double focal_length_ratio =
        focals[camera_id] / original_focals[camera_id];
    if (focal_length_ratio > options.max_focal_length_ratio ||
        focal_length_ratio < options.min_focal_length_ratio) {
      VLOG(2) << "Ignoring degenerate camera " << camera_id
              << " focal: " << focals[camera_id]
              << " original focal: " << original_focals[camera_id];
      rejected_cameras++;
      // Reset to original focal length.
      focals[camera_id] = original_focals[camera_id];
    }
  }
  LOG(INFO) << rejected_cameras
            << " cameras rejected in view graph calibration";

  result.focal_lengths = std::move(focals);

  // Evaluate calibration errors.
  ceres::Problem::EvaluateOptions eval_options;
  eval_options.num_threads = solver_options.num_threads;
  eval_options.apply_loss_function = false;
  std::vector<double> residuals;
  problem.Evaluate(eval_options, nullptr, &residuals, nullptr, nullptr);

  size_t residual_idx = 0;
  for (const auto& input : inputs) {
    const Eigen::Vector2d error(residuals[residual_idx],
                                residuals[residual_idx + 1]);
    result.calibration_errors_sq[input.pair_id] = error.squaredNorm();
    residual_idx += 2;
  }

  result.success = true;
  return result;
}

}  // namespace

bool CalibrateViewGraph(const ViewGraphCalibrationOptions& options,
                        Database* database) {
  THROW_CHECK_NOTNULL(database);

  // Read cameras and build image_id -> camera_id mapping.
  std::unordered_map<camera_t, Camera> cameras;
  for (const Camera& cam : database->ReadAllCameras()) {
    cameras[cam.camera_id] = cam;
  }
  std::unordered_map<image_t, camera_t> image_to_camera;
  for (const Image& img : database->ReadAllImages()) {
    image_to_camera[img.ImageId()] = img.CameraId();
  }

  // Read UNCALIBRATED and CALIBRATED two-view geometries.
  std::vector<std::pair<image_pair_t, TwoViewGeometry>> pairs;
  for (auto& [pair_id, tvg] : database->ReadTwoViewGeometries()) {
    if (tvg.config == TwoViewGeometry::UNCALIBRATED ||
        tvg.config == TwoViewGeometry::CALIBRATED) {
      pairs.emplace_back(pair_id, std::move(tvg));
    }
  }
  if (pairs.empty()) {
    LOG(WARNING) << "No image pairs to calibrate";
    return true;
  }
  LOG(INFO) << "Calibrating view graph with " << pairs.size() << " pairs";

  // Cross-validate prior focal lengths.
  if (options.cross_validate_prior_focal_lengths) {
    CrossValidatePriorFocalLengths(pairs, cameras, image_to_camera);
  }

  // Recompute F from E for CALIBRATED pairs using current calibration.
  for (auto& [pair_id, tvg] : pairs) {
    if (tvg.config != TwoViewGeometry::CALIBRATED) continue;
    if (!tvg.cam2_from_cam1.has_value()) continue;
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const Camera& camera1 = cameras.at(image_to_camera.at(image_id1));
    const Camera& camera2 = cameras.at(image_to_camera.at(image_id2));
    tvg.F = FundamentalFromEssentialMatrix(
        camera2.CalibrationMatrix(),
        EssentialMatrixFromPose(*tvg.cam2_from_cam1),
        camera1.CalibrationMatrix());
  }

  // Prepare inputs and run Ceres optimization.
  std::vector<ImagePairFundamental> inputs;
  inputs.reserve(pairs.size());
  for (const auto& [pair_id, tvg] : pairs) {
    if (tvg.config != TwoViewGeometry::CALIBRATED &&
        tvg.config != TwoViewGeometry::UNCALIBRATED)
      continue;
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    inputs.push_back({pair_id,
                      tvg.F,
                      image_to_camera.at(image_id1),
                      image_to_camera.at(image_id2)});
  }

  FocalLengthCalibrationResult calib_result =
      CalibrateFocalLengths(options, inputs, cameras);
  if (!calib_result.success) {
    return false;
  }

  // Update cameras in database with calibrated focal lengths.
  for (auto& [camera_id, camera] : cameras) {
    auto it = calib_result.focal_lengths.find(camera_id);
    if (it == calib_result.focal_lengths.end()) continue;
    for (const size_t idx : camera.FocalLengthIdxs()) {
      camera.params[idx] = it->second;
    }
    database->UpdateCamera(camera);
  }

  // Process pairs: tag degenerate or compute E matrix.
  const double max_calibration_error_sq =
      options.max_calibration_error * options.max_calibration_error;
  size_t invalid_counter = 0;
  std::vector<size_t> valid_pair_indices;

  for (size_t i = 0; i < pairs.size(); ++i) {
    auto& [pair_id, tvg] = pairs[i];
    if (tvg.config != TwoViewGeometry::CALIBRATED &&
        tvg.config != TwoViewGeometry::UNCALIBRATED)
      continue;

    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    auto error_it = calib_result.calibration_errors_sq.find(pair_id);
    if (error_it == calib_result.calibration_errors_sq.end()) continue;

    if (error_it->second > max_calibration_error_sq) {
      invalid_counter++;
      tvg.config = TwoViewGeometry::DEGENERATE_VGC;
      database->UpdateTwoViewGeometry(image_id1, image_id2, tvg);
    } else {
      const Camera& c1 = cameras.at(image_to_camera.at(image_id1));
      const Camera& c2 = cameras.at(image_to_camera.at(image_id2));
      tvg.E = c2.CalibrationMatrix().transpose() * tvg.F *
              c1.CalibrationMatrix();
      tvg.config = TwoViewGeometry::CALIBRATED;
      valid_pair_indices.push_back(i);
    }
  }
  LOG(INFO) << "Invalid / total number of two-view geometry: " << invalid_counter
            << " / " << pairs.size();

  // Re-estimate relative poses for valid pairs.
  if (options.reestimate_relative_pose && !valid_pair_indices.empty()) {
    std::vector<std::pair<image_pair_t, TwoViewGeometry>> valid_pairs;
    valid_pairs.reserve(valid_pair_indices.size());
    for (size_t idx : valid_pair_indices) {
      valid_pairs.push_back(std::move(pairs[idx]));
    }
    ReestimateRelativePoses(
        options, valid_pairs, cameras, image_to_camera, database);
    for (auto& [pair_id, tvg] : valid_pairs) {
      const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
      if (tvg.config == TwoViewGeometry::DEGENERATE ||
          tvg.config == TwoViewGeometry::UNDEFINED) {
        tvg.config = TwoViewGeometry::DEGENERATE_VGC;
      }
      database->UpdateTwoViewGeometry(image_id1, image_id2, tvg);
    }
  } else {
    for (size_t idx : valid_pair_indices) {
      auto& [pair_id, tvg] = pairs[idx];
      const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
      database->UpdateTwoViewGeometry(image_id1, image_id2, tvg);
    }
  }

  return true;
}

}  // namespace colmap
