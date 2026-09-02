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

#include "colmap/estimators/global_positioning.h"

#include "colmap/scene/database_cache.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <utility>

#include <gtest/gtest.h>

namespace colmap {
namespace {

std::vector<double> ExpectedObservationScales(
    const GlobalPositionerOptions& options,
    const Reconstruction& reconstruction) {
  std::vector<double> values;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options.min_num_view_per_track)) {
      continue;
    }
    for (const TrackElement& observation : point3D.track.Elements()) {
      if (!reconstruction.ExistsImage(observation.image_id)) continue;
      const Image& image = reconstruction.Image(observation.image_id);
      if (!image.HasPose()) continue;
      const std::optional<Eigen::Vector3d> cam_ray =
          image.CameraPtr()->CamRayFromImg(
              image.Point2D(observation.point2D_idx).xy);
      if (!cam_ray.has_value()) continue;
      const Eigen::Vector3d direction =
          image.CamFromWorld().rotation().inverse() * *cam_ray;
      const Eigen::Vector3d translation =
          point3D.xyz - image.FramePtr()->RigFromWorld().TgtOriginInSrc();
      values.push_back(std::max(
          1e-5, direction.dot(translation) / translation.squaredNorm()));
    }
  }
  return values;
}

std::vector<double> ObservationScaleValues(ceres::Problem& problem) {
  std::vector<double*> parameter_blocks;
  problem.GetParameterBlocks(&parameter_blocks);
  std::vector<double> values;
  for (double* parameter_block : parameter_blocks) {
    if (problem.ParameterBlockSize(parameter_block) == 1) {
      values.push_back(*parameter_block);
    }
  }
  std::sort(values.begin(), values.end());
  return values;
}

std::pair<size_t, size_t> ObservationScaleConstantCounts(
    const GlobalPositionerOptions& options) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 1;
  dataset_options.num_frames_per_rig = 4;
  dataset_options.num_points3D = 30;
  SynthesizeDataset(dataset_options, &reconstruction);

  auto positioner =
      CreateDefaultGlobalPositioner(options, PoseGraph(), &reconstruction);
  ceres::Problem& problem = positioner->Problem();
  std::vector<double*> parameter_blocks;
  problem.GetParameterBlocks(&parameter_blocks);
  size_t num_scales = 0;
  size_t num_constant_scales = 0;
  for (double* parameter_block : parameter_blocks) {
    if (problem.ParameterBlockSize(parameter_block) != 1) continue;
    ++num_scales;
    if (problem.IsParameterBlockConstant(parameter_block)) {
      ++num_constant_scales;
    }
  }
  return {num_constant_scales, num_scales};
}

std::pair<ObservationCovarianceMap, double> ObservationCovariancesAndCost(
    const GlobalPositionerOptions& options,
    const Reconstruction& reconstruction) {
  const Eigen::Vector3d standard_deviations(0.5, 1.0, 2.0);
  const Eigen::Matrix3d camera_covariance =
      standard_deviations.array().square().matrix().asDiagonal();
  const Eigen::Matrix3d camera_whitening =
      standard_deviations.cwiseInverse().asDiagonal();
  ObservationCovarianceMap covariances;
  double expected_cost = 0.0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options.min_num_view_per_track)) {
      continue;
    }
    const std::vector<TrackElement>& observations = point3D.track.Elements();
    for (std::size_t index = 0; index < observations.size(); ++index) {
      const TrackElement& observation = observations[index];
      if (!reconstruction.ExistsImage(observation.image_id)) continue;
      const Image& image = reconstruction.Image(observation.image_id);
      const std::optional<Eigen::Vector3d> camera_ray =
          image.CameraPtr()->CamRayFromImg(
              image.Point2D(observation.point2D_idx).xy);
      if (!image.HasPose() || !camera_ray.has_value()) {
        continue;
      }
      const Eigen::Matrix3d cam_from_world =
          image.CamFromWorld().rotation().toRotationMatrix();
      covariances.emplace(
          Point3DTrackElementKey{point3D_id, static_cast<uint64_t>(index)},
          cam_from_world.transpose() * camera_covariance * cam_from_world);

      const Eigen::Vector3d frame_center =
          image.FramePtr()->RigFromWorld().TgtOriginInSrc();
      Eigen::Vector3d point_from_center = point3D.xyz - frame_center;
      if (!image.IsRefInFrame()) {
        const Rig& rig = reconstruction.Rig(image.FramePtr()->RigId());
        const Rigid3d& cam_from_rig =
            rig.SensorFromRig(image.CameraPtr()->SensorId());
        point_from_center += image.CamFromWorld().rotation().inverse() *
                             cam_from_rig.translation();
      }
      const Eigen::Vector3d residual =
          image.CamFromWorld().rotation().inverse() * (*camera_ray) -
          point_from_center;
      expected_cost +=
          0.5 * (camera_whitening * cam_from_world * residual).squaredNorm();
    }
  }
  return {std::move(covariances), expected_cost};
}

Reconstruction CreateGlobalPositioningTestReconstruction() {
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 4;
  options.num_points3D = 30;
  SynthesizeDataset(options, &reconstruction);
  return reconstruction;
}

TEST(GlobalPositioning, Nominal) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  DatabaseCache database_cache;
  DatabaseCache::Options cache_options;
  database_cache.Load(*database, cache_options);

  PoseGraph pose_graph;
  pose_graph.Load(*database_cache.CorrespondenceGraph());

  // Copy GT reconstruction and keep only rotations (reset translations).
  Reconstruction reconstruction = gt_reconstruction;
  for (const auto& [frame_id, _] : reconstruction.Frames()) {
    Frame& frame = reconstruction.Frame(frame_id);
    frame.SetRigFromWorld(
        Rigid3d(frame.RigFromWorld().rotation(), Eigen::Vector3d::Zero()));
  }

  GlobalPositionerOptions options;
  options.use_gpu = false;
  options.random_seed = 42;
  options.solver_options.minimizer_progress_to_stdout = false;

  const bool success =
      RunGlobalPositioning(options, pose_graph, reconstruction);
  ASSERT_TRUE(success);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.5,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

TEST(GlobalPositioning, ComposableProblem) {
  Reconstruction reconstruction = CreateGlobalPositioningTestReconstruction();
  GlobalPositionerOptions options;
  options.use_gpu = false;
  auto loss = std::make_shared<ceres::CauchyLoss>(0.1);
  auto positioner = CreateDefaultGlobalPositioner(
      options, PoseGraph(), &reconstruction, {}, loss);
  loss.reset();

  const frame_t frame_id = reconstruction.Images().begin()->second.FrameId();
  double* center = positioner->FrameCenterParameterBlock(frame_id);
  ASSERT_NE(center, nullptr);
  EXPECT_TRUE(positioner->Problem().HasParameterBlock(center));
  EXPECT_EQ(positioner->FrameCenterParameterBlock(kInvalidFrameId), nullptr);

  double external_scale = 1.0;
  positioner->Problem().AddParameterBlock(&external_scale, 1);
  positioner->SetParameterBlockOrdering();
  const auto& ordering = *positioner->SolverOptions().linear_solver_ordering;
  EXPECT_EQ(ordering.GroupId(&external_scale), 0);
  EXPECT_EQ(ordering.NumElements(), positioner->Problem().NumParameterBlocks());

  ceres::Solver::Summary summary;
  ceres::Solve(positioner->SolverOptions(), &positioner->Problem(), &summary);
  EXPECT_TRUE(positioner->Finalize(summary));
}

TEST(GlobalPositioning, ObservationScaleGaugeOptions) {
  GlobalPositionerOptions options;
  options.use_gpu = false;
  EXPECT_TRUE(options.fix_observation_scale_gauge);

  const auto [default_constants, default_scales] =
      ObservationScaleConstantCounts(options);
  ASSERT_GT(default_scales, 0);
  EXPECT_EQ(default_constants, 1);

  options.fix_observation_scale_gauge = false;
  const auto [unfixed_constants, unfixed_scales] =
      ObservationScaleConstantCounts(options);
  EXPECT_EQ(unfixed_scales, default_scales);
  EXPECT_EQ(unfixed_constants, 0);

  options.optimize_scales = false;
  const auto [disabled_constants, disabled_scales] =
      ObservationScaleConstantCounts(options);
  EXPECT_EQ(disabled_scales, default_scales);
  EXPECT_EQ(disabled_constants, disabled_scales);
}

TEST(GlobalPositioning, UncalibratedObservationDownweightOption) {
  const Reconstruction reconstruction =
      CreateGlobalPositioningTestReconstruction();

  auto evaluate_cost = [&reconstruction](
                           const bool calibrated,
                           const bool downweight_uncalibrated_observations) {
    Reconstruction test_reconstruction = reconstruction;
    for (const auto& [camera_id, _] : test_reconstruction.Cameras()) {
      test_reconstruction.Camera(camera_id).has_prior_focal_length = calibrated;
    }

    GlobalPositionerOptions options;
    options.use_gpu = false;
    options.generate_random_positions = false;
    options.generate_random_points = false;
    options.downweight_uncalibrated_observations =
        downweight_uncalibrated_observations;
    auto positioner =
        CreateDefaultGlobalPositioner(options,
                                      PoseGraph(),
                                      &test_reconstruction,
                                      {},
                                      std::make_shared<ceres::TrivialLoss>());
    double cost = 0.0;
    EXPECT_TRUE(positioner->Problem().Evaluate(
        ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr));
    return cost;
  };

  GlobalPositionerOptions default_options;
  EXPECT_TRUE(default_options.downweight_uncalibrated_observations);
  const double calibrated_cost = evaluate_cost(true, true);
  const double stock_uncalibrated_cost = evaluate_cost(false, true);
  const double equal_weight_uncalibrated_cost = evaluate_cost(false, false);
  ASSERT_GT(calibrated_cost, 0.0);
  EXPECT_NEAR(
      stock_uncalibrated_cost, 0.5 * calibrated_cost, 1e-12 * calibrated_cost);
  EXPECT_NEAR(
      equal_weight_uncalibrated_cost, calibrated_cost, 1e-12 * calibrated_cost);
}

TEST(GlobalPositioning, InitializesWarmStartScalesFromCurrentScene) {
  Reconstruction reconstruction = CreateGlobalPositioningTestReconstruction();
  GlobalPositionerOptions options;
  options.use_gpu = false;
  options.generate_random_positions = false;
  options.generate_random_points = false;
  options.generate_scales = false;
  std::vector<double> expected_scales =
      ExpectedObservationScales(options, reconstruction);
  std::sort(expected_scales.begin(), expected_scales.end());

  auto positioner =
      CreateDefaultGlobalPositioner(options, PoseGraph(), &reconstruction);
  EXPECT_EQ(ObservationScaleValues(positioner->Problem()), expected_scales);
}

TEST(GlobalPositioning, KeyedObservationCovariances) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 2;
  dataset_options.num_frames_per_rig = 4;
  dataset_options.num_points3D = 30;
  SynthesizeDataset(dataset_options, &reconstruction);

  GlobalPositionerOptions options;
  options.use_gpu = false;
  options.generate_random_positions = false;
  options.generate_random_points = false;
  options.downweight_uncalibrated_observations = false;
  auto [covariances, expected_cost] =
      ObservationCovariancesAndCost(options, reconstruction);
  ASSERT_FALSE(covariances.empty());

  Reconstruction weighted_reconstruction = reconstruction;
  auto weighted =
      CreateDefaultGlobalPositioner(options,
                                    PoseGraph(),
                                    &weighted_reconstruction,
                                    covariances,
                                    std::make_shared<ceres::TrivialLoss>());
  double weighted_cost = 0.0;
  ASSERT_TRUE(weighted->Problem().Evaluate(ceres::Problem::EvaluateOptions(),
                                           &weighted_cost,
                                           nullptr,
                                           nullptr,
                                           nullptr));
  EXPECT_NEAR(weighted_cost, expected_cost, 1e-10);

  ObservationCovarianceMap missing = covariances;
  missing.erase(missing.begin());
  Reconstruction missing_reconstruction = reconstruction;
  EXPECT_THROW(CreateDefaultGlobalPositioner(
                   options, PoseGraph(), &missing_reconstruction, missing),
               std::invalid_argument);
}

TEST(GlobalPositioning, MultiCameraRig) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  DatabaseCache database_cache;
  DatabaseCache::Options cache_options;
  database_cache.Load(*database, cache_options);

  PoseGraph pose_graph;
  pose_graph.Load(*database_cache.CorrespondenceGraph());

  // Copy GT reconstruction and keep only rotations (reset translations).
  Reconstruction reconstruction = gt_reconstruction;
  for (const auto& [frame_id, _] : reconstruction.Frames()) {
    Frame& frame = reconstruction.Frame(frame_id);
    frame.SetRigFromWorld(
        Rigid3d(frame.RigFromWorld().rotation(), Eigen::Vector3d::Zero()));
  }

  GlobalPositionerOptions options;
  options.use_gpu = false;
  options.random_seed = 42;
  options.solver_options.minimizer_progress_to_stdout = false;

  const bool success =
      RunGlobalPositioning(options, pose_graph, reconstruction);
  ASSERT_TRUE(success);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.5,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

TEST(GlobalPositioning, RefineSensorFromRigFalsePreservesRig) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  // Multi-camera rig so the sensor offsets are non-trivial — both
  // rotation and translation must round-trip.
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  DatabaseCache database_cache;
  DatabaseCache::Options cache_options;
  database_cache.Load(*database, cache_options);

  PoseGraph pose_graph;
  pose_graph.Load(*database_cache.CorrespondenceGraph());

  // Copy GT reconstruction and keep only rotations on frames (reset
  // their translations); leave the rig calibration as-is.
  Reconstruction reconstruction = gt_reconstruction;
  for (const auto& [frame_id, _] : reconstruction.Frames()) {
    Frame& frame = reconstruction.Frame(frame_id);
    frame.SetRigFromWorld(
        Rigid3d(frame.RigFromWorld().rotation(), Eigen::Vector3d::Zero()));
  }

  // Snapshot the rig BEFORE GP.
  std::map<std::pair<rig_t, sensor_t>, Rigid3d> snapshot;
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
      ASSERT_TRUE(sensor_from_rig.has_value());
      snapshot[{rig_id, sensor_id}] = *sensor_from_rig;
    }
  }
  ASSERT_GT(snapshot.size(), 0u);

  GlobalPositionerOptions options;
  options.use_gpu = false;
  options.random_seed = 42;
  options.solver_options.minimizer_progress_to_stdout = false;
  options.refine_sensor_from_rig = false;

  ASSERT_TRUE(RunGlobalPositioning(options, pose_graph, reconstruction));

  // Every sensor_from_rig must match the snapshot exactly.
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor_from_rig_after] : rig.NonRefSensors()) {
      ASSERT_TRUE(sensor_from_rig_after.has_value())
          << "rig_id=" << rig_id << ", sensor_id=" << sensor_id.id;
      const auto& sensor_from_rig_before = snapshot.at({rig_id, sensor_id});
      EXPECT_EQ(*sensor_from_rig_after, sensor_from_rig_before)
          << "rig_id=" << rig_id << ", sensor_id=" << sensor_id.id;
    }
  }
}

}  // namespace
}  // namespace colmap
