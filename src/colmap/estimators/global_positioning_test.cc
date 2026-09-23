// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/global_positioning.h"

#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <limits>
#include <map>
#include <utility>

#include <gtest/gtest.h>

namespace colmap {
namespace {

struct ScaleDifferenceCostFunctor {
  template <typename T>
  bool operator()(const T* scale, const T* extension, T* residual) const {
    residual[0] = scale[0] - extension[0];
    return true;
  }
};

double ObservationWeightedCost(const GlobalPositionerOptions& options,
                               const Reconstruction& reconstruction) {
  double expected_cost = 0.0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options.min_num_view_per_track)) {
      continue;
    }
    for (const auto& observation : point3D.track.Elements()) {
      const Image& image = reconstruction.Image(observation.image_id);
      const Camera& camera = *image.CameraPtr();
      const Eigen::Vector2d pixel = image.Point2D(observation.point2D_idx).xy;
      const Eigen::Vector3d bearing = *camera.CamRayFromImg(pixel);
      // Independently differentiate unprojection, then whiten in a tangent
      // basis.
      Eigen::Matrix3x2d jacobian;
      constexpr double step = 1e-3;
      for (int axis = 0; axis < 2; ++axis) {
        const Eigen::Vector2d delta = step * Eigen::Vector2d::Unit(axis);
        jacobian.col(axis) = (*camera.CamRayFromImg(pixel + delta) -
                              *camera.CamRayFromImg(pixel - delta)) /
                             (2 * step);
      }
      Eigen::Matrix3x2d tangent;
      tangent.col(0) = bearing.unitOrthogonal();
      tangent.col(1) = bearing.cross(tangent.col(0));
      const Eigen::Matrix2d J = tangent.transpose() * jacobian;
      const double stddev = *options.experimental_observation_stddev;
      const Eigen::Matrix2d precision =
          (stddev * stddev * J * J.transpose()).inverse();
      const Eigen::Vector3d frame_center =
          image.FramePtr()->RigFromWorld().TgtOriginInSrc();
      Eigen::Vector3d displacement = point3D.xyz - frame_center;
      if (!image.IsRefInFrame()) {
        const Rigid3d& cam_from_rig =
            image.FramePtr()->RigPtr()->SensorFromRig(camera.SensorId());
        displacement += image.CamFromWorld().rotation().inverse() *
                        cam_from_rig.translation();
      }
      const Eigen::Vector3d residual =
          bearing - image.CamFromWorld().rotation() * displacement;
      const Eigen::Vector2d transverse = tangent.transpose() * residual;
      const double longitudinal = bearing.dot(residual);
      expected_cost +=
          0.5 * (transverse.dot(precision * transverse) +
                 0.5 * precision.trace() * longitudinal * longitudinal);
    }
  }
  return expected_cost;
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
  options.random_seed = 42;
  auto loss = std::make_shared<ceres::CauchyLoss>(0.1);
  auto positioner = GlobalPositioner::CreateDefault(
      options, PoseGraph(), reconstruction, loss);
  loss.reset();

  const frame_t frame_id = reconstruction.Images().begin()->second.FrameId();
  double* center = positioner->FrameCenterParameterBlock(frame_id);
  ASSERT_NE(center, nullptr);
  EXPECT_TRUE(positioner->Problem().HasParameterBlock(center));
  EXPECT_EQ(positioner->FrameCenterParameterBlock(kInvalidFrameId), nullptr);

  const auto initial_ordering =
      positioner->SolverOptions().linear_solver_ordering;
  ASSERT_NE(initial_ordering, nullptr);
  const auto stock = *initial_ordering;
  auto& problem = positioner->Problem();
  const auto& native_scales = stock.group_to_elements().at(0);
  const auto native = std::find_if(
      native_scales.begin(), native_scales.end(), [&](double* scale) {
        return !problem.IsParameterBlockConstant(scale);
      });
  ASSERT_NE(native, native_scales.end());
  const auto point_id = reconstruction.Points3D().begin()->first;
  double* point = reconstruction.Point3D(point_id).xyz.data();
  double independent = 1.0, shared = 1.0, blocked = 1.0;
  double coupled[2] = {1.0, 1.0};
  Eigen::Vector2d external_vector = Eigen::Vector2d::Zero();
  problem.AddParameterBlock(external_vector.data(), 2);
  for (double* scale : {&independent, &shared, &shared}) {
    problem.AddResidualBlock(
        BATAPairwiseDirectionCostFunctor::Create(Eigen::Vector3d::UnitX()),
        nullptr,
        center,
        point,
        scale);
  }
  for (const auto& [a, b] :
       {std::pair{*native, &blocked}, std::pair{&coupled[0], &coupled[1]}}) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ScaleDifferenceCostFunctor, 1, 1, 1>(
            new ScaleDifferenceCostFunctor()),
        nullptr,
        a,
        b);
  }

  positioner->ExtendParameterBlockOrdering();
  const auto ordering = positioner->SolverOptions().linear_solver_ordering;
  EXPECT_EQ(ordering, initial_ordering);
  EXPECT_EQ(ordering->GroupId(&independent), 0);
  EXPECT_EQ(ordering->GroupId(&shared), 3);
  EXPECT_EQ(ordering->GroupId(&blocked), 3);
  EXPECT_EQ(ordering->GroupId(external_vector.data()), 3);
  EXPECT_EQ(ordering->NumElements(), problem.NumParameterBlocks());
  for (const auto& [group, blocks] : stock.group_to_elements()) {
    for (double* block : blocks) EXPECT_EQ(ordering->GroupId(block), group);
  }
  EXPECT_EQ(
      std::min(ordering->GroupId(&coupled[0]), ordering->GroupId(&coupled[1])),
      0);
  EXPECT_EQ(
      std::max(ordering->GroupId(&coupled[0]), ordering->GroupId(&coupled[1])),
      3);
  positioner->ExtendParameterBlockOrdering({{&independent, 3},
                                            {&shared, 0},
                                            {&coupled[0], 3},
                                            {&coupled[1], 0},
                                            {center, 4}});
  positioner->ExtendParameterBlockOrdering();
  EXPECT_EQ(ordering->GroupId(&independent), 3);
  EXPECT_EQ(ordering->GroupId(&shared), 0);
  EXPECT_EQ(ordering->GroupId(&coupled[0]), 3);
  EXPECT_EQ(ordering->GroupId(&coupled[1]), 0);
  EXPECT_EQ(ordering->GroupId(center), 4);
  ceres::Solver::Summary summary;
  ceres::Solve(positioner->SolverOptions(), &problem, &summary);
  EXPECT_TRUE(positioner->Finalize(summary));

  options.use_parameter_block_ordering = false;
  auto unordered =
      GlobalPositioner::CreateDefault(options, PoseGraph(), reconstruction);
  EXPECT_EQ(unordered->SolverOptions().linear_solver_ordering, nullptr);
  EXPECT_TRUE(unordered->Solve().IsSolutionUsable());
}

TEST(GlobalPositioning, ObservationUncertainty) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 2;
  dataset_options.num_frames_per_rig = 4;
  dataset_options.num_points3D = 30;
  SynthesizeDataset(dataset_options, &reconstruction);
  for (const auto& [camera_id, _] : reconstruction.Cameras()) {
    reconstruction.Camera(camera_id).has_prior_focal_length = true;
  }
  GlobalPositionerOptions options;
  options.use_gpu = false;
  options.generate_random_positions = false;
  options.generate_random_points = false;
  for (const double stddev : {1.0, 8.0}) {
    options.experimental_observation_stddev = stddev;
    const double expected_cost =
        ObservationWeightedCost(options, reconstruction);
    auto positioner =
        GlobalPositioner::CreateDefault(options,
                                        PoseGraph(),
                                        reconstruction,
                                        std::make_shared<ceres::TrivialLoss>());
    double cost = 0.0;
    ASSERT_TRUE(positioner->Problem().Evaluate(
        ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr));
    EXPECT_GT(expected_cost, 0.0);
    EXPECT_NEAR(cost, expected_cost, 1e-7 * expected_cost);
  }
  for (const double stddev : {0.0,
                              -1.0,
                              std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()}) {
    options.experimental_observation_stddev = stddev;
    EXPECT_ANY_THROW(
        GlobalPositioner::CreateDefault(options, PoseGraph(), reconstruction));
  }

  // Force radial variance underflow after successful ray unprojection.
  for (const auto& [camera_id, _] : reconstruction.Cameras()) {
    reconstruction.Camera(camera_id).SetFocalLength(1e100);
  }
  options.experimental_observation_stddev = 1.0;
  auto positioner =
      GlobalPositioner::CreateDefault(options, PoseGraph(), reconstruction);
  EXPECT_EQ(positioner->Problem().NumResidualBlocks(), 0);
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
  options.experimental_observation_stddev = 8.0;
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
