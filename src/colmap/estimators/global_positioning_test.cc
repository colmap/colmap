// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/global_positioning.h"

#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/math/random.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <limits>
#include <map>
#include <utility>

#include <ceres/normal_prior.h>
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

Reconstruction CreateGlobalPositioningTestReconstruction(int num_cameras = 1) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = num_cameras;
  options.num_frames_per_rig = 4;
  options.num_points3D = 30;
  SynthesizeDataset(options, &reconstruction);
  return reconstruction;
}

TEST(GlobalPositioning, InvalidOptionsReportReason) {
  Reconstruction reconstruction;
  ObservationScaleMap scales;
  GlobalPositionerOptions invalid_weight, invalid_stddev;
  invalid_weight.uncalibrated_observation_weight = -1;
  invalid_stddev.experimental_observation_stddev = -1;
  for (const auto& [name, options] :
       {std::make_pair("uncalibrated_observation_weight", invalid_weight),
        std::make_pair("experimental_observation_stddev", invalid_stddev)}) {
    try {
      GlobalPositioner::CreateDefault(options, reconstruction, scales);
      FAIL() << "Invalid options were accepted";
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string(error.what()).find(name), std::string::npos);
    }
  }
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
  auto positioner_scales = GenerateObservationScales(
      reconstruction, options.initialize_scales_from_geometry);
  auto positioner = GlobalPositioner::CreateDefault(
      options, reconstruction, positioner_scales, loss);
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
  auto unordered_scales = GenerateObservationScales(
      reconstruction, options.initialize_scales_from_geometry);
  auto unordered = GlobalPositioner::CreateDefault(
      options, reconstruction, unordered_scales);
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
    reconstruction.Camera(camera_id).source = CameraSource::USER;
  }
  GlobalPositionerOptions options;
  options.use_gpu = false;
  options.generate_random_positions = false;
  options.generate_random_points = false;
  for (const double stddev : {1.0, 8.0}) {
    options.experimental_observation_stddev = stddev;
    const double expected_cost =
        ObservationWeightedCost(options, reconstruction);
    auto positioner_scales = GenerateObservationScales(
        reconstruction, options.initialize_scales_from_geometry);
    auto positioner =
        GlobalPositioner::CreateDefault(options,
                                        reconstruction,
                                        positioner_scales,
                                        std::make_shared<ceres::TrivialLoss>());
    double cost = 0.0;
    ASSERT_TRUE(positioner->Problem().Evaluate(
        ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr));
    EXPECT_GT(expected_cost, 0.0);
    EXPECT_NEAR(cost, expected_cost, 1e-7 * expected_cost);
  }
  for (const double stddev : {0.0, -1.0}) {
    options.experimental_observation_stddev = stddev;
    auto invalid_scales = GenerateObservationScales(reconstruction);
    EXPECT_ANY_THROW(GlobalPositioner::CreateDefault(
        options, reconstruction, invalid_scales));
  }

  // Force radial variance underflow after successful ray unprojection.
  for (const auto& [camera_id, _] : reconstruction.Cameras()) {
    reconstruction.Camera(camera_id).SetFocalLength(1e100);
  }
  options.experimental_observation_stddev = 1.0;
  auto positioner_scales = GenerateObservationScales(
      reconstruction, options.initialize_scales_from_geometry);
  auto positioner = GlobalPositioner::CreateDefault(
      options, reconstruction, positioner_scales);
  EXPECT_EQ(positioner->Problem().NumResidualBlocks(), 0);
}

TEST(GlobalPositioning, UncalibratedObservationWeight) {
  Reconstruction reconstruction = CreateGlobalPositioningTestReconstruction();
  for (const bool calibrated : {false, true}) {
    for (const auto& [camera_id, _] : reconstruction.Cameras()) {
      reconstruction.Camera(camera_id).source =
          calibrated ? CameraSource::USER : CameraSource::GUESS;
    }
    GlobalPositionerOptions options;
    options.use_gpu = false;
    options.uncalibrated_observation_weight = 0.2;
    auto positioner_scales = GenerateObservationScales(
        reconstruction, options.initialize_scales_from_geometry);
    auto positioner =
        GlobalPositioner::CreateDefault(options,
                                        reconstruction,
                                        positioner_scales,
                                        std::make_shared<ceres::TrivialLoss>());
    std::vector<ceres::ResidualBlockId> residuals;
    positioner->Problem().GetResidualBlocks(&residuals);
    ASSERT_FALSE(residuals.empty());
    double rho[3];
    positioner->Problem()
        .GetLossFunctionForResidualBlock(residuals.front())
        ->Evaluate(1.0, rho);
    EXPECT_DOUBLE_EQ(rho[0], calibrated ? 1.0 : 0.2);
  }
}

TEST(GlobalPositioning, ObservationScaleGauge) {
  for (const bool optimize : {false, true}) {
    for (const bool fix_gauge : {false, true}) {
      Reconstruction reconstruction =
          CreateGlobalPositioningTestReconstruction();
      GlobalPositionerOptions options;
      options.use_gpu = false;
      options.optimize_scales = optimize;
      options.fix_first_scale = fix_gauge;
      auto positioner_scales = GenerateObservationScales(
          reconstruction, options.initialize_scales_from_geometry);
      auto positioner = GlobalPositioner::CreateDefault(
          options, reconstruction, positioner_scales);
      auto& problem = positioner->Problem();
      std::vector<double*> blocks;
      problem.GetParameterBlocks(&blocks);
      int num_scales = 0;
      int num_constant = 0;
      for (double* block : blocks) {
        if (problem.ParameterBlockSize(block) != 1) continue;
        ++num_scales;
        num_constant += problem.IsParameterBlockConstant(block);
      }
      ASSERT_GT(num_scales, 1);
      EXPECT_EQ(num_constant,
                optimize ? static_cast<int>(fix_gauge) : num_scales);
    }
  }
}

TEST(GlobalPositioning, ScaleInitialization) {
  for (const bool from_geometry : {false, true}) {
    for (const bool optimize_points : {false, true}) {
      Reconstruction reconstruction =
          CreateGlobalPositioningTestReconstruction();
      GlobalPositionerOptions options;
      options.use_gpu = false;
      options.generate_random_positions = false;
      options.generate_random_points = false;
      options.optimize_points = optimize_points;
      options.initialize_scales_from_geometry = from_geometry;
      auto positioner_scales = GenerateObservationScales(
          reconstruction, options.initialize_scales_from_geometry);
      auto positioner = GlobalPositioner::CreateDefault(
          options, reconstruction, positioner_scales);
      std::vector<ceres::ResidualBlockId> residuals;
      positioner->Problem().GetResidualBlocks(&residuals);
      ASSERT_FALSE(residuals.empty());
      std::vector<double*> blocks;
      positioner->Problem().GetParameterBlocksForResidualBlock(
          residuals.front(), &blocks);
      ASSERT_EQ(blocks.size(), 3);
      // With noiseless observations, the geometric scale is inverse distance.
      const double distance = (Eigen::Map<Eigen::Vector3d>(blocks[1]) -
                               Eigen::Map<Eigen::Vector3d>(blocks[0]))
                                  .norm();
      EXPECT_NEAR(*blocks[2], from_geometry ? 1.0 / distance : 1.0, 1e-12);
    }
  }
}

TEST(GlobalPositioning, PreparationPreservesSuppliedState) {
  Reconstruction reconstruction = CreateGlobalPositioningTestReconstruction();
  const Reconstruction before = reconstruction;
  auto scales = GenerateObservationScales(reconstruction);
  for (auto& [observation, scale] : scales) scale = 0.25;
  const auto supplied = scales;
  GlobalPositionerOptions options;
  options.use_gpu = false;
  options.initialize_scales_from_geometry = true;
  auto positioner =
      GlobalPositioner::CreateDefault(options, reconstruction, scales);
  EXPECT_THAT(reconstruction, ReconstructionEq(before));
  EXPECT_EQ(scales, supplied);
}

TEST(GlobalPositioning, UnknownSensorOffsetInitialization) {
  Reconstruction reconstruction = CreateGlobalPositioningTestReconstruction(2);
  const rig_t rig_id = reconstruction.Rigs().begin()->first;
  const sensor_t sensor_id =
      reconstruction.Rig(rig_id).NonRefSensors().begin()->first;
  reconstruction.Rig(rig_id).SensorFromRig(sensor_id).translation().setConstant(
      std::numeric_limits<double>::quiet_NaN());
  GlobalPositionerOptions options;
  options.use_gpu = false;
  options.random_seed = 42;
  options.solver_options.max_num_iterations = 0;
  // Frame/point initialization must not restart the random sequence for
  // offsets.
  SetPRNGSeed(42);
  for (size_t i = 0;
       i < 3 * (reconstruction.NumFrames() + reconstruction.NumPoints3D());
       ++i) {
    RandomUniformReal(-1.0, 1.0);
  }
  const Eigen::Vector3d expected(RandomUniformReal(-1.0, 1.0),
                                 RandomUniformReal(-1.0, 1.0),
                                 RandomUniformReal(-1.0, 1.0));
  const Reconstruction original = reconstruction;
  for (const bool compose : {false, true}) {
    reconstruction = original;
    if (compose) {
      RandomizeReconstructionPositions(options, PoseGraph(), reconstruction);
      auto scales = GenerateObservationScales(reconstruction);
      auto prepared_options = options;
      prepared_options.random_seed = -1;
      auto positioner = GlobalPositioner::CreateDefault(
          prepared_options, reconstruction, scales);
      ASSERT_TRUE(positioner->Solve().IsSolutionUsable());
    } else {
      ASSERT_TRUE(RunGlobalPositioning(options, PoseGraph(), reconstruction));
    }
    EXPECT_LE(
        (reconstruction.Rig(rig_id).SensorFromRig(sensor_id).TgtOriginInSrc() -
         expected)
            .norm(),
        1e-14);
  }
}

TEST(GlobalPositioning, CustomFrameConstraint) {
  for (const bool optimize_positions : {false, true}) {
    Reconstruction reconstruction = CreateGlobalPositioningTestReconstruction();
    const Reconstruction original = reconstruction;
    auto scales = GenerateObservationScales(reconstruction);
    GlobalPositionerOptions options;
    options.use_gpu = false;
    options.optimize_positions = optimize_positions;
    options.min_num_view_per_track = reconstruction.NumImages() + 1;
    auto positioner =
        GlobalPositioner::CreateDefault(options, reconstruction, scales);
    const frame_t frame_id = reconstruction.Frames().begin()->first;
    double* center = positioner->FrameCenterParameterBlock(frame_id);
    ASSERT_NE(center, nullptr);
    EXPECT_EQ(positioner->Problem().NumParameterBlocks(),
              reconstruction.NumFrames());
    ASSERT_TRUE(positioner->Solve().IsSolutionUsable());
    EXPECT_THAT(reconstruction, ReconstructionEq(original));
    EXPECT_EQ(positioner->Problem().IsParameterBlockConstant(center),
              !optimize_positions);
    const Eigen::Vector3d initial = Eigen::Map<Eigen::Vector3d>(center);
    const Eigen::Vector3d target = initial + Eigen::Vector3d(1, 2, 3);
    positioner->Problem().AddResidualBlock(
        new ceres::NormalPrior(Eigen::Matrix3d::Identity(), target),
        nullptr,
        center);
    EXPECT_TRUE(
        positioner->SolverOptions().linear_solver_ordering->IsMember(center));
    ASSERT_TRUE(positioner->Solve().IsSolutionUsable());
    EXPECT_LT((reconstruction.Frame(frame_id).RigFromWorld().TgtOriginInSrc() -
               (optimize_positions ? target : initial))
                  .norm(),
              1e-6);
    for (const auto& [other_id, frame] : reconstruction.Frames()) {
      if (other_id == frame_id) continue;
      EXPECT_NE(positioner->FrameCenterParameterBlock(other_id), nullptr);
      EXPECT_EQ(frame.RigFromWorld().translation(),
                original.Frame(other_id).RigFromWorld().translation());
    }
  }
}

TEST(GlobalPositioning, RunnerInitializesOnlyOwnedState) {
  const Reconstruction original = CreateGlobalPositioningTestReconstruction();
  for (const bool supplied : {false, true}) {
    Reconstruction reconstruction = original;
    GlobalPositionerOptions options;
    options.use_gpu = false;
    options.random_seed = 42;
    options.solver_options.max_num_iterations = 0;
    auto scales = GenerateObservationScales(reconstruction);
    for (auto& [observation, scale] : scales) scale = 0.25;
    const auto initial_scales = scales;
    ASSERT_TRUE(RunGlobalPositioning(
        options, PoseGraph(), reconstruction, supplied ? &scales : nullptr));
    EXPECT_EQ(scales, initial_scales);
    for (const auto& [frame_id, frame] : reconstruction.Frames()) {
      const double change =
          (frame.RigFromWorld().translation() -
           original.Frame(frame_id).RigFromWorld().translation())
              .norm();
      if (supplied)
        EXPECT_LE(change, 1e-12);
      else
        EXPECT_GT(change, 1e-3);
    }
    for (const auto& [point_id, point] : reconstruction.Points3D()) {
      EXPECT_EQ(point.xyz == original.Point3D(point_id).xyz, supplied);
    }
  }
}

TEST(GlobalPositioning, CallerOwnedObservationScales) {
  Reconstruction reconstruction = CreateGlobalPositioningTestReconstruction(2);
  GlobalPositionerOptions options;
  options.use_gpu = false;
  options.optimize_positions = false;
  options.optimize_points = false;
  options.fix_first_scale = false;
  options.refine_sensor_from_rig = false;
  options.experimental_observation_stddev = 2.0;
  auto scales = GenerateObservationScales(reconstruction);
  const auto& point = reconstruction.Points3D().begin()->second;
  const auto observation = point.track.Element(0);
  const double expected =
      1.0 / (point.xyz -
             reconstruction.Image(observation.image_id).ProjectionCenter())
                .norm();
  double* scale = &scales.at(observation);
  *scale = 1.2 * expected;
  const TrackElement unused(kInvalidImageId, kInvalidPoint2DIdx);
  scales.emplace(unused, 3.0);
  auto positioner =
      GlobalPositioner::CreateDefault(options, reconstruction, scales);
  auto& problem = positioner->Problem();
  EXPECT_DOUBLE_EQ(*scale, 1.2 * expected);
  ASSERT_TRUE(problem.HasParameterBlock(scale));
  EXPECT_EQ(positioner->SolverOptions().linear_solver_ordering->GroupId(scale),
            0);
  EXPECT_FALSE(problem.HasParameterBlock(&scales.at(unused)));
  EXPECT_EQ(positioner->SolverOptions().linear_solver_ordering->GroupId(
                &scales.at(unused)),
            -1);
  scales.reserve(10000);
  EXPECT_EQ(&scales.at(observation), scale);
  ASSERT_TRUE(positioner->Solve().IsSolutionUsable());
  EXPECT_NEAR(*scale, expected, 1e-8);
  EXPECT_DOUBLE_EQ(scales.at(unused), 3.0);
}

TEST(GlobalPositioning, CallerOwnedObservationScaleValidation) {
  Reconstruction reconstruction = CreateGlobalPositioningTestReconstruction();
  const TrackElement observation =
      reconstruction.Points3D().begin()->second.track.Element(0);
  GlobalPositionerOptions options;
  options.use_gpu = false;
  auto scales = GenerateObservationScales(reconstruction);
  for (const double value : {0.5e-5,
                             0.0,
                             -1.0,
                             std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
    scales.at(observation) = value;
    EXPECT_THROW(
        GlobalPositioner::CreateDefault(options, reconstruction, scales),
        std::invalid_argument);
  }
  scales.erase(observation);
  EXPECT_THROW(GlobalPositioner::CreateDefault(options, reconstruction, scales),
               std::invalid_argument);
  scales.emplace(observation, 1e-5);
  auto positioner =
      GlobalPositioner::CreateDefault(options, reconstruction, scales);
  EXPECT_DOUBLE_EQ(scales.at(observation), 1e-5);
  EXPECT_DOUBLE_EQ(
      positioner->Problem().GetParameterLowerBound(&scales.at(observation), 0),
      1e-5);
  options.min_num_view_per_track = reconstruction.NumImages() + 1;
  ObservationScaleMap empty;
  auto filtered =
      GlobalPositioner::CreateDefault(options, reconstruction, empty);
  EXPECT_EQ(filtered->Problem().NumResidualBlocks(), 0);
}

TEST(GlobalPositioning, FirstScaleFollowsAdmittedObservationOrder) {
  Reconstruction reconstruction = CreateGlobalPositioningTestReconstruction();
  const auto& point = reconstruction.Points3D().begin()->second;
  const auto invalid = point.track.Element(0);
  reconstruction.Image(invalid.image_id)
      .Point2D(invalid.point2D_idx)
      .xy.setConstant(std::numeric_limits<double>::quiet_NaN());
  const auto expected = point.track.Element(1);
  auto scales = GenerateObservationScales(reconstruction);
  scales.emplace(TrackElement(kInvalidImageId, 0), 1.0);
  GlobalPositionerOptions options;
  options.use_gpu = false;
  auto positioner =
      GlobalPositioner::CreateDefault(options, reconstruction, scales);
  auto& problem = positioner->Problem();
  ASSERT_TRUE(problem.HasParameterBlock(&scales.at(expected)));
  for (auto& [observation, scale] : scales) {
    if (!problem.HasParameterBlock(&scale)) continue;
    EXPECT_EQ(problem.IsParameterBlockConstant(&scale),
              observation == expected);
  }
}

TEST(GlobalPositioning, ScaleInitializationUsesCameraCenter) {
  for (const double baseline : {0.0, 0.4}) {
    Reconstruction reconstruction;
    SyntheticDatasetOptions dataset_options;
    dataset_options.num_rigs = 1;
    dataset_options.num_cameras_per_rig = 2;
    dataset_options.num_frames_per_rig = 4;
    dataset_options.num_points3D = 10;
    dataset_options.camera_model_id = PinholeCameraModel::model_id;
    dataset_options.camera_params = {1000, 900, 512, 384};
    SynthesizeDataset(dataset_options, &reconstruction);
    reconstruction.Rig(1).SetSensorFromRig(
        sensor_t(SensorType::CAMERA, 2),
        Rigid3d(Eigen::Quaterniond(Eigen::AngleAxisd(
                    0.2, Eigen::Vector3d(1, 2, 3).normalized())),
                baseline * Eigen::Vector3d(1, -0.2, 0.3)));
    // Reproject after changing the extrinsics to obtain exact observations.
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      for (const auto& observation : point3D.track.Elements()) {
        Image& image = reconstruction.Image(observation.image_id);
        const auto pixel =
            image.CameraPtr()->ImgFromCam(image.CamFromWorld() * point3D.xyz);
        ASSERT_TRUE(pixel.has_value());
        image.Point2D(observation.point2D_idx).xy = *pixel;
      }
    }
    for (const bool weighted : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "baseline=" << baseline << " weighted=" << weighted);
      GlobalPositionerOptions options;
      options.use_gpu = false;
      options.generate_random_positions = false;
      options.generate_random_points = false;
      options.initialize_scales_from_geometry = true;
      if (weighted) options.experimental_observation_stddev = 1.0;
      auto scales = GenerateObservationScales(reconstruction, true);
      auto positioner =
          GlobalPositioner::CreateDefault(options, reconstruction, scales);
      auto& problem = positioner->Problem();
      ASSERT_GT(problem.NumResidualBlocks(), 0);
      ceres::Problem::EvaluateOptions evaluate_options;
      evaluate_options.apply_loss_function = false;
      double cost = 0;
      ASSERT_TRUE(
          problem.Evaluate(evaluate_options, &cost, nullptr, nullptr, nullptr));
      EXPECT_LT(cost, weighted ? 1e-20 : 1e-26);
    }
  }
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
