// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/scene/synthetic.h"

#include <limits>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

namespace colmap {
namespace {

using ImageDepths = FlatHashMap<point2D_t, std::pair<double, double>>;
using Depths = FlatHashMap<image_t, ImageDepths>;

Depths MakeDepths(const Reconstruction& reconstruction) {
  Depths depths;
  for (const auto& [id, point] : reconstruction.Points3D()) {
    for (const auto& obs : point.track.Elements()) {
      const double depth =
          (reconstruction.Image(obs.image_id).CamFromWorld() * point.xyz).z();
      depths[obs.image_id][obs.point2D_idx] = {depth, 0.1};
    }
  }
  return depths;
}

Reconstruction MakeRig() {
  Reconstruction reconstruction;
  SyntheticDatasetOptions data;
  data.num_rigs = 1;
  data.num_cameras_per_rig = 2;
  data.num_frames_per_rig = 3;
  data.num_points3D = 8;
  SynthesizeDataset(data, &reconstruction);
  auto& rig = reconstruction.Rig(reconstruction.Rigs().begin()->first);
  auto& transform = rig.SensorFromRig(rig.NonRefSensors().begin()->first);
  transform.rotation() = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY());
  transform.translation() = Eigen::Vector3d(0.3, -0.1, 0.2);
  for (image_t id : reconstruction.RegImageIds()) {
    auto& image = reconstruction.Image(id);
    for (auto& point : image.Points2D()) {
      if (point.HasPoint3D())
        point.xy = *image.CameraPtr()->ImgFromCam(
            image.CamFromWorld() *
            reconstruction.Point3D(point.point3D_id).xyz);
    }
  }
  return reconstruction;
}

TEST(AppendDepthConstraints, RigPosesAndParameterizations) {
  for (bool refine_frame : {false, true}) {
    for (bool refine_sensor : {false, true}) {
      auto reconstruction = MakeRig();
      auto depths = MakeDepths(reconstruction);
      BundleAdjustmentConfig config;
      for (image_t id : reconstruction.RegImageIds()) config.AddImage(id);
      BundleAdjustmentOptions options;
      options.refine_rig_from_world = refine_frame;
      options.refine_sensor_from_rig = refine_sensor;
      options.refine_focal_length = false;
      options.refine_extra_params = false;
      auto ba =
          CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
      auto& problem = *ba->Problem();
      std::vector<double*> blocks;
      problem.GetParameterBlocks(&blocks);
      FlatHashMap<double*, std::pair<const ceres::Manifold*, bool>> states;
      for (double* block : blocks)
        states.emplace(block,
                       std::make_pair(problem.GetManifold(block),
                                      problem.IsParameterBlockConstant(block)));
      const int residuals = problem.NumResiduals();
      double log_depth_scale = 0.0;
      size_t count = 0;
      for (const auto& [id, values] : depths)
        count += AppendBADepthConstraints(
            problem, reconstruction, id, values, &log_depth_scale);
      EXPECT_EQ(count, reconstruction.ComputeNumObservations());
      problem.SetParameterBlockConstant(&log_depth_scale);
      problem.GetParameterBlocks(&blocks);
      for (double* block : blocks) {
        const auto state = states.find(block);
        if (state == states.end()) {
          EXPECT_TRUE(problem.IsParameterBlockConstant(block));
        } else {
          EXPECT_EQ(problem.GetManifold(block), state->second.first);
          EXPECT_EQ(problem.IsParameterBlockConstant(block),
                    state->second.second);
        }
      }
      EXPECT_EQ(problem.NumResiduals(),
                residuals + reconstruction.ComputeNumObservations());
      double cost;
      ASSERT_TRUE(problem.Evaluate({}, &cost, nullptr, nullptr, nullptr));
      EXPECT_NEAR(cost, 0, 1e-15);
      for (const auto& [id, point] : reconstruction.Points3D())
        reconstruction.Point3D(id).xyz.z() += 0.1;
      auto result = ba->Solve();
      ASSERT_TRUE(result->IsSolutionUsable());
      ASSERT_TRUE(problem.Evaluate({}, &cost, nullptr, nullptr, nullptr));
      EXPECT_LT(cost, 1e-7);
    }
  }
}

TEST(AppendDepthConstraints, ScaledResiduals) {
  const Rigid3d cam_from_world;
  const double log_depth_scale = std::log(2.0);
  for (bool log_residual : {false, true}) {
    std::unique_ptr<ceres::CostFunction> cost(
        CamFromWorldDepthCostFunctor::Create(2.0,
                                             0.25,
                                             log_residual
                                                 ? DepthResidualType::kLog
                                                 : DepthResidualType::kLinear));
    for (double z : {3.0, -1.0}) {
      const Eigen::Vector3d point(0, 0, z);
      const double* parameters[] = {
          point.data(), cam_from_world.params.data(), &log_depth_scale};
      double residual, derivative;
      double* jacobians[] = {nullptr, nullptr, &derivative};
      ASSERT_TRUE(cost->Evaluate(parameters, &residual, jacobians));
      EXPECT_NEAR(residual,
                  z <= 0         ? 0
                  : log_residual ? 8 * std::log(0.75)
                                 : (z - 4) / 0.5,
                  1e-12);
      EXPECT_NEAR(derivative, z <= 0 ? 0 : log_residual ? -8 : -2 * z, 1e-12);
    }
  }
}

class BADepthConstraintsTest : public ::testing::Test {
 protected:
  size_t Append(const ImageDepths& depths,
                ceres::LossFunction* loss = nullptr) {
    return AppendBADepthConstraints(problem,
                                    reconstruction,
                                    image.ImageId(),
                                    depths,
                                    &log_depth_scale,
                                    options,
                                    loss);
  }

  Reconstruction reconstruction = MakeRig();
  const Image& image =
      reconstruction.Image(reconstruction.RegImageIds().back());
  ImageDepths depths = MakeDepths(reconstruction).at(image.ImageId());
  double log_depth_scale = 0.0;
  BADepthConstraintOptions options;
  ceres::Problem problem;
};

TEST_F(BADepthConstraintsTest, PartialDepthAndLosses) {
  std::vector<point2D_t> indices;
  for (point2D_t idx = 0; idx < image.NumPoints2D(); ++idx)
    if (image.Point2D(idx).HasPoint3D()) indices.push_back(idx);
  ASSERT_GE(indices.size(), 3);
  indices.resize(3);
  EXPECT_EQ(Append({}), 0);
  EXPECT_EQ(Append(depths), 0);
  EXPECT_EQ(problem.NumParameterBlocks(), 0);
  for (size_t i = 0; i < indices.size(); ++i) {
    auto& prior = depths.at(indices[i]);
    prior.first += 1;
    prior.second = 1;
    auto& point = reconstruction.Point3D(image.Point2D(indices[i]).point3D_id);
    problem.AddParameterBlock(point.xyz.data(), 3);
  }
  options.residual_type = DepthResidualType::kLinear;
  EXPECT_EQ(Append({{indices[0], depths.at(indices[0])}},
                   new ceres::ScaledLoss(nullptr, 10, ceres::TAKE_OWNERSHIP)),
            1);
  EXPECT_EQ(Append({{indices[2], depths.at(indices[2])}},
                   new ceres::ScaledLoss(nullptr, 30, ceres::TAKE_OWNERSHIP)),
            1);
  EXPECT_EQ(problem.NumParameterBlocks(), 5 + !image.IsRefInFrame());
  double cost;
  ASSERT_TRUE(problem.Evaluate({}, &cost, nullptr, nullptr, nullptr));
  EXPECT_NEAR(cost, 20, 1e-12);
}

TEST_F(BADepthConstraintsTest, InvalidInputs) {
  const auto idx = depths.begin()->first;
  auto& point = reconstruction.Point3D(image.Point2D(idx).point3D_id);
  problem.AddParameterBlock(point.xyz.data(), 3);
  for (double invalid : {0.0,
                         -1.0,
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(Append({{idx, {invalid, 1}}}), std::exception);
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
    log_depth_scale = invalid;
    EXPECT_THROW(Append({}), std::exception);
  }
  log_depth_scale = 0.0;
  EXPECT_THROW(Append({{idx, {2, 0}}}), std::exception);
  EXPECT_THROW(Append({{static_cast<point2D_t>(image.NumPoints2D()), {2, 1}}}),
               std::exception);
}

}  // namespace
}  // namespace colmap
