// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/rotation_averaging_ceres.h"

#include "colmap/scene/frame.h"
#include "colmap/scene/image.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

Eigen::Quaterniond ZRotation(const double angle) {
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()));
}

PoseGraph::Edge Edge(const Eigen::Quaterniond& rotation,
                     const int num_matches = 10) {
  PoseGraph::Edge edge;
  edge.cam2_from_cam1 = Rigid3d(rotation, Eigen::Vector3d::Zero());
  edge.num_matches = num_matches;
  return edge;
}

Reconstruction MakeTrivialReconstruction(const std::vector<image_t>& ids,
                                         const bool posed = true) {
  Reconstruction reconstruction;
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 10, 10, 5);
  reconstruction.AddCameraWithTrivialRig(camera);
  for (const image_t image_id : ids) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(camera.camera_id);
    reconstruction.AddImageWithTrivialFrame(image);
    if (posed) reconstruction.Frame(image_id).SetRigFromWorld(Rigid3d());
  }
  return reconstruction;
}

Reconstruction MakeRigReconstruction(const int num_rigs = 1,
                                     const int num_cameras = 2) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = num_rigs;
  options.num_cameras_per_rig = num_cameras;
  options.num_frames_per_rig = 3;
  options.num_points3D = 5;
  SynthesizeDataset(options, &reconstruction);
  for (const auto& [id, frame] : reconstruction.Frames()) {
    reconstruction.Frame(id).RigFromWorld().rotation() =
        Eigen::Quaterniond(
            Eigen::AngleAxisd(0.12 * id, Eigen::Vector3d::UnitX())) *
        ZRotation(0.17 * id);
  }
  for (const auto& [id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, pose] : rig.NonRefSensors()) {
      reconstruction.Rig(id).SensorFromRig(sensor_id).rotation() =
          Eigen::Quaterniond(
              Eigen::AngleAxisd(0.1 * sensor_id.id, Eigen::Vector3d::UnitY())) *
          ZRotation(0.07 * sensor_id.id);
    }
  }
  return reconstruction;
}

std::vector<std::vector<image_t>> RigImages(
    const Reconstruction& reconstruction) {
  std::map<frame_t, std::map<camera_t, image_t>> images;
  for (const auto& [id, image] : reconstruction.Images()) {
    images[image.FrameId()][image.CameraId()] = id;
  }
  std::vector<std::vector<image_t>> result;
  for (const auto& [_, frame] : images) {
    result.emplace_back();
    for (const auto& [camera, image] : frame) result.back().push_back(image);
  }
  return result;
}

Eigen::Quaterniond RelativeRotation(const Reconstruction& reconstruction,
                                    const image_t image1,
                                    const image_t image2) {
  return reconstruction.Image(image2).CamFromWorld().rotation() *
         reconstruction.Image(image1).CamFromWorld().rotation().inverse();
}

PoseGraph MakePoseGraph(const Reconstruction& reconstruction) {
  PoseGraph graph;
  for (const auto& [id1, image1] : reconstruction.Images()) {
    for (const auto& [id2, image2] : reconstruction.Images()) {
      if (id1 < id2) {
        graph.AddEdge(
            id1, id2, Edge(RelativeRotation(reconstruction, id1, id2)));
      }
    }
  }
  return graph;
}

void ExpectRelativeRotations(const Reconstruction& reconstruction,
                             const PoseGraph& graph,
                             const double tolerance) {
  for (const auto& [pair, edge] : graph.ValidEdges()) {
    const auto [id1, id2] = PairIdToImagePair(pair);
    EXPECT_NEAR(RelativeRotation(reconstruction, id1, id2)
                    .angularDistance(edge.cam2_from_cam1.rotation()),
                0,
                tolerance);
  }
}

TEST(CeresRotationAverager, RecoversNominalRotations) {
  Reconstruction reconstruction = MakeTrivialReconstruction({1, 2, 3});
  PoseGraph pose_graph;
  pose_graph.AddEdge(1, 2, Edge(ZRotation(0.2)));
  pose_graph.AddEdge(2, 3, Edge(ZRotation(-0.1)));
  CeresRotationAveragerOptions options;

  auto averager =
      CreateDefaultCeresRotationAverager(options, pose_graph, reconstruction);
  EXPECT_EQ(averager->Problem().NumParameterBlocks(), 3);
  EXPECT_EQ(averager->Problem().NumResidualBlocks(), 2);
  EXPECT_EQ(averager->SolverOptions().linear_solver_type,
            ceres::SPARSE_NORMAL_CHOLESKY);
  ASSERT_TRUE(averager->Solve().IsSolutionUsable());
  ExpectRelativeRotations(reconstruction, pose_graph, 1e-10);
}

TEST(CeresRotationAverager, MatchCountReweighting) {
  Reconstruction reconstruction = MakeTrivialReconstruction({1, 2, 3});
  CeresRotationAveragerOptions options;
  options.skip_initialization = true;
  options.loss_function_scale = 0.1;
  for (const int num_matches : {10, 0}) {
    PoseGraph graph;
    graph.AddEdge(1, 2, Edge(ZRotation(0.2), num_matches));
    graph.AddEdge(2, 3, Edge(ZRotation(0.2), num_matches / 2));
    graph.AddEdge(1, 3, Edge(ZRotation(0.2), 0));
    for (const auto reweighting :
         {RotationAveragingReweighting::UNIFORM,
          RotationAveragingReweighting::INLIER_MATCH_COUNT}) {
      options.reweighting = reweighting;
      auto averager =
          CreateDefaultCeresRotationAverager(options, graph, reconstruction);
      double cost;
      ASSERT_TRUE(averager->Problem().Evaluate(
          ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr));
      // Each Huber cost is 0.015; weights are either (1, 1, 1) or (1, 0.5, 0).
      const bool uniform =
          reweighting == RotationAveragingReweighting::UNIFORM ||
          num_matches == 0;
      EXPECT_NEAR(cost, uniform ? 0.045 : 0.0225, 1e-12);
    }
  }
}

TEST(CeresRotationAverager, AddsIndividualRelativeRotationResidual) {
  Reconstruction reconstruction = MakeTrivialReconstruction({1, 2});
  PoseGraph pose_graph;
  pose_graph.AddEdge(1, 2, Edge(ZRotation(0.2)));
  auto averager = CreateDefaultCeresRotationAverager(
      CeresRotationAveragerOptions(), pose_graph, reconstruction);

  auto loss = std::make_shared<ceres::CauchyLoss>(0.05);
  averager->AddRelativeRotationResidual(1, 2, ZRotation(0.2), loss);
  EXPECT_EQ(averager->Problem().NumResidualBlocks(), 2);
  std::weak_ptr<ceres::LossFunction> retained_loss = loss;
  loss.reset();
  EXPECT_FALSE(retained_loss.expired());
  EXPECT_TRUE(averager->Solve().IsSolutionUsable());
}

TEST(CeresRotationAverager, RejectsUnconfiguredRotationBlocks) {
  Reconstruction reconstruction = MakeRigReconstruction();
  const auto images = RigImages(reconstruction);
  PoseGraph graph;
  graph.AddEdge(images[0][0], images[1][0], Edge(ZRotation(0.2)));
  CeresRotationAveragerOptions options;
  options.skip_initialization = true;
  auto averager =
      CreateDefaultCeresRotationAverager(options, graph, reconstruction);

  for (const auto& [image1, image2] : {std::pair{images[2][0], images[0][0]},
                                       std::pair{images[0][0], images[2][0]},
                                       std::pair{images[0][1], images[1][0]},
                                       std::pair{images[0][0], images[1][1]}}) {
    EXPECT_THROW(averager->AddRelativeRotationResidual(
                     image1, image2, ZRotation(0.2), nullptr),
                 std::invalid_argument);
    EXPECT_EQ(averager->Problem().NumParameterBlocks(), 2);
    EXPECT_EQ(averager->Problem().NumResidualBlocks(), 1);
  }
}

TEST(CeresRotationAverager, SelectsMstOrSuppliedInitialization) {
  Reconstruction supplied = MakeTrivialReconstruction({1, 2});
  supplied.Frame(1).RigFromWorld().rotation() = ZRotation(0.7);
  supplied.Frame(2).RigFromWorld().rotation() = ZRotation(0.9);
  Reconstruction initialized = MakeTrivialReconstruction({1, 2}, false);
  PoseGraph pose_graph;
  pose_graph.AddEdge(1, 2, Edge(ZRotation(0.2)));

  auto mst = CreateDefaultCeresRotationAverager(
      CeresRotationAveragerOptions(), pose_graph, initialized);
  EXPECT_NEAR((initialized.Frame(2).RigFromWorld().rotation() *
               initialized.Frame(1).RigFromWorld().rotation().inverse())
                  .angularDistance(ZRotation(0.2)),
              0.0,
              1e-12);
  EXPECT_TRUE(
      initialized.Frame(1).RigFromWorld().translation().array().isNaN().all());

  CeresRotationAveragerOptions options;
  options.skip_initialization = true;
  auto preserved =
      CreateDefaultCeresRotationAverager(options, pose_graph, supplied);
  EXPECT_EQ(supplied.Frame(1).RigFromWorld().rotation().coeffs(),
            ZRotation(0.7).coeffs());
  EXPECT_EQ(supplied.Frame(2).RigFromWorld().rotation().coeffs(),
            ZRotation(0.9).coeffs());
}

TEST(CeresRotationAverager, RejectsDisconnectedPoseGraph) {
  Reconstruction reconstruction = MakeTrivialReconstruction({1, 2, 3, 4});
  PoseGraph pose_graph;
  pose_graph.AddEdge(1, 2, Edge(ZRotation(0.1)));
  pose_graph.AddEdge(3, 4, Edge(ZRotation(0.2)));
  EXPECT_THROW(CreateDefaultCeresRotationAverager(
                   CeresRotationAveragerOptions(), pose_graph, reconstruction),
               std::invalid_argument);
}

TEST(CeresRotationAverager, CalibratedRigs) {
  const Reconstruction truth = MakeRigReconstruction(2);
  const PoseGraph graph = MakePoseGraph(truth);
  for (const bool skip_initialization : {false, true}) {
    Reconstruction reconstruction = truth;
    const auto images = RigImages(reconstruction);
    for (const auto& [id, rig] : reconstruction.Rigs()) {
      for (const auto& [sensor, pose] : rig.NonRefSensors()) {
        reconstruction.Rig(id).SensorFromRig(sensor).translation()[0] =
            std::numeric_limits<double>::quiet_NaN();
      }
    }
    CeresRotationAveragerOptions options;
    options.skip_initialization = skip_initialization;
    options.refine_sensor_from_rig = false;
    auto averager =
        CreateDefaultCeresRotationAverager(options, graph, reconstruction);
    EXPECT_EQ(averager->Problem().NumParameterBlocks(),
              truth.NumFrames() + truth.NumRigs());
    EXPECT_EQ(averager->Problem().NumResidualBlocks(), graph.NumEdges());
    averager->AddRelativeRotationResidual(
        images[1][1],
        images[0][0],
        RelativeRotation(truth, images[1][1], images[0][0]),
        nullptr);
    EXPECT_EQ(averager->Problem().NumResidualBlocks(), graph.NumEdges() + 1);
    ASSERT_TRUE(averager->Solve().IsSolutionUsable());
    ExpectRelativeRotations(reconstruction, graph, 1e-8);
    for (const auto& [id, rig] : truth.Rigs()) {
      for (const auto& [sensor, pose] : rig.NonRefSensors()) {
        const auto rotation =
            reconstruction.Rig(id).SensorFromRig(sensor).rotation();
        EXPECT_TRUE(averager->Problem().IsParameterBlockConstant(
            rotation.coeffs().data()));
        EXPECT_EQ(rotation.coeffs(), pose->rotation().coeffs());
      }
    }
  }
}

TEST(CeresRotationAverager, CalibratedRigWithDisconnectedImageGraph) {
  Reconstruction reconstruction = MakeRigReconstruction();
  const auto images = RigImages(reconstruction);
  PoseGraph graph;
  graph.AddEdge(
      images[0][0],
      images[1][0],
      Edge(RelativeRotation(reconstruction, images[0][0], images[1][0])));
  graph.AddEdge(
      images[1][1],
      images[2][1],
      Edge(RelativeRotation(reconstruction, images[1][1], images[2][1])));
  for (const auto& [id, frame] : reconstruction.Frames()) {
    reconstruction.DeRegisterFrame(id);
  }
  auto averager = CreateDefaultCeresRotationAverager(
      CeresRotationAveragerOptions(), graph, reconstruction);
  ASSERT_TRUE(averager->Solve().IsSolutionUsable());
  ExpectRelativeRotations(reconstruction, graph, 1e-7);
}

TEST(CeresRotationAverager, EstimatesUncalibratedRigs) {
  const Reconstruction truth = MakeRigReconstruction(2);
  const PoseGraph graph = MakePoseGraph(truth);
  // Missing calibration, rotation-only calibration, and supplied warm start.
  for (const int mode : {0, 1, 2}) {
    Reconstruction reconstruction = truth;
    for (const auto& [id, rig] : truth.Rigs()) {
      for (const auto& [sensor, pose] : rig.NonRefSensors()) {
        if (mode == 0) {
          reconstruction.Rig(id).ResetSensorFromRig(sensor);
        } else {
          auto& initial = reconstruction.Rig(id).SensorFromRig(sensor);
          initial.rotation() = ZRotation(0.15) * initial.rotation();
          initial.translation()[0] = std::numeric_limits<double>::quiet_NaN();
        }
      }
    }
    CeresRotationAveragerOptions options;
    options.skip_initialization = mode == 2;
    options.solver_options.function_tolerance = 1e-12;
    options.solver_options.gradient_tolerance = 1e-12;
    options.solver_options.parameter_tolerance = 1e-12;
    auto averager =
        CreateDefaultCeresRotationAverager(options, graph, reconstruction);
    EXPECT_EQ(averager->Problem().NumParameterBlocks(),
              truth.NumFrames() + truth.NumRigs());
    ASSERT_TRUE(averager->Solve().IsSolutionUsable());
    for (const auto& [id, rig] : truth.Rigs()) {
      for (const auto& [sensor, pose] : rig.NonRefSensors()) {
        const auto& estimated = reconstruction.Rig(id).SensorFromRig(sensor);
        EXPECT_NEAR(
            estimated.rotation().angularDistance(pose->rotation()), 0, 1e-7);
        if (mode == 0) {
          EXPECT_TRUE(estimated.translation().array().isNaN().all());
        }
      }
    }
    ExpectRelativeRotations(reconstruction, graph, 1e-7);
  }
}

TEST(CeresRotationAverager, RefinesCalibrationThroughPreparedProblem) {
  const Reconstruction truth = MakeRigReconstruction();
  Reconstruction reconstruction = truth;
  const auto images = RigImages(truth);
  PoseGraph graph;
  graph.AddEdge(images[0][0],
                images[0][1],
                Edge(RelativeRotation(truth, images[0][0], images[0][1])));
  const Image& image = reconstruction.Image(images[0][1]);
  auto& sensor =
      image.FramePtr()->RigPtr()->SensorFromRig(image.CameraPtr()->SensorId());
  sensor.rotation() = ZRotation(0.2) * sensor.rotation();
  auto averager = CreateDefaultCeresRotationAverager(
      CeresRotationAveragerOptions(), graph, reconstruction);
  double* rotation = sensor.rotation().coeffs().data();
  EXPECT_TRUE(averager->Problem().IsParameterBlockConstant(rotation));
  averager->Problem().SetParameterBlockVariable(rotation);
  ASSERT_TRUE(averager->Solve().IsSolutionUsable());
  EXPECT_NEAR(
      RelativeRotation(reconstruction, images[0][0], images[0][1])
          .angularDistance(RelativeRotation(truth, images[0][0], images[0][1])),
      0,
      1e-8);
}

TEST(CeresRotationAverager, MissingSensorRotation) {
  Reconstruction reconstruction = MakeRigReconstruction();
  const auto images = RigImages(reconstruction);
  PoseGraph graph;
  // The sensor has no reference image from the same frame in the graph.
  for (const auto& [id1, id2] : {std::pair{images[0][0], images[1][0]},
                                 std::pair{images[1][0], images[2][1]}}) {
    graph.AddEdge(id1, id2, Edge(RelativeRotation(reconstruction, id1, id2)));
  }
  const Image& image = reconstruction.Image(images[0][1]);
  Rig& rig = *image.FramePtr()->RigPtr();
  const sensor_t sensor_id = image.CameraPtr()->SensorId();
  rig.ResetSensorFromRig(sensor_id);
  reconstruction.DeRegisterFrame(reconstruction.Image(images[2][1]).FrameId());
  const Reconstruction before = reconstruction;
  CeresRotationAveragerOptions options;
  options.skip_initialization = true;
  options.refine_sensor_from_rig = false;
  EXPECT_THROW(
      CreateDefaultCeresRotationAverager(options, graph, reconstruction),
      std::invalid_argument);
  EXPECT_EQ(reconstruction.Frames(), before.Frames());
  EXPECT_EQ(reconstruction.Rigs(), before.Rigs());
  EXPECT_EQ(reconstruction.RegFrameIds(), before.RegFrameIds());

  options.refine_sensor_from_rig = true;
  options.skip_initialization = false;
  auto averager =
      CreateDefaultCeresRotationAverager(options, graph, reconstruction);
  const auto rotation = rig.SensorFromRig(sensor_id).rotation();
  EXPECT_EQ(rotation.coeffs(), Eigen::Quaterniond::Identity().coeffs());
  EXPECT_FALSE(
      averager->Problem().IsParameterBlockConstant(rotation.coeffs().data()));
  ASSERT_TRUE(averager->Solve().IsSolutionUsable());
  ExpectRelativeRotations(reconstruction, graph, 1e-7);
}

}  // namespace
}  // namespace colmap
