#include "colmap/estimators/rotation_averaging_ceres.h"

#include "colmap/scene/frame.h"
#include "colmap/scene/image.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <memory>
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
    if (posed) {
      reconstruction.AddImageWithTrivialFrame(
          image,
          Rigid3d(Eigen::Quaterniond::Identity(),
                  Eigen::Vector3d(image_id, image_id + 1, image_id + 2)));
    } else {
      reconstruction.AddImageWithTrivialFrame(image);
    }
  }
  return reconstruction;
}

void SetRotation(Reconstruction& reconstruction,
                 const frame_t frame_id,
                 const Eigen::Quaterniond& rotation) {
  Frame& frame = reconstruction.Frame(frame_id);
  frame.RigFromWorld().rotation() = rotation;
}

TEST(CeresRotationAverager, RecoversNominalRotations) {
  Reconstruction reconstruction = MakeTrivialReconstruction({1, 2, 3});
  const Eigen::Vector3d translation =
      reconstruction.Frame(2).RigFromWorld().translation();
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
  EXPECT_NEAR((reconstruction.Image(2).CamFromWorld().rotation() *
               reconstruction.Image(1).CamFromWorld().rotation().inverse())
                  .angularDistance(ZRotation(0.2)),
              0.0,
              1e-10);
  EXPECT_EQ(reconstruction.Frame(2).RigFromWorld().translation(), translation);
}

TEST(CeresRotationAverager, AddsIndividualRelativeRotationResidual) {
  Reconstruction reconstruction = MakeTrivialReconstruction({1, 2, 3});
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
  EXPECT_THROW(averager->AddRelativeRotationResidual(
                   1, 3, ZRotation(0.2), retained_loss.lock()),
               std::invalid_argument);
  EXPECT_TRUE(averager->Solve().IsSolutionUsable());
}

TEST(CeresRotationAverager, SelectsMstOrSuppliedInitialization) {
  Reconstruction supplied = MakeTrivialReconstruction({1, 2});
  SetRotation(supplied, 1, ZRotation(0.7));
  SetRotation(supplied, 2, ZRotation(0.9));
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

TEST(CeresRotationAverager, RejectsMultiCameraRigs) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 2;
  dataset_options.num_frames_per_rig = 2;
  dataset_options.num_points3D = 10;
  SynthesizeDataset(dataset_options, &reconstruction);
  std::vector<image_t> ref_image_ids;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (image.IsRefInFrame()) ref_image_ids.push_back(image_id);
  }
  ASSERT_EQ(ref_image_ids.size(), 2);
  PoseGraph pose_graph;
  pose_graph.AddEdge(ref_image_ids[0], ref_image_ids[1], Edge(ZRotation(0.2)));
  EXPECT_THROW(CreateDefaultCeresRotationAverager(
                   CeresRotationAveragerOptions(), pose_graph, reconstruction),
               std::invalid_argument);
}

}  // namespace
}  // namespace colmap
