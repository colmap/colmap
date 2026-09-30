// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/gravity_refinement.h"

#include "colmap/geometry/triangulation.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

void LoadReconstructionAndPoseGraph(const Database& database,
                                    Reconstruction* reconstruction,
                                    PoseGraph* pose_graph) {
  DatabaseCache database_cache;
  DatabaseCache::Options options;
  database_cache.Load(database, options);
  reconstruction->Load(database_cache);
  pose_graph->Load(*database_cache.CorrespondenceGraph());
}

void SynthesizeGravityOutliers(std::vector<PosePrior>& pose_priors,
                               double outlier_ratio = 0.0) {
  for (auto& pose_prior : pose_priors) {
    if (pose_prior.HasGravity() &&
        RandomUniformReal<double>(0, 1) < outlier_ratio) {
      pose_prior.gravity = RandomEigenVectord<3>().normalized();
    }
  }
}

void ExpectEqualGravity(const Eigen::Vector3d& gravity_in_world,
                        const Reconstruction& gt,
                        const std::vector<PosePrior>& pose_priors,
                        const double max_gravity_error_deg) {
  const double max_gravity_error_rad = DegToRad(max_gravity_error_deg);
  NodeHashMap<image_t, const PosePrior*> image_to_pose_prior;
  for (const auto& pose_prior : pose_priors) {
    if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
      image_to_pose_prior.emplace(pose_prior.corr_data_id.id, &pose_prior);
    }
  }
  for (const auto& image_id : gt.RegImageIds()) {
    const auto& image = gt.Image(image_id);
    if (!image.IsRefInFrame()) {
      continue;
    }
    const Eigen::Vector3d gravity_gt =
        gt.Image(image_id).CamFromWorld().rotation() * gravity_in_world;
    const Eigen::Vector3d gravity_computed =
        image_to_pose_prior.at(image_id)->gravity;
    const double gravity_error_rad =
        CalculateAngleBetweenVectors(gravity_gt, gravity_computed);
    EXPECT_LT(gravity_error_rad, max_gravity_error_rad);
  }
}

struct GravityRefinementTestData {
  SyntheticDatasetOptions synthetic_options;
  Reconstruction gt_reconstruction;
  Reconstruction reconstruction;
  PoseGraph pose_graph;
  std::vector<PosePrior> pose_priors;
};

GravityRefinementTestData SynthesizeGravityRefinementTestData(
    int num_cameras_per_rig) {
  GravityRefinementTestData data;
  data.synthetic_options.num_rigs = 2;
  data.synthetic_options.num_cameras_per_rig = num_cameras_per_rig;
  data.synthetic_options.num_frames_per_rig = 25;
  data.synthetic_options.num_points3D = 100;
  data.synthetic_options.prior_gravity = true;
  data.synthetic_options.two_view_geometry_has_relative_pose = true;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  SynthesizeDataset(
      data.synthetic_options, &data.gt_reconstruction, database.get());
  LoadReconstructionAndPoseGraph(
      *database, &data.reconstruction, &data.pose_graph);
  data.pose_priors = database->ReadAllPosePriors();
  return data;
}

void RunRefineGravityTest(int num_cameras_per_rig) {
  GravityRefinementTestData data =
      SynthesizeGravityRefinementTestData(num_cameras_per_rig);
  SynthesizeGravityOutliers(data.pose_priors, /*outlier_ratio=*/0.3);

  GravityRefinerOptions options;
  RunGravityRefinement(
      options, data.pose_graph, data.reconstruction, data.pose_priors);

  ExpectEqualGravity(data.synthetic_options.prior_gravity_in_world,
                     data.gt_reconstruction,
                     data.pose_priors,
                     /*max_gravity_error_deg=*/1e-2);
}

TEST(GravityRefinement, RefineGravity) {
  RunRefineGravityTest(/*num_cameras_per_rig=*/1);
}

TEST(GravityRefinement, RefineGravityWithNonTrivialRigs) {
  RunRefineGravityTest(/*num_cameras_per_rig=*/2);
}

// Rotates every neighbor vote for the gravity of `image_id` by `angle_deg`
// about an axis orthogonal to its true gravity. With `split`, the votes
// alternate between +angle_deg and -angle_deg, so they form two clusters
// that no single gravity can agree with. Returns the true gravity of the
// image and the number of corrupted edges.
std::pair<Eigen::Vector3d, int> CorruptNeighborVotes(
    const Eigen::Vector3d& gravity_in_world,
    const Reconstruction& gt,
    const image_t image_id,
    const double angle_deg,
    const bool split,
    PoseGraph& pose_graph) {
  const Eigen::Vector3d gravity_true =
      gt.Image(image_id).CamFromWorld().rotation() * gravity_in_world;
  const Eigen::Vector3d axis = gravity_true.unitOrthogonal();

  std::vector<image_pair_t> pair_ids;
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (image_id1 == image_id || image_id2 == image_id) {
      pair_ids.push_back(pair_id);
    }
  }
  // Two equally sized clusters, so that neither one holds a majority.
  if (split && pair_ids.size() % 2 == 1) {
    pose_graph.SetInvalidEdge(pair_ids.back());
    pair_ids.pop_back();
  }

  for (size_t i = 0; i < pair_ids.size(); ++i) {
    const double sign = (split && i % 2 == 1) ? -1. : 1.;
    const Rigid3d delta(
        Eigen::Quaterniond(Eigen::AngleAxisd(DegToRad(sign * angle_deg), axis)),
        Eigen::Vector3d::Zero());
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_ids[i]);
    PoseGraph::Edge& edge = pose_graph.EdgeRef(image_id1, image_id2).first;
    if (image_id1 == image_id) {
      // vote = cam1_from_cam2 * gravity2 = delta * gravity_true
      edge.cam2_from_cam1 = edge.cam2_from_cam1 * Inverse(delta);
    } else {
      // vote = cam2_from_cam1 * gravity1 = delta * gravity_true
      edge.cam2_from_cam1 = delta * edge.cam2_from_cam1;
    }
  }
  return {gravity_true, static_cast<int>(pair_ids.size())};
}

PosePrior& FindImagePosePrior(std::vector<PosePrior>& pose_priors,
                              const image_t image_id) {
  PosePrior* found = nullptr;
  for (auto& pose_prior : pose_priors) {
    if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA &&
        pose_prior.corr_data_id.id == image_id) {
      found = &pose_prior;
      break;
    }
  }
  return *THROW_CHECK_NOTNULL(found);
}

class GravityRefinementAcceptanceTest : public ::testing::TestWithParam<bool> {
};

// One image gets a wrong gravity prior, so the refiner picks it up. Its
// neighbor votes are all rotated by the same angle (consistent), or by
// alternating angles (split). A consistent vote must be accepted, a split
// vote must be rejected and leave the prior as it was.
TEST_P(GravityRefinementAcceptanceTest, AcceptsOnlyConsistentVotes) {
  const bool split = GetParam();
  GravityRefinementTestData data =
      SynthesizeGravityRefinementTestData(/*num_cameras_per_rig=*/1);

  const image_t image_id = data.gt_reconstruction.RegImageIds().front();
  GravityRefinerOptions options;
  const double vote_angle_deg = 40.;
  const auto [gravity_true, num_votes] =
      CorruptNeighborVotes(data.synthetic_options.prior_gravity_in_world,
                           data.gt_reconstruction,
                           image_id,
                           vote_angle_deg,
                           split,
                           data.pose_graph);
  ASSERT_GE(num_votes, options.min_num_neighbors);

  // Move the prior so that every edge flags the image as error prone.
  PosePrior& pose_prior = FindImagePosePrior(data.pose_priors, image_id);
  const Eigen::Vector3d gravity_before =
      Eigen::AngleAxisd(DegToRad(10.), gravity_true.unitOrthogonal()) *
      gravity_true;
  pose_prior.gravity = gravity_before;

  RunGravityRefinement(
      options, data.pose_graph, data.reconstruction, data.pose_priors);

  const Eigen::Vector3d& gravity_after =
      FindImagePosePrior(data.pose_priors, image_id).gravity;
  if (split) {
    // The votes disagree with any refined gravity, so the prior stays.
    EXPECT_LT(CalculateAngleBetweenVectors(gravity_after, gravity_before),
              DegToRad(1e-6));
  } else {
    // The votes agree with the refined gravity, so the prior follows them.
    const Eigen::Vector3d gravity_expected =
        Eigen::AngleAxisd(DegToRad(vote_angle_deg),
                          gravity_true.unitOrthogonal()) *
        gravity_true;
    EXPECT_LT(CalculateAngleBetweenVectors(gravity_after, gravity_expected),
              DegToRad(1e-2));
  }
}

INSTANTIATE_TEST_SUITE_P(GravityRefinement,
                         GravityRefinementAcceptanceTest,
                         ::testing::Bool());

}  // namespace
}  // namespace colmap
