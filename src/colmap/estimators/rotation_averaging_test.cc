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

#include "colmap/estimators/rotation_averaging.h"

#include "colmap/estimators/rotation_averaging_impl.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/synthetic.h"

#include <map>
#include <utility>

#include <algorithm>
#include <array>
#include <numeric>
#include <set>

#include <Eigen/Geometry>
#include <ceres/rotation.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void LoadReconstructionAndPoseGraph(const Database& database,
                                    Reconstruction* reconstruction,
                                    PoseGraph* pose_graph,
                                    DatabaseCache* database_cache) {
  DatabaseCache::Options options;
  database_cache->Load(database, options);
  reconstruction->Load(*database_cache);
  pose_graph->Load(*database_cache->CorrespondenceGraph());
}

struct TestData {
  std::shared_ptr<Database> database;
  DatabaseCache database_cache;
  Reconstruction gt_reconstruction;
  Reconstruction reconstruction;
  PoseGraph pose_graph;
  std::vector<PosePrior> pose_priors;
};

TestData CreateTestData(const SyntheticDatasetOptions& dataset_options,
                        const SyntheticNoiseOptions* noise_options = nullptr) {
  TestData data;
  data.database = Database::Open(kInMemorySqliteDatabasePath);
  SynthesizeDataset(
      dataset_options, &data.gt_reconstruction, data.database.get());
  if (noise_options) {
    SynthesizeNoise(
        *noise_options, &data.gt_reconstruction, data.database.get());
  }
  LoadReconstructionAndPoseGraph(*data.database,
                                 &data.reconstruction,
                                 &data.pose_graph,
                                 &data.database_cache);
  data.pose_priors = data.database->ReadAllPosePriors();
  return data;
}

RotationEstimatorOptions CreateRATestOptions(bool use_gravity = false) {
  RotationEstimatorOptions options;
  options.skip_initialization = false;
  options.use_gravity = use_gravity;
  options.use_stratified = true;
  return options;
}

void ExpectEqualRotations(const Reconstruction& gt,
                          const Reconstruction& computed,
                          const double max_rotation_error_deg) {
  const double max_rotation_error_rad = DegToRad(max_rotation_error_deg);
  const std::vector<image_t> reg_image_ids = gt.RegImageIds();
  for (size_t i = 0; i < reg_image_ids.size(); i++) {
    const image_t image_id1 = reg_image_ids[i];
    for (size_t j = 0; j < i; j++) {
      const image_t image_id2 = reg_image_ids[j];
      const Eigen::Quaterniond cam2_from_cam1 =
          computed.Image(image_id2).CamFromWorld().rotation() *
          computed.Image(image_id1).CamFromWorld().rotation().inverse();
      const Eigen::Quaterniond cam2_from_cam1_gt =
          gt.Image(image_id2).CamFromWorld().rotation() *
          gt.Image(image_id1).CamFromWorld().rotation().inverse();
      EXPECT_LE(cam2_from_cam1.angularDistance(cam2_from_cam1_gt),
                max_rotation_error_rad);
    }
  }
}

void ResetSensorsFromRig(Reconstruction& reconstruction) {
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction.Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }
}

void RunAndVerifyRotationAveraging(const Reconstruction& gt_reconstruction,
                                   const Reconstruction& reconstruction,
                                   const PoseGraph& pose_graph,
                                   const std::vector<PosePrior>& pose_priors,
                                   const std::vector<bool>& use_gravity_values,
                                   const double max_rotation_error_deg) {
  for (const bool use_gravity : use_gravity_values) {
    Reconstruction reconstruction_copy = reconstruction;
    PoseGraph pose_graph_copy = pose_graph;
    RunRotationAveraging(CreateRATestOptions(use_gravity),
                         pose_graph_copy,
                         reconstruction_copy,
                         pose_priors);

    ExpectEqualRotations(
        gt_reconstruction, reconstruction_copy, max_rotation_error_deg);
  }
}

TEST(RotationAveraging, WithoutNoise) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {true, false},
                                /*max_rotation_error_deg=*/1e-2);
}

TEST(RotationAveraging, WithoutNoiseWithNonTrivialKnownRig) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {true, false},
                                /*max_rotation_error_deg=*/1e-2);
}

TEST(RotationAveraging, WithoutNoiseWithNonTrivialUnknownRig) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  ResetSensorsFromRig(data.reconstruction);

  // For unknown rigs, it is not supported to use gravity.
  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {false},
                                /*max_rotation_error_deg=*/1e-2);
}

TEST(RotationAveraging, WithNoiseAndOutliers) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  synthetic_noise_options.prior_gravity_stddev = 3e-1;
  auto data =
      CreateTestData(synthetic_dataset_options, &synthetic_noise_options);

  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {true, false},
                                /*max_rotation_error_deg=*/3);
}

TEST(RotationAveraging, WithNoiseAndOutliersWithNonTrivialKnownRigs) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  synthetic_noise_options.prior_gravity_stddev = 3e-1;
  auto data =
      CreateTestData(synthetic_dataset_options, &synthetic_noise_options);

  RunAndVerifyRotationAveraging(data.gt_reconstruction,
                                data.reconstruction,
                                data.pose_graph,
                                data.pose_priors,
                                {true, false},
                                /*max_rotation_error_deg=*/2.);
}

TEST(RotationAveraging, DeterministicRandomSeed) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  RotationEstimatorOptions options = CreateRATestOptions();
  options.random_seed = 42;

  // Run twice with the same seed and verify identical results.
  Reconstruction reconstruction1 = data.reconstruction;
  PoseGraph pose_graph1 = data.pose_graph;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph1, reconstruction1, data.pose_priors));

  Reconstruction reconstruction2 = data.reconstruction;
  PoseGraph pose_graph2 = data.pose_graph;
  EXPECT_TRUE(RunRotationAveraging(
      options, pose_graph2, reconstruction2, data.pose_priors));

  ExpectEqualRotations(
      reconstruction1, reconstruction2, /*max_rotation_error_deg=*/0);
}

TEST(RotationAveraging, EmptyPoseGraph) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 20;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  // Invalidate all edges so connected components are empty.
  for (auto& [pair_id, edge] : data.pose_graph.Edges()) {
    edge.valid = false;
  }

  RotationEstimatorOptions options = CreateRATestOptions();
  EXPECT_FALSE(RunRotationAveraging(
      options, data.pose_graph, data.reconstruction, data.pose_priors));
}

TEST(RotationAveraging, MultiImageRigFrameDeregisterDoesNotCrashOnSecondVisit) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  std::vector<frame_t> frame_ids;
  for (const auto& [fid, _] : data.reconstruction.Frames()) {
    frame_ids.push_back(fid);
  }
  std::sort(frame_ids.begin(), frame_ids.end());
  ASSERT_EQ(frame_ids.size(), 4);
  const frame_t isolated_frame_id = frame_ids.back();

  // 1. Collect every image_id that belongs to the isolated frame.
  std::unordered_set<image_t> isolated_image_ids;
  for (const auto& data_id :
       data.reconstruction.Frame(isolated_frame_id).ImageIds()) {
    isolated_image_ids.insert(data_id.id);
  }
  ASSERT_GE(isolated_image_ids.size(), size_t{2})
      << "Bug only fires when the deregistered frame carries >= 2 images.";

  // 2. Strip every pose-graph edge touching the isolated frame. After
  //    this the frame is unreachable from any other frame in the
  //    pose-graph CC.
  std::vector<std::pair<image_t, image_t>> edges_to_remove;
  for (const auto& [pair_id, edge] : data.pose_graph.Edges()) {
    const auto [id1, id2] = PairIdToImagePair(pair_id);
    if (isolated_image_ids.count(id1) || isolated_image_ids.count(id2)) {
      edges_to_remove.emplace_back(id1, id2);
    }
  }
  ASSERT_FALSE(edges_to_remove.empty());
  for (const auto& [id1, id2] : edges_to_remove) {
    data.pose_graph.DeleteEdge(id1, id2);
  }

  // 3. Pre-register the isolated frame with its GT pose. This puts it
  //    into reg_frame_ids_ even though no edges touch it.
  ASSERT_TRUE(data.gt_reconstruction.Frame(isolated_frame_id).HasPose());
  data.reconstruction.Frame(isolated_frame_id)
      .SetRigFromWorld(
          data.gt_reconstruction.Frame(isolated_frame_id).RigFromWorld());
  data.reconstruction.RegisterFrame(isolated_frame_id);

  RotationEstimatorOptions options = CreateRATestOptions();
  options.max_rotation_error_deg = 1.0;

  EXPECT_TRUE(RunRotationAveraging(
      options, data.pose_graph, data.reconstruction, data.pose_priors));

  // Post-condition: the isolated frame was deregistered cleanly.
  // The other frames remain registered with poses recovered by RA.
  EXPECT_FALSE(data.reconstruction.Frame(isolated_frame_id).HasPose());
  for (size_t i = 0; i + 1 < frame_ids.size(); ++i) {
    EXPECT_TRUE(data.reconstruction.Frame(frame_ids[i]).HasPose())
        << "Frame " << frame_ids[i]
        << " should remain registered after deregistration of the "
        << "isolated frame.";
  }
}

TEST(RotationAveraging, GravityWithUnknownRigSensorsReturnsFalse) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  ResetSensorsFromRig(data.reconstruction);

  // With gravity enabled and unknown rig sensors, EstimateRotations should
  // fail inside RunRotationAveraging because AllSensorsFromRigKnown returns
  // false. However, RunRotationAveraging takes the HasUnknownCamsFromRig path
  // which creates an expanded reconstruction (singleton rigs) that avoids the
  // AllSensorsFromRigKnown check. To directly hit the
  // AllSensorsFromRigKnown check, we use RotationEstimator directly.
  RotationEstimatorOptions options = CreateRATestOptions(/*use_gravity=*/true);

  std::unordered_set<image_t> active_image_ids;
  for (const auto& [image_id, image] : data.reconstruction.Images()) {
    active_image_ids.insert(image_id);
  }

  RotationEstimator estimator(options);
  EXPECT_FALSE(estimator.EstimateRotations(data.pose_graph,
                                           data.pose_priors,
                                           active_image_ids,
                                           data.reconstruction));
}

TEST(RotationAveraging, SkipRiskyLcPairsWithUnknownRigUsesCorrespondenceGraph) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  ResetSensorsFromRig(data.reconstruction);

  RotationEstimatorOptions options = CreateRATestOptions();
  options.skip_risky_lc_pairs = true;

  EXPECT_TRUE(
      RunRotationAveraging(options,
                           data.pose_graph,
                           data.reconstruction,
                           data.pose_priors,
                           nullptr,
                           data.database_cache.CorrespondenceGraph().get()));
}

TEST(RotationAveraging,
     SkipRiskyLcPairsWithStratifiedGravityUsesCorrespondenceGraph) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  // Remove one gravity prior so the stratified subset path is used.
  data.pose_priors.pop_back();

  RotationEstimatorOptions options = CreateRATestOptions(/*use_gravity=*/true);
  options.skip_risky_lc_pairs = true;
  options.use_stratified = true;

  EXPECT_TRUE(
      RunRotationAveraging(options,
                           data.pose_graph,
                           data.reconstruction,
                           data.pose_priors,
                           nullptr,
                           data.database_cache.CorrespondenceGraph().get()));
}

// LC-penalty branch inside ComputeMaximumPoseGraphSpanningTree.
//
// With ``prioritize_tracking=false`` the MST runs vanilla maximum-weight
// Kruskal. With ``prioritize_tracking=true`` it subtracts
// ``kLCPenalty=1e9`` from edges whose ``are_lc`` true count exceeds non-LC
// inliers, routing the tree away from LC-dominated pairs.
//
// We construct a minimal three-image graph where the highest-weight edge is
// LC-dominated and verify the parent map flips between the two modes.

namespace {

// Build a minimal PoseGraph with three valid edges:
//   1-2: weight 100 (LC-dominated when correspondence_graph is annotated)
//   1-3: weight 10
//   2-3: weight 10
// With prioritize_tracking=false the MST picks 1-2 plus one of {1-3, 2-3}.
// With prioritize_tracking=true and the LC annotation below, the 1-2 edge
// has its weight reduced by kLCPenalty=1e9 so the MST picks 1-3 and 2-3.
struct LcMstFixture {
  PoseGraph pose_graph;
  CorrespondenceGraph correspondence_graph;
  std::unordered_set<image_t> image_ids;
};

LcMstFixture BuildLcMstFixture() {
  LcMstFixture data;
  for (image_t image_id : {static_cast<image_t>(1),
                           static_cast<image_t>(2),
                           static_cast<image_t>(3)}) {
    data.image_ids.insert(image_id);
    // The CG ``AddImage`` only requires per-image num_points2D; pose-graph
    // edges below carry the actual weights.
    data.correspondence_graph.AddImage(image_id, /*num_points=*/0);
  }

  auto AddEdge = [&](image_t a, image_t b, int num_matches) {
    PoseGraph::Edge edge;
    edge.cam2_from_cam1 = Rigid3d();
    edge.num_matches = num_matches;
    edge.valid = true;
    data.pose_graph.AddEdge(a, b, std::move(edge));
  };

  AddEdge(1, 2, /*num_matches=*/100);
  AddEdge(1, 3, /*num_matches=*/10);
  AddEdge(2, 3, /*num_matches=*/10);

  // Mark the 1-2 pair as LC-dominated: more than half of its inliers carry
  // are_lc=true. The MST helper inspects ImagePairsMap() entries indexed by
  // pair_id, and (when a CG is plumbed through) reads the edge weight from
  // ``inliers.size()`` rather than ``edge.num_matches``. Size each inliers
  // vector to match the corresponding pose-graph edge so the two weight
  // sources agree.
  const image_pair_t pair_12 = ImagePairToPairId(1, 2);
  auto& cg_pair = data.correspondence_graph.MutableImagePairs()[pair_12];
  cg_pair.image_id1 = 1;
  cg_pair.image_id2 = 2;
  cg_pair.pair_id = pair_12;
  cg_pair.inliers.resize(100);
  std::iota(cg_pair.inliers.begin(), cg_pair.inliers.end(), 0);
  // 80 of 100 inliers are LC -> dominated.
  cg_pair.are_lc.assign(100, true);
  std::fill(cg_pair.are_lc.begin() + 80, cg_pair.are_lc.end(), false);

  // The other two pairs are tracking-dominated, weight 10 each.
  const std::array<std::pair<image_t, image_t>, 2> tracking_pairs = {
      {{1, 3}, {2, 3}}};
  for (const auto& pair : tracking_pairs) {
    const image_t a = pair.first;
    const image_t b = pair.second;
    const image_pair_t pair_id = ImagePairToPairId(a, b);
    auto& other = data.correspondence_graph.MutableImagePairs()[pair_id];
    other.image_id1 = a;
    other.image_id2 = b;
    other.pair_id = pair_id;
    other.inliers.resize(10);
    std::iota(other.inliers.begin(), other.inliers.end(), 0);
    other.are_lc.assign(10, false);
  }
  return data;
}

}  // namespace

// Returns the set of (child, parent) edges in the spanning tree, excluding
// the root self-loop (``parents[root] == root``).
std::set<std::pair<image_t, image_t>> CollectTreeEdges(
    const std::unordered_map<image_t, image_t>& parents, image_t root) {
  std::set<std::pair<image_t, image_t>> edges;
  for (const auto& [child, parent] : parents) {
    if (child == root) continue;  // root's self-loop is not a tree edge
    // Canonicalise (a, b) so we can compare regardless of orientation.
    edges.emplace(std::min(child, parent), std::max(child, parent));
  }
  return edges;
}

TEST(RotationAveraging, Gate_LcPenaltyMst_Off_KeepsLcDominantEdge) {
  LcMstFixture data = BuildLcMstFixture();

  std::unordered_map<image_t, image_t> parents;
  const image_t root =
      ComputeMaximumPoseGraphSpanningTree(data.pose_graph,
                                          data.image_ids,
                                          parents,
                                          /*prioritize_tracking=*/false,
                                          &data.correspondence_graph);

  // 3 nodes -> 3 entries in the parent map (one is the root self-loop).
  EXPECT_EQ(parents.size(), 3u);

  // The 1-2 edge (weight 100) is the heaviest, so it must appear in the
  // tree regardless of LC status when the gate is OFF.
  const auto tree_edges = CollectTreeEdges(parents, root);
  EXPECT_EQ(tree_edges.size(), 2u);
  EXPECT_GT(tree_edges.count({1, 2}), 0u)
      << "Expected the 1-2 edge in the MST when prioritize_tracking=false";

  EXPECT_TRUE(root == 1 || root == 2 || root == 3);
}

TEST(RotationAveraging, Gate_LcPenaltyMst_On_RoutesAroundLcEdge) {
  LcMstFixture data = BuildLcMstFixture();

  std::unordered_map<image_t, image_t> parents;
  const image_t root =
      ComputeMaximumPoseGraphSpanningTree(data.pose_graph,
                                          data.image_ids,
                                          parents,
                                          /*prioritize_tracking=*/true,
                                          &data.correspondence_graph);

  // With the LC penalty active, the 1-2 edge's effective weight is
  // 100 - kLCPenalty (=1e9), so the MST must pick the 1-3 + 2-3 path.
  EXPECT_EQ(parents.size(), 3u);
  const auto tree_edges = CollectTreeEdges(parents, root);
  EXPECT_EQ(tree_edges.size(), 2u);
  EXPECT_EQ(tree_edges.count({1, 2}), 0u)
      << "Expected the 1-2 edge to be skipped when prioritize_tracking=true";
  EXPECT_GT(tree_edges.count({1, 3}), 0u)
      << "Expected the 1-3 edge in the LC-penalty MST";
  EXPECT_GT(tree_edges.count({2, 3}), 0u)
      << "Expected the 2-3 edge in the LC-penalty MST";

  EXPECT_TRUE(root == 1 || root == 2 || root == 3);
}

TEST(RotationAveraging, Gate_LcPenaltyMst_OnWithoutCgFallsBackToVanilla) {
  // With the gate ON but no correspondence graph, the helper should ignore
  // the LC penalty entirely (the second guard in the ``cg_map_ptr`` ternary).
  // Verifies that the gate alone — without a CG — does not silently change
  // behaviour relative to the OFF branch.
  LcMstFixture data = BuildLcMstFixture();

  std::unordered_map<image_t, image_t> parents_on;
  ComputeMaximumPoseGraphSpanningTree(data.pose_graph,
                                      data.image_ids,
                                      parents_on,
                                      /*prioritize_tracking=*/true,
                                      /*correspondence_graph=*/nullptr);

  std::unordered_map<image_t, image_t> parents_off;
  ComputeMaximumPoseGraphSpanningTree(data.pose_graph,
                                      data.image_ids,
                                      parents_off,
                                      /*prioritize_tracking=*/false,
                                      /*correspondence_graph=*/nullptr);

  EXPECT_EQ(parents_on, parents_off);
}

// Covers: InitializeRigRotationsFromImages standalone (lines 465-564) with
// multi-camera rig to exercise cam_from_rig estimation and rig_from_world
// averaging.
TEST(RotationAveraging, InitializeSensorFromRigUsingCamsFromWorld) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  // Build cams_from_world from the ground truth.
  std::unordered_map<image_t, Rigid3d> cams_from_world;
  for (const auto& [image_id, image] : data.gt_reconstruction.Images()) {
    if (image.HasPose()) {
      cams_from_world[image_id] = image.CamFromWorld();
    }
  }

  ResetSensorsFromRig(data.reconstruction);

  EXPECT_TRUE(
      InitializeRigRotationsFromImages(cams_from_world, data.reconstruction));

  for (const auto& [rig_id, rig] : data.reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
      EXPECT_LT(sensor_from_rig->rotation().angularDistance(
                    data.gt_reconstruction.Rig(rig_id)
                        .SensorFromRig(sensor_id)
                        .rotation()),
                1e-6);
    }
  }
}

// When a sensor_from_rig is already fully calibrated (valid rotation AND
// translation), InitializeRigRotationsFromImages must preserve it rather than
// resetting the translation to NaN.
TEST(RotationAveraging, InitializeSensorFromRigPreservesCalibratedRig) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  std::unordered_map<image_t, Rigid3d> cams_from_world;
  for (const auto& [image_id, image] : data.gt_reconstruction.Images()) {
    if (image.HasPose()) {
      cams_from_world[image_id] = image.CamFromWorld();
    }
  }

  // Snapshot the (already-calibrated) rig BEFORE initialization.
  std::map<std::pair<rig_t, sensor_t>, Rigid3d> snapshot;
  for (const auto& [rig_id, rig] : data.reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
      ASSERT_TRUE(sensor_from_rig.has_value());
      snapshot[{rig_id, sensor_id}] = *sensor_from_rig;
    }
  }
  ASSERT_GT(snapshot.size(), 0u);

  EXPECT_TRUE(
      InitializeRigRotationsFromImages(cams_from_world, data.reconstruction));

  for (const auto& [rig_id, rig] : data.reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor_from_rig_after] : rig.NonRefSensors()) {
      ASSERT_TRUE(sensor_from_rig_after.has_value())
          << "rig_id=" << rig_id << ", sensor_id=" << sensor_id.id;
      const auto& sensor_from_rig_before = snapshot.at({rig_id, sensor_id});
      EXPECT_EQ(*sensor_from_rig_after, sensor_from_rig_before)
          << "rig_id=" << rig_id << ", sensor_id=" << sensor_id.id;
    }
  }
}

TEST(RotationAveraging, RefineSensorFromRigFalsePreservesRig) {
  SetPRNGSeed(1);

  // A non-trivial multi-camera rig so both rotation AND translation are
  // non-zero
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  auto data = CreateTestData(synthetic_dataset_options);

  // Snapshot the rig BEFORE RA so we can compare element-wise.
  std::map<std::pair<rig_t, sensor_t>, Rigid3d> snapshot;
  for (const auto& [rig_id, rig] : data.reconstruction.Rigs()) {
    for (const auto& [sensor_id, sfr] : rig.NonRefSensors()) {
      ASSERT_TRUE(sfr.has_value());
      snapshot[{rig_id, sensor_id}] = *sfr;
    }
  }
  // Sanity check: at least one sensor should have a non-zero translation
  // so the test would actually catch the old "reset to zero" behaviour.
  ASSERT_GT(snapshot.size(), 0u);

  // Run RA with refine_sensor_from_rig=false.
  RotationEstimatorOptions options = CreateRATestOptions(/*use_gravity=*/true);
  options.refine_sensor_from_rig = false;
  ASSERT_TRUE(RunRotationAveraging(
      options, data.pose_graph, data.reconstruction, data.pose_priors));

  // Every sensor_from_rig must match the snapshot exactly.
  for (const auto& [rig_id, rig] : data.reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor_from_rig_after] : rig.NonRefSensors()) {
      ASSERT_TRUE(sensor_from_rig_after.has_value())
          << "rig_id=" << rig_id << ", sensor_id=" << sensor_id.id;
      const auto& sensor_from_rig_before = snapshot.at({rig_id, sensor_id});
      EXPECT_EQ(*sensor_from_rig_after, sensor_from_rig_before)
          << "rig_id=" << rig_id << ", sensor_id=" << sensor_id.id;
    }
  }
}

// RelativeRotationError functor tests (video-aware Ceres path).

namespace {

// Convert a rotation matrix to a 3-DOF angle-axis vector.
Eigen::Vector3d RotationMatrixToAngleAxisVec(const Eigen::Matrix3d& R) {
  Eigen::Vector3d aa;
  ceres::RotationMatrixToAngleAxis(R.data(), aa.data());
  return aa;
}

// Convert a quaternion to a 3-DOF angle-axis vector.
Eigen::Vector3d QuaternionToAngleAxisVec(const Eigen::Quaterniond& q) {
  return RotationMatrixToAngleAxisVec(q.toRotationMatrix());
}

}  // namespace

TEST(RelativeRotationError, ZeroResidualWhenRotationsConsistent) {
  // Construct a consistent triple (R1, R2, R_rel) such that
  // R2 = R_rel * R1. The residual should vanish.
  for (int trial = 0; trial < 16; ++trial) {
    const Eigen::Quaterniond q1 = Eigen::Quaterniond::UnitRandom();
    const Eigen::Quaterniond q_rel = Eigen::Quaterniond::UnitRandom();
    const Eigen::Matrix3d R1 = q1.toRotationMatrix();
    const Eigen::Matrix3d R_rel = q_rel.toRotationMatrix();
    const Eigen::Matrix3d R2 = R_rel * R1;

    const Eigen::Vector3d aa1 = RotationMatrixToAngleAxisVec(R1);
    const Eigen::Vector3d aa2 = RotationMatrixToAngleAxisVec(R2);
    const Eigen::Vector3d rel_aa = RotationMatrixToAngleAxisVec(R_rel);

    RelativeRotationError functor(rel_aa);
    Eigen::Vector3d residual;
    ASSERT_TRUE(functor(aa1.data(), aa2.data(), residual.data()));
    EXPECT_LT(residual.norm(), 1e-9) << "trial=" << trial;
  }
}

TEST(RelativeRotationError, NonZeroResidualWhenInconsistent) {
  // Inject a known angular discrepancy R_err on top of an otherwise
  // consistent triple, so the functor's residual norm should match
  // the discrepancy angle.
  for (int trial = 0; trial < 16; ++trial) {
    const Eigen::Quaterniond q1 = Eigen::Quaterniond::UnitRandom();
    const Eigen::Quaterniond q_rel = Eigen::Quaterniond::UnitRandom();

    // Build a known small-angle perturbation about a random axis.
    const double err_angle = 0.05 + 0.1 * trial / 16.0;  // 0.05..0.15 rad
    Eigen::Vector3d axis = Eigen::Vector3d::Random().normalized();
    const Eigen::AngleAxisd q_err(err_angle, axis);

    const Eigen::Matrix3d R1 = q1.toRotationMatrix();
    const Eigen::Matrix3d R_rel = q_rel.toRotationMatrix();
    // R2 deliberately rotated *off* the consistent value by R_err. The
    // residual ends up being the inverse-transformed perturbation:
    //   R_residual = R2^T * R_rel * R1
    //              = (R_err * R_rel * R1)^T * R_rel * R1
    //              = R1^T * R_rel^T * R_err^T * R_rel * R1.
    // That is a similarity transform of R_err^T, so its rotation angle
    // equals err_angle exactly.
    const Eigen::Matrix3d R2 = q_err.toRotationMatrix() * R_rel * R1;

    const Eigen::Vector3d aa1 = RotationMatrixToAngleAxisVec(R1);
    const Eigen::Vector3d aa2 = RotationMatrixToAngleAxisVec(R2);
    const Eigen::Vector3d rel_aa = RotationMatrixToAngleAxisVec(R_rel);

    RelativeRotationError functor(rel_aa);
    Eigen::Vector3d residual;
    ASSERT_TRUE(functor(aa1.data(), aa2.data(), residual.data()));
    EXPECT_NEAR(residual.norm(), err_angle, 1e-9) << "trial=" << trial;
  }
}

TEST(RelativeRotationError, SymmetryUnderSwapAndInvert) {
  // Swapping (R1, R2) and inverting R_rel must give a residual of equal
  // magnitude (sign flips because angle-axis of inverse is negated).
  for (int trial = 0; trial < 16; ++trial) {
    const Eigen::Quaterniond q1 = Eigen::Quaterniond::UnitRandom();
    const Eigen::Quaterniond q2 = Eigen::Quaterniond::UnitRandom();
    const Eigen::Quaterniond q_rel = Eigen::Quaterniond::UnitRandom();

    const Eigen::Vector3d aa1 = QuaternionToAngleAxisVec(q1);
    const Eigen::Vector3d aa2 = QuaternionToAngleAxisVec(q2);
    const Eigen::Vector3d rel_aa = QuaternionToAngleAxisVec(q_rel);
    const Eigen::Vector3d rel_aa_inv =
        QuaternionToAngleAxisVec(q_rel.conjugate());

    RelativeRotationError functor_orig(rel_aa);
    RelativeRotationError functor_swap(rel_aa_inv);
    Eigen::Vector3d residual_orig, residual_swap;
    ASSERT_TRUE(functor_orig(aa1.data(), aa2.data(), residual_orig.data()));
    ASSERT_TRUE(functor_swap(aa2.data(), aa1.data(), residual_swap.data()));

    // Norms must agree to high precision.
    EXPECT_NEAR(residual_orig.norm(), residual_swap.norm(), 1e-9)
        << "trial=" << trial;
    // The two residuals are angle-axis vectors of mutually inverse
    // rotations, hence sum to zero.
    EXPECT_LT((residual_orig + residual_swap).norm(), 1e-9)
        << "trial=" << trial;
  }
}

}  // namespace
}  // namespace colmap
