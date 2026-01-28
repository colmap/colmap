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

#include "colmap/scene/pose_graph.h"

#include "colmap/scene/database_cache.h"
#include "colmap/util/testing.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

PoseGraph::Edge SynthesizeEdge(int num_matches = 50) {
  PoseGraph::Edge edge;
  edge.cam2_from_cam1 = Rigid3d();
  edge.num_matches = num_matches;
  return edge;
}

TEST(PoseGraph, Nominal) {
  PoseGraph pose_graph;

  // Empty view graph.
  EXPECT_TRUE(pose_graph.Empty());
  EXPECT_EQ(pose_graph.NumEdges(), 0);

  // Add some pairs.
  pose_graph.AddEdge(1, 2, SynthesizeEdge());
  pose_graph.AddEdge(1, 3, SynthesizeEdge());
  pose_graph.AddEdge(2, 3, SynthesizeEdge());

  EXPECT_FALSE(pose_graph.Empty());
  EXPECT_EQ(pose_graph.NumEdges(), 3);

  // Invalidate one pair.
  pose_graph.SetInvalidEdge(ImagePairToPairId(1, 2));
  EXPECT_EQ(pose_graph.NumEdges(), 3);
  EXPECT_FALSE(pose_graph.IsValid(ImagePairToPairId(1, 2)));

  // Clear the view graph.
  pose_graph.Clear();
  EXPECT_TRUE(pose_graph.Empty());
  EXPECT_EQ(pose_graph.NumEdges(), 0);
}

TEST(PoseGraph, AddEdge) {
  PoseGraph pose_graph;

  // Normal add.
  PoseGraph::Edge edge = SynthesizeEdge();
  edge.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0));
  pose_graph.AddEdge(1, 2, edge);

  EXPECT_EQ(pose_graph.NumEdges(), 1);
  const auto& [stored, swapped] = pose_graph.EdgeRef(1, 2);
  EXPECT_FALSE(swapped);
  EXPECT_EQ(stored.cam2_from_cam1.translation().x(), 1);

  // Add with swapped IDs should invert the pair.
  PoseGraph::Edge edge2 = SynthesizeEdge();
  edge2.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(2, 0, 0));
  pose_graph.AddEdge(4, 3, edge2);  // 4 > 3, should swap and invert

  EXPECT_EQ(pose_graph.NumEdges(), 2);
  const auto& [stored2, swapped2] = pose_graph.EdgeRef(3, 4);
  EXPECT_FALSE(swapped2);
  EXPECT_EQ(stored2.cam2_from_cam1.translation().x(), -2);

  // Duplicate should throw.
  EXPECT_THROW(pose_graph.AddEdge(1, 2, SynthesizeEdge()), std::runtime_error);
  EXPECT_THROW(pose_graph.AddEdge(2, 1, SynthesizeEdge()), std::runtime_error);
}

TEST(PoseGraph, HasEdge) {
  PoseGraph pose_graph;
  pose_graph.AddEdge(1, 2, SynthesizeEdge());

  EXPECT_TRUE(pose_graph.HasEdge(1, 2));
  EXPECT_TRUE(pose_graph.HasEdge(2, 1));  // Order doesn't matter
  EXPECT_FALSE(pose_graph.HasEdge(1, 3));
}

TEST(PoseGraph, EdgeRef) {
  PoseGraph pose_graph;
  PoseGraph::Edge edge = SynthesizeEdge();
  edge.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0));
  pose_graph.AddEdge(1, 2, edge);

  // Normal order: swapped = false.
  auto [ref1, swapped1] = pose_graph.EdgeRef(1, 2);
  EXPECT_FALSE(swapped1);
  EXPECT_EQ(ref1.cam2_from_cam1.translation().x(), 1);

  // Reversed order: swapped = true.
  auto [ref2, swapped2] = pose_graph.EdgeRef(2, 1);
  EXPECT_TRUE(swapped2);
  EXPECT_EQ(ref2.cam2_from_cam1.translation().x(), 1);  // Same reference

  // Modify validity through PoseGraph.
  pose_graph.SetInvalidEdge(ImagePairToPairId(1, 2));
  EXPECT_FALSE(pose_graph.IsValid(ImagePairToPairId(1, 2)));

  // Non-existent pair should throw.
  EXPECT_THROW(pose_graph.EdgeRef(1, 3), std::out_of_range);
}

TEST(PoseGraph, GetEdge) {
  PoseGraph pose_graph;
  PoseGraph::Edge edge = SynthesizeEdge();
  edge.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0));
  pose_graph.AddEdge(1, 2, edge);

  // Normal order: returns as-is.
  PoseGraph::Edge copy1 = pose_graph.GetEdge(1, 2);
  EXPECT_EQ(copy1.cam2_from_cam1.translation().x(), 1);

  // Reversed order: returns inverted copy.
  PoseGraph::Edge copy2 = pose_graph.GetEdge(2, 1);
  EXPECT_EQ(copy2.cam2_from_cam1.translation().x(), -1);

  // Original unchanged.
  EXPECT_EQ(pose_graph.EdgeRef(1, 2).first.cam2_from_cam1.translation().x(), 1);

  // Non-existent pair should throw.
  EXPECT_THROW(pose_graph.GetEdge(1, 3), std::out_of_range);
}

TEST(PoseGraph, DeleteEdge) {
  PoseGraph pose_graph;
  pose_graph.AddEdge(1, 2, SynthesizeEdge());
  pose_graph.AddEdge(1, 3, SynthesizeEdge());

  EXPECT_TRUE(pose_graph.DeleteEdge(1, 2));
  EXPECT_FALSE(pose_graph.HasEdge(1, 2));
  EXPECT_EQ(pose_graph.NumEdges(), 1);

  // Delete with reversed order.
  EXPECT_TRUE(pose_graph.DeleteEdge(3, 1));
  EXPECT_EQ(pose_graph.NumEdges(), 0);

  // Delete non-existent returns false.
  EXPECT_FALSE(pose_graph.DeleteEdge(1, 2));
}

TEST(PoseGraph, UpdateEdge) {
  PoseGraph pose_graph;
  PoseGraph::Edge edge = SynthesizeEdge();
  edge.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0));
  pose_graph.AddEdge(1, 2, edge);

  // Update with normal order.
  PoseGraph::Edge updated = SynthesizeEdge();
  updated.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(5, 0, 0));
  pose_graph.UpdateEdge(1, 2, updated);

  EXPECT_EQ(pose_graph.EdgeRef(1, 2).first.cam2_from_cam1.translation().x(), 5);

  // Update with reversed order should invert.
  PoseGraph::Edge updated2 = SynthesizeEdge();
  updated2.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(3, 0, 0));
  pose_graph.UpdateEdge(2, 1, updated2);

  EXPECT_EQ(pose_graph.EdgeRef(1, 2).first.cam2_from_cam1.translation().x(),
            -3);

  // Update non-existent should throw.
  EXPECT_THROW(pose_graph.UpdateEdge(1, 3, SynthesizeEdge()),
               std::runtime_error);
}

TEST(PoseGraph, ValidEdges) {
  PoseGraph pose_graph;

  const image_pair_t pair_id1 = ImagePairToPairId(1, 2);
  const image_pair_t pair_id2 = ImagePairToPairId(1, 3);
  const image_pair_t pair_id3 = ImagePairToPairId(2, 3);
  pose_graph.AddEdge(1, 2, SynthesizeEdge());
  pose_graph.AddEdge(1, 3, SynthesizeEdge());
  pose_graph.AddEdge(2, 3, SynthesizeEdge());

  auto GetValidPairIds = [&]() {
    std::vector<image_pair_t> ids;
    for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
      ids.push_back(pair_id);
    }
    return ids;
  };

  // All pairs start valid.
  EXPECT_THAT(GetValidPairIds(),
              testing::UnorderedElementsAre(pair_id1, pair_id2, pair_id3));

  // Invalidate one pair.
  pose_graph.SetInvalidEdge(pair_id2);
  EXPECT_THAT(GetValidPairIds(),
              testing::UnorderedElementsAre(pair_id1, pair_id3));

  // Re-validate the pair.
  pose_graph.SetValidEdge(pair_id2);
  EXPECT_THAT(GetValidPairIds(),
              testing::UnorderedElementsAre(pair_id1, pair_id2, pair_id3));
}

TEST(PoseGraph, Load) {
  const auto test_dir = CreateTestDir();
  auto database = Database::Open(test_dir / "database.db");

  Camera camera = Camera::CreateFromModelId(
      kInvalidCameraId, SimplePinholeCameraModel::model_id, 1, 1, 1);
  const camera_t camera_id = database->WriteCamera(camera);

  // Create images.
  for (int i = 1; i <= 3; ++i) {
    Image image;
    image.SetName("image" + std::to_string(i));
    image.SetCameraId(camera_id);
    database->WriteImage(image);
  }

  TwoViewGeometry two_view;
  two_view.config = TwoViewGeometry::CALIBRATED;
  two_view.inlier_matches = {{0, 0}, {1, 1}};
  two_view.cam2_from_cam1 = Rigid3d(Eigen::Quaterniond::UnitRandom(),
                                    Eigen::Vector3d::Random().normalized());

  // Create pairs (1,2) and (2,3)
  database->WriteMatches(1, 2, FeatureMatches(10));
  database->WriteMatches(2, 3, FeatureMatches(10));
  database->WriteTwoViewGeometry(1, 2, two_view);
  database->WriteTwoViewGeometry(2, 3, two_view);

  // Load into DatabaseCache with relative poses.
  DatabaseCache cache;
  DatabaseCache::Options options;
  cache.Load(*database, options);

  PoseGraph pose_graph;
  pose_graph.Load(*cache.CorrespondenceGraph());

  EXPECT_EQ(pose_graph.NumEdges(), 2);
  EXPECT_TRUE(pose_graph.HasEdge(1, 2));
  EXPECT_TRUE(pose_graph.HasEdge(2, 3));
}

}  // namespace
}  // namespace colmap
