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

#include "glomap/scene/view_graph.h"

#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace glomap {
namespace {

colmap::Rigid3d TestCam2FromCam1() {
  return colmap::Rigid3d(Eigen::Quaterniond::UnitRandom(),
                         Eigen::Vector3d::Random().normalized());
}

ImagePair SynthesizeImagePair(int num_inliers = 50, int num_matches = 100) {
  THROW_CHECK_LE(num_inliers, num_matches);
  ImagePair pair;
  // Set default identity pose.
  pair.cam2_from_cam1 = colmap::Rigid3d();
  // Match feature i in image 1 to feature i in image 2.
  pair.matches.resize(num_matches, 2);
  for (int i = 0; i < num_matches; ++i) {
    pair.matches(i, 0) = i;
    pair.matches(i, 1) = i;
  }
  // First num_inliers matches are inliers.
  pair.inlier_matches.reserve(num_inliers);
  for (int i = 0; i < num_inliers; ++i) {
    pair.inlier_matches.emplace_back(i, i);
  }
  return pair;
}

colmap::Rigid3d AddRotationError(const colmap::Rigid3d& pose,
                                 double error_deg) {
  const Eigen::Quaterniond error_rotation(
      Eigen::AngleAxisd(colmap::DegToRad(error_deg), Eigen::Vector3d::UnitZ()));
  return colmap::Rigid3d(error_rotation * pose.rotation, pose.translation);
}

TEST(ViewGraph, Nominal) {
  ViewGraph view_graph;

  // Empty view graph.
  EXPECT_TRUE(view_graph.Empty());
  EXPECT_EQ(view_graph.NumImagePairs(), 0);
  EXPECT_EQ(view_graph.NumValidImagePairs(), 0);

  // Add some pairs.
  view_graph.AddImagePair(1, 2, SynthesizeImagePair());
  view_graph.AddImagePair(1, 3, SynthesizeImagePair());
  view_graph.AddImagePair(2, 3, SynthesizeImagePair());

  EXPECT_FALSE(view_graph.Empty());
  EXPECT_EQ(view_graph.NumImagePairs(), 3);
  EXPECT_EQ(view_graph.NumValidImagePairs(), 3);

  // Invalidate one pair.
  view_graph.SetInvalidImagePair(colmap::ImagePairToPairId(1, 2));
  EXPECT_EQ(view_graph.NumImagePairs(), 3);
  EXPECT_EQ(view_graph.NumValidImagePairs(), 2);

  // Clear the view graph.
  view_graph.Clear();
  EXPECT_TRUE(view_graph.Empty());
  EXPECT_EQ(view_graph.NumImagePairs(), 0);
  EXPECT_EQ(view_graph.NumValidImagePairs(), 0);
}

TEST(ViewGraph, AddImagePair) {
  ViewGraph view_graph;

  // Normal add.
  ImagePair pair = SynthesizeImagePair();
  pair.cam2_from_cam1 =
      colmap::Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0));
  view_graph.AddImagePair(1, 2, pair);

  EXPECT_EQ(view_graph.NumImagePairs(), 1);
  const auto& [stored, swapped] = view_graph.ImagePair(1, 2);
  EXPECT_FALSE(swapped);
  EXPECT_TRUE(stored.cam2_from_cam1.has_value());
  EXPECT_EQ(stored.cam2_from_cam1->translation.x(), 1);

  // Add with swapped IDs should invert the pair.
  ImagePair pair2 = SynthesizeImagePair();
  pair2.cam2_from_cam1 =
      colmap::Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(2, 0, 0));
  view_graph.AddImagePair(4, 3, pair2);  // 4 > 3, should swap and invert

  EXPECT_EQ(view_graph.NumImagePairs(), 2);
  const auto& [stored2, swapped2] = view_graph.ImagePair(3, 4);
  EXPECT_FALSE(swapped2);
  EXPECT_TRUE(stored2.cam2_from_cam1.has_value());
  EXPECT_EQ(stored2.cam2_from_cam1->translation.x(), -2);

  // Duplicate should throw.
  EXPECT_THROW(view_graph.AddImagePair(1, 2, SynthesizeImagePair()),
               std::runtime_error);
  EXPECT_THROW(view_graph.AddImagePair(2, 1, SynthesizeImagePair()),
               std::runtime_error);
}

TEST(ViewGraph, HasImagePair) {
  ViewGraph view_graph;
  view_graph.AddImagePair(1, 2, SynthesizeImagePair());

  EXPECT_TRUE(view_graph.HasImagePair(1, 2));
  EXPECT_TRUE(view_graph.HasImagePair(2, 1));  // Order doesn't matter
  EXPECT_FALSE(view_graph.HasImagePair(1, 3));
}

TEST(ViewGraph, Pair) {
  ViewGraph view_graph;
  ImagePair pair = SynthesizeImagePair();
  pair.cam2_from_cam1 =
      colmap::Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0));
  view_graph.AddImagePair(1, 2, pair);

  // Normal order: swapped = false.
  auto [ref1, swapped1] = view_graph.ImagePair(1, 2);
  EXPECT_FALSE(swapped1);
  EXPECT_TRUE(ref1.cam2_from_cam1.has_value());
  EXPECT_EQ(ref1.cam2_from_cam1->translation.x(), 1);

  // Reversed order: swapped = true.
  auto [ref2, swapped2] = view_graph.ImagePair(2, 1);
  EXPECT_TRUE(swapped2);
  EXPECT_TRUE(ref2.cam2_from_cam1.has_value());
  EXPECT_EQ(ref2.cam2_from_cam1->translation.x(), 1);  // Same reference

  // Modify validity through ViewGraph.
  view_graph.SetInvalidImagePair(colmap::ImagePairToPairId(1, 2));
  EXPECT_FALSE(view_graph.IsValid(colmap::ImagePairToPairId(1, 2)));

  // Non-existent pair should throw.
  EXPECT_THROW(view_graph.ImagePair(1, 3), std::out_of_range);
}

TEST(ViewGraph, GetImagePair) {
  ViewGraph view_graph;
  ImagePair pair = SynthesizeImagePair();
  pair.cam2_from_cam1 =
      colmap::Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0));
  view_graph.AddImagePair(1, 2, pair);

  // Normal order: returns as-is.
  ImagePair copy1 = view_graph.GetImagePair(1, 2);
  EXPECT_TRUE(copy1.cam2_from_cam1.has_value());
  EXPECT_EQ(copy1.cam2_from_cam1->translation.x(), 1);

  // Reversed order: returns inverted copy.
  ImagePair copy2 = view_graph.GetImagePair(2, 1);
  EXPECT_TRUE(copy2.cam2_from_cam1.has_value());
  EXPECT_EQ(copy2.cam2_from_cam1->translation.x(), -1);

  // Original unchanged.
  EXPECT_TRUE(view_graph.ImagePair(1, 2).first.cam2_from_cam1.has_value());
  EXPECT_EQ(view_graph.ImagePair(1, 2).first.cam2_from_cam1->translation.x(),
            1);

  // Non-existent pair should throw.
  EXPECT_THROW(view_graph.GetImagePair(1, 3), std::out_of_range);
}

TEST(ViewGraph, DeleteImagePair) {
  ViewGraph view_graph;
  view_graph.AddImagePair(1, 2, SynthesizeImagePair());
  view_graph.AddImagePair(1, 3, SynthesizeImagePair());

  EXPECT_TRUE(view_graph.DeleteImagePair(1, 2));
  EXPECT_FALSE(view_graph.HasImagePair(1, 2));
  EXPECT_EQ(view_graph.NumImagePairs(), 1);

  // Delete with reversed order.
  EXPECT_TRUE(view_graph.DeleteImagePair(3, 1));
  EXPECT_EQ(view_graph.NumImagePairs(), 0);

  // Delete non-existent returns false.
  EXPECT_FALSE(view_graph.DeleteImagePair(1, 2));
}

TEST(ViewGraph, UpdateImagePair) {
  ViewGraph view_graph;
  ImagePair pair = SynthesizeImagePair();
  pair.cam2_from_cam1 =
      colmap::Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0));
  view_graph.AddImagePair(1, 2, pair);

  // Update with normal order.
  ImagePair updated = SynthesizeImagePair();
  updated.cam2_from_cam1 =
      colmap::Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(5, 0, 0));
  view_graph.UpdateImagePair(1, 2, updated);

  EXPECT_TRUE(view_graph.ImagePair(1, 2).first.cam2_from_cam1.has_value());
  EXPECT_EQ(view_graph.ImagePair(1, 2).first.cam2_from_cam1->translation.x(),
            5);

  // Update with reversed order should invert.
  ImagePair updated2 = SynthesizeImagePair();
  updated2.cam2_from_cam1 =
      colmap::Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(3, 0, 0));
  view_graph.UpdateImagePair(2, 1, updated2);

  EXPECT_TRUE(view_graph.ImagePair(1, 2).first.cam2_from_cam1.has_value());
  EXPECT_EQ(view_graph.ImagePair(1, 2).first.cam2_from_cam1->translation.x(),
            -3);

  // Update non-existent should throw.
  EXPECT_THROW(view_graph.UpdateImagePair(1, 3, SynthesizeImagePair()),
               std::runtime_error);
}

TEST(ViewGraph, ValidImagePairs) {
  ViewGraph view_graph;

  const image_pair_t pair_id1 = colmap::ImagePairToPairId(1, 2);
  const image_pair_t pair_id2 = colmap::ImagePairToPairId(1, 3);
  const image_pair_t pair_id3 = colmap::ImagePairToPairId(2, 3);
  view_graph.AddImagePair(1, 2, SynthesizeImagePair());
  view_graph.AddImagePair(1, 3, SynthesizeImagePair());
  view_graph.AddImagePair(2, 3, SynthesizeImagePair());

  auto GetValidPairIds = [&]() {
    std::vector<image_pair_t> ids;
    for (const auto& [pair_id, image_pair] : view_graph.ValidImagePairs()) {
      ids.push_back(pair_id);
    }
    return ids;
  };

  // All pairs start valid.
  EXPECT_THAT(GetValidPairIds(),
              testing::UnorderedElementsAre(pair_id1, pair_id2, pair_id3));

  // Invalidate one pair.
  view_graph.SetInvalidImagePair(pair_id2);
  EXPECT_THAT(GetValidPairIds(),
              testing::UnorderedElementsAre(pair_id1, pair_id3));

  // Re-validate the pair.
  view_graph.SetValidImagePair(pair_id2);
  EXPECT_THAT(GetValidPairIds(),
              testing::UnorderedElementsAre(pair_id1, pair_id2, pair_id3));
}

TEST(ViewGraph, FilterByRelativeRotation) {
  colmap::Reconstruction reconstruction;
  colmap::SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 4;
  colmap::SynthesizeDataset(options, &reconstruction);

  const std::vector<image_t> image_ids = reconstruction.RegImageIds();
  const image_t id1 = image_ids[0];
  const image_t id2 = image_ids[1];
  const image_t id3 = image_ids[2];
  const image_t id4 = image_ids[3];

  auto GetRelativePose = [&](image_t i, image_t j) {
    return reconstruction.Image(j).CamFromWorld() *
           colmap::Inverse(reconstruction.Image(i).CamFromWorld());
  };

  ViewGraph view_graph;
  ImagePair pair1 = SynthesizeImagePair();
  pair1.cam2_from_cam1 = AddRotationError(GetRelativePose(id1, id2), 3.0);
  ImagePair pair2 = SynthesizeImagePair();
  pair2.cam2_from_cam1 = AddRotationError(GetRelativePose(id1, id3), 10.0);
  ImagePair pair3 = SynthesizeImagePair();
  pair3.cam2_from_cam1 = AddRotationError(GetRelativePose(id1, id4), 90.0);
  ImagePair pair4 = SynthesizeImagePair(50);
  pair4.cam2_from_cam1 = GetRelativePose(id2, id3);

  const image_pair_t pair_id1 = colmap::ImagePairToPairId(id1, id2);
  const image_pair_t pair_id2 = colmap::ImagePairToPairId(id1, id3);
  const image_pair_t pair_id3 = colmap::ImagePairToPairId(id1, id4);
  const image_pair_t pair_id4 = colmap::ImagePairToPairId(id2, id3);
  view_graph.AddImagePair(id1, id2, std::move(pair1));
  view_graph.AddImagePair(id1, id3, std::move(pair2));
  view_graph.AddImagePair(id1, id4, std::move(pair3));
  view_graph.AddImagePair(id2, id3, std::move(pair4));
  view_graph.SetInvalidImagePair(pair_id4);  // Already invalid

  reconstruction.DeRegisterFrame(reconstruction.Image(id4).FrameId());

  view_graph.FilterByRelativeRotation(reconstruction, 5.0);

  EXPECT_TRUE(view_graph.IsValid(pair_id1));
  EXPECT_FALSE(view_graph.IsValid(pair_id2));
  EXPECT_TRUE(view_graph.IsValid(pair_id3));
  EXPECT_FALSE(view_graph.IsValid(pair_id4));
}

TEST(ViewGraph, LoadFromDatabase) {
  const std::string test_dir = colmap::CreateTestDir();

  // Create two databases with overlapping image pairs.
  auto database1 = colmap::Database::Open(test_dir + "/database1.db");
  auto database2 = colmap::Database::Open(test_dir + "/database2.db");

  colmap::Camera camera = colmap::Camera::CreateFromModelId(
      colmap::kInvalidCameraId,
      colmap::SimplePinholeCameraModel::model_id,
      1,
      1,
      1);
  const colmap::camera_t camera_id1 = database1->WriteCamera(camera);
  const colmap::camera_t camera_id2 = database2->WriteCamera(camera);

  // Create images in both databases.
  for (int i = 1; i <= 4; ++i) {
    colmap::Image image;
    image.SetName("image" + std::to_string(i));
    image.SetCameraId(camera_id1);
    database1->WriteImage(image);
    image.SetCameraId(camera_id2);
    database2->WriteImage(image);
  }

  colmap::TwoViewGeometry two_view;
  two_view.config = colmap::TwoViewGeometry::CALIBRATED;
  two_view.inlier_matches = {{0, 0}, {1, 1}};
  two_view.cam2_from_cam1 = TestCam2FromCam1();

  // Database1: pairs (1,2) and (2,3)
  database1->WriteMatches(1, 2, colmap::FeatureMatches(10));
  database1->WriteMatches(2, 3, colmap::FeatureMatches(10));
  database1->WriteTwoViewGeometry(1, 2, two_view);
  database1->WriteTwoViewGeometry(2, 3, two_view);

  // Database2: pairs (2,3) and (3,4) - pair (2,3) overlaps with database1
  database2->WriteMatches(2, 3, colmap::FeatureMatches(10));
  database2->WriteMatches(3, 4, colmap::FeatureMatches(10));
  database2->WriteTwoViewGeometry(2, 3, two_view);
  database2->WriteTwoViewGeometry(3, 4, two_view);

  ViewGraph view_graph;

  // First read from database1 should succeed.
  view_graph.LoadFromDatabase(*database1);
  EXPECT_EQ(view_graph.NumImagePairs(), 2);

  // Second read from database2 with allow_duplicate=false should throw
  // because pair (2,3) already exists.
  EXPECT_ANY_THROW(
      view_graph.LoadFromDatabase(*database2, /*allow_duplicate=*/false));

  // Second read from database2 with allow_duplicate=true should succeed.
  view_graph.LoadFromDatabase(*database2, /*allow_duplicate=*/true);
  // Should now have 3 pairs: (1,2), (2,3), (3,4)
  EXPECT_EQ(view_graph.NumImagePairs(), 3);
}

}  // namespace
}  // namespace glomap
