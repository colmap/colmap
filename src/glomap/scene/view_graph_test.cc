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

#include <gtest/gtest.h>

namespace glomap {
namespace {

ImagePair SynthesizeImagePair(int num_inliers = 50,
                              int num_matches = 100,
                              bool is_valid = true) {
  ImagePair pair;
  // Match feature i in image 1 to feature i in image 2.
  pair.matches.resize(num_matches, 2);
  for (int i = 0; i < num_matches; ++i) {
    pair.matches(i, 0) = i;
    pair.matches(i, 1) = i;
  }
  // First num_inliers matches are inliers.
  pair.inliers.resize(num_inliers);
  for (int i = 0; i < num_inliers; ++i) {
    pair.inliers[i] = i;
  }
  pair.is_valid = is_valid;
  return pair;
}

colmap::Rigid3d AddRotationError(const colmap::Rigid3d& pose,
                                 double error_deg) {
  const Eigen::Quaterniond error_rotation(
      Eigen::AngleAxisd(colmap::DegToRad(error_deg), Eigen::Vector3d::UnitZ()));
  return colmap::Rigid3d(error_rotation * pose.rotation, pose.translation);
}

TEST(ViewGraph, FilterByNumInliers) {
  ViewGraph view_graph;

  const image_pair_t pair_id1 = colmap::ImagePairToPairId(1, 2);
  const image_pair_t pair_id2 = colmap::ImagePairToPairId(1, 3);
  const image_pair_t pair_id3 = colmap::ImagePairToPairId(2, 3);
  const image_pair_t pair_id4 = colmap::ImagePairToPairId(2, 4);
  view_graph.AddImagePair(1, 2, SynthesizeImagePair(50));
  view_graph.AddImagePair(1, 3, SynthesizeImagePair(20));
  view_graph.AddImagePair(2, 3, SynthesizeImagePair(30));
  view_graph.AddImagePair(2, 4, SynthesizeImagePair(50, 100, false));

  view_graph.FilterByNumInliers(30);

  EXPECT_TRUE(view_graph.image_pairs.at(pair_id1).is_valid);
  EXPECT_FALSE(view_graph.image_pairs.at(pair_id2).is_valid);
  EXPECT_TRUE(view_graph.image_pairs.at(pair_id3).is_valid);
  EXPECT_FALSE(view_graph.image_pairs.at(pair_id4).is_valid);
}

TEST(ViewGraph, FilterByInlierRatio) {
  ViewGraph view_graph;

  const image_pair_t pair_id1 = colmap::ImagePairToPairId(1, 2);
  const image_pair_t pair_id2 = colmap::ImagePairToPairId(1, 3);
  const image_pair_t pair_id3 = colmap::ImagePairToPairId(2, 3);
  const image_pair_t pair_id4 = colmap::ImagePairToPairId(2, 4);
  view_graph.AddImagePair(1, 2, SynthesizeImagePair(50));  // 50% ratio
  view_graph.AddImagePair(1, 3, SynthesizeImagePair(10));  // 10% ratio
  view_graph.AddImagePair(2, 3, SynthesizeImagePair(25));  // 25% ratio
  view_graph.AddImagePair(
      2, 4, SynthesizeImagePair(50, 100, false));  // invalid

  view_graph.FilterByInlierRatio(0.25);

  EXPECT_TRUE(view_graph.image_pairs.at(pair_id1).is_valid);
  EXPECT_FALSE(view_graph.image_pairs.at(pair_id2).is_valid);
  EXPECT_TRUE(view_graph.image_pairs.at(pair_id3).is_valid);
  EXPECT_FALSE(view_graph.image_pairs.at(pair_id4).is_valid);
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
  ImagePair pair4 = SynthesizeImagePair(50, 100, false);
  pair4.cam2_from_cam1 = GetRelativePose(id2, id3);

  const image_pair_t pair_id1 = colmap::ImagePairToPairId(id1, id2);
  const image_pair_t pair_id2 = colmap::ImagePairToPairId(id1, id3);
  const image_pair_t pair_id3 = colmap::ImagePairToPairId(id1, id4);
  const image_pair_t pair_id4 = colmap::ImagePairToPairId(id2, id3);
  view_graph.AddImagePair(id1, id2, std::move(pair1));
  view_graph.AddImagePair(id1, id3, std::move(pair2));
  view_graph.AddImagePair(id1, id4, std::move(pair3));
  view_graph.AddImagePair(id2, id3, std::move(pair4));

  reconstruction.DeRegisterFrame(reconstruction.Image(id4).FrameId());

  view_graph.FilterByRelativeRotation(reconstruction, 5.0);

  EXPECT_TRUE(view_graph.image_pairs.at(pair_id1).is_valid);
  EXPECT_FALSE(view_graph.image_pairs.at(pair_id2).is_valid);
  EXPECT_TRUE(view_graph.image_pairs.at(pair_id3).is_valid);
  EXPECT_FALSE(view_graph.image_pairs.at(pair_id4).is_valid);
}

}  // namespace
}  // namespace glomap
