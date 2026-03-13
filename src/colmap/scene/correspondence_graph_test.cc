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

#include "colmap/scene/correspondence_graph.h"

#include "colmap/geometry/rigid3_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

int CountNumTransitiveCorrespondences(const CorrespondenceGraph& graph,
                                      const image_t image_id,
                                      const point2D_t point2D_idx,
                                      const size_t transitivity) {
  std::vector<CorrespondenceGraph::Correspondence> corrs;
  graph.ExtractTransitiveCorrespondences(
      image_id, point2D_idx, transitivity, &corrs);
  return corrs.size();
}

TEST(Correspondence, Print) {
  CorrespondenceGraph::Correspondence correspondence(1, 2);
  std::ostringstream stream;
  stream << correspondence;
  EXPECT_EQ(stream.str(), "Correspondence(image_id=1, point2D_idx=2)");
}

TEST(CorrespondenceGraph, Empty) {
  CorrespondenceGraph correspondence_graph;
  EXPECT_EQ(correspondence_graph.NumImages(), 0);
  EXPECT_EQ(correspondence_graph.NumImagePairs(), 0);
  EXPECT_EQ(correspondence_graph.NumMatchesBetweenAllImages().size(), 0);
  EXPECT_EQ(correspondence_graph.ImagePairs().size(), 0);
}

TEST(CorrespondenceGraph, Print) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 10);
  correspondence_graph.AddTwoViewGeometry(0, 1, TwoViewGeometry());
  std::ostringstream stream;
  stream << correspondence_graph;
  EXPECT_EQ(stream.str(),
            "CorrespondenceGraph(num_images=2, num_image_pairs=1)");
}

TEST(CorrespondenceGraph, TwoView) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 10);
  EXPECT_TRUE(correspondence_graph.ExistsImage(0));
  EXPECT_TRUE(correspondence_graph.ExistsImage(1));
  EXPECT_FALSE(correspondence_graph.ExistsImage(2));
  EXPECT_EQ(correspondence_graph.NumImages(), 2);
  TwoViewGeometry two_view_geometry01;
  two_view_geometry01.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  two_view_geometry01.inlier_matches = {
      {0, 0},
      {1, 2},
      {3, 7},
      {4, 8},
  };
  correspondence_graph.AddTwoViewGeometry(0, 1, two_view_geometry01);
  correspondence_graph.Finalize();
  EXPECT_EQ(correspondence_graph.NumCorrespondencesForImage(0), 4);
  EXPECT_EQ(correspondence_graph.NumCorrespondencesForImage(1), 4);
  const image_pair_t pair_id = ImagePairToPairId(0, 1);
  EXPECT_EQ(correspondence_graph.NumMatchesBetweenAllImages().size(), 1);
  EXPECT_EQ(correspondence_graph.NumMatchesBetweenAllImages().at(pair_id), 4);
  EXPECT_THAT(correspondence_graph.ImagePairs(),
              testing::UnorderedElementsAre(pair_id));
  const TwoViewGeometry two_view_geometry01_stored =
      correspondence_graph.ExtractTwoViewGeometry(
          0, 1, /*extract_inlier_matches=*/true);
  EXPECT_THAT(two_view_geometry01_stored.cam2_from_cam1.value(),
              Rigid3dNear(two_view_geometry01.cam2_from_cam1.value(),
                          /*rtol=*/1e-6,
                          /*ttol=*/1e-6));
  EXPECT_EQ(two_view_geometry01_stored.inlier_matches,
            two_view_geometry01.inlier_matches);
  FeatureMatches matches01_stored;
  correspondence_graph.ExtractMatchesBetweenImages(0, 1, matches01_stored);
  EXPECT_EQ(matches01_stored, two_view_geometry01.inlier_matches);
  TwoViewGeometry two_view_geometry10 = two_view_geometry01;
  two_view_geometry10.Invert();
  const TwoViewGeometry two_view_geometry10_stored =
      correspondence_graph.ExtractTwoViewGeometry(
          1, 0, /*extract_inlier_matches=*/true);
  EXPECT_THAT(two_view_geometry10_stored.cam2_from_cam1.value(),
              Rigid3dNear(two_view_geometry10.cam2_from_cam1.value(),
                          /*rtol=*/1e-6,
                          /*ttol=*/1e-6));
  EXPECT_EQ(two_view_geometry10_stored.inlier_matches,
            two_view_geometry10.inlier_matches);
  FeatureMatches matches10_stored;
  correspondence_graph.ExtractMatchesBetweenImages(1, 0, matches10_stored);
  EXPECT_EQ(matches10_stored, two_view_geometry10.inlier_matches);
  const TwoViewGeometry two_view_geometry01_without_matches_stored =
      correspondence_graph.ExtractTwoViewGeometry(
          0, 1, /*extract_inlier_matches=*/false);
  EXPECT_THAT(two_view_geometry01_without_matches_stored.inlier_matches,
              testing::IsEmpty());

  std::vector<CorrespondenceGraph::Correspondence> corrs;

  correspondence_graph.ExtractCorrespondences(0, 0, &corrs);
  EXPECT_EQ(corrs.size(), 1);
  EXPECT_TRUE(correspondence_graph.HasCorrespondences(0, 0));
  EXPECT_TRUE(correspondence_graph.IsTwoViewObservation(0, 0));
  EXPECT_EQ(corrs.at(0).image_id, 1);
  EXPECT_EQ(corrs.at(0).point2D_idx, 0);

  correspondence_graph.ExtractCorrespondences(1, 0, &corrs);
  EXPECT_EQ(corrs.size(), 1);
  EXPECT_TRUE(correspondence_graph.HasCorrespondences(1, 0));
  EXPECT_TRUE(correspondence_graph.IsTwoViewObservation(1, 0));
  EXPECT_EQ(corrs.at(0).image_id, 0);
  EXPECT_EQ(corrs.at(0).point2D_idx, 0);

  correspondence_graph.ExtractCorrespondences(0, 1, &corrs);
  EXPECT_EQ(corrs.size(), 1);
  EXPECT_TRUE(correspondence_graph.HasCorrespondences(0, 1));
  EXPECT_TRUE(correspondence_graph.IsTwoViewObservation(0, 1));
  EXPECT_EQ(corrs.at(0).image_id, 1);
  EXPECT_EQ(corrs.at(0).point2D_idx, 2);

  correspondence_graph.ExtractCorrespondences(1, 2, &corrs);
  EXPECT_EQ(corrs.size(), 1);
  EXPECT_TRUE(correspondence_graph.HasCorrespondences(1, 2));
  EXPECT_TRUE(correspondence_graph.IsTwoViewObservation(1, 2));
  EXPECT_EQ(corrs.at(0).image_id, 0);
  EXPECT_EQ(corrs.at(0).point2D_idx, 1);

  correspondence_graph.ExtractCorrespondences(0, 4, &corrs);
  EXPECT_EQ(corrs.size(), 1);
  EXPECT_TRUE(correspondence_graph.HasCorrespondences(0, 4));
  EXPECT_TRUE(correspondence_graph.IsTwoViewObservation(0, 4));
  EXPECT_EQ(corrs.at(0).image_id, 1);
  EXPECT_EQ(corrs.at(0).point2D_idx, 8);

  correspondence_graph.ExtractCorrespondences(0, 3, &corrs);
  EXPECT_EQ(corrs.size(), 1);
  EXPECT_TRUE(correspondence_graph.HasCorrespondences(0, 3));
  EXPECT_EQ(corrs.at(0).image_id, 1);
  EXPECT_EQ(corrs.at(0).point2D_idx, 7);

  correspondence_graph.ExtractCorrespondences(1, 7, &corrs);
  EXPECT_EQ(corrs.size(), 1);
  EXPECT_TRUE(correspondence_graph.HasCorrespondences(1, 7));
  EXPECT_TRUE(correspondence_graph.IsTwoViewObservation(1, 7));
  EXPECT_EQ(corrs.at(0).image_id, 0);
  EXPECT_EQ(corrs.at(0).point2D_idx, 3);

  correspondence_graph.ExtractCorrespondences(1, 8, &corrs);
  EXPECT_EQ(corrs.size(), 1);
  EXPECT_TRUE(correspondence_graph.HasCorrespondences(1, 8));
  EXPECT_TRUE(correspondence_graph.IsTwoViewObservation(1, 8));
  EXPECT_EQ(corrs.at(0).image_id, 0);
  EXPECT_EQ(corrs.at(0).point2D_idx, 4);

  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(CountNumTransitiveCorrespondences(correspondence_graph, 0, i, 0),
              0);
    correspondence_graph.ExtractCorrespondences(0, i, &corrs);
    EXPECT_EQ(corrs.size(),
              CountNumTransitiveCorrespondences(correspondence_graph, 0, i, 2));
    EXPECT_EQ(CountNumTransitiveCorrespondences(correspondence_graph, 1, i, 0),
              0);
    correspondence_graph.ExtractCorrespondences(1, i, &corrs);
    EXPECT_EQ(corrs.size(),
              CountNumTransitiveCorrespondences(correspondence_graph, 1, i, 2));
  }
  FeatureMatches matches01;
  correspondence_graph.ExtractMatchesBetweenImages(0, 1, matches01);
  EXPECT_EQ(matches01, two_view_geometry01.inlier_matches);
  FeatureMatches matches10;
  correspondence_graph.ExtractMatchesBetweenImages(1, 0, matches10);
  EXPECT_EQ(matches10, two_view_geometry10.inlier_matches);
  EXPECT_EQ(correspondence_graph.NumObservationsForImage(0), 4);
  EXPECT_EQ(correspondence_graph.NumObservationsForImage(1), 4);
  EXPECT_EQ(correspondence_graph.NumCorrespondencesForImage(0), 4);
  EXPECT_EQ(correspondence_graph.NumCorrespondencesForImage(1), 4);
}

TEST(CorrespondenceGraph, ThreeView) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 10);
  correspondence_graph.AddImage(2, 10);
  TwoViewGeometry two_view_geometry01;
  two_view_geometry01.inlier_matches = {{0, 0}};
  correspondence_graph.AddTwoViewGeometry(0, 1, two_view_geometry01);
  TwoViewGeometry two_view_geometry02;
  two_view_geometry02.inlier_matches = {{0, 0}};
  correspondence_graph.AddTwoViewGeometry(0, 2, two_view_geometry02);
  TwoViewGeometry two_view_geometry12;
  two_view_geometry12.inlier_matches = {{0, 0}, {5, 5}};
  correspondence_graph.AddTwoViewGeometry(1, 2, two_view_geometry12);
  correspondence_graph.Finalize();
  EXPECT_EQ(correspondence_graph.NumObservationsForImage(0), 1);
  EXPECT_EQ(correspondence_graph.NumObservationsForImage(1), 2);
  EXPECT_EQ(correspondence_graph.NumObservationsForImage(2), 2);
  EXPECT_EQ(correspondence_graph.NumCorrespondencesForImage(0), 2);
  EXPECT_EQ(correspondence_graph.NumCorrespondencesForImage(1), 3);
  EXPECT_EQ(correspondence_graph.NumCorrespondencesForImage(2), 3);
  const image_pair_t pair_id01 = ImagePairToPairId(0, 1);
  const image_pair_t pair_id02 = ImagePairToPairId(0, 2);
  const image_pair_t pair_id12 = ImagePairToPairId(1, 2);
  EXPECT_EQ(correspondence_graph.NumMatchesBetweenAllImages().size(), 3);
  EXPECT_EQ(correspondence_graph.NumMatchesBetweenAllImages().at(pair_id01), 1);
  EXPECT_EQ(correspondence_graph.NumMatchesBetweenAllImages().at(pair_id02), 1);
  EXPECT_EQ(correspondence_graph.NumMatchesBetweenAllImages().at(pair_id12), 2);
  EXPECT_THAT(correspondence_graph.ImagePairs(),
              testing::UnorderedElementsAre(pair_id01, pair_id02, pair_id12));

  std::vector<CorrespondenceGraph::Correspondence> corrs;
  correspondence_graph.ExtractCorrespondences(0, 0, &corrs);
  EXPECT_EQ(corrs.size(), 2);
  EXPECT_EQ(corrs.at(0).image_id, 1);
  EXPECT_EQ(corrs.at(0).point2D_idx, 0);
  EXPECT_EQ(corrs.at(1).image_id, 2);
  EXPECT_EQ(corrs.at(1).point2D_idx, 0);

  correspondence_graph.ExtractCorrespondences(1, 0, &corrs);
  EXPECT_EQ(corrs.size(), 2);
  EXPECT_EQ(corrs.at(0).image_id, 0);
  EXPECT_EQ(corrs.at(0).point2D_idx, 0);
  EXPECT_EQ(corrs.at(1).image_id, 2);
  EXPECT_EQ(corrs.at(1).point2D_idx, 0);

  correspondence_graph.ExtractCorrespondences(2, 0, &corrs);
  EXPECT_EQ(corrs.size(), 2);
  EXPECT_EQ(corrs.at(0).image_id, 0);
  EXPECT_EQ(corrs.at(0).point2D_idx, 0);
  EXPECT_EQ(corrs.at(1).image_id, 1);
  EXPECT_EQ(corrs.at(1).point2D_idx, 0);

  correspondence_graph.ExtractCorrespondences(1, 5, &corrs);
  EXPECT_EQ(corrs.size(), 1);
  EXPECT_EQ(corrs.at(0).image_id, 2);
  EXPECT_EQ(corrs.at(0).point2D_idx, 5);

  correspondence_graph.ExtractCorrespondences(2, 5, &corrs);
  EXPECT_EQ(corrs.size(), 1);
  EXPECT_EQ(corrs.at(0).image_id, 1);
  EXPECT_EQ(corrs.at(0).point2D_idx, 5);

  EXPECT_EQ(CountNumTransitiveCorrespondences(correspondence_graph, 0, 0, 2),
            2);
  EXPECT_EQ(CountNumTransitiveCorrespondences(correspondence_graph, 1, 0, 2),
            2);
  EXPECT_EQ(CountNumTransitiveCorrespondences(correspondence_graph, 2, 0, 2),
            2);
  EXPECT_EQ(CountNumTransitiveCorrespondences(correspondence_graph, 0, 0, 3),
            2);
  EXPECT_EQ(CountNumTransitiveCorrespondences(correspondence_graph, 1, 0, 3),
            2);
  EXPECT_EQ(CountNumTransitiveCorrespondences(correspondence_graph, 2, 0, 3),
            2);
}

TEST(CorrespondenceGraph, OutOfBounds) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 4);
  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches = {
      {9, 3},
      {10, 3},
      {9, 4},
  };
  correspondence_graph.AddTwoViewGeometry(0, 1, two_view_geometry);
  EXPECT_EQ(correspondence_graph.NumCorrespondencesForImage(0), 1);
  EXPECT_EQ(correspondence_graph.NumCorrespondencesForImage(1), 1);
  const image_pair_t pair_id = ImagePairToPairId(0, 1);
  EXPECT_EQ(correspondence_graph.NumMatchesBetweenAllImages().at(pair_id), 1);
}

TEST(CorrespondenceGraph, Duplicate) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 10);
  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches = {
      {0, 0},
      {1, 1},
      {1, 1},
      {3, 3},
      {3, 4},
  };
  correspondence_graph.AddTwoViewGeometry(0, 1, two_view_geometry);
  EXPECT_EQ(correspondence_graph.NumCorrespondencesForImage(0), 4);
  EXPECT_EQ(correspondence_graph.NumCorrespondencesForImage(1), 4);
  const image_pair_t pair_id = ImagePairToPairId(0, 1);
  EXPECT_EQ(correspondence_graph.NumMatchesBetweenAllImages().at(pair_id), 4);
}

TEST(CorrespondenceGraph, UpdateTwoViewGeometry) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 10);
  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.inlier_matches = {{0, 0}, {1, 2}, {3, 7}};
  correspondence_graph.AddTwoViewGeometry(0, 1, two_view_geometry);
  correspondence_graph.Finalize();

  // Verify initial state has no relative pose.
  TwoViewGeometry extracted =
      correspondence_graph.ExtractTwoViewGeometry(0, 1, false);
  EXPECT_FALSE(extracted.cam2_from_cam1.has_value());
  EXPECT_EQ(extracted.config, TwoViewGeometry::CALIBRATED);

  // Update with a decomposed relative pose.
  TwoViewGeometry updated = extracted;
  updated.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  correspondence_graph.UpdateTwoViewGeometry(0, 1, updated);

  // Verify the updated geometry is returned.
  TwoViewGeometry after_update =
      correspondence_graph.ExtractTwoViewGeometry(0, 1, false);
  EXPECT_TRUE(after_update.cam2_from_cam1.has_value());
  EXPECT_THAT(after_update.cam2_from_cam1.value(),
              Rigid3dNear(updated.cam2_from_cam1.value(), 1e-6, 1e-6));
  EXPECT_EQ(after_update.config, TwoViewGeometry::CALIBRATED);

  // Verify matches are preserved (stored separately from geometry).
  FeatureMatches matches;
  correspondence_graph.ExtractMatchesBetweenImages(0, 1, matches);
  EXPECT_EQ(matches, two_view_geometry.inlier_matches);
}

TEST(CorrespondenceGraph, UpdateTwoViewGeometrySwapped) {
  CorrespondenceGraph correspondence_graph;
  // Use image IDs where ShouldSwapImagePair(1, 0) is true, i.e. id1 > id2.
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 10);
  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.inlier_matches = {{0, 0}, {1, 2}};
  correspondence_graph.AddTwoViewGeometry(0, 1, two_view_geometry);
  correspondence_graph.Finalize();

  // Update using the swapped order (1, 0).
  const Rigid3d cam1_from_cam0(Eigen::Quaterniond::UnitRandom(),
                               Eigen::Vector3d::Random());
  TwoViewGeometry updated;
  updated.config = TwoViewGeometry::CALIBRATED;
  updated.cam2_from_cam1 = cam1_from_cam0;
  correspondence_graph.UpdateTwoViewGeometry(1, 0, updated);

  // Extract in the same swapped order and verify it matches.
  const TwoViewGeometry extracted_10 =
      correspondence_graph.ExtractTwoViewGeometry(1, 0, false);
  EXPECT_THAT(extracted_10.cam2_from_cam1.value(),
              Rigid3dNear(cam1_from_cam0, 1e-6, 1e-6));

  // Extract in canonical order and verify it's the inverse.
  const TwoViewGeometry extracted_01 =
      correspondence_graph.ExtractTwoViewGeometry(0, 1, false);
  EXPECT_THAT(extracted_01.cam2_from_cam1.value(),
              Rigid3dNear(Inverse(cam1_from_cam0), 1e-6, 1e-6));
}

}  // namespace
}  // namespace colmap
