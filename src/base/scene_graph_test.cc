// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#define TEST_NAME "base/scene_graph"
#include "util/testing.h"

#include "base/scene_graph.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestDefault) {
  SceneGraph scene_graph;
  BOOST_CHECK_EQUAL(scene_graph.NumImages(), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesBetweenImages().size(), 0);
}

BOOST_AUTO_TEST_CASE(TestTwoView) {
  SceneGraph scene_graph;
  scene_graph.AddImage(0, 10);
  scene_graph.AddImage(1, 10);
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(0), true);
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(1), true);
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(2), false);
  BOOST_CHECK_EQUAL(scene_graph.NumImages(), 2);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesBetweenImages().size(), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(0), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(1), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesBetweenImages(0, 1), 0);
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK(!scene_graph.HasCorrespondences(0, i));
    BOOST_CHECK(!scene_graph.HasCorrespondences(1, i));
    BOOST_CHECK(!scene_graph.IsTwoViewObservation(0, i));
    BOOST_CHECK(!scene_graph.IsTwoViewObservation(1, i));
  }
  FeatureMatches matches(4);
  matches[0].point2D_idx1 = 0;
  matches[0].point2D_idx2 = 0;
  matches[1].point2D_idx1 = 1;
  matches[1].point2D_idx2 = 2;
  matches[2].point2D_idx1 = 3;
  matches[2].point2D_idx2 = 7;
  matches[3].point2D_idx1 = 4;
  matches[3].point2D_idx2 = 8;
  scene_graph.AddCorrespondences(0, 1, matches);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(0), 4);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(1), 4);
  const image_pair_t pair_id = Database::ImagePairToPairId(0, 1);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesBetweenImages().size(), 1);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesBetweenImages().at(pair_id),
                    4);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 0).size(), 1);
  BOOST_CHECK(scene_graph.HasCorrespondences(0, 0));
  BOOST_CHECK(scene_graph.IsTwoViewObservation(0, 0));
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 0).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 0).size(), 1);
  BOOST_CHECK(scene_graph.HasCorrespondences(1, 0));
  BOOST_CHECK(scene_graph.IsTwoViewObservation(1, 0));
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 0).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 1).size(), 1);
  BOOST_CHECK(scene_graph.HasCorrespondences(0, 1));
  BOOST_CHECK(scene_graph.IsTwoViewObservation(0, 1));
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 1).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 1).at(0).point2D_idx, 2);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 2).size(), 1);
  BOOST_CHECK(scene_graph.HasCorrespondences(1, 2));
  BOOST_CHECK(scene_graph.IsTwoViewObservation(1, 2));
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 2).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 2).at(0).point2D_idx, 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 4).size(), 1);
  BOOST_CHECK(scene_graph.HasCorrespondences(0, 3));
  BOOST_CHECK(scene_graph.IsTwoViewObservation(0, 4));
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 3).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 3).at(0).point2D_idx, 7);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 4).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 4).at(0).point2D_idx, 8);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 7).size(), 1);
  BOOST_CHECK(scene_graph.HasCorrespondences(1, 7));
  BOOST_CHECK(scene_graph.IsTwoViewObservation(1, 7));
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 7).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 7).at(0).point2D_idx, 3);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 8).size(), 1);
  BOOST_CHECK(scene_graph.HasCorrespondences(1, 8));
  BOOST_CHECK(scene_graph.IsTwoViewObservation(1, 8));
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 8).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 8).at(0).point2D_idx, 4);
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(scene_graph.FindTransitiveCorrespondences(0, i, 0).size(),
                      0);
    BOOST_CHECK_EQUAL(
        scene_graph.FindCorrespondences(0, i).size(),
        scene_graph.FindTransitiveCorrespondences(0, i, 1).size());
    BOOST_CHECK_EQUAL(
        scene_graph.FindCorrespondences(0, i).size(),
        scene_graph.FindTransitiveCorrespondences(0, i, 2).size());
    BOOST_CHECK_EQUAL(scene_graph.FindTransitiveCorrespondences(1, i, 0).size(),
                      0);
    BOOST_CHECK_EQUAL(
        scene_graph.FindCorrespondences(1, i).size(),
        scene_graph.FindTransitiveCorrespondences(1, i, 1).size());
    BOOST_CHECK_EQUAL(
        scene_graph.FindCorrespondences(1, i).size(),
        scene_graph.FindTransitiveCorrespondences(1, i, 2).size());
  }
  const auto corrs01 = scene_graph.FindCorrespondencesBetweenImages(0, 1);
  const auto corrs10 = scene_graph.FindCorrespondencesBetweenImages(1, 0);
  BOOST_CHECK_EQUAL(corrs01.size(), matches.size());
  BOOST_CHECK_EQUAL(corrs10.size(), matches.size());
  for (size_t i = 0; i < corrs01.size(); ++i) {
    BOOST_CHECK_EQUAL(corrs01[i].first, corrs10[i].second);
    BOOST_CHECK_EQUAL(corrs01[i].second, corrs10[i].first);
    BOOST_CHECK_EQUAL(matches[i].point2D_idx1, corrs01[i].first);
    BOOST_CHECK_EQUAL(matches[i].point2D_idx2, corrs01[i].second);
  }
  scene_graph.Finalize();
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(0), 4);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(1), 4);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(0), 4);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(1), 4);
}

BOOST_AUTO_TEST_CASE(TestThreeView) {
  SceneGraph scene_graph;
  scene_graph.AddImage(0, 10);
  scene_graph.AddImage(1, 10);
  scene_graph.AddImage(2, 10);
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(0), true);
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(1), true);
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(2), true);
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(3), false);
  BOOST_CHECK_EQUAL(scene_graph.NumImages(), 3);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesBetweenImages().size(), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(2), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(0), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(1), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(2), 0);
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK(!scene_graph.HasCorrespondences(0, i));
    BOOST_CHECK(!scene_graph.HasCorrespondences(1, i));
    BOOST_CHECK(!scene_graph.HasCorrespondences(2, i));
    BOOST_CHECK(!scene_graph.IsTwoViewObservation(0, i));
    BOOST_CHECK(!scene_graph.IsTwoViewObservation(1, i));
    BOOST_CHECK(!scene_graph.IsTwoViewObservation(2, i));
  }
  FeatureMatches matches01(1);
  matches01[0].point2D_idx1 = 0;
  matches01[0].point2D_idx2 = 0;
  scene_graph.AddCorrespondences(0, 1, matches01);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(2), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(0), 1);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(1), 1);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(2), 0);
  FeatureMatches matches02(1);
  matches02[0].point2D_idx1 = 0;
  matches02[0].point2D_idx2 = 0;
  scene_graph.AddCorrespondences(0, 2, matches02);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(2), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(1), 1);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(2), 1);
  FeatureMatches matches12(2);
  matches12[0].point2D_idx1 = 0;
  matches12[0].point2D_idx2 = 0;
  matches12[1].point2D_idx1 = 5;
  matches12[1].point2D_idx2 = 5;
  scene_graph.AddCorrespondences(1, 2, matches12);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(2), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(1), 3);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(2), 3);
  const image_pair_t pair_id01 = Database::ImagePairToPairId(0, 1);
  const image_pair_t pair_id02 = Database::ImagePairToPairId(0, 2);
  const image_pair_t pair_id12 = Database::ImagePairToPairId(1, 2);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesBetweenImages().size(), 3);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesBetweenImages().at(pair_id01),
                    1);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesBetweenImages().at(pair_id02),
                    1);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesBetweenImages().at(pair_id12),
                    2);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 0).size(), 2);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 0).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 0).at(1).image_id, 2);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 0).at(1).point2D_idx, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 0).size(), 2);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 0).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 0).at(1).image_id, 2);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 0).at(1).point2D_idx, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(2, 0).size(), 2);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(2, 0).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(2, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(2, 0).at(1).image_id, 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(2, 0).at(1).point2D_idx, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 5).size(), 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 5).at(0).image_id, 2);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 5).at(0).point2D_idx, 5);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(2, 5).size(), 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(2, 5).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(2, 5).at(0).point2D_idx, 5);
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(
        scene_graph.FindCorrespondences(0, i).size(),
        scene_graph.FindTransitiveCorrespondences(0, i, 1).size());
    BOOST_CHECK_EQUAL(
        scene_graph.FindCorrespondences(1, i).size(),
        scene_graph.FindTransitiveCorrespondences(1, i, 1).size());
    BOOST_CHECK_EQUAL(
        scene_graph.FindCorrespondences(2, i).size(),
        scene_graph.FindTransitiveCorrespondences(2, i, 1).size());
  }
  BOOST_CHECK_EQUAL(scene_graph.FindTransitiveCorrespondences(0, 0, 2).size(),
                    2);
  BOOST_CHECK_EQUAL(scene_graph.FindTransitiveCorrespondences(1, 0, 2).size(),
                    2);
  BOOST_CHECK_EQUAL(scene_graph.FindTransitiveCorrespondences(2, 0, 2).size(),
                    2);
  BOOST_CHECK_EQUAL(scene_graph.FindTransitiveCorrespondences(0, 0, 3).size(),
                    2);
  BOOST_CHECK_EQUAL(scene_graph.FindTransitiveCorrespondences(1, 0, 3).size(),
                    2);
  BOOST_CHECK_EQUAL(scene_graph.FindTransitiveCorrespondences(2, 0, 3).size(),
                    2);
  scene_graph.Finalize();
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(0), 1);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(1), 2);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(2), 2);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(1), 3);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(2), 3);
  scene_graph.AddImage(3, 10);
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(0), true);
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(1), true);
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(2), true);
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(3), true);
  BOOST_CHECK_EQUAL(scene_graph.NumImages(), 4);
  scene_graph.Finalize();
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(0), true);
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(1), true);
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(2), true);
  BOOST_CHECK_EQUAL(scene_graph.ExistsImage(3), false);
  BOOST_CHECK_EQUAL(scene_graph.NumImages(), 3);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(0), 1);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(1), 2);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(2), 2);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(1), 3);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(2), 3);
}

BOOST_AUTO_TEST_CASE(TestOutOfBounds) {
  SceneGraph scene_graph;
  scene_graph.AddImage(0, 10);
  scene_graph.AddImage(1, 4);
  FeatureMatches matches(3);
  matches[0].point2D_idx1 = 9;
  matches[0].point2D_idx2 = 3;
  matches[1].point2D_idx1 = 10;
  matches[1].point2D_idx2 = 3;
  matches[2].point2D_idx1 = 9;
  matches[2].point2D_idx2 = 4;
  scene_graph.AddCorrespondences(0, 1, matches);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(0), 1);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(1), 1);
  const image_pair_t pair_id = Database::ImagePairToPairId(0, 1);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesBetweenImages().at(pair_id),
                    1);
}

BOOST_AUTO_TEST_CASE(TestDuplicate) {
  SceneGraph scene_graph;
  scene_graph.AddImage(0, 10);
  scene_graph.AddImage(1, 10);
  FeatureMatches matches(5);
  matches[0].point2D_idx1 = 0;
  matches[0].point2D_idx2 = 0;
  matches[1].point2D_idx1 = 1;
  matches[1].point2D_idx2 = 1;
  matches[2].point2D_idx1 = 1;
  matches[2].point2D_idx2 = 1;
  matches[3].point2D_idx1 = 3;
  matches[3].point2D_idx2 = 3;
  matches[4].point2D_idx1 = 3;
  matches[4].point2D_idx2 = 4;
  scene_graph.AddCorrespondences(0, 1, matches);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(0), 3);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(1), 3);
  const image_pair_t pair_id = Database::ImagePairToPairId(0, 1);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesBetweenImages().at(pair_id),
                    3);
}
