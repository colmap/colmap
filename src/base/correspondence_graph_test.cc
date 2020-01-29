// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "base/correspondence_graph"
#include "util/testing.h"

#include "base/correspondence_graph.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestDefault) {
  CorrespondenceGraph correspondence_graph;
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().size(), 0);
}

BOOST_AUTO_TEST_CASE(TestTwoView) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 10);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(0), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(1), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(2), false);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().size(), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesBetweenImages(0, 1),
                    0);
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK(!correspondence_graph.HasCorrespondences(0, i));
    BOOST_CHECK(!correspondence_graph.HasCorrespondences(1, i));
    BOOST_CHECK(!correspondence_graph.IsTwoViewObservation(0, i));
    BOOST_CHECK(!correspondence_graph.IsTwoViewObservation(1, i));
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
  correspondence_graph.AddCorrespondences(0, 1, matches);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 4);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 4);
  const image_pair_t pair_id = Database::ImagePairToPairId(0, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().size(), 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().at(pair_id), 4);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(0, 0).size(), 1);
  BOOST_CHECK(correspondence_graph.HasCorrespondences(0, 0));
  BOOST_CHECK(correspondence_graph.IsTwoViewObservation(0, 0));
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 0).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(1, 0).size(), 1);
  BOOST_CHECK(correspondence_graph.HasCorrespondences(1, 0));
  BOOST_CHECK(correspondence_graph.IsTwoViewObservation(1, 0));
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 0).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(0, 1).size(), 1);
  BOOST_CHECK(correspondence_graph.HasCorrespondences(0, 1));
  BOOST_CHECK(correspondence_graph.IsTwoViewObservation(0, 1));
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 1).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 1).at(0).point2D_idx, 2);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(1, 2).size(), 1);
  BOOST_CHECK(correspondence_graph.HasCorrespondences(1, 2));
  BOOST_CHECK(correspondence_graph.IsTwoViewObservation(1, 2));
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 2).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 2).at(0).point2D_idx, 1);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(0, 4).size(), 1);
  BOOST_CHECK(correspondence_graph.HasCorrespondences(0, 3));
  BOOST_CHECK(correspondence_graph.IsTwoViewObservation(0, 4));
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 3).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 3).at(0).point2D_idx, 7);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 4).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 4).at(0).point2D_idx, 8);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(1, 7).size(), 1);
  BOOST_CHECK(correspondence_graph.HasCorrespondences(1, 7));
  BOOST_CHECK(correspondence_graph.IsTwoViewObservation(1, 7));
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 7).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 7).at(0).point2D_idx, 3);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(1, 8).size(), 1);
  BOOST_CHECK(correspondence_graph.HasCorrespondences(1, 8));
  BOOST_CHECK(correspondence_graph.IsTwoViewObservation(1, 8));
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 8).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 8).at(0).point2D_idx, 4);
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(
        correspondence_graph.FindTransitiveCorrespondences(0, i, 0).size(), 0);
    BOOST_CHECK_EQUAL(
        correspondence_graph.FindCorrespondences(0, i).size(),
        correspondence_graph.FindTransitiveCorrespondences(0, i, 1).size());
    BOOST_CHECK_EQUAL(
        correspondence_graph.FindCorrespondences(0, i).size(),
        correspondence_graph.FindTransitiveCorrespondences(0, i, 2).size());
    BOOST_CHECK_EQUAL(
        correspondence_graph.FindTransitiveCorrespondences(1, i, 0).size(), 0);
    BOOST_CHECK_EQUAL(
        correspondence_graph.FindCorrespondences(1, i).size(),
        correspondence_graph.FindTransitiveCorrespondences(1, i, 1).size());
    BOOST_CHECK_EQUAL(
        correspondence_graph.FindCorrespondences(1, i).size(),
        correspondence_graph.FindTransitiveCorrespondences(1, i, 2).size());
  }
  const auto corrs01 =
      correspondence_graph.FindCorrespondencesBetweenImages(0, 1);
  const auto corrs10 =
      correspondence_graph.FindCorrespondencesBetweenImages(1, 0);
  BOOST_CHECK_EQUAL(corrs01.size(), matches.size());
  BOOST_CHECK_EQUAL(corrs10.size(), matches.size());
  for (size_t i = 0; i < corrs01.size(); ++i) {
    BOOST_CHECK_EQUAL(corrs01[i].point2D_idx1, corrs10[i].point2D_idx2);
    BOOST_CHECK_EQUAL(corrs01[i].point2D_idx2, corrs10[i].point2D_idx1);
    BOOST_CHECK_EQUAL(matches[i].point2D_idx1, corrs01[i].point2D_idx1);
    BOOST_CHECK_EQUAL(matches[i].point2D_idx2, corrs01[i].point2D_idx2);
  }
  correspondence_graph.Finalize();
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 4);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 4);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 4);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 4);
}

BOOST_AUTO_TEST_CASE(TestThreeView) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 10);
  correspondence_graph.AddImage(2, 10);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(0), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(1), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(2), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(3), false);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 3);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().size(), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 0);
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK(!correspondence_graph.HasCorrespondences(0, i));
    BOOST_CHECK(!correspondence_graph.HasCorrespondences(1, i));
    BOOST_CHECK(!correspondence_graph.HasCorrespondences(2, i));
    BOOST_CHECK(!correspondence_graph.IsTwoViewObservation(0, i));
    BOOST_CHECK(!correspondence_graph.IsTwoViewObservation(1, i));
    BOOST_CHECK(!correspondence_graph.IsTwoViewObservation(2, i));
  }
  FeatureMatches matches01(1);
  matches01[0].point2D_idx1 = 0;
  matches01[0].point2D_idx2 = 0;
  correspondence_graph.AddCorrespondences(0, 1, matches01);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 0);
  FeatureMatches matches02(1);
  matches02[0].point2D_idx1 = 0;
  matches02[0].point2D_idx2 = 0;
  correspondence_graph.AddCorrespondences(0, 2, matches02);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 1);
  FeatureMatches matches12(2);
  matches12[0].point2D_idx1 = 0;
  matches12[0].point2D_idx2 = 0;
  matches12[1].point2D_idx1 = 5;
  matches12[1].point2D_idx2 = 5;
  correspondence_graph.AddCorrespondences(1, 2, matches12);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 3);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 3);
  const image_pair_t pair_id01 = Database::ImagePairToPairId(0, 1);
  const image_pair_t pair_id02 = Database::ImagePairToPairId(0, 2);
  const image_pair_t pair_id12 = Database::ImagePairToPairId(1, 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().size(), 3);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().at(pair_id01), 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().at(pair_id02), 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().at(pair_id12), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(0, 0).size(), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 0).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 0).at(1).image_id, 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 0).at(1).point2D_idx, 0);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(1, 0).size(), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 0).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 0).at(1).image_id, 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 0).at(1).point2D_idx, 0);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(2, 0).size(), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(2, 0).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(2, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(2, 0).at(1).image_id, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(2, 0).at(1).point2D_idx, 0);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(1, 5).size(), 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 5).at(0).image_id, 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 5).at(0).point2D_idx, 5);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(2, 5).size(), 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(2, 5).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(2, 5).at(0).point2D_idx, 5);
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(
        correspondence_graph.FindCorrespondences(0, i).size(),
        correspondence_graph.FindTransitiveCorrespondences(0, i, 1).size());
    BOOST_CHECK_EQUAL(
        correspondence_graph.FindCorrespondences(1, i).size(),
        correspondence_graph.FindTransitiveCorrespondences(1, i, 1).size());
    BOOST_CHECK_EQUAL(
        correspondence_graph.FindCorrespondences(2, i).size(),
        correspondence_graph.FindTransitiveCorrespondences(2, i, 1).size());
  }
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindTransitiveCorrespondences(0, 0, 2).size(), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindTransitiveCorrespondences(1, 0, 2).size(), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindTransitiveCorrespondences(2, 0, 2).size(), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindTransitiveCorrespondences(0, 0, 3).size(), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindTransitiveCorrespondences(1, 0, 3).size(), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindTransitiveCorrespondences(2, 0, 3).size(), 2);
  correspondence_graph.Finalize();
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 3);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 3);
  correspondence_graph.AddImage(3, 10);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(0), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(1), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(2), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(3), true);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 4);
  correspondence_graph.Finalize();
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(0), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(1), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(2), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(3), false);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 3);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 3);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 3);
}

BOOST_AUTO_TEST_CASE(TestOutOfBounds) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 4);
  FeatureMatches matches(3);
  matches[0].point2D_idx1 = 9;
  matches[0].point2D_idx2 = 3;
  matches[1].point2D_idx1 = 10;
  matches[1].point2D_idx2 = 3;
  matches[2].point2D_idx1 = 9;
  matches[2].point2D_idx2 = 4;
  correspondence_graph.AddCorrespondences(0, 1, matches);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 1);
  const image_pair_t pair_id = Database::ImagePairToPairId(0, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().at(pair_id), 1);
}

BOOST_AUTO_TEST_CASE(TestDuplicate) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 10);
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
  correspondence_graph.AddCorrespondences(0, 1, matches);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 3);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 3);
  const image_pair_t pair_id = Database::ImagePairToPairId(0, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().at(pair_id), 3);
}
