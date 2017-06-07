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

#define TEST_NAME "retrieval/visual_index"
#include "util/testing.h"

#include "retrieval/visual_index.h"

using namespace colmap::retrieval;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  VisualIndex visual_index;
  BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 0);
}

BOOST_AUTO_TEST_CASE(TestBuild) {
  VisualIndex::DescType descriptors = VisualIndex::DescType::Random(1000, 128);
  VisualIndex visual_index;
  BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 0);
  VisualIndex::BuildOptions build_options;
  build_options.num_visual_words = 10;
  build_options.branching = 10;
  visual_index.Build(build_options, descriptors);
  BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 10);
}

BOOST_AUTO_TEST_CASE(TestQuery) {
  VisualIndex::DescType descriptors = VisualIndex::DescType::Random(1000, 128);
  VisualIndex visual_index;
  BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 0);
  VisualIndex::BuildOptions build_options;
  build_options.num_visual_words = 10;
  build_options.branching = 10;
  visual_index.Build(build_options, descriptors);
  BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 10);

  VisualIndex::IndexOptions index_options;
  VisualIndex::GeomType keypoints1(100);
  VisualIndex::DescType descriptors1 = VisualIndex::DescType::Random(100, 128);
  visual_index.Add(index_options, 1, keypoints1, descriptors1);
  VisualIndex::GeomType keypoints2(100);
  VisualIndex::DescType descriptors2 = VisualIndex::DescType::Random(100, 128);
  visual_index.Add(index_options, 2, keypoints2, descriptors2);
  visual_index.Prepare();

  VisualIndex::QueryOptions query_options;
  std::vector<ImageScore> image_scores;
  visual_index.Query(query_options, descriptors1, &image_scores);
  BOOST_CHECK_EQUAL(image_scores.size(), 2);
  BOOST_CHECK_EQUAL(image_scores[0].image_id, 1);
  BOOST_CHECK_EQUAL(image_scores[1].image_id, 2);
  BOOST_CHECK_GT(image_scores[0].score, image_scores[1].score);

  query_options.max_num_images = 1;
  visual_index.Query(query_options, descriptors1, &image_scores);
  BOOST_CHECK_EQUAL(image_scores.size(), 1);
  BOOST_CHECK_EQUAL(image_scores[0].image_id, 1);

  query_options.max_num_images = 3;
  visual_index.Query(query_options, descriptors1, &image_scores);
  BOOST_CHECK_EQUAL(image_scores.size(), 2);
  BOOST_CHECK_EQUAL(image_scores[0].image_id, 1);
  BOOST_CHECK_EQUAL(image_scores[1].image_id, 2);
  BOOST_CHECK_GT(image_scores[0].score, image_scores[1].score);
}
