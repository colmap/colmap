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

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void TestVocabTreeType() {
  typedef VisualIndex<kDescType, kDescDim, kEmbeddingDim> VisualIndexType;

  {
    VisualIndexType visual_index;
    BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 0);
  }

  {
    typename VisualIndexType::DescType descriptors =
        VisualIndexType::DescType::Random(1000, kDescDim);
    VisualIndexType visual_index;
    BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 0);
    typename VisualIndexType::BuildOptions build_options;
    build_options.num_visual_words = 10;
    build_options.branching = 10;
    visual_index.Build(build_options, descriptors);
    BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 10);
  }

  {
    typename VisualIndexType::DescType descriptors =
        VisualIndexType::DescType::Random(1000, kDescDim);
    VisualIndexType visual_index;
    BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 0);
    typename VisualIndexType::BuildOptions build_options;
    build_options.num_visual_words = 10;
    build_options.branching = 10;
    visual_index.Build(build_options, descriptors);
    BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 10);

    typename VisualIndexType::IndexOptions index_options;
    typename VisualIndexType::GeomType keypoints1(100);
    typename VisualIndexType::DescType descriptors1 =
        VisualIndexType::DescType::Random(100, kDescDim);
    visual_index.Add(index_options, 1, keypoints1, descriptors1);
    typename VisualIndexType::GeomType keypoints2(100);
    typename VisualIndexType::DescType descriptors2 =
        VisualIndexType::DescType::Random(100, kDescDim);
    visual_index.Add(index_options, 2, keypoints2, descriptors2);
    visual_index.Prepare();

    typename VisualIndexType::QueryOptions query_options;
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
}

BOOST_AUTO_TEST_CASE(TestVocabTree) {
  TestVocabTreeType<uint8_t, 128, 64>();
  TestVocabTreeType<uint8_t, 64, 64>();
  TestVocabTreeType<uint8_t, 64, 32>();
  TestVocabTreeType<int, 128, 64>();
  TestVocabTreeType<int, 64, 64>();
  TestVocabTreeType<int, 64, 32>();
  TestVocabTreeType<float, 128, 64>();
  TestVocabTreeType<float, 64, 64>();
  TestVocabTreeType<float, 64, 32>();
  TestVocabTreeType<double, 128, 64>();
  TestVocabTreeType<double, 64, 64>();
  TestVocabTreeType<double, 64, 32>();
}
