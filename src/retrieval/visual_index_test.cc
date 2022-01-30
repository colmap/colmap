// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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

#define TEST_NAME "retrieval/visual_index"
#include "util/testing.h"

#include "retrieval/visual_index.h"

using namespace colmap;
using namespace colmap::retrieval;

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void TestVocabTreeType() {
  typedef VisualIndex<kDescType, kDescDim, kEmbeddingDim> VisualIndexType;

  SetPRNGSeed(0);

  {
    VisualIndexType visual_index;
    BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 0);
  }

  {
    typename VisualIndexType::DescType descriptors =
        VisualIndexType::DescType::Random(50, kDescDim);
    VisualIndexType visual_index;
    BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 0);
    typename VisualIndexType::BuildOptions build_options;
    build_options.num_visual_words = 5;
    build_options.branching = 5;
    visual_index.Build(build_options, descriptors);
    BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 5);
  }

  {
    typename VisualIndexType::DescType descriptors =
        VisualIndexType::DescType::Random(1000, kDescDim);
    VisualIndexType visual_index;
    BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 0);
    typename VisualIndexType::BuildOptions build_options;
    build_options.num_visual_words = 100;
    build_options.branching = 10;
    visual_index.Build(build_options, descriptors);
    BOOST_CHECK_EQUAL(visual_index.NumVisualWords(), 100);

    typename VisualIndexType::IndexOptions index_options;
    typename VisualIndexType::GeomType keypoints1(50);
    typename VisualIndexType::DescType descriptors1 =
        VisualIndexType::DescType::Random(50, kDescDim);
    visual_index.Add(index_options, 1, keypoints1, descriptors1);
    typename VisualIndexType::GeomType keypoints2(50);
    typename VisualIndexType::DescType descriptors2 =
        VisualIndexType::DescType::Random(50, kDescDim);
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
  TestVocabTreeType<uint8_t, 32, 16>();
  TestVocabTreeType<int, 32, 16>();
  TestVocabTreeType<float, 32, 16>();
  TestVocabTreeType<double, 32, 16>();
}
