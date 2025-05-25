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

#include "colmap/retrieval/visual_index.h"

#include "colmap/math/random.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

class ParameterizedVisualIndexTests
    : public ::testing::TestWithParam<
          std::pair</*embedding_dim=*/int, /*embedding_dim=*/int>> {};

TEST_P(ParameterizedVisualIndexTests, Nominal) {
  const auto [desc_dim, embedding_dim] = GetParam();

  SetPRNGSeed(0);

  {
    auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
    EXPECT_EQ(visual_index->NumVisualWords(), 0);
  }

  VisualIndex::BuildOptions build_options;
  // Keep test runtimes low.
  build_options.num_iterations = 10;
  build_options.num_rounds = 1;

  {
    VisualIndex::Descriptors descriptors =
        VisualIndex::Descriptors::Random(200, desc_dim);
    auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
    EXPECT_EQ(visual_index->NumVisualWords(), 0);
    EXPECT_EQ(visual_index->NumImages(), 0);
    EXPECT_EQ(visual_index->DescDim(), desc_dim);
    EXPECT_EQ(visual_index->EmbeddingDim(), embedding_dim);
    build_options.num_visual_words = 5;
    visual_index->Build(build_options, descriptors);
    EXPECT_EQ(visual_index->NumVisualWords(), 5);
  }

  {
    VisualIndex::Descriptors descriptors =
        VisualIndex::Descriptors::Random(4096, desc_dim);
    auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
    EXPECT_EQ(visual_index->NumVisualWords(), 0);
    EXPECT_EQ(visual_index->NumImages(), 0);
    build_options.num_visual_words = 512;
    visual_index->Build(build_options, descriptors);
    EXPECT_EQ(visual_index->NumVisualWords(), 512);
  }

  {
    VisualIndex::Descriptors descriptors =
        VisualIndex::Descriptors::Random(1000, desc_dim);
    auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
    EXPECT_EQ(visual_index->NumVisualWords(), 0);
    EXPECT_EQ(visual_index->NumImages(), 0);
    build_options.num_visual_words = 100;
    visual_index->Build(build_options, descriptors);
    EXPECT_EQ(visual_index->NumVisualWords(), 100);

    VisualIndex::IndexOptions index_options;
    VisualIndex::Geometries keypoints1(50);
    VisualIndex::Descriptors descriptors1 =
        VisualIndex::Descriptors::Random(50, desc_dim);
    visual_index->Add(index_options, 1, keypoints1, descriptors1);
    EXPECT_EQ(visual_index->NumImages(), 1);
    VisualIndex::Geometries keypoints2(50);
    VisualIndex::Descriptors descriptors2 =
        VisualIndex::Descriptors::Random(50, desc_dim);
    visual_index->Add(index_options, 2, keypoints2, descriptors2);
    EXPECT_EQ(visual_index->NumImages(), 2);
    visual_index->Prepare();

    VisualIndex::QueryOptions query_options;
    std::vector<ImageScore> image_scores;
    visual_index->Query(query_options, descriptors1, &image_scores);
    EXPECT_EQ(image_scores.size(), 2);
    EXPECT_EQ(image_scores[0].image_id, 1);
    EXPECT_EQ(image_scores[1].image_id, 2);
    EXPECT_GT(image_scores[0].score, image_scores[1].score);

    query_options.max_num_images = 1;
    visual_index->Query(query_options, descriptors1, &image_scores);
    EXPECT_EQ(image_scores.size(), 1);
    EXPECT_EQ(image_scores[0].image_id, 1);

    query_options.max_num_images = 3;
    visual_index->Query(query_options, descriptors1, &image_scores);
    EXPECT_EQ(image_scores.size(), 2);
    EXPECT_EQ(image_scores[0].image_id, 1);
    EXPECT_EQ(image_scores[1].image_id, 2);
    EXPECT_GT(image_scores[0].score, image_scores[1].score);
  }
}

TEST_P(ParameterizedVisualIndexTests, ReadWrite) {
  const auto [desc_dim, embedding_dim] = GetParam();
  const std::string test_dir = CreateTestDir();
  const std::string vocab_tree_path = test_dir + "/vocab_tree.bin";

  VisualIndex::BuildOptions build_options;
  // Keep test runtimes low.
  build_options.num_iterations = 10;
  build_options.num_rounds = 1;

  VisualIndex::Descriptors descriptors =
      VisualIndex::Descriptors::Random(200, desc_dim);
  auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
  EXPECT_EQ(visual_index->NumVisualWords(), 0);
  EXPECT_EQ(visual_index->DescDim(), desc_dim);
  EXPECT_EQ(visual_index->EmbeddingDim(), embedding_dim);
  build_options.num_visual_words = 5;
  visual_index->Build(build_options, descriptors);
  EXPECT_EQ(visual_index->NumVisualWords(), 5);

  VisualIndex::IndexOptions index_options;
  VisualIndex::Geometries keypoints1(50);
  VisualIndex::Descriptors descriptors1 =
      VisualIndex::Descriptors::Random(50, desc_dim);
  visual_index->Add(index_options, 1, keypoints1, descriptors1);
  VisualIndex::Geometries keypoints2(50);
  VisualIndex::Descriptors descriptors2 =
      VisualIndex::Descriptors::Random(50, desc_dim);
  visual_index->Add(index_options, 2, keypoints2, descriptors2);

  EXPECT_EQ(visual_index->NumImages(), 2);

  visual_index->Write(vocab_tree_path);
  auto read_visual_index = VisualIndex::Read(vocab_tree_path);
  EXPECT_EQ(visual_index->NumVisualWords(),
            read_visual_index->NumVisualWords());
  EXPECT_EQ(visual_index->NumImages(), read_visual_index->NumImages());
  EXPECT_EQ(visual_index->DescDim(), read_visual_index->DescDim());
  EXPECT_EQ(visual_index->EmbeddingDim(), read_visual_index->EmbeddingDim());
  EXPECT_TRUE(visual_index->IsImageIndexed(1));
  EXPECT_TRUE(visual_index->IsImageIndexed(2));
  EXPECT_FALSE(visual_index->IsImageIndexed(3));
}

INSTANTIATE_TEST_SUITE_P(VisualIndexTests,
                         ParameterizedVisualIndexTests,
                         ::testing::Values(std::make_pair(128, 64),
                                           std::make_pair(32, 8)));

#ifdef COLMAP_DOWNLOAD_ENABLED

TEST(VisualIndex, Download) {
  const std::string test_dir = CreateTestDir();
  const std::string vocab_tree_path = test_dir + "/server_vocab_tree.bin";
  OverwriteDownloadCacheDir(test_dir);

  VisualIndex::Descriptors descriptors =
      VisualIndex::Descriptors::Random(50, 32);
  auto visual_index = VisualIndex::Create(32, 8);
  VisualIndex::BuildOptions build_options;
  build_options.num_visual_words = 5;
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  visual_index->Build(build_options, descriptors);
  visual_index->Write(vocab_tree_path);

  std::vector<char> vocab_tree_data;
  ReadBinaryBlob(vocab_tree_path, &vocab_tree_data);
  const std::string vocab_tree_uri =
      "file://" + std::filesystem::absolute(vocab_tree_path).string() +
      ";vocab_tree.bin;" +
      ComputeSHA256({vocab_tree_data.data(), vocab_tree_data.size()});
  auto downloaded_visual_index = VisualIndex::Read(vocab_tree_uri);
  EXPECT_EQ(downloaded_visual_index->NumVisualWords(),
            visual_index->NumVisualWords());
}

#endif

}  // namespace
}  // namespace retrieval
}  // namespace colmap
