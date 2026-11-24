// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/retrieval/inverted_file.h"

#include "colmap/retrieval/inverted_index.h"

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

TEST(InvertedFile, RequiresEmbeddingAndSortedEntries) {
  InvertedFile<64> file;
  EXPECT_FALSE(file.IsUsable());
  Eigen::Matrix<float, Eigen::Dynamic, 64> training(2, 64);
  training.row(0).setConstant(-1);
  training.row(1).setConstant(1);
  file.ComputeHammingEmbedding(training);
  EXPECT_TRUE(file.HasHammingEmbedding());
  EXPECT_FALSE(file.EntriesSorted());
  EXPECT_FALSE(file.IsUsable());

  const Eigen::VectorXf descriptor = Eigen::VectorXf::Ones(64);
  file.AddEntry(7, 0, descriptor, {});
  file.ComputeIDFWeight(2);
  std::vector<ImageScore> scores;
  file.ScoreFeature(descriptor, &scores);
  EXPECT_TRUE(scores.empty());

  file.SortEntries();
  EXPECT_TRUE(file.IsUsable());
  file.ScoreFeature(descriptor, &scores);
  ASSERT_EQ(scores.size(), 1);
  EXPECT_EQ(scores[0].image_id, 7);
  EXPECT_GT(scores[0].score, 0);

  file.AddEntry(3, 0, descriptor, {});
  EXPECT_TRUE(file.HasHammingEmbedding());
  EXPECT_FALSE(file.EntriesSorted());
  EXPECT_FALSE(file.IsUsable());
  file.ScoreFeature(descriptor, &scores);
  EXPECT_TRUE(scores.empty());

  file.SortEntries();
  EXPECT_TRUE(file.IsUsable());
  file.ScoreFeature(descriptor, &scores);
  ASSERT_EQ(scores.size(), 2);
  EXPECT_EQ(scores[0].image_id, 3);
  EXPECT_EQ(scores[1].image_id, 7);

  file.ClearEntries();
  EXPECT_TRUE(file.HasHammingEmbedding());
  EXPECT_FALSE(file.EntriesSorted());
  EXPECT_FALSE(file.IsUsable());
  file.Reset();
  EXPECT_FALSE(file.HasHammingEmbedding());
  EXPECT_FALSE(file.IsUsable());
  file.AddEntry(7, 0, descriptor, {});
  file.SortEntries();
  file.ComputeIDFWeight(2);
  EXPECT_TRUE(file.EntriesSorted());
  EXPECT_FALSE(file.IsUsable());
  file.ScoreFeature(descriptor, &scores);
  EXPECT_TRUE(scores.empty());
}

TEST(InvertedFile, SkipWordsWithoutEnoughTrainingDescriptors) {
  using Index = InvertedIndex<float, 128, 64>;
  const Index::DescType descriptor = Index::DescType::Ones(1, 128);
  for (const int num_training_descriptors : {4, 5}) {
    SCOPED_TRACE(num_training_descriptors);
    Index index;
    index.Initialize(2);
    const Index::DescType training =
        Index::DescType::Ones(num_training_descriptors + 5, 128);
    Index::WordIds word_ids(num_training_descriptors + 5, 1);
    word_ids.topRows(5).setZero();
    word_ids.bottomRows(num_training_descriptors).setOnes();
    index.ComputeHammingEmbedding(training, word_ids);
    index.AddEntry(7, 1, 0, descriptor, {});
    index.AddEntry(9, 0, 0, descriptor, {});
    index.Finalize();

    std::vector<ImageScore> scores;
    index.Query(descriptor, Index::WordIds::Ones(1, 1), &scores);
    if (num_training_descriptors < 5) {
      EXPECT_TRUE(scores.empty());
    } else {
      ASSERT_EQ(scores.size(), 1);
      EXPECT_EQ(scores[0].image_id, 7);
      EXPECT_GT(scores[0].score, 0);
    }
    index.Query(descriptor, Index::WordIds::Zero(1, 1), &scores);
    ASSERT_EQ(scores.size(), 1);
    EXPECT_EQ(scores[0].image_id, 9);
    EXPECT_GT(scores[0].score, 0);
  }
}

}  // namespace
}  // namespace retrieval
}  // namespace colmap
