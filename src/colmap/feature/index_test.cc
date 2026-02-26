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

#include "colmap/feature/index.h"

#include "colmap/feature/utils.h"
#include "colmap/math/random.h"

#include <gtest/gtest.h>
#include <omp.h>

namespace colmap {
namespace {

FeatureDescriptorsFloat CreateRandomFeatureDescriptors(
    const size_t num_features) {
  SetPRNGSeed(0);
  FeatureDescriptorsFloat descriptors;
  descriptors.type = FeatureExtractorType::SIFT;
  descriptors.data.resize(num_features, 128);
  for (size_t i = 0; i < num_features; ++i) {
    for (size_t j = 0; j < 128; ++j) {
      descriptors.data(i, j) = std::pow(RandomUniformReal(0.0f, 1.0f), 2);
    }
  }
  L2NormalizeFeatureDescriptors(&descriptors.data);
  return descriptors;
}

class ParameterizedFeatureDescriptorIndexTests
    : public ::testing::TestWithParam<
          std::pair<FeatureDescriptorIndex::Type, /*num_descriptors=*/int>> {};

TEST_P(ParameterizedFeatureDescriptorIndexTests, Nominal) {
  const auto [type, num_descriptors] = GetParam();

  std::unique_ptr<FeatureDescriptorIndex> index;
  try {
    index = FeatureDescriptorIndex::Create(type);
  } catch (const std::runtime_error& e) {
    GTEST_SKIP() << "Skipping test due to: " << e.what();
  }

  EXPECT_NE(index, nullptr);

  const FeatureDescriptorsFloat index_descriptors =
      CreateRandomFeatureDescriptors(num_descriptors);
  const FeatureDescriptorsFloat& query_descriptors = index_descriptors;
  index->Build(index_descriptors);

  Eigen::RowMajorMatrixXi indices;
  Eigen::RowMajorMatrixXf distances;
  index->Search(/*num_neighbors=*/1, query_descriptors, indices, distances);

  ASSERT_EQ(indices.rows(), query_descriptors.data.rows());
  ASSERT_EQ(indices.cols(), 1);
  ASSERT_EQ(distances.rows(), query_descriptors.data.rows());
  ASSERT_EQ(distances.cols(), 1);

  for (int i = 0; i < query_descriptors.data.rows(); ++i) {
    EXPECT_EQ(indices(i, 0), i);
    EXPECT_NEAR(distances(i, 0), 0, 1e-6);
  }

  index->Search(/*num_neighbors=*/2, query_descriptors, indices, distances);
  ASSERT_EQ(indices.rows(), query_descriptors.data.rows());
  ASSERT_EQ(indices.cols(), 2);
  ASSERT_EQ(distances.rows(), query_descriptors.data.rows());
  ASSERT_EQ(distances.cols(), 2);

  for (int i = 0; i < query_descriptors.data.rows(); ++i) {
    EXPECT_EQ(indices(i, 0), i);
    EXPECT_NEAR(distances(i, 0), 0, 1e-6);
    EXPECT_NE(indices(i, 1), i);
    EXPECT_NEAR(distances(i, 1),
                (query_descriptors.data.row(i) -
                 index_descriptors.data.row(indices(i, 1)))
                    .squaredNorm(),
                1e-6);
  }

  index->Search(/*num_neighbors=*/index_descriptors.data.rows() + 1,
                query_descriptors,
                indices,
                distances);
  ASSERT_EQ(indices.rows(), query_descriptors.data.rows());
  ASSERT_EQ(indices.cols(), index_descriptors.data.rows());
  ASSERT_EQ(distances.rows(), query_descriptors.data.rows());
  ASSERT_EQ(distances.cols(), index_descriptors.data.rows());
}

TEST(FeatureDescriptorIndexTests, TypeMismatch) {
  constexpr int kNumDescriptors = 100;

  std::unique_ptr<FeatureDescriptorIndex> index;
  try {
    index = FeatureDescriptorIndex::Create(FeatureDescriptorIndex::Type::FAISS);
  } catch (const std::runtime_error& e) {
    GTEST_SKIP() << "Skipping test due to: " << e.what();
  }
  ASSERT_NE(index, nullptr);

  // Prepare SIFT descriptors for index build.
  FeatureDescriptorsFloat sift_desc =
      CreateRandomFeatureDescriptors(kNumDescriptors);

  // Prepare descriptors with a different type.
  FeatureDescriptorsFloat undefined_desc =
      CreateRandomFeatureDescriptors(kNumDescriptors);
  undefined_desc.type = FeatureExtractorType::UNDEFINED;

  // Build should throw when descriptor types are inconsistent.
  index->Build(undefined_desc);

  // Build correctly with SIFT so we can test Query mismatch.
  index->Build(sift_desc);

  // Query should throw when descriptor types are inconsistent.
  Eigen::RowMajorMatrixXi indices;
  Eigen::RowMajorMatrixXf distances;
  EXPECT_THROW(index->Search(
                   /*num_neighbors=*/1, undefined_desc, indices, distances),
               std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    FeatureDescriptorIndexTests,
    ParameterizedFeatureDescriptorIndexTests,
    ::testing::Values(std::make_pair(FeatureDescriptorIndex::Type::FAISS, 100),
                      std::make_pair(FeatureDescriptorIndex::Type::FAISS,
                                     1000)));

}  // namespace
}  // namespace colmap
