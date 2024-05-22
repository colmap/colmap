// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#pragma once

#include "colmap/controllers/feature_matching.h"
#include "colmap/controllers/feature_matching_utils.h"
#include "colmap/retrieval/visual_index.h"
#include "colmap/scene/database.h"
#include "colmap/util/types.h"

namespace colmap {

class PairGenerator {
 public:
  virtual ~PairGenerator() = default;

  virtual void Reset() = 0;

  virtual bool HasFinished() const = 0;

  virtual std::vector<std::pair<image_t, image_t>> Next() = 0;

  std::vector<std::pair<image_t, image_t>> AllPairs();
};

class ExhaustivePairGenerator : public PairGenerator {
 public:
  using PairOptions = ExhaustiveMatchingOptions;
  static size_t CacheSize(const ExhaustiveMatchingOptions& options) {
    return 5 * options.block_size;
  }

  ExhaustivePairGenerator(const ExhaustiveMatchingOptions& options,
                          const std::shared_ptr<Database>& database);

  ExhaustivePairGenerator(const ExhaustiveMatchingOptions& options,
                          const std::shared_ptr<FeatureMatcherCache>& cache);

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  const ExhaustiveMatchingOptions options_;
  const std::vector<image_t> image_ids_;
  const size_t block_size_;
  const size_t num_blocks_;
  size_t start_idx1_;
  size_t start_idx2_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
};

class VocabTreePairGenerator : public PairGenerator {
 public:
  using PairOptions = VocabTreeMatchingOptions;
  static size_t CacheSize(const VocabTreeMatchingOptions& options) {
    return 5 * options.num_images;
  }

  VocabTreePairGenerator(const VocabTreeMatchingOptions& options,
                         std::shared_ptr<FeatureMatcherCache> cache,
                         const std::vector<image_t>& query_image_ids = {});

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  void IndexImages(const std::vector<image_t>& image_ids);

  struct Retrieval {
    image_t image_id = kInvalidImageId;
    std::vector<retrieval::ImageScore> image_scores;
  };

  void Query(image_t image_id);

  const VocabTreeMatchingOptions options_;
  const std::shared_ptr<FeatureMatcherCache> cache_;
  ThreadPool thread_pool;
  JobQueue<Retrieval> queue;
  retrieval::VisualIndex<> visual_index_;
  std::vector<image_t> query_image_ids_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  size_t query_idx_;
  size_t result_idx_;
  retrieval::VisualIndex<>::QueryOptions query_options_;
};

class SequentialPairGenerator : public PairGenerator {
 public:
  using PairOptions = SequentialMatchingOptions;
  static size_t CacheSize(const SequentialMatchingOptions& options) {
    return std::max(5 * options.loop_detection_num_images, 5 * options.overlap);
  }

  SequentialPairGenerator(const SequentialMatchingOptions& options,
                          std::shared_ptr<FeatureMatcherCache> cache);

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  std::vector<image_t> GetOrderedImageIds() const;

  const SequentialMatchingOptions options_;
  const std::shared_ptr<FeatureMatcherCache> cache_;
  std::vector<image_t> image_ids_;
  std::unique_ptr<VocabTreePairGenerator> vocab_tree_pair_generator_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  size_t image_idx_;
};

class SpatialPairGenerator : public PairGenerator {
 public:
  using PairOptions = SpatialMatchingOptions;
  static size_t CacheSize(const SpatialMatchingOptions& options) {
    return 5 * options.max_num_neighbors;
  }

  SpatialPairGenerator(const SpatialMatchingOptions& options,
                       const std::shared_ptr<FeatureMatcherCache>& cache);

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> ReadLocationData(
      const FeatureMatcherCache& cache);

  const SpatialMatchingOptions options_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      index_matrix_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      distance_matrix_;
  std::vector<image_t> image_ids_;
  std::vector<size_t> location_idxs_;
  size_t current_idx_;
};

class ImportedPairGenerator : public PairGenerator {
 public:
  using PairOptions = ImagePairsMatchingOptions;
  static size_t CacheSize(const ImagePairsMatchingOptions& options) {
    return options.block_size;
  }

  ImportedPairGenerator(const ImagePairsMatchingOptions& options,
                        const std::shared_ptr<FeatureMatcherCache>& cache);

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  const ImagePairsMatchingOptions options_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  std::vector<std::pair<image_t, image_t>> block_image_pairs_;
  size_t pair_idx_;
};

}  // namespace colmap
