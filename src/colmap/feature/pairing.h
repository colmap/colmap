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

#include "colmap/feature/matcher.h"
#include "colmap/retrieval/visual_index.h"
#include "colmap/scene/database.h"
#include "colmap/util/threading.h"
#include "colmap/util/types.h"

namespace colmap {

struct ExhaustiveMatchingOptions {
  // Block size, i.e. number of images to simultaneously load into memory.
  int block_size = 50;

  bool Check() const;

  inline size_t CacheSize() const { return block_size; }
};

struct VocabTreeMatchingOptions {
  // Number of images to retrieve for each query image.
  int num_images = 100;

  // Number of nearest neighbors to retrieve per query feature.
  int num_nearest_neighbors = 5;

  // Number of nearest-neighbor checks to use in retrieval.
  int num_checks = 256;

  // How many images to return after spatial verification. Set to 0 to turn off
  // spatial verification.
  int num_images_after_verification = 0;

  // The maximum number of features to use for indexing an image. If an
  // image has more features, only the largest-scale features will be indexed.
  int max_num_features = -1;

  // Path to the vocabulary tree.
  std::string vocab_tree_path = "";

  // Optional path to file with specific image names to match.
  std::string match_list_path = "";

  // Number of threads for indexing and retrieval.
  int num_threads = -1;

  bool Check() const;

  inline size_t CacheSize() const { return 5 * num_images; }
};

struct SequentialMatchingOptions {
  // Number of overlapping image pairs.
  int overlap = 10;

  // Whether to match images against their quadratic neighbors.
  bool quadratic_overlap = true;

  // Whether to enable vocabulary tree based loop detection.
  bool loop_detection = false;

  // Loop detection is invoked every `loop_detection_period` images.
  int loop_detection_period = 10;

  // The number of images to retrieve in loop detection. This number should
  // be significantly bigger than the sequential matching overlap.
  int loop_detection_num_images = 50;

  // Number of nearest neighbors to retrieve per query feature.
  int loop_detection_num_nearest_neighbors = 1;

  // Number of nearest-neighbor checks to use in retrieval.
  int loop_detection_num_checks = 256;

  // How many images to return after spatial verification. Set to 0 to turn off
  // spatial verification.
  int loop_detection_num_images_after_verification = 0;

  // The maximum number of features to use for indexing an image. If an
  // image has more features, only the largest-scale features will be indexed.
  int loop_detection_max_num_features = -1;

  // Path to the vocabulary tree.
  std::string vocab_tree_path = "";

  bool Check() const;

  VocabTreeMatchingOptions VocabTreeOptions() const;

  inline size_t CacheSize() const {
    return std::max(5 * loop_detection_num_images, 5 * overlap);
  }
};

struct SpatialMatchingOptions {
  // Whether to ignore the Z-component of the location prior.
  bool ignore_z = true;

  // The maximum number of nearest neighbors to match.
  int max_num_neighbors = 50;

  // The maximum distance between the query and nearest neighbor. For GPS
  // coordinates the unit is Euclidean distance in meters.
  double max_distance = 100;

  // Number of threads for indexing and retrieval.
  int num_threads = -1;

  bool Check() const;

  inline size_t CacheSize() const { return 5 * max_num_neighbors; }
};

struct TransitiveMatchingOptions {
  // The maximum number of image pairs to process in one batch.
  int batch_size = 1000;

  // The number of transitive closure iterations.
  int num_iterations = 3;

  bool Check() const;

  inline size_t CacheSize() const { return 2 * batch_size; }
};

struct ImagePairsMatchingOptions {
  // Number of image pairs to match in one batch.
  int block_size = 1225;

  // Path to the file with the matches.
  std::string match_list_path = "";

  bool Check() const;

  inline size_t CacheSize() const { return block_size; }
};

struct FeaturePairsMatchingOptions {
  // Whether to geometrically verify the given matches.
  bool verify_matches = true;

  // Path to the file with the matches.
  std::string match_list_path = "";

  bool Check() const;
};

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

  ExhaustivePairGenerator(const ExhaustiveMatchingOptions& options,
                          const std::shared_ptr<FeatureMatcherCache>& cache);

  ExhaustivePairGenerator(const ExhaustiveMatchingOptions& options,
                          const std::shared_ptr<Database>& database);

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  const ExhaustiveMatchingOptions options_;
  const std::vector<image_t> image_ids_;
  const size_t block_size_;
  const size_t num_blocks_;
  size_t start_idx1_ = 0;
  size_t start_idx2_ = 0;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
};

class VocabTreePairGenerator : public PairGenerator {
 public:
  using PairOptions = VocabTreeMatchingOptions;

  VocabTreePairGenerator(const VocabTreeMatchingOptions& options,
                         const std::shared_ptr<FeatureMatcherCache>& cache,
                         const std::vector<image_t>& query_image_ids = {});

  VocabTreePairGenerator(const VocabTreeMatchingOptions& options,
                         const std::shared_ptr<Database>& database,
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
  retrieval::VisualIndex<>::QueryOptions query_options_;
  std::vector<image_t> query_image_ids_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  size_t query_idx_ = 0;
  size_t result_idx_ = 0;
};

class SequentialPairGenerator : public PairGenerator {
 public:
  using PairOptions = SequentialMatchingOptions;

  SequentialPairGenerator(const SequentialMatchingOptions& options,
                          const std::shared_ptr<FeatureMatcherCache>& cache);

  SequentialPairGenerator(const SequentialMatchingOptions& options,
                          const std::shared_ptr<Database>& database);

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
  size_t image_idx_ = 0;
};

class SpatialPairGenerator : public PairGenerator {
 public:
  using PairOptions = SpatialMatchingOptions;

  SpatialPairGenerator(const SpatialMatchingOptions& options,
                       const std::shared_ptr<FeatureMatcherCache>& cache);

  SpatialPairGenerator(const SpatialMatchingOptions& options,
                       const std::shared_ptr<Database>& database);

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>
  ReadPositionPriorData(FeatureMatcherCache& cache);

  const SpatialMatchingOptions options_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      index_matrix_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      distance_matrix_;
  std::vector<image_t> image_ids_;
  std::vector<size_t> position_idxs_;
  size_t current_idx_ = 0;
  int knn_ = 0;
};

class TransitivePairGenerator : public PairGenerator {
 public:
  using PairOptions = TransitiveMatchingOptions;

  TransitivePairGenerator(const TransitiveMatchingOptions& options,
                          const std::shared_ptr<FeatureMatcherCache>& cache);

  TransitivePairGenerator(const TransitiveMatchingOptions& options,
                          const std::shared_ptr<Database>& database);

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  const TransitiveMatchingOptions options_;
  const std::shared_ptr<FeatureMatcherCache> cache_;
  int current_iteration_ = 0;
  int current_batch_idx_ = 0;
  int current_num_batches_ = 0;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  std::unordered_set<image_pair_t> image_pair_ids_;
};

class ImportedPairGenerator : public PairGenerator {
 public:
  using PairOptions = ImagePairsMatchingOptions;

  ImportedPairGenerator(const ImagePairsMatchingOptions& options,
                        const std::shared_ptr<FeatureMatcherCache>& cache);

  ImportedPairGenerator(const ImagePairsMatchingOptions& options,
                        const std::shared_ptr<Database>& database);

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  const ImagePairsMatchingOptions options_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  std::vector<std::pair<image_t, image_t>> block_image_pairs_;
  size_t pair_idx_ = 0;
};

}  // namespace colmap
