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

#pragma once

#include "colmap/controllers/matcher_cache.h"
#include "colmap/retrieval/global_descriptor_index.h"
#include "colmap/retrieval/global_descriptor_model.h"
#include "colmap/retrieval/visual_index.h"
#include "colmap/scene/database.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/threading.h"
#include "colmap/util/types.h"

#include <filesystem>

namespace colmap {

struct GlobalDescriptorPairingOptions;

struct ExhaustivePairingOptions {
  // Block size, i.e. number of images to simultaneously load into memory.
  int block_size = 50;

  bool Check() const;

  // Each block matches two sets of images with size block_size. To hold all
  // images in the block, the cache thus needs to hold 2 * block_size.
  inline size_t CacheSize() const { return 2 * block_size; }
};

struct VocabTreePairingOptions {
  // Number of images to retrieve for each query image.
  int num_images = 100;

  // Number of nearest neighbors to retrieve per query feature.
  int num_nearest_neighbors = 5;

  // Number of nearest-neighbor checks to use in retrieval.
  int num_checks = 64;

  // How many images to return after spatial verification. Set to 0 to turn off
  // spatial verification.
  int num_images_after_verification = 0;

  // The maximum number of features to use for indexing an image. If an
  // image has more features, only the largest-scale features will be indexed.
  int max_num_features = -1;

  // Path to the vocabulary tree.
  std::filesystem::path vocab_tree_path;

  // Optional path to file with specific image names to match.
  std::filesystem::path match_list_path = "";

  // Number of threads for indexing and retrieval.
  int num_threads = -1;

  bool Check() const;

  inline size_t CacheSize() const { return 5 * num_images; }
};

struct SequentialPairingOptions {
  // Number of overlapping image pairs.
  int overlap = 10;

  // Whether to match images against their quadratic neighbors.
  bool quadratic_overlap = true;

  // Whether to match an image against all images within the same rig frame
  // and all images in neighboring rig frames. Note that this assumes that
  // images are appropriate named according to the following scheme:
  //
  //    rig1/
  //      camera1/
  //        image0001.jpg
  //        image0002.jpg
  //        image0003.jpg
  //        ...
  //      camera2/
  //        image0001.jpg
  //        image0002.jpg
  //        image0003.jpg
  //        ...
  //      camera3/
  //        image0001.jpg
  //        image0002.jpg
  //        image0003.jpg
  //        ...
  //      ...
  //
  // where, for overlap=1, rig1/camera1/image0001.jpg will be matched against:
  //
  //    rig1/camera2/image0001.jpg  # same frame
  //    rig1/camera3/image0001.jpg  # same frame
  //    rig1/camera1/image0002.jpg  # neighboring frame
  //    rig1/camera2/image0002.jpg  # neighboring frame
  //    rig1/camera3/image0002.jpg  # neighboring frame
  //
  // If no rigs/frames are configured in the database, this option is ignored.
  bool expand_rig_images = true;

  // Whether to enable vocabulary tree based loop detection.
  bool loop_detection = false;

  // The frequency at which loop detection is triggered, in number of images.
  int loop_detection_period = 10;

  // The number of images to retrieve in loop detection. This number should
  // be significantly larger than the sequential matching overlap.
  int loop_detection_num_images = 50;

  // Number of nearest neighbors to retrieve per query feature.
  int loop_detection_num_nearest_neighbors = 1;

  // Number of nearest-neighbor checks to use in retrieval.
  int loop_detection_num_checks = 64;

  // How many images to return after spatial verification. Set to 0 to turn off
  // spatial verification.
  int loop_detection_num_images_after_verification = 0;

  // The maximum number of features to use for indexing an image. If an
  // image has more features, only the largest-scale features will be indexed.
  int loop_detection_max_num_features = -1;

  // Number of threads for loop detection indexing and retrieval.
  int num_threads = -1;

  // Path to the vocabulary tree (used when loop_detection_model_path is empty).
  std::filesystem::path vocab_tree_path;

  // Global descriptor model type for loop detection (e.g. "MixVPR", "MegaLoc").
  std::string loop_detection_model_type;

  // Path to a global descriptor ONNX model for loop detection.
  // When empty, auto-downloads from the model type's default URI.
  std::filesystem::path loop_detection_model_path;

  // Path to the image directory, required for global-descriptor loop detection.
  std::filesystem::path loop_detection_image_path;

  bool Check() const;

  VocabTreePairingOptions VocabTreeOptions() const;
  // Returns options suitable for GlobalDescriptorPairGenerator.
  // Defined in pairing.cc to avoid circular dependency with the full
  // GlobalDescriptorPairingOptions definition.
  GlobalDescriptorPairingOptions GlobalDescriptorOptions() const;

  inline size_t CacheSize() const {
    return std::max(5 * loop_detection_num_images, 5 * overlap);
  }
};

struct SpatialPairingOptions {
  // Whether to ignore the Z-component of the location prior.
  bool ignore_z = true;

  // The maximum number of nearest neighbors to match.
  int max_num_neighbors = 50;

  // The minimum number of nearest neighbors to match. Neighbors include those
  // within max_distance or to satisfy min_num_neighbors.
  int min_num_neighbors = 0;

  // The maximum distance between the query and nearest neighbor. For GPS
  // coordinates the unit is Euclidean distance in meters.
  double max_distance = 100;

  // Number of threads for indexing and retrieval.
  int num_threads = -1;

  bool Check() const;

  inline size_t CacheSize() const { return 5 * max_num_neighbors; }
};

struct TransitivePairingOptions {
  // The maximum number of image pairs to process in one batch.
  int batch_size = 1000;

  // The number of transitive closure iterations.
  int num_iterations = 3;

  bool Check() const;

  inline size_t CacheSize() const { return 2 * batch_size; }
};

struct ImportedPairingOptions {
  // Number of image pairs to match in one batch.
  int block_size = 1225;

  // Path to the file with the matches.
  std::filesystem::path match_list_path = "";

  bool Check() const;

  inline size_t CacheSize() const { return block_size; }
};

struct FeaturePairsMatchingOptions {
  // Whether to geometrically verify the given matches.
  bool verify_matches = true;

  // Path to the file with the matches.
  std::filesystem::path match_list_path = "";

  bool Check() const;
};

struct ExistingMatchedPairingOptions {
  // The number of image pairs to match in one batch.
  int batch_size = 1000;

  bool Check() const;

  inline size_t CacheSize() const {
    return std::max<size_t>(10, static_cast<size_t>(2 * std::sqrt(batch_size)));
  }
};

struct GlobalDescriptorPairingOptions {
  // Number of most similar images to retrieve for each query image.
  int num_images = 100;

  // Global descriptor model type (e.g. "MixVPR", "MegaLoc").
  // If empty, defaults to "MixVPR".
  std::string model_type = "MixVPR";

  // Path to the global descriptor ONNX model file.
  // If empty, auto-downloads from the model's default URI.
  std::filesystem::path model_path;

  // Path to the directory containing the images.
  std::filesystem::path image_path;

  // Optional path to file with specific image names to use as queries.
  // If empty, all images are used as queries.
  std::filesystem::path match_list_path = "";

  // Number of threads for ONNX inference.
  int num_threads = -1;

  // Whether to use GPU / CoreML for ONNX inference (default true on macOS
  // for CoreML acceleration; falls back to CPU if unavailable).
  bool use_gpu = true;

  // GPU index for ONNX inference.
  std::string gpu_index = "-1";

  // Batch size for ONNX inference.
  int batch_size = 16;

  bool Check() const;

  inline size_t CacheSize() const { return 5 * num_images; }
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
  using PairingOptions = ExhaustivePairingOptions;

  ExhaustivePairGenerator(const ExhaustivePairingOptions& options,
                          const std::shared_ptr<FeatureMatcherCache>& cache);

  ExhaustivePairGenerator(const ExhaustivePairingOptions& options,
                          const std::shared_ptr<Database>& database);

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  const ExhaustivePairingOptions options_;
  const std::vector<image_t> image_ids_;
  const size_t block_size_;
  const size_t num_blocks_;
  size_t start_idx1_ = 0;
  size_t start_idx2_ = 0;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
};

class VocabTreePairGenerator : public PairGenerator {
 public:
  using PairingOptions = VocabTreePairingOptions;

  VocabTreePairGenerator(const VocabTreePairingOptions& options,
                         const std::shared_ptr<FeatureMatcherCache>& cache,
                         const std::vector<image_t>& query_image_ids = {});

  VocabTreePairGenerator(const VocabTreePairingOptions& options,
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

  const VocabTreePairingOptions options_;
  const std::shared_ptr<FeatureMatcherCache> cache_;
  ThreadPool thread_pool_;
  JobQueue<Retrieval> queue_;
  std::unique_ptr<retrieval::VisualIndex> visual_index_;
  retrieval::VisualIndex::QueryOptions query_options_;
  std::vector<image_t> query_image_ids_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  size_t query_idx_ = 0;
  size_t result_idx_ = 0;
};

class SequentialPairGenerator : public PairGenerator {
 public:
  using PairingOptions = SequentialPairingOptions;

  SequentialPairGenerator(const SequentialPairingOptions& options,
                          const std::shared_ptr<FeatureMatcherCache>& cache);

  SequentialPairGenerator(const SequentialPairingOptions& options,
                          const std::shared_ptr<Database>& database);

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  std::vector<image_t> GetOrderedImageIds() const;

  const SequentialPairingOptions options_;
  const std::shared_ptr<FeatureMatcherCache> cache_;
  std::vector<image_t> image_ids_;
  // Optional mapping from frames to images and vice versa.
  NodeHashMap<frame_t, std::vector<image_t>> frame_to_image_ids_;
  NodeHashMap<image_t, frame_t> image_to_frame_ids_;
  std::unique_ptr<PairGenerator> loop_detection_pair_generator_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  size_t image_idx_ = 0;
};

class SpatialPairGenerator : public PairGenerator {
 public:
  using PairingOptions = SpatialPairingOptions;

  SpatialPairGenerator(const SpatialPairingOptions& options,
                       const std::shared_ptr<FeatureMatcherCache>& cache);

  SpatialPairGenerator(const SpatialPairingOptions& options,
                       const std::shared_ptr<Database>& database);

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

  Eigen::RowMajorMatrixXf ReadPositionPriorData(FeatureMatcherCache& cache);

 private:
  const SpatialPairingOptions options_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      index_matrix_;
  Eigen::RowMajorMatrixXf distance_squared_matrix_;
  std::vector<image_t> image_ids_;
  std::vector<size_t> position_idxs_;
  size_t current_idx_ = 0;
  int knn_ = 0;
};

class TransitivePairGenerator : public PairGenerator {
 public:
  using PairingOptions = TransitivePairingOptions;

  TransitivePairGenerator(const TransitivePairingOptions& options,
                          const std::shared_ptr<FeatureMatcherCache>& cache);

  TransitivePairGenerator(const TransitivePairingOptions& options,
                          const std::shared_ptr<Database>& database);

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  const TransitivePairingOptions options_;
  const std::shared_ptr<FeatureMatcherCache> cache_;
  int current_iteration_ = 0;
  int current_batch_idx_ = 0;
  int current_num_batches_ = 0;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  FlatHashSet<image_pair_t> image_pair_ids_;
};

class ImportedPairGenerator : public PairGenerator {
 public:
  using PairingOptions = ImportedPairingOptions;

  ImportedPairGenerator(const ImportedPairingOptions& options,
                        const std::shared_ptr<FeatureMatcherCache>& cache);

  ImportedPairGenerator(const ImportedPairingOptions& options,
                        const std::shared_ptr<Database>& database);

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  const ImportedPairingOptions options_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  std::vector<std::pair<image_t, image_t>> block_image_pairs_;
  size_t pair_idx_ = 0;
};

class ExistingMatchedPairGenerator : public PairGenerator {
 public:
  using PairingOptions = ExistingMatchedPairingOptions;

  ExistingMatchedPairGenerator(
      const ExistingMatchedPairingOptions& options,
      const std::shared_ptr<FeatureMatcherCache>& cache);

  ExistingMatchedPairGenerator(const ExistingMatchedPairingOptions& options,
                               const std::shared_ptr<Database>& database);

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  const ExistingMatchedPairingOptions options_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  size_t start_idx_ = 0;
  size_t num_batches_ = 0;
};

class GlobalDescriptorPairGenerator : public PairGenerator {
 public:
  using PairingOptions = GlobalDescriptorPairingOptions;

  GlobalDescriptorPairGenerator(
      const GlobalDescriptorPairingOptions& options,
      const std::shared_ptr<FeatureMatcherCache>& cache,
      const std::vector<image_t>& query_image_ids = {});

  GlobalDescriptorPairGenerator(
      const GlobalDescriptorPairingOptions& options,
      const std::shared_ptr<Database>& database,
      const std::vector<image_t>& query_image_ids = {});

  void Reset() override;

  bool HasFinished() const override;

  std::vector<std::pair<image_t, image_t>> Next() override;

 private:
  // Preprocess a bitmap for the given model. Returns serialized NCHW float data.
  static std::vector<float> PreprocessImage(
      const Bitmap& bitmap,
      const retrieval::GlobalDescriptorModel& model);

  // Compute global descriptors for all images using batched ONNX inference
  // and add them to the global descriptor index.
  void ComputeAndIndexDescriptors();

  const GlobalDescriptorPairingOptions options_;
  const std::shared_ptr<FeatureMatcherCache> cache_;
  std::vector<image_t> query_image_ids_;
  retrieval::GlobalDescriptorIndex global_descriptor_index_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  size_t pair_idx_ = 0;
};

}  // namespace colmap
