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

#include "colmap/controllers/feature_matching.h"

#include "colmap/controllers/feature_matching_utils.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/pairing.h"
#include "colmap/feature/utils.h"
#include "colmap/geometry/gps.h"
#include "colmap/retrieval/visual_index.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <fstream>
#include <numeric>

namespace colmap {
namespace {

void PrintElapsedTime(const Timer& timer) {
  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
}

template <typename DerivedPairOptions>
class PairGenerator {
 public:
  using PairOptions = DerivedPairOptions;

  virtual ~PairGenerator() = default;

  virtual void Reset() = 0;

  virtual bool HasFinished() = 0;

  virtual std::vector<std::pair<image_t, image_t>> Next() = 0;

  std::vector<std::pair<image_t, image_t>> AllPairs() {
    std::vector<std::pair<image_t, image_t>> image_pairs;
    while (!HasFinished()) {
      std::vector<std::pair<image_t, image_t>> image_pairs_block = Next();
      image_pairs.insert(image_pairs.end(),
                         std::make_move_iterator(image_pairs_block.begin()),
                         std::make_move_iterator(image_pairs_block.end()));
    }
    return image_pairs;
  };
};

template <typename DerivedPairGenerator>
class GenericFeatureMatcher : public Thread {
 public:
  GenericFeatureMatcher(
      const typename DerivedPairGenerator::PairOptions& pair_options,
      const SiftMatchingOptions& matching_options,
      const TwoViewGeometryOptions& geometry_options,
      const std::string& database_path)
      : pair_options_(pair_options),
        database_(new Database(database_path)),
        cache_(new FeatureMatcherCache(
            DerivedPairGenerator::CacheSize(pair_options_), database_.get())),
        matcher_(
            matching_options, geometry_options, database_.get(), cache_.get()) {
    THROW_CHECK(matching_options.Check());
    THROW_CHECK(geometry_options.Check());
  }

 private:
  void Run() override {
    PrintHeading1("Feature matching");
    Timer run_timer;
    run_timer.Start();

    if (!matcher_.Setup()) {
      return;
    }

    cache_->Setup();

    DerivedPairGenerator pair_generator(pair_options_, cache_);
    while (!pair_generator.HasFinished()) {
      if (IsStopped()) {
        run_timer.PrintMinutes();
        return;
      }
      Timer timer;
      timer.Start();
      std::vector<std::pair<image_t, image_t>> image_pairs =
          pair_generator.Next();
      DatabaseTransaction database_transaction(database_.get());
      matcher_.Match(image_pairs);
      PrintElapsedTime(timer);
    }
    run_timer.PrintMinutes();
  }

  const typename DerivedPairGenerator::PairOptions pair_options_;
  const std::shared_ptr<Database> database_;
  const std::shared_ptr<FeatureMatcherCache> cache_;
  FeatureMatcherController matcher_;
};

class ExhaustivePairGenerator
    : public PairGenerator<ExhaustiveMatchingOptions> {
 public:
  static size_t CacheSize(const ExhaustiveMatchingOptions& options) {
    return 5 * options.block_size;
  }

  ExhaustivePairGenerator(const ExhaustiveMatchingOptions& options,
                          const std::shared_ptr<Database>& database)
      : ExhaustivePairGenerator(
            options,
            std::make_shared<FeatureMatcherCache>(
                CacheSize(options), THROW_CHECK_NOTNULL(database).get())) {}

  ExhaustivePairGenerator(const ExhaustiveMatchingOptions& options,
                          const std::shared_ptr<FeatureMatcherCache>& cache)
      : options_(options),
        image_ids_(THROW_CHECK_NOTNULL(cache)->GetImageIds()),
        block_size_(static_cast<size_t>(options_.block_size)),
        num_blocks_(static_cast<size_t>(
            std::ceil(static_cast<double>(image_ids_.size()) / block_size_))) {
    THROW_CHECK(options.Check());
    const size_t num_pairs_per_block = block_size_ * (block_size_ - 1) / 2;
    image_pairs_.reserve(num_pairs_per_block);
    Reset();
  }

  void Reset() {
    start_idx1_ = 0;
    start_idx2_ = 0;
  }

  bool HasFinished() { return start_idx1_ >= image_ids_.size(); }

  virtual std::vector<std::pair<image_t, image_t>> Next() {
    image_pairs_.clear();
    if (HasFinished()) {
      return image_pairs_;
    }

    const size_t end_idx1 =
        std::min(image_ids_.size(), start_idx1_ + block_size_) - 1;
    const size_t end_idx2 =
        std::min(image_ids_.size(), start_idx2_ + block_size_) - 1;

    LOG(INFO) << StringPrintf("Matching block [%d/%d, %d/%d]",
                              start_idx1_ / block_size_ + 1,
                              num_blocks_,
                              start_idx2_ / block_size_ + 1,
                              num_blocks_)
              << std::flush;

    for (size_t idx1 = start_idx1_; idx1 <= end_idx1; ++idx1) {
      for (size_t idx2 = start_idx2_; idx2 <= end_idx2; ++idx2) {
        const size_t block_id1 = idx1 % block_size_;
        const size_t block_id2 = idx2 % block_size_;
        if ((idx1 > idx2 && block_id1 <= block_id2) ||
            (idx1 < idx2 && block_id1 < block_id2)) {  // Avoid duplicate pairs
          image_pairs_.emplace_back(image_ids_[idx1], image_ids_[idx2]);
        }
      }
    }
    start_idx2_ += block_size_;
    if (start_idx2_ >= image_ids_.size()) {
      start_idx2_ = 0;
      start_idx1_ += block_size_;
    }
    return image_pairs_;
  }

 private:
  const ExhaustiveMatchingOptions options_;
  const std::vector<image_t> image_ids_;
  const size_t block_size_;
  const size_t num_blocks_;
  size_t start_idx1_;
  size_t start_idx2_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
};

}  // namespace

bool ExhaustiveMatchingOptions::Check() const {
  CHECK_OPTION_GT(block_size, 1);
  return true;
}

std::unique_ptr<Thread> CreateExhaustiveFeatureMatcher(
    const ExhaustiveMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return std::make_unique<GenericFeatureMatcher<ExhaustivePairGenerator>>(
      options, matching_options, geometry_options, database_path);
}

namespace {

class VocabTreePairGenerator : public PairGenerator<VocabTreeMatchingOptions> {
 public:
  static size_t CacheSize(const VocabTreeMatchingOptions& options) {
    return 5 * options.num_images;
  }

  VocabTreePairGenerator(const VocabTreeMatchingOptions& options,
                         std::shared_ptr<FeatureMatcherCache> cache,
                         const std::vector<image_t>& query_image_ids = {})
      : options_(options),
        cache_(std::move(THROW_CHECK_NOTNULL(cache))),
        thread_pool(-1),  // TODO: fixme  matching_options.num_threads
        queue(-1) {
    THROW_CHECK(options.Check());

    // Read the pre-trained vocabulary tree from disk.
    visual_index_.Read(options_.vocab_tree_path);

    const std::vector<image_t> all_image_ids = cache_->GetImageIds();
    if (query_image_ids.size() > 0) {
      query_image_ids_ = query_image_ids;
    } else if (options_.match_list_path == "") {
      query_image_ids_ = cache_->GetImageIds();
    } else {
      // Map image names to image identifiers.
      std::unordered_map<std::string, image_t> image_name_to_image_id;
      image_name_to_image_id.reserve(all_image_ids.size());
      for (const auto image_id : all_image_ids) {
        const auto& image = cache_->GetImage(image_id);
        image_name_to_image_id.emplace(image.Name(), image_id);
      }

      // Read the match list path.
      std::ifstream file(options_.match_list_path);
      THROW_CHECK_FILE_OPEN(file, options_.match_list_path);
      std::string line;
      while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
          continue;
        }

        if (image_name_to_image_id.count(line) == 0) {
          LOG(ERROR) << "Image " << line << " does not exist.";
        } else {
          query_image_ids_.push_back(image_name_to_image_id.at(line));
        }
      }
    }

    IndexImages(all_image_ids);

    query_options_.max_num_images = options_.num_images;
    query_options_.num_neighbors = options_.num_nearest_neighbors;
    query_options_.num_checks = options_.num_checks;
    query_options_.num_images_after_verification =
        options_.num_images_after_verification;

    Reset();
  }

  void Reset() {
    query_idx_ = 0;
    result_idx_ = 0;
  }

  bool HasFinished() { return result_idx_ >= query_image_ids_.size(); }

  virtual std::vector<std::pair<image_t, image_t>> Next() {
    image_pairs_.clear();
    if (HasFinished()) {
      return image_pairs_;
    }
    if (query_idx_ == 0) {
      // Initially, make all retrieval threads busy and continue with the
      // matching.
      const size_t init_num_tasks =
          std::min(query_image_ids_.size(), 2 * thread_pool.NumThreads());
      for (; query_idx_ < init_num_tasks; ++query_idx_) {
        thread_pool.AddTask(
            &VocabTreePairGenerator::Query, this, query_image_ids_[query_idx_]);
      }
    }

    LOG(INFO) << StringPrintf("Matching image [%d/%d]",
                              result_idx_ + 1,
                              query_image_ids_.size())
              << std::flush;

    // Push the next image to the retrieval queue.
    if (query_idx_ < query_image_ids_.size()) {
      thread_pool.AddTask(
          &VocabTreePairGenerator::Query, this, query_image_ids_[query_idx_++]);
    }

    // Pop the next results from the retrieval queue.
    auto retrieval = queue.Pop();
    THROW_CHECK(retrieval.IsValid());

    const auto& image_id = retrieval.Data().image_id;
    const auto& image_scores = retrieval.Data().image_scores;

    // Compose the image pairs from the scores.
    image_pairs_.reserve(image_scores.size());
    for (const auto image_score : image_scores) {
      image_pairs_.emplace_back(image_id, image_score.image_id);
    }
    ++result_idx_;
    return image_pairs_;
  }

 private:
  void IndexImages(const std::vector<image_t>& image_ids) {
    retrieval::VisualIndex<>::IndexOptions index_options;
    index_options.num_threads =
        -1;  // TODO: fixme matching_options.num_threads;
    index_options.num_checks = options_.num_checks;

    for (size_t i = 0; i < image_ids.size(); ++i) {
      Timer timer;
      timer.Start();
      LOG(INFO) << StringPrintf(
                       "Indexing image [%d/%d]", i + 1, image_ids.size())
                << std::flush;
      auto keypoints = *cache_->GetKeypoints(image_ids[i]);
      auto descriptors = *cache_->GetDescriptors(image_ids[i]);
      if (options_.max_num_features > 0 &&
          descriptors.rows() > options_.max_num_features) {
        ExtractTopScaleFeatures(
            &keypoints, &descriptors, options_.max_num_features);
      }
      visual_index_.Add(index_options, image_ids[i], keypoints, descriptors);
      PrintElapsedTime(timer);
    }

    // Compute the TF-IDF weights, etc.
    visual_index_.Prepare();
  }

  struct Retrieval {
    image_t image_id = kInvalidImageId;
    std::vector<retrieval::ImageScore> image_scores;
  };

  void Query(const image_t image_id) {
    auto keypoints = *cache_->GetKeypoints(image_id);
    auto descriptors = *cache_->GetDescriptors(image_id);
    if (options_.max_num_features > 0 &&
        descriptors.rows() > options_.max_num_features) {
      ExtractTopScaleFeatures(
          &keypoints, &descriptors, options_.max_num_features);
    }

    Retrieval retrieval;
    retrieval.image_id = image_id;
    visual_index_.Query(
        query_options_, keypoints, descriptors, &retrieval.image_scores);

    THROW_CHECK(queue.Push(std::move(retrieval)));
  }

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

}  // namespace

bool VocabTreeMatchingOptions::Check() const {
  CHECK_OPTION_GT(num_images, 0);
  CHECK_OPTION_GT(num_nearest_neighbors, 0);
  CHECK_OPTION_GT(num_checks, 0);
  return true;
}

std::unique_ptr<Thread> CreateVocabTreeFeatureMatcher(
    const VocabTreeMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return std::make_unique<GenericFeatureMatcher<VocabTreePairGenerator>>(
      options, matching_options, geometry_options, database_path);
}

namespace {

class SequentialPairGenerator
    : public PairGenerator<SequentialMatchingOptions> {
 public:
  static size_t CacheSize(const SequentialMatchingOptions& options) {
    return std::max(5 * options.loop_detection_num_images, 5 * options.overlap);
  }

  SequentialPairGenerator(const SequentialMatchingOptions& options,
                          std::shared_ptr<FeatureMatcherCache> cache)
      : options_(options), cache_(std::move(THROW_CHECK_NOTNULL(cache))) {
    THROW_CHECK(options.Check());
    image_ids_ = GetOrderedImageIds();
    image_pairs_.reserve(options_.overlap);

    if (options_.loop_detection) {
      std::vector<image_t> query_image_ids;
      for (size_t i = 0; i < image_ids_.size();
           i += options_.loop_detection_period) {
        query_image_ids.push_back(image_ids_[i]);
      }
      vocab_tree_pair_generator_ = std::make_unique<VocabTreePairGenerator>(
          options_.VocabTreeOptions(), cache, query_image_ids);
    }

    Reset();
  }

  void Reset() {
    image_idx_ = 0;
    if (vocab_tree_pair_generator_) {
      vocab_tree_pair_generator_->Reset();
    }
  }

  bool HasFinished() {
    return image_idx_ >= image_ids_.size() &&
           (vocab_tree_pair_generator_
                ? vocab_tree_pair_generator_->HasFinished()
                : true);
  }

  virtual std::vector<std::pair<image_t, image_t>> Next() {
    image_pairs_.clear();
    if (HasFinished()) {
      if (vocab_tree_pair_generator_) {
        return vocab_tree_pair_generator_->Next();
      }
      return image_pairs_;
    }
    LOG(INFO) << StringPrintf("Matching image [%d/%d]",
                              image_idx_ + 1,
                              image_ids_.size())
              << std::flush;

    const auto image_id1 = image_ids_.at(image_idx_);
    for (int i = 0; i < options_.overlap; ++i) {
      const size_t image_idx_2 = image_idx_ + i;
      if (image_idx_2 < image_ids_.size()) {
        image_pairs_.emplace_back(image_id1, image_ids_.at(image_idx_2));
        if (options_.quadratic_overlap) {
          const size_t image_idx_2_quadratic = image_idx_ + (1ull << i);
          if (image_idx_2_quadratic < image_ids_.size()) {
            image_pairs_.emplace_back(image_id1,
                                      image_ids_.at(image_idx_2_quadratic));
          }
        }
      } else {
        break;
      }
    }

    return image_pairs_;
  }

 private:
  std::vector<image_t> GetOrderedImageIds() const {
    const std::vector<image_t> image_ids = cache_->GetImageIds();

    std::vector<Image> ordered_images;
    ordered_images.reserve(image_ids.size());
    for (const auto image_id : image_ids) {
      ordered_images.push_back(cache_->GetImage(image_id));
    }

    std::sort(ordered_images.begin(),
              ordered_images.end(),
              [](const Image& image1, const Image& image2) {
                return image1.Name() < image2.Name();
              });

    std::vector<image_t> ordered_image_ids;
    ordered_image_ids.reserve(image_ids.size());
    for (const auto& image : ordered_images) {
      ordered_image_ids.push_back(image.ImageId());
    }

    return ordered_image_ids;
  }

  const SequentialMatchingOptions options_;
  const std::shared_ptr<FeatureMatcherCache> cache_;
  std::vector<image_t> image_ids_;
  std::unique_ptr<VocabTreePairGenerator> vocab_tree_pair_generator_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  size_t image_idx_;
};

}  // namespace

bool SequentialMatchingOptions::Check() const {
  CHECK_OPTION_GT(overlap, 0);
  CHECK_OPTION_GT(loop_detection_period, 0);
  CHECK_OPTION_GT(loop_detection_num_images, 0);
  CHECK_OPTION_GT(loop_detection_num_nearest_neighbors, 0);
  CHECK_OPTION_GT(loop_detection_num_checks, 0);
  return true;
}

VocabTreeMatchingOptions SequentialMatchingOptions::VocabTreeOptions() const {
  VocabTreeMatchingOptions options;
  options.num_images = loop_detection_num_images;
  options.num_nearest_neighbors = loop_detection_num_nearest_neighbors;
  options.num_checks = loop_detection_num_checks;
  options.num_images_after_verification =
      loop_detection_num_images_after_verification;
  options.max_num_features = loop_detection_max_num_features;
  options.vocab_tree_path = vocab_tree_path;
  return options;
}

std::unique_ptr<Thread> CreateSequentialFeatureMatcher(
    const SequentialMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return std::make_unique<GenericFeatureMatcher<SequentialPairGenerator>>(
      options, matching_options, geometry_options, database_path);
}

namespace {

class SpatialPairGenerator : public PairGenerator<SpatialMatchingOptions> {
 public:
  static size_t CacheSize(const SpatialMatchingOptions& options) {
    return 5 * options.max_num_neighbors;
  }

  SpatialPairGenerator(const SpatialMatchingOptions& options,
                       const std::shared_ptr<FeatureMatcherCache>& cache)
      : options_(options), image_ids_(cache->GetImageIds()) {
    THROW_CHECK(options.Check());
    ////////////////////////////////////////////////////////////////////////////
    // Spatial indexing
    ////////////////////////////////////////////////////////////////////////////

    Timer timer;
    timer.Start();
    LOG(INFO) << "Indexing images..." << std::flush;

    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> location_matrix =
        ReadLocationData(*cache);
    size_t num_locations = location_idxs_.size();

    PrintElapsedTime(timer);
    if (num_locations == 0) {
      LOG(INFO) << "=> No images with location data.";
      return;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Building spatial index
    ////////////////////////////////////////////////////////////////////////////

    timer.Restart();
    LOG(INFO) << "Building search index..." << std::flush;

    flann::Matrix<float> locations(
        location_matrix.data(), num_locations, location_matrix.cols());

    flann::LinearIndexParams index_params;
    flann::LinearIndex<flann::L2<float>> search_index(index_params);
    search_index.buildIndex(locations);

    PrintElapsedTime(timer);

    //////////////////////////////////////////////////////////////////////////////
    // Searching spatial index
    //////////////////////////////////////////////////////////////////////////////

    timer.Restart();

    LOG(INFO) << "Searching for nearest neighbors..." << std::flush;

    const int knn = std::min<int>(options_.max_num_neighbors, num_locations);
    image_pairs_.reserve(knn);

    index_matrix_.resize(num_locations, knn);
    flann::Matrix<size_t> indices(index_matrix_.data(), num_locations, knn);

    distance_matrix_.resize(num_locations, knn);
    flann::Matrix<float> distances(distance_matrix_.data(), num_locations, knn);

    flann::SearchParams search_params(flann::FLANN_CHECKS_AUTOTUNED);
    const int num_threads = -1;  // TODO: get from options_
    if (num_threads == ThreadPool::kMaxNumThreads) {
      search_params.cores = std::thread::hardware_concurrency();
    } else {
      search_params.cores = num_threads;
    }
    if (search_params.cores <= 0) {
      search_params.cores = 1;
    }

    search_index.knnSearch(locations, indices, distances, knn, search_params);

    PrintElapsedTime(timer);
    Reset();
  }

  void Reset() { current_idx_ = 0; }

  bool HasFinished() { return current_idx_ >= location_idxs_.size(); }

  virtual std::vector<std::pair<image_t, image_t>> Next() {
    image_pairs_.clear();
    if (HasFinished()) {
      return image_pairs_;
    }

    LOG(INFO) << StringPrintf("Matching image [%d/%d]",
                              current_idx_ + 1,
                              location_idxs_.size())
              << std::flush;
    const int knn =
        std::min<int>(options_.max_num_neighbors, location_idxs_.size());
    const float max_distance =
        static_cast<float>(options_.max_distance * options_.max_distance);
    for (int j = 0; j < knn; ++j) {
      // Check if query equals result.
      if (index_matrix_(current_idx_, j) == current_idx_) {
        continue;
      }

      // Since the nearest neighbors are sorted by distance, we can break.
      if (distance_matrix_(current_idx_, j) > max_distance) {
        break;
      }

      const image_t image_id = image_ids_.at(location_idxs_[current_idx_]);
      const size_t nn_idx = location_idxs_.at(index_matrix_(current_idx_, j));
      const image_t nn_image_id = image_ids_.at(nn_idx);
      image_pairs_.emplace_back(image_id, nn_image_id);
    }
    ++current_idx_;
    return image_pairs_;
  }

 private:
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> ReadLocationData(
      const FeatureMatcherCache& cache) {
    GPSTransform gps_transform;
    std::vector<Eigen::Vector3d> ells(1);

    size_t num_locations = 0;
    location_idxs_.clear();
    location_idxs_.reserve(image_ids_.size());
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> location_matrix(
        image_ids_.size(), 3);

    for (size_t i = 0; i < image_ids_.size(); ++i) {
      const auto& image = cache.GetImage(image_ids_[i]);
      const Eigen::Vector3d& translation_prior =
          image.CamFromWorldPrior().translation;

      if ((translation_prior(0) == 0 && translation_prior(1) == 0 &&
           options_.ignore_z) ||
          (translation_prior(0) == 0 && translation_prior(1) == 0 &&
           translation_prior(2) == 0 && !options_.ignore_z)) {
        continue;
      }

      location_idxs_.push_back(i);

      if (options_.is_gps) {
        ells[0](0) = translation_prior(0);
        ells[0](1) = translation_prior(1);
        ells[0](2) = options_.ignore_z ? 0 : translation_prior(2);

        const auto xyzs = gps_transform.EllToXYZ(ells);

        location_matrix(num_locations, 0) = static_cast<float>(xyzs[0](0));
        location_matrix(num_locations, 1) = static_cast<float>(xyzs[0](1));
        location_matrix(num_locations, 2) = static_cast<float>(xyzs[0](2));
      } else {
        location_matrix(num_locations, 0) =
            static_cast<float>(translation_prior(0));
        location_matrix(num_locations, 1) =
            static_cast<float>(translation_prior(1));
        location_matrix(num_locations, 2) =
            static_cast<float>(options_.ignore_z ? 0 : translation_prior(2));
      }

      num_locations += 1;
    }
    return location_matrix;
  }

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

}  // namespace

bool SpatialMatchingOptions::Check() const {
  CHECK_OPTION_GT(max_num_neighbors, 0);
  CHECK_OPTION_GT(max_distance, 0.0);
  return true;
}

std::unique_ptr<Thread> CreateSpatialFeatureMatcher(
    const SpatialMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return std::make_unique<GenericFeatureMatcher<SpatialPairGenerator>>(
      options, matching_options, geometry_options, database_path);
}

namespace {

class TransitiveFeatureMatcher : public Thread {
 public:
  TransitiveFeatureMatcher(const TransitiveMatchingOptions& options,
                           const SiftMatchingOptions& matching_options,
                           const TwoViewGeometryOptions& geometry_options,
                           const std::string& database_path)
      : options_(options),
        matching_options_(matching_options),
        database_(database_path),
        cache_(options_.batch_size, &database_),
        matcher_(matching_options, geometry_options, &database_, &cache_) {
    THROW_CHECK(options.Check());
    THROW_CHECK(matching_options.Check());
    THROW_CHECK(geometry_options.Check());
  }

 private:
  void Run() override {
    PrintHeading1("Transitive feature matching");
    Timer run_timer;
    run_timer.Start();

    if (!matcher_.Setup()) {
      return;
    }

    cache_.Setup();

    const std::vector<image_t> image_ids = cache_.GetImageIds();

    std::vector<std::pair<image_t, image_t>> image_pairs;
    std::unordered_set<image_pair_t> image_pair_ids;

    for (int iteration = 0; iteration < options_.num_iterations; ++iteration) {
      if (IsStopped()) {
        run_timer.PrintMinutes();
        return;
      }

      Timer timer;
      timer.Start();

      LOG(INFO) << StringPrintf(
          "Iteration [%d/%d]", iteration + 1, options_.num_iterations);

      std::vector<std::pair<image_t, image_t>> existing_image_pairs;
      std::vector<int> existing_num_inliers;
      database_.ReadTwoViewGeometryNumInliers(&existing_image_pairs,
                                              &existing_num_inliers);

      THROW_CHECK_EQ(existing_image_pairs.size(), existing_num_inliers.size());

      std::unordered_map<image_t, std::vector<image_t>> adjacency;
      for (const auto& image_pair : existing_image_pairs) {
        adjacency[image_pair.first].push_back(image_pair.second);
        adjacency[image_pair.second].push_back(image_pair.first);
      }

      const size_t batch_size = static_cast<size_t>(options_.batch_size);

      size_t num_batches = 0;
      image_pairs.clear();
      image_pair_ids.clear();
      for (const auto& image : adjacency) {
        const auto image_id1 = image.first;
        for (const auto& image_id2 : image.second) {
          if (adjacency.count(image_id2) > 0) {
            for (const auto& image_id3 : adjacency.at(image_id2)) {
              const auto image_pair_id =
                  Database::ImagePairToPairId(image_id1, image_id3);
              if (image_pair_ids.count(image_pair_id) == 0) {
                image_pairs.emplace_back(image_id1, image_id3);
                image_pair_ids.insert(image_pair_id);
                if (image_pairs.size() >= batch_size) {
                  num_batches += 1;
                  LOG(INFO)
                      << StringPrintf("  Batch %d", num_batches) << std::flush;
                  DatabaseTransaction database_transaction(&database_);
                  matcher_.Match(image_pairs);
                  image_pairs.clear();
                  PrintElapsedTime(timer);
                  timer.Restart();

                  if (IsStopped()) {
                    run_timer.PrintMinutes();
                    return;
                  }
                }
              }
            }
          }
        }
      }

      num_batches += 1;
      LOG(INFO) << StringPrintf("  Batch %d", num_batches) << std::flush;
      DatabaseTransaction database_transaction(&database_);
      matcher_.Match(image_pairs);
      PrintElapsedTime(timer);
    }

    run_timer.PrintMinutes();
  }

  const TransitiveMatchingOptions options_;
  const SiftMatchingOptions matching_options_;
  Database database_;
  FeatureMatcherCache cache_;
  FeatureMatcherController matcher_;
};

}  // namespace

bool TransitiveMatchingOptions::Check() const {
  CHECK_OPTION_GT(batch_size, 0);
  CHECK_OPTION_GT(num_iterations, 0);
  return true;
}

std::unique_ptr<Thread> CreateTransitiveFeatureMatcher(
    const TransitiveMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return std::make_unique<TransitiveFeatureMatcher>(
      options, matching_options, geometry_options, database_path);
}

namespace {

class ImportedPairGenerator : public PairGenerator<ImagePairsMatchingOptions> {
 public:
  static size_t CacheSize(const ImagePairsMatchingOptions& options) {
    return options.block_size;
  }

  ImportedPairGenerator(const ImagePairsMatchingOptions& options,
                        const std::shared_ptr<FeatureMatcherCache>& cache)
      : options_(options) {
    THROW_CHECK(options.Check());

    const std::vector<image_t> image_ids = cache->GetImageIds();
    std::unordered_map<std::string, image_t> image_name_to_image_id;
    image_name_to_image_id.reserve(image_ids.size());
    for (const auto image_id : image_ids) {
      const auto& image = cache->GetImage(image_id);
      image_name_to_image_id.emplace(image.Name(), image_id);
    }
    image_pairs_ =
        ReadImagePairsText(options_.match_list_path, image_name_to_image_id);
    block_image_pairs_.reserve(options_.block_size);
    Reset();
  }

  void Reset() { pair_idx_ = 0; }

  bool HasFinished() { return pair_idx_ >= image_pairs_.size(); }

  virtual std::vector<std::pair<image_t, image_t>> Next() {
    block_image_pairs_.clear();
    if (HasFinished()) {
      return block_image_pairs_;
    }

    LOG(INFO) << StringPrintf("Matching block [%d/%d]",
                              pair_idx_ / options_.block_size + 1,
                              image_pairs_.size() / options_.block_size + 1)
              << std::flush;

    const size_t block_end =
        std::min(pair_idx_ + options_.block_size, image_pairs_.size());
    for (size_t j = pair_idx_; j < block_end; ++j) {
      block_image_pairs_.push_back(image_pairs_[j]);
    }
    pair_idx_ += options_.block_size;
    return block_image_pairs_;
  }

 private:
  const ImagePairsMatchingOptions options_;
  std::vector<std::pair<image_t, image_t>> image_pairs_;
  std::vector<std::pair<image_t, image_t>> block_image_pairs_;
  size_t pair_idx_;
};

}  // namespace

bool ImagePairsMatchingOptions::Check() const {
  CHECK_OPTION_GT(block_size, 0);
  return true;
}

std::unique_ptr<Thread> CreateImagePairsFeatureMatcher(
    const ImagePairsMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return std::make_unique<GenericFeatureMatcher<ImportedPairGenerator>>(
      options, matching_options, geometry_options, database_path);
}

namespace {

class FeaturePairsFeatureMatcher : public Thread {
 public:
  FeaturePairsFeatureMatcher(const FeaturePairsMatchingOptions& options,
                             const SiftMatchingOptions& matching_options,
                             const TwoViewGeometryOptions& geometry_options,
                             const std::string& database_path)
      : options_(options),
        matching_options_(matching_options),
        geometry_options_(geometry_options),
        database_(database_path),
        cache_(kCacheSize, &database_) {
    THROW_CHECK(options.Check());
    THROW_CHECK(matching_options.Check());
    THROW_CHECK(geometry_options.Check());
  }

 private:
  const static size_t kCacheSize = 100;

  void Run() override {
    PrintHeading1("Importing matches");
    Timer run_timer;
    run_timer.Start();

    cache_.Setup();

    std::unordered_map<std::string, const Image*> image_name_to_image;
    image_name_to_image.reserve(cache_.GetImageIds().size());
    for (const auto image_id : cache_.GetImageIds()) {
      const auto& image = cache_.GetImage(image_id);
      image_name_to_image.emplace(image.Name(), &image);
    }

    std::ifstream file(options_.match_list_path);
    THROW_CHECK_FILE_OPEN(file, options_.match_list_path);

    std::string line;
    while (std::getline(file, line)) {
      if (IsStopped()) {
        run_timer.PrintMinutes();
        return;
      }

      StringTrim(&line);
      if (line.empty()) {
        continue;
      }

      std::istringstream line_stream(line);

      std::string image_name1, image_name2;
      try {
        line_stream >> image_name1 >> image_name2;
      } catch (...) {
        LOG(ERROR) << "Could not read image pair.";
        break;
      }

      LOG(INFO) << StringPrintf(
          "%s - %s", image_name1.c_str(), image_name2.c_str());

      if (image_name_to_image.count(image_name1) == 0) {
        LOG(INFO) << StringPrintf("SKIP: Image %s not found in database.",
                                  image_name1.c_str());
        break;
      }
      if (image_name_to_image.count(image_name2) == 0) {
        LOG(INFO) << StringPrintf("SKIP: Image %s not found in database.",
                                  image_name2.c_str());
        break;
      }

      const Image& image1 = *image_name_to_image[image_name1];
      const Image& image2 = *image_name_to_image[image_name2];

      bool skip_pair = false;
      if (database_.ExistsInlierMatches(image1.ImageId(), image2.ImageId())) {
        LOG(INFO) << "SKIP: Matches for image pair already exist in database.";
        skip_pair = true;
      }

      FeatureMatches matches;
      while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty()) {
          break;
        }

        std::istringstream line_stream(line);

        FeatureMatch match;
        try {
          line_stream >> match.point2D_idx1 >> match.point2D_idx2;
        } catch (...) {
          LOG(ERROR) << "Cannot read feature matches.";
          break;
        }

        matches.push_back(match);
      }

      if (skip_pair) {
        continue;
      }

      const Camera& camera1 = cache_.GetCamera(image1.CameraId());
      const Camera& camera2 = cache_.GetCamera(image2.CameraId());

      if (options_.verify_matches) {
        database_.WriteMatches(image1.ImageId(), image2.ImageId(), matches);

        const auto keypoints1 = cache_.GetKeypoints(image1.ImageId());
        const auto keypoints2 = cache_.GetKeypoints(image2.ImageId());

        TwoViewGeometry two_view_geometry =
            EstimateTwoViewGeometry(camera1,
                                    FeatureKeypointsToPointsVector(*keypoints1),
                                    camera2,
                                    FeatureKeypointsToPointsVector(*keypoints2),
                                    matches,
                                    geometry_options_);

        database_.WriteTwoViewGeometry(
            image1.ImageId(), image2.ImageId(), two_view_geometry);
      } else {
        TwoViewGeometry two_view_geometry;

        if (camera1.has_prior_focal_length && camera2.has_prior_focal_length) {
          two_view_geometry.config = TwoViewGeometry::CALIBRATED;
        } else {
          two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
        }

        two_view_geometry.inlier_matches = matches;

        database_.WriteTwoViewGeometry(
            image1.ImageId(), image2.ImageId(), two_view_geometry);
      }
    }

    run_timer.PrintMinutes();
  }

  const FeaturePairsMatchingOptions options_;
  const SiftMatchingOptions matching_options_;
  const TwoViewGeometryOptions geometry_options_;
  Database database_;
  FeatureMatcherCache cache_;
};

}  // namespace

bool FeaturePairsMatchingOptions::Check() const { return true; }

std::unique_ptr<Thread> CreateFeaturePairsFeatureMatcher(
    const FeaturePairsMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return std::make_unique<FeaturePairsFeatureMatcher>(
      options, matching_options, geometry_options, database_path);
}

}  // namespace colmap
