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

#include "colmap/feature/pairing.h"

#include "colmap/feature/utils.h"
#include "colmap/geometry/gps.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <fstream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace colmap {
namespace {

std::vector<std::pair<image_t, image_t>> ReadImagePairsText(
    const std::string& path,
    const std::unordered_map<std::string, image_t>& image_name_to_image_id) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);

  std::string line;
  std::vector<std::pair<image_t, image_t>> image_pairs;
  std::unordered_set<image_pair_t> image_pairs_set;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream line_stream(line);

    std::string image_name1;
    std::string image_name2;

    std::getline(line_stream, image_name1, ' ');
    StringTrim(&image_name1);
    std::getline(line_stream, image_name2, ' ');
    StringTrim(&image_name2);

    if (image_name_to_image_id.count(image_name1) == 0) {
      LOG(ERROR) << "Image " << image_name1 << " does not exist.";
      continue;
    }
    if (image_name_to_image_id.count(image_name2) == 0) {
      LOG(ERROR) << "Image " << image_name2 << " does not exist.";
      continue;
    }

    const image_t image_id1 = image_name_to_image_id.at(image_name1);
    const image_t image_id2 = image_name_to_image_id.at(image_name2);
    const image_pair_t image_pair =
        Database::ImagePairToPairId(image_id1, image_id2);
    const bool image_pair_exists = image_pairs_set.insert(image_pair).second;
    if (image_pair_exists) {
      image_pairs.emplace_back(image_id1, image_id2);
    }
  }
  return image_pairs;
}

}  // namespace

bool ExhaustiveMatchingOptions::Check() const {
  CHECK_OPTION_GT(block_size, 1);
  return true;
}

bool VocabTreeMatchingOptions::Check() const {
  CHECK_OPTION_GT(num_images, 0);
  CHECK_OPTION_GT(num_nearest_neighbors, 0);
  CHECK_OPTION_GT(num_checks, 0);
  return true;
}

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

bool SpatialMatchingOptions::Check() const {
  CHECK_OPTION_GT(max_num_neighbors, 0);
  CHECK_OPTION_GT(max_distance, 0.0);
  return true;
}

bool TransitiveMatchingOptions::Check() const {
  CHECK_OPTION_GT(batch_size, 0);
  CHECK_OPTION_GT(num_iterations, 0);
  return true;
}

bool ImagePairsMatchingOptions::Check() const {
  CHECK_OPTION_GT(block_size, 0);
  return true;
}

bool FeaturePairsMatchingOptions::Check() const { return true; }

std::vector<std::pair<image_t, image_t>> PairGenerator::AllPairs() {
  std::vector<std::pair<image_t, image_t>> image_pairs;
  while (!this->HasFinished()) {
    std::vector<std::pair<image_t, image_t>> image_pairs_block = this->Next();
    image_pairs.insert(image_pairs.end(),
                       std::make_move_iterator(image_pairs_block.begin()),
                       std::make_move_iterator(image_pairs_block.end()));
  }
  return image_pairs;
}

ExhaustivePairGenerator::ExhaustivePairGenerator(
    const ExhaustiveMatchingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache)
    : options_(options),
      image_ids_(THROW_CHECK_NOTNULL(cache)->GetImageIds()),
      block_size_(static_cast<size_t>(options_.block_size)),
      num_blocks_(static_cast<size_t>(
          std::ceil(static_cast<double>(image_ids_.size()) / block_size_))) {
  THROW_CHECK(options.Check());
  LOG(INFO) << "Generating exhaustive image pairs...";
  const size_t num_pairs_per_block = block_size_ * (block_size_ - 1) / 2;
  image_pairs_.reserve(num_pairs_per_block);
}

ExhaustivePairGenerator::ExhaustivePairGenerator(
    const ExhaustiveMatchingOptions& options,
    const std::shared_ptr<Database>& database)
    : ExhaustivePairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(CacheSize(options),
                                                THROW_CHECK_NOTNULL(database),
                                                /*do_setup=*/true)) {}

void ExhaustivePairGenerator::Reset() {
  start_idx1_ = 0;
  start_idx2_ = 0;
}

bool ExhaustivePairGenerator::HasFinished() const {
  return start_idx1_ >= image_ids_.size();
}

std::vector<std::pair<image_t, image_t>> ExhaustivePairGenerator::Next() {
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
                            num_blocks_);

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

VocabTreePairGenerator::VocabTreePairGenerator(
    const VocabTreeMatchingOptions& options,
    std::shared_ptr<FeatureMatcherCache> cache,
    const std::vector<image_t>& query_image_ids)
    : options_(options),
      cache_(std::move(THROW_CHECK_NOTNULL(cache))),
      thread_pool(options_.num_threads),
      queue(options_.num_threads) {
  THROW_CHECK(options.Check());
  LOG(INFO) << "Generating image pairs with vocabulary tree...";

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
}

VocabTreePairGenerator::VocabTreePairGenerator(
    const VocabTreeMatchingOptions& options,
    const std::shared_ptr<Database>& database,
    const std::vector<image_t>& query_image_ids)
    : VocabTreePairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(CacheSize(options),
                                                THROW_CHECK_NOTNULL(database),
                                                /*do_setup=*/true),
          query_image_ids) {}

void VocabTreePairGenerator::Reset() {
  query_idx_ = 0;
  result_idx_ = 0;
}

bool VocabTreePairGenerator::HasFinished() const {
  return result_idx_ >= query_image_ids_.size();
}

std::vector<std::pair<image_t, image_t>> VocabTreePairGenerator::Next() {
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

  LOG(INFO) << StringPrintf(
      "Matching image [%d/%d]", result_idx_ + 1, query_image_ids_.size());

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

void VocabTreePairGenerator::IndexImages(
    const std::vector<image_t>& image_ids) {
  retrieval::VisualIndex<>::IndexOptions index_options;
  index_options.num_threads = options_.num_threads;
  index_options.num_checks = options_.num_checks;

  for (size_t i = 0; i < image_ids.size(); ++i) {
    Timer timer;
    timer.Start();
    LOG(INFO) << StringPrintf(
        "Indexing image [%d/%d]", i + 1, image_ids.size());
    auto keypoints = *cache_->GetKeypoints(image_ids[i]);
    auto descriptors = *cache_->GetDescriptors(image_ids[i]);
    if (options_.max_num_features > 0 &&
        descriptors.rows() > options_.max_num_features) {
      ExtractTopScaleFeatures(
          &keypoints, &descriptors, options_.max_num_features);
    }
    visual_index_.Add(index_options, image_ids[i], keypoints, descriptors);
    LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
  }

  // Compute the TF-IDF weights, etc.
  visual_index_.Prepare();
}

void VocabTreePairGenerator::Query(const image_t image_id) {
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

SequentialPairGenerator::SequentialPairGenerator(
    const SequentialMatchingOptions& options,
    std::shared_ptr<FeatureMatcherCache> cache)
    : options_(options), cache_(std::move(THROW_CHECK_NOTNULL(cache))) {
  THROW_CHECK(options.Check());
  LOG(INFO) << "Generating sequential image pairs...";
  image_ids_ = GetOrderedImageIds();
  image_pairs_.reserve(options_.overlap);

  if (options_.loop_detection) {
    std::vector<image_t> query_image_ids;
    for (size_t i = 0; i < image_ids_.size();
         i += options_.loop_detection_period) {
      query_image_ids.push_back(image_ids_[i]);
    }
    vocab_tree_pair_generator_ = std::make_unique<VocabTreePairGenerator>(
        options_.VocabTreeOptions(), cache_, query_image_ids);
  }
}

SequentialPairGenerator::SequentialPairGenerator(
    const SequentialMatchingOptions& options,
    const std::shared_ptr<Database>& database)
    : SequentialPairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(CacheSize(options),
                                                THROW_CHECK_NOTNULL(database),
                                                /*do_setup=*/true)) {}

void SequentialPairGenerator::Reset() {
  image_idx_ = 0;
  if (vocab_tree_pair_generator_) {
    vocab_tree_pair_generator_->Reset();
  }
}

bool SequentialPairGenerator::HasFinished() const {
  return image_idx_ >= image_ids_.size() &&
         (vocab_tree_pair_generator_ ? vocab_tree_pair_generator_->HasFinished()
                                     : true);
}

std::vector<std::pair<image_t, image_t>> SequentialPairGenerator::Next() {
  image_pairs_.clear();
  if (image_idx_ >= image_ids_.size()) {
    if (vocab_tree_pair_generator_) {
      return vocab_tree_pair_generator_->Next();
    }
    return image_pairs_;
  }
  LOG(INFO) << StringPrintf(
      "Matching image [%d/%d]", image_idx_ + 1, image_ids_.size());

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
  ++image_idx_;
  return image_pairs_;
}

std::vector<image_t> SequentialPairGenerator::GetOrderedImageIds() const {
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

SpatialPairGenerator::SpatialPairGenerator(
    const SpatialMatchingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache)
    : options_(options), image_ids_(cache->GetImageIds()) {
  LOG(INFO) << "Generating spatial image pairs...";
  THROW_CHECK(options.Check());

  Timer timer;
  timer.Start();
  LOG(INFO) << "Indexing images...";

  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> location_matrix =
      ReadLocationData(*cache);
  size_t num_locations = location_idxs_.size();

  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
  if (num_locations == 0) {
    LOG(INFO) << "=> No images with location data.";
    return;
  }

  timer.Restart();
  LOG(INFO) << "Building search index...";

  flann::Matrix<float> locations(
      location_matrix.data(), num_locations, location_matrix.cols());

  flann::LinearIndexParams index_params;
  flann::LinearIndex<flann::L2<float>> search_index(index_params);
  search_index.buildIndex(locations);

  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());

  timer.Restart();
  LOG(INFO) << "Searching for nearest neighbors...";

  const int knn = std::min<int>(options_.max_num_neighbors, num_locations);
  image_pairs_.reserve(knn);

  index_matrix_.resize(num_locations, knn);
  flann::Matrix<size_t> indices(index_matrix_.data(), num_locations, knn);

  distance_matrix_.resize(num_locations, knn);
  flann::Matrix<float> distances(distance_matrix_.data(), num_locations, knn);

  flann::SearchParams search_params(flann::FLANN_CHECKS_AUTOTUNED);
  if (options_.num_threads == ThreadPool::kMaxNumThreads) {
    search_params.cores = std::thread::hardware_concurrency();
  } else {
    search_params.cores = options_.num_threads;
  }
  if (search_params.cores <= 0) {
    search_params.cores = 1;
  }

  search_index.knnSearch(locations, indices, distances, knn, search_params);

  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
}

SpatialPairGenerator::SpatialPairGenerator(
    const SpatialMatchingOptions& options,
    const std::shared_ptr<Database>& database)
    : SpatialPairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(CacheSize(options),
                                                THROW_CHECK_NOTNULL(database),
                                                /*do_setup=*/true)) {}

void SpatialPairGenerator::Reset() { current_idx_ = 0; }

bool SpatialPairGenerator::HasFinished() const {
  return current_idx_ >= location_idxs_.size();
}

std::vector<std::pair<image_t, image_t>> SpatialPairGenerator::Next() {
  image_pairs_.clear();
  if (HasFinished()) {
    return image_pairs_;
  }

  LOG(INFO) << StringPrintf(
      "Matching image [%d/%d]", current_idx_ + 1, location_idxs_.size());
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

Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>
SpatialPairGenerator::ReadLocationData(const FeatureMatcherCache& cache) {
  GPSTransform gps_transform;
  std::vector<Eigen::Vector3d> ells(1);

  size_t num_locations = 0;
  location_idxs_.clear();
  location_idxs_.reserve(image_ids_.size());
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> location_matrix(
      image_ids_.size(), 3);

  for (size_t i = 0; i < image_ids_.size(); ++i) {
    if (!cache.ExistsPosePrior(image_ids_[i])) {
      continue;
    }
    const auto& pose_prior = cache.GetPosePrior(image_ids_[i]);
    const Eigen::Vector3d& translation_prior = pose_prior.position;
    if ((translation_prior(0) == 0 && translation_prior(1) == 0 &&
         options_.ignore_z) ||
        (translation_prior(0) == 0 && translation_prior(1) == 0 &&
         translation_prior(2) == 0 && !options_.ignore_z)) {
      continue;
    }

    location_idxs_.push_back(i);

    switch (pose_prior.coordinate_system) {
      case PosePrior::CoordinateSystem::WGS84: {
        ells[0](0) = translation_prior(0);
        ells[0](1) = translation_prior(1);
        ells[0](2) = options_.ignore_z ? 0 : translation_prior(2);

        const auto xyzs = gps_transform.EllToXYZ(ells);
        location_matrix(num_locations, 0) = static_cast<float>(xyzs[0](0));
        location_matrix(num_locations, 1) = static_cast<float>(xyzs[0](1));
        location_matrix(num_locations, 2) = static_cast<float>(xyzs[0](2));
      } break;
      case PosePrior::CoordinateSystem::UNDEFINED:
        LOG(INFO) << "Unknown coordinate system for image " << image_ids_[i]
                  << ", assuming cartesian.";
      case PosePrior::CoordinateSystem::CARTESIAN:
      default:
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

ImportedPairGenerator::ImportedPairGenerator(
    const ImagePairsMatchingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache)
    : options_(options) {
  LOG(INFO) << "Importing image pairs...";
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
}

ImportedPairGenerator::ImportedPairGenerator(
    const ImagePairsMatchingOptions& options,
    const std::shared_ptr<Database>& database)
    : ImportedPairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(CacheSize(options),
                                                THROW_CHECK_NOTNULL(database),
                                                /*do_setup=*/true)) {}

void ImportedPairGenerator::Reset() { pair_idx_ = 0; }

bool ImportedPairGenerator::HasFinished() const {
  return pair_idx_ >= image_pairs_.size();
}

std::vector<std::pair<image_t, image_t>> ImportedPairGenerator::Next() {
  block_image_pairs_.clear();
  if (HasFinished()) {
    return block_image_pairs_;
  }

  LOG(INFO) << StringPrintf("Matching block [%d/%d]",
                            pair_idx_ / options_.block_size + 1,
                            image_pairs_.size() / options_.block_size + 1);

  const size_t block_end =
      std::min(pair_idx_ + options_.block_size, image_pairs_.size());
  for (size_t j = pair_idx_; j < block_end; ++j) {
    block_image_pairs_.push_back(image_pairs_[j]);
  }
  pair_idx_ += options_.block_size;
  return block_image_pairs_;
}

}  // namespace colmap
