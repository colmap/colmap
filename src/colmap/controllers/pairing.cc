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

#include "colmap/controllers/pairing.h"

#include "colmap/feature/utils.h"
#include "colmap/geometry/gps.h"
#include "colmap/retrieval/global_descriptor_model.h"
#include "colmap/retrieval/resources.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#ifdef COLMAP_ONNX_ENABLED
#include "colmap/feature/onnx_utils.h"
#endif

#include <fstream>
#include <vector>

#include <faiss/IndexFlat.h>
#include <omp.h>


namespace colmap {
namespace {

std::vector<std::pair<image_t, image_t>> ReadImagePairsText(
    const std::filesystem::path& path,
    const NodeHashMap<std::string, image_t>& image_name_to_image_id) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);

  std::string line;
  std::vector<std::pair<image_t, image_t>> image_pairs;
  FlatHashSet<image_pair_t> image_pairs_set;
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
    const image_pair_t image_pair = ImagePairToPairId(image_id1, image_id2);
    const bool image_pair_exists = image_pairs_set.insert(image_pair).second;
    if (image_pair_exists) {
      image_pairs.emplace_back(image_id1, image_id2);
    }
  }
  return image_pairs;
}

}  // namespace

bool ExistingMatchedPairingOptions::Check() const {
  CHECK_OPTION_GT(batch_size, 1);
  return true;
}

bool ExhaustivePairingOptions::Check() const {
  CHECK_OPTION_GT(block_size, 1);
  return true;
}

bool VocabTreePairingOptions::Check() const {
  CHECK_OPTION_GT(num_images, 0);
  CHECK_OPTION_GT(num_nearest_neighbors, 0);
  CHECK_OPTION_GT(num_checks, 0);
  return true;
}

bool SequentialPairingOptions::Check() const {
  CHECK_OPTION_GT(overlap, 0);
  CHECK_OPTION_GT(loop_detection_period, 0);
  CHECK_OPTION_GT(loop_detection_num_images, 0);
  CHECK_OPTION_GT(loop_detection_num_nearest_neighbors, 0);
  CHECK_OPTION_GT(loop_detection_num_checks, 0);
  return true;
}

VocabTreePairingOptions SequentialPairingOptions::VocabTreeOptions() const {
  VocabTreePairingOptions options;
  options.num_images = loop_detection_num_images;
  options.num_nearest_neighbors = loop_detection_num_nearest_neighbors;
  options.num_checks = loop_detection_num_checks;
  options.num_images_after_verification =
      loop_detection_num_images_after_verification;
  options.max_num_features = loop_detection_max_num_features;
  options.vocab_tree_path = vocab_tree_path;
  options.num_threads = num_threads;
  return options;
}

GlobalDescriptorPairingOptions
SequentialPairingOptions::GlobalDescriptorOptions() const {
  GlobalDescriptorPairingOptions options;
  options.num_images = loop_detection_num_images;
  options.model_type = loop_detection_model_type.empty()
                           ? "MixVPR"
                           : loop_detection_model_type;
  options.model_path = loop_detection_model_path;
  options.image_path = loop_detection_image_path;
  options.num_threads = num_threads;
  return options;
}

bool SpatialPairingOptions::Check() const {
  CHECK_OPTION_GE(max_distance, 0.0);
  CHECK_OPTION_GT(max_num_neighbors, 0);
  CHECK_OPTION_LE(min_num_neighbors, max_num_neighbors);
  CHECK_OPTION_GE(min_num_neighbors, 0);
  CHECK_OPTION(max_distance > 0.0 || min_num_neighbors > 0);
  return true;
}

bool TransitivePairingOptions::Check() const {
  CHECK_OPTION_GT(batch_size, 0);
  CHECK_OPTION_GT(num_iterations, 0);
  return true;
}

bool ImportedPairingOptions::Check() const {
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
    const ExhaustivePairingOptions& options,
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
    const ExhaustivePairingOptions& options,
    const std::shared_ptr<Database>& database)
    : ExhaustivePairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(
              options.CacheSize(), THROW_CHECK_NOTNULL(database))) {}

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

  LOG(INFO) << StringPrintf("Processing block [%d/%d, %d/%d]",
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
    const VocabTreePairingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache,
    const std::vector<image_t>& query_image_ids)
    : options_(options),
      cache_(THROW_CHECK_NOTNULL(cache)),
      thread_pool_(options_.num_threads),
      queue_(options_.num_threads) {
  THROW_CHECK(options.Check());
  LOG(INFO) << "Generating image pairs with vocabulary tree...";

  const std::vector<image_t> all_image_ids = cache_->GetImageIds();
  if (query_image_ids.size() > 0) {
    query_image_ids_ = query_image_ids;
  } else if (options_.match_list_path == "") {
    query_image_ids_ = cache_->GetImageIds();
  } else {
    // Map image names to image identifiers.
    NodeHashMap<std::string, image_t> image_name_to_image_id;
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

  // Since we parallelize over the query images, there is no need to parallelize
  // the nearest neighbor search over the query descriptors.
  query_options_.num_threads = 1;
  query_options_.max_num_images = options_.num_images;
  query_options_.num_neighbors = options_.num_nearest_neighbors;
  query_options_.num_checks = options_.num_checks;
  query_options_.num_images_after_verification =
      options_.num_images_after_verification;
}

VocabTreePairGenerator::VocabTreePairGenerator(
    const VocabTreePairingOptions& options,
    const std::shared_ptr<Database>& database,
    const std::vector<image_t>& query_image_ids)
    : VocabTreePairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(options.CacheSize(),
                                                THROW_CHECK_NOTNULL(database)),
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
    return {};
  }
  if (query_idx_ == 0) {
    // Initially, make all retrieval threads busy and continue with the
    // matching.
    const size_t init_num_tasks =
        std::min(query_image_ids_.size(), 2 * thread_pool_.NumThreads());
    for (; query_idx_ < init_num_tasks; ++query_idx_) {
      thread_pool_.AddTask(
          &VocabTreePairGenerator::Query, this, query_image_ids_[query_idx_]);
    }
  }

  LOG(INFO) << StringPrintf(
      "Processing image [%d/%d]", result_idx_ + 1, query_image_ids_.size());

  // Push the next image to the retrieval queue.
  if (query_idx_ < query_image_ids_.size()) {
    thread_pool_.AddTask(
        &VocabTreePairGenerator::Query, this, query_image_ids_[query_idx_++]);
  }

  // Pop the next results from the retrieval queue.
  auto retrieval = queue_.Pop();
  THROW_CHECK(retrieval.IsValid());

  const auto& image_id = retrieval.Data().image_id;
  const auto& image_scores = retrieval.Data().image_scores;

  // Compose the image pairs from the scores.
  image_pairs_.reserve(image_scores.size());
  for (const auto& image_score : image_scores) {
    image_pairs_.emplace_back(image_id, image_score.image_id);
  }
  ++result_idx_;
  return image_pairs_;
}

void VocabTreePairGenerator::IndexImages(
    const std::vector<image_t>& image_ids) {
  retrieval::VisualIndex::IndexOptions index_options;
  // We only assign each feature to a single visual word in the indexing phase.
  // During the query phase, we check for overlap in possibly multiple nearest
  // neighbor visual words. We could do it symmetrically but experiments showed
  // only marginal improvements that do not justify the memory/compute increase.
  index_options.num_neighbors = 1;
  index_options.num_checks = options_.num_checks;
  index_options.num_threads = options_.num_threads;

  for (size_t i = 0; i < image_ids.size(); ++i) {
    Timer timer;
    timer.Start();
    LOG(INFO) << StringPrintf(
        "Indexing image [%d/%d]", i + 1, image_ids.size());
    auto keypoints = *cache_->GetKeypoints(image_ids[i]);
    auto descriptors = *cache_->GetDescriptors(image_ids[i]);
    if (visual_index_ == nullptr) {
      visual_index_ = retrieval::VisualIndex::Read(
          options_.vocab_tree_path.empty()
              ? GetVocabTreeUriForFeatureType(descriptors.type)
              : options_.vocab_tree_path);
    }
    if (options_.max_num_features > 0 &&
        descriptors.data.rows() > options_.max_num_features) {
      ExtractTopScaleFeatures(
          &keypoints, &descriptors, options_.max_num_features);
    }
    visual_index_->Add(
        index_options, image_ids[i], keypoints, descriptors.ToFloat());
    LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
  }

  // Compute the TF-IDF weights, etc.
  visual_index_->Prepare();
}

void VocabTreePairGenerator::Query(const image_t image_id) {
  Retrieval retrieval;
  retrieval.image_id = image_id;

  // Each query must push exactly one result, because the consuming Next() pops
  // exactly one result per query. If a query fails (e.g., due to corrupt
  // features or an out-of-memory error during spatial verification), we still
  // push an empty result and skip retrieval for this image. Otherwise, the
  // consumer would block indefinitely waiting for a result that never arrives.
  try {
    auto keypoints = *cache_->GetKeypoints(image_id);
    auto descriptors = *cache_->GetDescriptors(image_id);
    if (options_.max_num_features > 0 &&
        descriptors.data.rows() > options_.max_num_features) {
      ExtractTopScaleFeatures(
          &keypoints, &descriptors, options_.max_num_features);
    }

    visual_index_->Query(query_options_,
                         keypoints,
                         descriptors.ToFloat(),
                         &retrieval.image_scores);
  } catch (const std::exception& error) {
    LOG(ERROR) << "Failed to query image " << image_id
               << " against vocabulary tree, skipping: " << error.what();
    retrieval.image_scores.clear();
  }

  THROW_CHECK(queue_.Push(std::move(retrieval)));
}

SequentialPairGenerator::SequentialPairGenerator(
    const SequentialPairingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache)
    : options_(options), cache_(THROW_CHECK_NOTNULL(cache)) {
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
    // Vocab tree path is cleared by the GUI/CLI when MixVPR is selected,
    // and vice-versa.  If vocab_tree_path is empty the user chose global
    // descriptor mode (model_path may be empty = auto-download).
    if (options_.vocab_tree_path.empty()) {
      LOG(INFO) << "Using global descriptor for loop detection";
      loop_detection_pair_generator_ =
          std::make_unique<GlobalDescriptorPairGenerator>(
              options_.GlobalDescriptorOptions(), cache_, query_image_ids);
    } else {
      loop_detection_pair_generator_ =
          std::make_unique<VocabTreePairGenerator>(
              options_.VocabTreeOptions(), cache_, query_image_ids);
    }
  }

  if (options_.expand_rig_images) {
    const std::vector<frame_t> frame_ids = cache_->GetFrameIds();
    frame_to_image_ids_.reserve(frame_ids.size());
    image_to_frame_ids_.reserve(image_ids_.size());
    for (const frame_t frame_id : frame_ids) {
      const Frame& frame = cache_->GetFrame(frame_id);
      auto& frame_image_ids = frame_to_image_ids_[frame_id];
      for (const data_t& data_id : frame.ImageIds()) {
        frame_image_ids.push_back(data_id.id);
        image_to_frame_ids_[data_id.id] = frame_id;
      }
    }
  }
}

SequentialPairGenerator::SequentialPairGenerator(
    const SequentialPairingOptions& options,
    const std::shared_ptr<Database>& database)
    : SequentialPairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(
              options.CacheSize(), THROW_CHECK_NOTNULL(database))) {}

void SequentialPairGenerator::Reset() {
  image_idx_ = 0;
  if (loop_detection_pair_generator_) {
    loop_detection_pair_generator_->Reset();
  }
}

bool SequentialPairGenerator::HasFinished() const {
  return image_idx_ >= image_ids_.size() &&
         (loop_detection_pair_generator_ ? loop_detection_pair_generator_->HasFinished()
                                     : true);
}

std::vector<std::pair<image_t, image_t>> SequentialPairGenerator::Next() {
  image_pairs_.clear();
  if (image_idx_ >= image_ids_.size()) {
    if (loop_detection_pair_generator_) {
      return loop_detection_pair_generator_->Next();
    }
    return image_pairs_;
  }
  LOG(INFO) << StringPrintf(
      "Processing image [%d/%d]", image_idx_ + 1, image_ids_.size());

  const auto image_id1 = image_ids_.at(image_idx_);

  // If image is part of a rig, then pair the other images in the same frame.
  if (options_.expand_rig_images) {
    if (const auto frame_id1_it = image_to_frame_ids_.find(image_id1);
        frame_id1_it != image_to_frame_ids_.end()) {
      for (const image_t frame_image_id2 :
           frame_to_image_ids_.at(frame_id1_it->second)) {
        if (image_id1 != frame_image_id2) {
          image_pairs_.emplace_back(image_id1, frame_image_id2);
        }
      }
    }
  }

  auto MaybeExpandRigImages = [this](image_t image_id1, image_t image_id2) {
    if (!options_.expand_rig_images) {
      return;
    }
    const auto frame_id2_it = image_to_frame_ids_.find(image_id2);
    if (frame_id2_it != image_to_frame_ids_.end()) {
      // Pair with all images in second frame.
      for (const image_t frame_image_id2 :
           frame_to_image_ids_.at(frame_id2_it->second)) {
        if (image_id1 != frame_image_id2 && image_id2 != frame_image_id2) {
          image_pairs_.emplace_back(image_id1, frame_image_id2);
        }
      }
    }
  };

  for (int i = 0; i < options_.overlap; ++i) {
    if (options_.quadratic_overlap) {
      const size_t image_idx_2_quadratic = image_idx_ + (1ull << i);
      if (image_idx_2_quadratic < image_ids_.size()) {
        const image_t image_id2 = image_ids_.at(image_idx_2_quadratic);
        image_pairs_.emplace_back(image_id1, image_id2);
        MaybeExpandRigImages(image_id1, image_id2);
      } else {
        break;
      }
    } else {
      const size_t image_idx_2 = image_idx_ + i + 1;
      if (image_idx_2 < image_ids_.size()) {
        const image_t image_id2 = image_ids_.at(image_idx_2);
        image_pairs_.emplace_back(image_id1, image_id2);
        MaybeExpandRigImages(image_id1, image_id2);
      } else {
        break;
      }
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
    const SpatialPairingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache)
    : options_(options), image_ids_(THROW_CHECK_NOTNULL(cache)->GetImageIds()) {
  LOG(INFO) << "Generating spatial image pairs...";
  THROW_CHECK(options.Check());

  Timer timer;
  timer.Start();
  LOG(INFO) << "Indexing images...";

  Eigen::RowMajorMatrixXf position_matrix = ReadPositionPriorData(*cache);
  const int num_positions = position_idxs_.size();

  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
  if (num_positions == 0) {
    LOG(INFO) << "=> No images with location data.";
    return;
  }
  if (num_positions <= options_.min_num_neighbors) {
    LOG(WARNING) << StringPrintf(
        "min_num_neighbors (%d) exceeds number of images with location data "
        "(%zu), this may limit the number of matched pairs.",
        options_.min_num_neighbors,
        num_positions);
  }

  timer.Restart();
  LOG(INFO) << "Building search index...";

  faiss::IndexFlatL2 search_index(/*d=*/3);
  search_index.add(position_matrix.rows(), position_matrix.data());

  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());

  timer.Restart();
  LOG(INFO) << "Searching for nearest neighbors...";

  knn_ = std::min(options_.max_num_neighbors + 1, num_positions);
  image_pairs_.reserve(knn_);

  index_matrix_.resize(num_positions, knn_);
  distance_squared_matrix_.resize(num_positions, knn_);

  omp_set_num_threads(GetEffectiveNumThreads(options_.num_threads));

  search_index.search(position_matrix.rows(),
                      position_matrix.data(),
                      knn_,
                      distance_squared_matrix_.data(),
                      index_matrix_.data());

  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
}

SpatialPairGenerator::SpatialPairGenerator(
    const SpatialPairingOptions& options,
    const std::shared_ptr<Database>& database)
    : SpatialPairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(
              options.CacheSize(), THROW_CHECK_NOTNULL(database))) {}

void SpatialPairGenerator::Reset() { current_idx_ = 0; }

bool SpatialPairGenerator::HasFinished() const {
  return current_idx_ >= position_idxs_.size();
}

std::vector<std::pair<image_t, image_t>> SpatialPairGenerator::Next() {
  image_pairs_.clear();
  if (HasFinished()) {
    return image_pairs_;
  }

  LOG(INFO) << StringPrintf(
      "Processing image [%d/%d]", current_idx_ + 1, position_idxs_.size());
  const float max_distance_squared =
      static_cast<float>(options_.max_distance * options_.max_distance);
  for (int j = 0; j < knn_; ++j) {
    // Check if query equals result.
    if (index_matrix_(current_idx_, j) == static_cast<int>(current_idx_)) {
      continue;
    }

    // Since the nearest neighbors are sorted by distance, we can break
    // once the distance is too large and enough neighbors are collected.
    if (distance_squared_matrix_(current_idx_, j) > max_distance_squared &&
        j > options_.min_num_neighbors) {
      break;
    }

    const image_t image_id = image_ids_.at(position_idxs_[current_idx_]);
    const size_t nn_idx = position_idxs_.at(index_matrix_(current_idx_, j));
    const image_t nn_image_id = image_ids_.at(nn_idx);
    image_pairs_.emplace_back(image_id, nn_image_id);
  }
  ++current_idx_;
  return image_pairs_;
}

Eigen::RowMajorMatrixXf SpatialPairGenerator::ReadPositionPriorData(
    FeatureMatcherCache& cache) {
  GPSTransform gps_transform;
  std::vector<Eigen::Vector3d> ells(1);

  Eigen::RowMajorMatrixXd position_matrix(image_ids_.size(), 3);
  position_idxs_.clear();
  position_idxs_.reserve(image_ids_.size());

  for (size_t i = 0; i < image_ids_.size(); ++i) {
    const PosePrior* pose_prior = cache.FindImagePosePriorOrNull(image_ids_[i]);
    if (pose_prior == nullptr) {
      continue;
    }

    if ((!options_.ignore_z && !pose_prior->HasPosition()) ||
        (options_.ignore_z && !pose_prior->position.head<2>().allFinite())) {
      continue;
    }

    const size_t position_idx = position_idxs_.size();
    position_idxs_.push_back(i);

    switch (pose_prior->coordinate_system) {
      case PosePrior::CoordinateSystem::WGS84: {
        ells[0](0) = pose_prior->position(0);
        ells[0](1) = pose_prior->position(1);
        ells[0](2) = options_.ignore_z ? 0 : pose_prior->position(2);

        const std::vector<Eigen::Vector3d> xyzs =
            gps_transform.EllipsoidToECEF(ells);
        position_matrix(position_idx, 0) = xyzs[0](0);
        position_matrix(position_idx, 1) = xyzs[0](1);
        position_matrix(position_idx, 2) = xyzs[0](2);
      } break;
      case PosePrior::CoordinateSystem::UNDEFINED:
      default:
        LOG(WARNING) << "Unknown coordinate system for image " << image_ids_[i]
                     << ", assuming cartesian.";
      case PosePrior::CoordinateSystem::CARTESIAN:
        position_matrix(position_idx, 0) = pose_prior->position(0);
        position_matrix(position_idx, 1) = pose_prior->position(1);
        position_matrix(position_idx, 2) =
            options_.ignore_z ? 0 : pose_prior->position(2);
    }
  }

  // Subtract the mean coordinate before casting to float for better numerical
  // precision when dealing with large coordinates (e.g. GPS). For even better
  // precision, we could also rescale the coordinates.
  position_matrix.rowwise() -= position_matrix.colwise().mean();
  return position_matrix.topRows(position_idxs_.size()).cast<float>();
}

TransitivePairGenerator::TransitivePairGenerator(
    const TransitivePairingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache)
    : options_(options), cache_(cache) {
  THROW_CHECK(options.Check());
}

TransitivePairGenerator::TransitivePairGenerator(
    const TransitivePairingOptions& options,
    const std::shared_ptr<Database>& database)
    : TransitivePairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(
              options.CacheSize(), THROW_CHECK_NOTNULL(database))) {}

void TransitivePairGenerator::Reset() {
  current_iteration_ = 0;
  current_batch_idx_ = 0;
  image_pairs_.clear();
  image_pair_ids_.clear();
}

bool TransitivePairGenerator::HasFinished() const {
  return current_iteration_ >= options_.num_iterations && image_pairs_.empty();
}

std::vector<std::pair<image_t, image_t>> TransitivePairGenerator::Next() {
  if (!image_pairs_.empty()) {
    current_batch_idx_++;
    std::vector<std::pair<image_t, image_t>> batch;
    while (!image_pairs_.empty() &&
           static_cast<int>(batch.size()) < options_.batch_size) {
      batch.push_back(image_pairs_.back());
      image_pairs_.pop_back();
    }
    LOG(INFO) << StringPrintf(
        "Processing batch [%d/%d]", current_batch_idx_, current_num_batches_);
    return batch;
  }

  if (current_iteration_ >= options_.num_iterations) {
    return {};
  }

  current_batch_idx_ = 0;
  current_num_batches_ = 0;
  current_iteration_++;

  LOG(INFO) << StringPrintf(
      "Iteration [%d/%d]", current_iteration_, options_.num_iterations);

  std::vector<std::pair<image_pair_t, int>> existing_pair_ids_and_num_inliers;
  cache_->AccessDatabase(
      [&existing_pair_ids_and_num_inliers](Database& database) {
        existing_pair_ids_and_num_inliers =
            database.ReadTwoViewGeometryNumInliers();
      });

  std::map<image_t, std::vector<image_t>> adjacency;
  for (const auto& [pair_id, _] : existing_pair_ids_and_num_inliers) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    adjacency[image_id1].push_back(image_id2);
    adjacency[image_id2].push_back(image_id1);
    image_pair_ids_.insert(pair_id);
  }

  for (const auto& image : adjacency) {
    const auto image_id1 = image.first;
    for (const auto& image_id2 : image.second) {
      const auto it = adjacency.find(image_id2);
      if (it == adjacency.end()) {
        continue;
      }
      for (const auto& image_id3 : it->second) {
        if (image_id1 == image_id3) {
          continue;
        }
        const auto image_pair_id = ImagePairToPairId(image_id1, image_id3);
        if (image_pair_ids_.count(image_pair_id) != 0) {
          continue;
        }
        image_pairs_.emplace_back(std::minmax(image_id1, image_id3));
        image_pair_ids_.insert(image_pair_id);
      }
    }
  }

  current_num_batches_ =
      std::ceil(static_cast<double>(image_pairs_.size()) / options_.batch_size);

  return Next();
}

ImportedPairGenerator::ImportedPairGenerator(
    const ImportedPairingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache)
    : options_(options) {
  THROW_CHECK(options.Check());

  LOG(INFO) << "Importing image pairs...";
  const std::vector<image_t> image_ids = cache->GetImageIds();
  NodeHashMap<std::string, image_t> image_name_to_image_id;
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
    const ImportedPairingOptions& options,
    const std::shared_ptr<Database>& database)
    : ImportedPairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(
              options.CacheSize(), THROW_CHECK_NOTNULL(database))) {}

void ImportedPairGenerator::Reset() { pair_idx_ = 0; }

bool ImportedPairGenerator::HasFinished() const {
  return pair_idx_ >= image_pairs_.size();
}

std::vector<std::pair<image_t, image_t>> ImportedPairGenerator::Next() {
  block_image_pairs_.clear();
  if (HasFinished()) {
    return block_image_pairs_;
  }

  LOG(INFO) << StringPrintf("Processing block [%d/%d]",
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

ExistingMatchedPairGenerator::ExistingMatchedPairGenerator(
    const ExistingMatchedPairingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache)
    : options_(options) {
  THROW_CHECK(options.Check());
  LOG(INFO) << "Generating existing image pairs...";
  cache->AccessDatabase([this](Database& database) {
    auto num_matches = database.ReadNumMatches();
    image_pairs_.reserve(num_matches.size());
    for (const auto& [pair_id, _] : num_matches) {
      image_pairs_.emplace_back(PairIdToImagePair(pair_id));
    }
  });
  num_batches_ =
      std::ceil(static_cast<double>(image_pairs_.size()) / options_.batch_size);
}

ExistingMatchedPairGenerator::ExistingMatchedPairGenerator(
    const ExistingMatchedPairingOptions& options,
    const std::shared_ptr<Database>& database)
    : ExistingMatchedPairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(
              options.CacheSize(), THROW_CHECK_NOTNULL(database))) {}

void ExistingMatchedPairGenerator::Reset() { start_idx_ = 0; }

bool ExistingMatchedPairGenerator::HasFinished() const {
  return start_idx_ >= image_pairs_.size();
}

std::vector<std::pair<image_t, image_t>> ExistingMatchedPairGenerator::Next() {
  if (HasFinished()) {
    return {};
  }

  const size_t end_idx =
      std::min(start_idx_ + options_.batch_size, image_pairs_.size());

  std::vector<std::pair<image_t, image_t>> batch;
  batch.reserve(end_idx - start_idx_);
  for (size_t idx = start_idx_; idx < end_idx; ++idx) {
    batch.emplace_back(image_pairs_[idx]);
  }

  LOG(INFO) << StringPrintf("Processing batch [%d/%d]",
                            start_idx_ / options_.batch_size + 1,
                            num_batches_);

  start_idx_ = end_idx;

  return batch;
}

// ---------------------------------------------------------------------------
// GlobalDescriptorPairGenerator
// ---------------------------------------------------------------------------

bool GlobalDescriptorPairingOptions::Check() const {
  CHECK_OPTION_GT(num_images, 0);
  CHECK_OPTION_GT(batch_size, 0);
  return true;
}

GlobalDescriptorPairGenerator::GlobalDescriptorPairGenerator(
    const GlobalDescriptorPairingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache,
    const std::vector<image_t>& query_image_ids)
    : options_(options)
    , cache_(THROW_CHECK_NOTNULL(cache))
    , global_descriptor_index_(4096) {
  THROW_CHECK(options.Check());
  LOG(INFO) << "Generating image pairs with global descriptors (MixVPR)...";

  const std::vector<image_t> all_image_ids = cache_->GetImageIds();
  if (query_image_ids.size() > 0) {
    query_image_ids_ = query_image_ids;
  } else if (options_.match_list_path.empty()) {
    query_image_ids_ = cache_->GetImageIds();
  } else {
    // Map image names to image identifiers.
    NodeHashMap<std::string, image_t> image_name_to_image_id;
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
      if (line.empty() || line[0] == '#') continue;
      if (image_name_to_image_id.count(line) == 0) {
        LOG(ERROR) << "Image " << line << " does not exist.";
      } else {
        query_image_ids_.push_back(image_name_to_image_id.at(line));
      }
    }
  }

  ComputeAndIndexDescriptors();

  // Pre-compute all image pairs by querying each query image against the index.
  LOG(INFO) << "Computing image pairs for " << query_image_ids_.size()
            << " query images...";
  retrieval::GlobalDescriptorIndex::QueryOptions query_opts;
  query_opts.max_num_images = options_.num_images;

  FlatHashSet<image_pair_t> seen_pairs;
  for (size_t i = 0; i < query_image_ids_.size(); ++i) {
    const image_t query_id = query_image_ids_[i];
    std::vector<retrieval::ImageScore> scores;
    global_descriptor_index_.Query(query_opts, query_id, &scores);

    for (const auto& score : scores) {
      const image_pair_t pair_id =
          ImagePairToPairId(query_id, score.image_id);
      if (seen_pairs.insert(pair_id).second) {
        image_pairs_.emplace_back(query_id, score.image_id);
      }
    }
  }

  LOG(INFO) << "Generated " << image_pairs_.size() << " image pairs from "
            << query_image_ids_.size() << " query images.";
}

GlobalDescriptorPairGenerator::GlobalDescriptorPairGenerator(
    const GlobalDescriptorPairingOptions& options,
    const std::shared_ptr<Database>& database,
    const std::vector<image_t>& query_image_ids)
    : GlobalDescriptorPairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(
              options.CacheSize(), THROW_CHECK_NOTNULL(database)),
          query_image_ids) {}

void GlobalDescriptorPairGenerator::Reset() { pair_idx_ = 0; }

bool GlobalDescriptorPairGenerator::HasFinished() const {
  return pair_idx_ >= image_pairs_.size();
}

std::vector<std::pair<image_t, image_t>>
GlobalDescriptorPairGenerator::Next() {
  if (HasFinished()) {
    return {};
  }

  const size_t end_idx =
      std::min(pair_idx_ + static_cast<size_t>(options_.num_images),
               image_pairs_.size());

  LOG(INFO) << StringPrintf(
      "Processing pairs [%d/%d]",
      pair_idx_ + 1,
      image_pairs_.size());

  std::vector<std::pair<image_t, image_t>> batch;
  batch.reserve(end_idx - pair_idx_);
  for (size_t i = pair_idx_; i < end_idx; ++i) {
    batch.push_back(image_pairs_[i]);
  }
  pair_idx_ = end_idx;
  return batch;
}

// static
std::vector<float> GlobalDescriptorPairGenerator::PreprocessImage(
    const Bitmap& bitmap,
    const retrieval::GlobalDescriptorModel& model) {
  // Input: RGB bitmap. Output: float32 NCHW tensor normalized per model config.
  const int input_w =
      model.input_width > 0 ? model.input_width : bitmap.Width();
  const int input_h =
      model.input_height > 0 ? model.input_height : bitmap.Height();
  const int kChannels = 3;

  Bitmap resized = bitmap.CloneAsRGB();
  if (resized.Width() != input_w || resized.Height() != input_h) {
    resized.Rescale(input_w, input_h, Bitmap::RescaleFilter::kBilinear);
  }

  const int kNumPixels = input_w * input_h;
  std::vector<float> tensor(kChannels * kNumPixels);
  const auto& data = resized.RowMajorData();

  for (int y = 0; y < input_h; ++y) {
    for (int x = 0; x < input_w; ++x) {
      for (int c = 0; c < kChannels; ++c) {
        const float val = static_cast<float>(
            data[(y * input_w + x) * kChannels + c]) / 255.0f;
        tensor[c * kNumPixels + y * input_w + x] =
            (val - model.mean[c]) / model.std[c];
      }
    }
  }

  return tensor;
}

void GlobalDescriptorPairGenerator::ComputeAndIndexDescriptors() {
#ifdef COLMAP_ONNX_ENABLED
  const std::vector<image_t> all_image_ids = cache_->GetImageIds();

  // Look up the model config.
  const retrieval::GlobalDescriptorModel* model_info =
      retrieval::GlobalDescriptorModel::GetModel(options_.model_type);
  if (!model_info) {
    LOG(FATAL_THROW) << "Unknown global descriptor model: '"
                     << options_.model_type
                     << "'. Available models: MixVPR, MegaLoc.";
  }
  LOG(INFO) << "Using global descriptor model: " << model_info->name;

  const int kInputW = model_info->input_width > 0 ? model_info->input_width
                                                    : 0;  // dynamic
  const int kInputH = model_info->input_height > 0 ? model_info->input_height
                                                    : 0;
  const int kChannels = 3;
  const int kDescriptorDim = model_info->descriptor_dim;
  const int kBatchSize =
      model_info->supports_batching ? options_.batch_size : 1;

  // Resize the index if descriptor dim changed (e.g. model switch).
  global_descriptor_index_ =
      retrieval::GlobalDescriptorIndex(kDescriptorDim);

  // Try to load cached descriptors from disk.
  const std::filesystem::path cache_path =
      options_.image_path /
      ("global_descriptors_" + model_info->name + ".bin");
  if (ExistsFile(cache_path)) {
    try {
      global_descriptor_index_.Read(cache_path);
      if (global_descriptor_index_.NumImages() == all_image_ids.size()) {
        LOG(INFO) << "Loaded cached " << model_info->name
                  << " descriptors for " << all_image_ids.size()
                  << " images from " << cache_path;
        return;
      }
      LOG(INFO) << "Cached descriptors have "
                << global_descriptor_index_.NumImages() << " images, but "
                << all_image_ids.size() << " are needed. Re-extracting...";
      global_descriptor_index_ = retrieval::GlobalDescriptorIndex(
          kDescriptorDim);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to read cached descriptors (" << e.what()
                   << "), re-extracting...";
      global_descriptor_index_ = retrieval::GlobalDescriptorIndex(
          kDescriptorDim);
    }
  }

  // Resolve model path.
  std::string model_path = options_.model_path.string();
  if (model_path.empty()) {
    model_path = model_info->default_model_uri;
    LOG(INFO) << "Auto-downloading " << model_info->name
              << " model from default URI";
  } else {
    LOG(INFO) << "Loading ONNX model from " << model_path;
  }
  std::unique_ptr<ONNXModel> model;
  try {
    model = std::make_unique<ONNXModel>(model_path,
                                        options_.num_threads,
                                        options_.use_gpu,
                                        options_.gpu_index);
  } catch (const std::exception& e) {
    LOG(FATAL_THROW)
        << "Failed to load " << model_info->name << " model. "
        << "If you left the model path empty, COLMAP attempted to download "
        << "from the default URI but it may have failed (network issue). "
        << "You can manually download the model and specify its path. "
        << "Original error: " << e.what();
  }

  // Validate model I/O.
  THROW_CHECK_EQ(model->input_shapes().size(), 1);
  ThrowCheckONNXNode(model->input_names()[0],
                     model_info->input_name,
                     model->input_shapes()[0],
                     model_info->expected_input_shape);
  THROW_CHECK_EQ(model->output_shapes().size(), 1);
  ThrowCheckONNXNode(model->output_names()[0],
                     model_info->output_name,
                     model->output_shapes()[0],
                     model_info->expected_output_shape);

  const int kPixelsPerImage =
      kInputW > 0 ? kChannels * kInputW * kInputH : 0;

  // Process images in batches.
  std::vector<image_t> batch_image_ids;
  std::vector<float> batch_data;
  batch_image_ids.reserve(kBatchSize);
  if (kPixelsPerImage > 0) {
    batch_data.reserve(kBatchSize * kPixelsPerImage);
  }

  size_t total_processed = 0;
  const size_t total_images = all_image_ids.size();

  for (size_t i = 0; i < total_images; ++i) {
    const image_t image_id = all_image_ids[i];
    const Image& image = cache_->GetImage(image_id);

    const std::filesystem::path full_path =
        options_.image_path / image.Name();
    Bitmap bitmap;
    if (!bitmap.Read(full_path, /*as_rgb=*/true)) {
      LOG(ERROR) << "Failed to read image: " << full_path
                 << ", skipping image " << image_id;
      continue;
    }

    std::vector<float> tensor = PreprocessImage(bitmap, *model_info);
    batch_data.insert(batch_data.end(), tensor.begin(), tensor.end());
    batch_image_ids.push_back(image_id);

    if (static_cast<int>(batch_image_ids.size()) >= kBatchSize ||
        i == total_images - 1) {
      const int current_batch_size =
          static_cast<int>(batch_image_ids.size());
      const int per_image_size =
          static_cast<int>(tensor.size());  // last image's size

      std::vector<int64_t> input_shape = {
          current_batch_size, kChannels, kInputH, kInputW};
      // Handle dynamic input sizes.
      if (kInputW == 0) {
        input_shape = {current_batch_size,
                        kChannels,
                        bitmap.Height(),
                        bitmap.Width()};
        // For models without batching, we still create the batch dim = 1.
        if (!model_info->supports_batching) {
          input_shape[0] = 1;
        }
      }

      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                     OrtMemType::OrtMemTypeCPU),
          batch_data.data(),
          batch_data.size(),
          input_shape.data(),
          input_shape.size());

      std::vector<Ort::Value> input_tensors;
      input_tensors.emplace_back(std::move(input_tensor));
      std::vector<Ort::Value> output_tensors = model->Run(input_tensors);
      THROW_CHECK_EQ(output_tensors.size(), 1);

      const float* output_data =
          output_tensors[0].GetTensorData<float>();
      const auto output_shape =
          output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

      int output_batch = current_batch_size;
      if (output_shape.size() >= 1 && output_shape[0] != -1) {
        output_batch = static_cast<int>(output_shape[0]);
      }

      for (int j = 0; j < output_batch; ++j) {
        const float* desc = output_data + j * kDescriptorDim;
        std::vector<float> descriptor(desc, desc + kDescriptorDim);
        global_descriptor_index_.Add(batch_image_ids[j], descriptor);
      }

      total_processed += output_batch;
      LOG(INFO) << StringPrintf("Extracted descriptors [%d/%d]",
                                total_processed,
                                total_images);

      batch_image_ids.clear();
      batch_data.clear();
    }
  }

  if (global_descriptor_index_.NumImages() > 0) {
    global_descriptor_index_.Write(cache_path);
    LOG(INFO) << "Cached " << model_info->name << " descriptors to "
              << cache_path;
  }

  global_descriptor_index_.Prepare();
#else
  LOG(FATAL_THROW)
      << "Global descriptor matching requires ONNX Runtime support. "
      << "Please rebuild COLMAP with ONNX_ENABLED=ON.";
#endif  // COLMAP_ONNX_ENABLED
}

}  // namespace colmap
