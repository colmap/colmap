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

#include "colmap/feature/pairing.h"

#include "colmap/feature/utils.h"
#include "colmap/geometry/gps.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <fstream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <faiss/IndexFlat.h>
#include <omp.h>

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
    const VocabTreePairingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache,
    const std::vector<image_t>& query_image_ids)
    : options_(options),
      cache_(THROW_CHECK_NOTNULL(cache)),
      thread_pool_(options_.num_threads),
      queue_(options_.num_threads) {
  THROW_CHECK(options.Check());
  LOG(INFO) << "Generating image pairs with vocabulary tree...";

  // Read the pre-trained vocabulary tree from disk.
  visual_index_ = retrieval::VisualIndex::Read(options_.vocab_tree_path);

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
      "Matching image [%d/%d]", result_idx_ + 1, query_image_ids_.size());

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
  for (const auto image_score : image_scores) {
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
    if (options_.max_num_features > 0 &&
        descriptors.rows() > options_.max_num_features) {
      ExtractTopScaleFeatures(
          &keypoints, &descriptors, options_.max_num_features);
    }
    visual_index_->Add(
        index_options, image_ids[i], keypoints, descriptors.cast<float>());
    LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
  }

  // Compute the TF-IDF weights, etc.
  visual_index_->Prepare();
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
  visual_index_->Query(query_options_,
                       keypoints,
                       descriptors.cast<float>(),
                       &retrieval.image_scores);

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
    vocab_tree_pair_generator_ = std::make_unique<VocabTreePairGenerator>(
        options_.VocabTreeOptions(), cache_, query_image_ids);
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
  const size_t num_positions = position_idxs_.size();

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

  knn_ = std::min<int>(options_.max_num_neighbors + 1, num_positions);
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
      "Matching image [%d/%d]", current_idx_ + 1, position_idxs_.size());
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
    const PosePrior* pose_prior = cache.GetPosePriorOrNull(image_ids_[i]);
    if (pose_prior == nullptr) {
      continue;
    }

    const Eigen::Vector3d& position_prior = pose_prior->position;
    if ((position_prior(0) == 0 && position_prior(1) == 0 &&
         position_prior(2) == 0) ||
        (options_.ignore_z && position_prior(0) == 0 &&
         position_prior(1) == 0)) {
      continue;
    }

    const size_t position_idx = position_idxs_.size();
    position_idxs_.push_back(i);

    switch (pose_prior->coordinate_system) {
      case PosePrior::CoordinateSystem::WGS84: {
        ells[0](0) = position_prior(0);
        ells[0](1) = position_prior(1);
        ells[0](2) = options_.ignore_z ? 0 : position_prior(2);

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
        position_matrix(position_idx, 0) = position_prior(0);
        position_matrix(position_idx, 1) = position_prior(1);
        position_matrix(position_idx, 2) =
            options_.ignore_z ? 0 : position_prior(2);
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
        "Matching batch [%d/%d]", current_batch_idx_, current_num_batches_);
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
    const auto [image_id1, image_id2] = Database::PairIdToImagePair(pair_id);
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
        const auto image_pair_id =
            Database::ImagePairToPairId(image_id1, image_id3);
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
