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
#include "colmap/feature/utils.h"
#include "colmap/geometry/gps.h"
#include "colmap/retrieval/visual_index.h"
#include "colmap/util/misc.h"

#include <fstream>
#include <numeric>

namespace colmap {
namespace {

void PrintElapsedTime(const Timer& timer) {
  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
}

void IndexImagesInVisualIndex(const int num_threads,
                              const int num_checks,
                              const int max_num_features,
                              const std::vector<image_t>& image_ids,
                              Thread* thread,
                              FeatureMatcherCache* cache,
                              retrieval::VisualIndex<>* visual_index) {
  retrieval::VisualIndex<>::IndexOptions index_options;
  index_options.num_threads = num_threads;
  index_options.num_checks = num_checks;

  for (size_t i = 0; i < image_ids.size(); ++i) {
    if (thread->IsStopped()) {
      return;
    }

    Timer timer;
    timer.Start();

    LOG(INFO) << StringPrintf("Indexing image [%d/%d]", i + 1, image_ids.size())
              << std::flush;

    auto keypoints = *cache->GetKeypoints(image_ids[i]);
    auto descriptors = *cache->GetDescriptors(image_ids[i]);
    if (max_num_features > 0 && descriptors.rows() > max_num_features) {
      ExtractTopScaleFeatures(&keypoints, &descriptors, max_num_features);
    }

    visual_index->Add(index_options, image_ids[i], keypoints, descriptors);

    PrintElapsedTime(timer);
  }

  // Compute the TF-IDF weights, etc.
  visual_index->Prepare();
}

void MatchNearestNeighborsInVisualIndex(const int num_threads,
                                        const int num_images,
                                        const int num_neighbors,
                                        const int num_checks,
                                        const int num_images_after_verification,
                                        const int max_num_features,
                                        const std::vector<image_t>& image_ids,
                                        Thread* thread,
                                        FeatureMatcherCache* cache,
                                        retrieval::VisualIndex<>* visual_index,
                                        FeatureMatcherController* matcher) {
  struct Retrieval {
    image_t image_id = kInvalidImageId;
    std::vector<retrieval::ImageScore> image_scores;
  };

  // Create a thread pool to retrieve the nearest neighbors.
  ThreadPool retrieval_thread_pool(num_threads);
  JobQueue<Retrieval> retrieval_queue(num_threads);

  // The retrieval thread kernel function. Note that the descriptors should be
  // extracted outside of this function sequentially to avoid any concurrent
  // access to the database causing race conditions.
  retrieval::VisualIndex<>::QueryOptions query_options;
  query_options.max_num_images = num_images;
  query_options.num_neighbors = num_neighbors;
  query_options.num_checks = num_checks;
  query_options.num_images_after_verification = num_images_after_verification;
  auto QueryFunc = [&](const image_t image_id) {
    auto keypoints = *cache->GetKeypoints(image_id);
    auto descriptors = *cache->GetDescriptors(image_id);
    if (max_num_features > 0 && descriptors.rows() > max_num_features) {
      ExtractTopScaleFeatures(&keypoints, &descriptors, max_num_features);
    }

    Retrieval retrieval;
    retrieval.image_id = image_id;
    visual_index->Query(
        query_options, keypoints, descriptors, &retrieval.image_scores);

    CHECK(retrieval_queue.Push(std::move(retrieval)));
  };

  // Initially, make all retrieval threads busy and continue with the matching.
  size_t image_idx = 0;
  const size_t init_num_tasks =
      std::min(image_ids.size(), 2 * retrieval_thread_pool.NumThreads());
  for (; image_idx < init_num_tasks; ++image_idx) {
    retrieval_thread_pool.AddTask(QueryFunc, image_ids[image_idx]);
  }

  std::vector<std::pair<image_t, image_t>> image_pairs;

  // Pop the finished retrieval results and enqueue them for feature matching.
  for (size_t i = 0; i < image_ids.size(); ++i) {
    if (thread->IsStopped()) {
      retrieval_queue.Stop();
      return;
    }

    Timer timer;
    timer.Start();

    LOG(INFO) << StringPrintf("Matching image [%d/%d]", i + 1, image_ids.size())
              << std::flush;

    // Push the next image to the retrieval queue.
    if (image_idx < image_ids.size()) {
      retrieval_thread_pool.AddTask(QueryFunc, image_ids[image_idx]);
      image_idx += 1;
    }

    // Pop the next results from the retrieval queue.
    auto retrieval = retrieval_queue.Pop();
    CHECK(retrieval.IsValid());

    const auto& image_id = retrieval.Data().image_id;
    const auto& image_scores = retrieval.Data().image_scores;

    // Compose the image pairs from the scores.
    image_pairs.clear();
    image_pairs.reserve(image_scores.size());
    for (const auto image_score : image_scores) {
      image_pairs.emplace_back(image_id, image_score.image_id);
    }

    matcher->Match(image_pairs);

    PrintElapsedTime(timer);
  }
}

class ExhaustiveFeatureMatcher : public Thread {
 public:
  ExhaustiveFeatureMatcher(const ExhaustiveMatchingOptions& options,
                           const SiftMatchingOptions& matching_options,
                           const TwoViewGeometryOptions& geometry_options,
                           const std::string& database_path)
      : options_(options),
        matching_options_(matching_options),
        database_(database_path),
        cache_(5 * options_.block_size, &database_),
        matcher_(matching_options, geometry_options, &database_, &cache_) {
    CHECK(options.Check());
    CHECK(matching_options.Check());
    CHECK(geometry_options.Check());
  }

 private:
  void Run() override {
    PrintHeading1("Exhaustive feature matching");

    if (!matcher_.Setup()) {
      return;
    }

    cache_.Setup();

    const std::vector<image_t> image_ids = cache_.GetImageIds();

    const size_t block_size = static_cast<size_t>(options_.block_size);
    const size_t num_blocks = static_cast<size_t>(
        std::ceil(static_cast<double>(image_ids.size()) / block_size));
    const size_t num_pairs_per_block = block_size * (block_size - 1) / 2;

    std::vector<std::pair<image_t, image_t>> image_pairs;
    image_pairs.reserve(num_pairs_per_block);

    for (size_t start_idx1 = 0; start_idx1 < image_ids.size();
         start_idx1 += block_size) {
      const size_t end_idx1 =
          std::min(image_ids.size(), start_idx1 + block_size) - 1;
      for (size_t start_idx2 = 0; start_idx2 < image_ids.size();
           start_idx2 += block_size) {
        const size_t end_idx2 =
            std::min(image_ids.size(), start_idx2 + block_size) - 1;

        if (IsStopped()) {
          GetTimer().PrintMinutes();
          return;
        }

        Timer timer;
        timer.Start();

        LOG(INFO) << StringPrintf("Matching block [%d/%d, %d/%d]",
                                  start_idx1 / block_size + 1,
                                  num_blocks,
                                  start_idx2 / block_size + 1,
                                  num_blocks)
                  << std::flush;

        image_pairs.clear();
        for (size_t idx1 = start_idx1; idx1 <= end_idx1; ++idx1) {
          for (size_t idx2 = start_idx2; idx2 <= end_idx2; ++idx2) {
            const size_t block_id1 = idx1 % block_size;
            const size_t block_id2 = idx2 % block_size;
            if ((idx1 > idx2 && block_id1 <= block_id2) ||
                (idx1 < idx2 &&
                 block_id1 < block_id2)) {  // Avoid duplicate pairs
              image_pairs.emplace_back(image_ids[idx1], image_ids[idx2]);
            }
          }
        }

        DatabaseTransaction database_transaction(&database_);
        matcher_.Match(image_pairs);

        PrintElapsedTime(timer);
      }
    }

    GetTimer().PrintMinutes();
  }

  const ExhaustiveMatchingOptions options_;
  const SiftMatchingOptions matching_options_;
  Database database_;
  FeatureMatcherCache cache_;
  FeatureMatcherController matcher_;
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
  return std::make_unique<ExhaustiveFeatureMatcher>(
      options, matching_options, geometry_options, database_path);
}

namespace {

class SequentialFeatureMatcher : public Thread {
 public:
  SequentialFeatureMatcher(const SequentialMatchingOptions& options,
                           const SiftMatchingOptions& matching_options,
                           const TwoViewGeometryOptions& geometry_options,
                           const std::string& database_path)
      : options_(options),
        matching_options_(matching_options),
        database_(database_path),
        cache_(std::max(5 * options_.loop_detection_num_images,
                        5 * options_.overlap),
               &database_),
        matcher_(matching_options, geometry_options, &database_, &cache_) {
    CHECK(options.Check());
    CHECK(matching_options.Check());
    CHECK(geometry_options.Check());
  }

 private:
  void Run() override {
    PrintHeading1("Sequential feature matching");

    if (!matcher_.Setup()) {
      return;
    }

    cache_.Setup();

    const std::vector<image_t> ordered_image_ids = GetOrderedImageIds();

    RunSequentialMatching(ordered_image_ids);
    if (options_.loop_detection) {
      RunLoopDetection(ordered_image_ids);
    }

    GetTimer().PrintMinutes();
  }

  std::vector<image_t> GetOrderedImageIds() const {
    const std::vector<image_t> image_ids = cache_.GetImageIds();

    std::vector<Image> ordered_images;
    ordered_images.reserve(image_ids.size());
    for (const auto image_id : image_ids) {
      ordered_images.push_back(cache_.GetImage(image_id));
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

  void RunSequentialMatching(const std::vector<image_t>& image_ids) {
    std::vector<std::pair<image_t, image_t>> image_pairs;
    image_pairs.reserve(options_.overlap);

    for (size_t image_idx1 = 0; image_idx1 < image_ids.size(); ++image_idx1) {
      if (IsStopped()) {
        return;
      }

      const auto image_id1 = image_ids.at(image_idx1);

      Timer timer;
      timer.Start();

      LOG(INFO) << StringPrintf("Matching image [%d/%d]",
                                image_idx1 + 1,
                                image_ids.size())
                << std::flush;

      image_pairs.clear();
      for (int i = 0; i < options_.overlap; ++i) {
        const size_t image_idx2 = image_idx1 + i;
        if (image_idx2 < image_ids.size()) {
          image_pairs.emplace_back(image_id1, image_ids.at(image_idx2));
          if (options_.quadratic_overlap) {
            const size_t image_idx2_quadratic = image_idx1 + (1 << i);
            if (image_idx2_quadratic < image_ids.size()) {
              image_pairs.emplace_back(image_id1,
                                       image_ids.at(image_idx2_quadratic));
            }
          }
        } else {
          break;
        }
      }

      DatabaseTransaction database_transaction(&database_);
      matcher_.Match(image_pairs);

      PrintElapsedTime(timer);
    }
  }

  void RunLoopDetection(const std::vector<image_t>& image_ids) {
    // Read the pre-trained vocabulary tree from disk.
    retrieval::VisualIndex<> visual_index;
    visual_index.Read(options_.vocab_tree_path);

    // Index all images in the visual index.
    IndexImagesInVisualIndex(matching_options_.num_threads,
                             options_.loop_detection_num_checks,
                             options_.loop_detection_max_num_features,
                             image_ids,
                             this,
                             &cache_,
                             &visual_index);

    if (IsStopped()) {
      return;
    }

    // Only perform loop detection for every n-th image.
    std::vector<image_t> match_image_ids;
    for (size_t i = 0; i < image_ids.size();
         i += options_.loop_detection_period) {
      match_image_ids.push_back(image_ids[i]);
    }

    MatchNearestNeighborsInVisualIndex(
        matching_options_.num_threads,
        options_.loop_detection_num_images,
        options_.loop_detection_num_nearest_neighbors,
        options_.loop_detection_num_checks,
        options_.loop_detection_num_images_after_verification,
        options_.loop_detection_max_num_features,
        match_image_ids,
        this,
        &cache_,
        &visual_index,
        &matcher_);
  }

  const SequentialMatchingOptions options_;
  const SiftMatchingOptions matching_options_;
  Database database_;
  FeatureMatcherCache cache_;
  FeatureMatcherController matcher_;
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

std::unique_ptr<Thread> CreateSequentialFeatureMatcher(
    const SequentialMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return std::make_unique<SequentialFeatureMatcher>(
      options, matching_options, geometry_options, database_path);
}

namespace {

class VocabTreeFeatureMatcher : public Thread {
 public:
  VocabTreeFeatureMatcher(const VocabTreeMatchingOptions& options,
                          const SiftMatchingOptions& matching_options,
                          const TwoViewGeometryOptions& geometry_options,
                          const std::string& database_path)
      : options_(options),
        matching_options_(matching_options),
        database_(database_path),
        cache_(5 * options_.num_images, &database_),
        matcher_(matching_options, geometry_options, &database_, &cache_) {
    CHECK(options.Check());
    CHECK(matching_options.Check());
    CHECK(geometry_options.Check());
  }

 private:
  void Run() override {
    PrintHeading1("Vocabulary tree feature matching");

    if (!matcher_.Setup()) {
      return;
    }

    cache_.Setup();

    // Read the pre-trained vocabulary tree from disk.
    retrieval::VisualIndex<> visual_index;
    visual_index.Read(options_.vocab_tree_path);

    const std::vector<image_t> all_image_ids = cache_.GetImageIds();
    std::vector<image_t> image_ids;
    if (options_.match_list_path == "") {
      image_ids = cache_.GetImageIds();
    } else {
      // Map image names to image identifiers.
      std::unordered_map<std::string, image_t> image_name_to_image_id;
      image_name_to_image_id.reserve(all_image_ids.size());
      for (const auto image_id : all_image_ids) {
        const auto& image = cache_.GetImage(image_id);
        image_name_to_image_id.emplace(image.Name(), image_id);
      }

      // Read the match list path.
      std::ifstream file(options_.match_list_path);
      CHECK(file.is_open()) << options_.match_list_path;
      std::string line;
      while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
          continue;
        }

        if (image_name_to_image_id.count(line) == 0) {
          LOG(ERROR) << "Image " << line << " does not exist.";
        } else {
          image_ids.push_back(image_name_to_image_id.at(line));
        }
      }
    }

    // Index all images in the visual index.
    IndexImagesInVisualIndex(matching_options_.num_threads,
                             options_.num_checks,
                             options_.max_num_features,
                             all_image_ids,
                             this,
                             &cache_,
                             &visual_index);

    if (IsStopped()) {
      GetTimer().PrintMinutes();
      return;
    }

    // Match all images in the visual index.
    MatchNearestNeighborsInVisualIndex(matching_options_.num_threads,
                                       options_.num_images,
                                       options_.num_nearest_neighbors,
                                       options_.num_checks,
                                       options_.num_images_after_verification,
                                       options_.max_num_features,
                                       image_ids,
                                       this,
                                       &cache_,
                                       &visual_index,
                                       &matcher_);

    GetTimer().PrintMinutes();
  }

  const VocabTreeMatchingOptions options_;
  const SiftMatchingOptions matching_options_;
  Database database_;
  FeatureMatcherCache cache_;
  FeatureMatcherController matcher_;
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
  return std::make_unique<VocabTreeFeatureMatcher>(
      options, matching_options, geometry_options, database_path);
}

namespace {

class SpatialFeatureMatcher : public Thread {
 public:
  SpatialFeatureMatcher(const SpatialMatchingOptions& options,
                        const SiftMatchingOptions& matching_options,
                        const TwoViewGeometryOptions& geometry_options,
                        const std::string& database_path)
      : options_(options),
        matching_options_(matching_options),
        database_(database_path),
        cache_(5 * options_.max_num_neighbors, &database_),
        matcher_(matching_options, geometry_options, &database_, &cache_) {
    CHECK(options.Check());
    CHECK(matching_options.Check());
    CHECK(geometry_options.Check());
  }

 private:
  void Run() override {
    PrintHeading1("Spatial feature matching");

    if (!matcher_.Setup()) {
      return;
    }

    cache_.Setup();

    const std::vector<image_t> image_ids = cache_.GetImageIds();

    //////////////////////////////////////////////////////////////////////////////
    // Spatial indexing
    //////////////////////////////////////////////////////////////////////////////

    Timer timer;
    timer.Start();

    LOG(INFO) << "Indexing images..." << std::flush;

    GPSTransform gps_transform;

    size_t num_locations = 0;
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> location_matrix(
        image_ids.size(), 3);

    std::vector<size_t> location_idxs;
    location_idxs.reserve(image_ids.size());

    std::vector<Eigen::Vector3d> ells(1);

    for (size_t i = 0; i < image_ids.size(); ++i) {
      const auto image_id = image_ids[i];
      const auto& image = cache_.GetImage(image_id);
      const Eigen::Vector3d& translation_prior =
          image.CamFromWorldPrior().translation;

      if ((translation_prior(0) == 0 && translation_prior(1) == 0 &&
           options_.ignore_z) ||
          (translation_prior(0) == 0 && translation_prior(1) == 0 &&
           translation_prior(2) == 0 && !options_.ignore_z)) {
        continue;
      }

      location_idxs.push_back(i);

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

    PrintElapsedTime(timer);

    if (num_locations == 0) {
      LOG(INFO) << "=> No images with location data.";
      GetTimer().PrintMinutes();
      return;
    }

    //////////////////////////////////////////////////////////////////////////////
    // Building spatial index
    //////////////////////////////////////////////////////////////////////////////

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

    Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        index_matrix(num_locations, knn);
    flann::Matrix<size_t> indices(index_matrix.data(), num_locations, knn);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        distance_matrix(num_locations, knn);
    flann::Matrix<float> distances(distance_matrix.data(), num_locations, knn);

    flann::SearchParams search_params(flann::FLANN_CHECKS_AUTOTUNED);
    if (matching_options_.num_threads == ThreadPool::kMaxNumThreads) {
      search_params.cores = std::thread::hardware_concurrency();
    } else {
      search_params.cores = matching_options_.num_threads;
    }
    if (search_params.cores <= 0) {
      search_params.cores = 1;
    }

    search_index.knnSearch(locations, indices, distances, knn, search_params);

    PrintElapsedTime(timer);

    //////////////////////////////////////////////////////////////////////////////
    // Matching
    //////////////////////////////////////////////////////////////////////////////

    const float max_distance =
        static_cast<float>(options_.max_distance * options_.max_distance);

    std::vector<std::pair<image_t, image_t>> image_pairs;
    image_pairs.reserve(knn);

    for (size_t i = 0; i < num_locations; ++i) {
      if (IsStopped()) {
        GetTimer().PrintMinutes();
        return;
      }

      timer.Restart();

      LOG(INFO) << StringPrintf("Matching image [%d/%d]", i + 1, num_locations)
                << std::flush;

      image_pairs.clear();

      for (int j = 0; j < knn; ++j) {
        // Check if query equals result.
        if (index_matrix(i, j) == i) {
          continue;
        }

        // Since the nearest neighbors are sorted by distance, we can break.
        if (distance_matrix(i, j) > max_distance) {
          break;
        }

        const size_t idx = location_idxs[i];
        const image_t image_id = image_ids.at(idx);
        const size_t nn_idx = location_idxs.at(index_matrix(i, j));
        const image_t nn_image_id = image_ids.at(nn_idx);
        image_pairs.emplace_back(image_id, nn_image_id);
      }

      DatabaseTransaction database_transaction(&database_);
      matcher_.Match(image_pairs);

      PrintElapsedTime(timer);
    }

    GetTimer().PrintMinutes();
  }

  const SpatialMatchingOptions options_;
  const SiftMatchingOptions matching_options_;
  Database database_;
  FeatureMatcherCache cache_;
  FeatureMatcherController matcher_;
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
  return std::make_unique<SpatialFeatureMatcher>(
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
    CHECK(options.Check());
    CHECK(matching_options.Check());
    CHECK(geometry_options.Check());
  }

 private:
  void Run() override {
    PrintHeading1("Transitive feature matching");

    if (!matcher_.Setup()) {
      return;
    }

    cache_.Setup();

    const std::vector<image_t> image_ids = cache_.GetImageIds();

    std::vector<std::pair<image_t, image_t>> image_pairs;
    std::unordered_set<image_pair_t> image_pair_ids;

    for (int iteration = 0; iteration < options_.num_iterations; ++iteration) {
      if (IsStopped()) {
        GetTimer().PrintMinutes();
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

      CHECK_EQ(existing_image_pairs.size(), existing_num_inliers.size());

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
                    GetTimer().PrintMinutes();
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

    GetTimer().PrintMinutes();
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

class ImagePairsFeatureMatcher : public Thread {
 public:
  ImagePairsFeatureMatcher(const ImagePairsMatchingOptions& options,
                           const SiftMatchingOptions& matching_options,
                           const TwoViewGeometryOptions& geometry_options,
                           const std::string& database_path)
      : options_(options),
        matching_options_(matching_options),
        database_(database_path),
        cache_(options.block_size, &database_),
        matcher_(matching_options, geometry_options, &database_, &cache_) {
    CHECK(options.Check());
    CHECK(matching_options.Check());
    CHECK(geometry_options.Check());
  }

 private:
  void Run() override {
    PrintHeading1("Custom feature matching");

    if (!matcher_.Setup()) {
      return;
    }

    cache_.Setup();

    //////////////////////////////////////////////////////////////////////////////
    // Reading image pairs list
    //////////////////////////////////////////////////////////////////////////////

    std::unordered_map<std::string, image_t> image_name_to_image_id;
    image_name_to_image_id.reserve(cache_.GetImageIds().size());
    for (const auto image_id : cache_.GetImageIds()) {
      const auto& image = cache_.GetImage(image_id);
      image_name_to_image_id.emplace(image.Name(), image_id);
    }

    std::ifstream file(options_.match_list_path);
    CHECK(file.is_open()) << options_.match_list_path;

    std::string line;
    std::vector<std::pair<image_t, image_t>> image_pairs;
    std::unordered_set<colmap::image_pair_t> image_pairs_set;
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

    //////////////////////////////////////////////////////////////////////////////
    // Feature matching
    //////////////////////////////////////////////////////////////////////////////

    const size_t num_match_blocks =
        image_pairs.size() / options_.block_size + 1;
    std::vector<std::pair<image_t, image_t>> block_image_pairs;
    block_image_pairs.reserve(options_.block_size);

    for (size_t i = 0; i < image_pairs.size(); i += options_.block_size) {
      if (IsStopped()) {
        GetTimer().PrintMinutes();
        return;
      }

      Timer timer;
      timer.Start();

      LOG(INFO) << StringPrintf("Matching block [%d/%d]",
                                i / options_.block_size + 1,
                                num_match_blocks)
                << std::flush;

      const size_t block_end = i + options_.block_size <= image_pairs.size()
                                   ? i + options_.block_size
                                   : image_pairs.size();
      std::vector<std::pair<image_t, image_t>> block_image_pairs;
      block_image_pairs.reserve(options_.block_size);
      for (size_t j = i; j < block_end; ++j) {
        block_image_pairs.push_back(image_pairs[j]);
      }

      DatabaseTransaction database_transaction(&database_);
      matcher_.Match(block_image_pairs);

      PrintElapsedTime(timer);
    }

    GetTimer().PrintMinutes();
  }

  const ImagePairsMatchingOptions options_;
  const SiftMatchingOptions matching_options_;
  Database database_;
  FeatureMatcherCache cache_;
  FeatureMatcherController matcher_;
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
  return std::make_unique<ImagePairsFeatureMatcher>(
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
    CHECK(options.Check());
    CHECK(matching_options.Check());
    CHECK(geometry_options.Check());
  }

 private:
  const static size_t kCacheSize = 100;

  void Run() override {
    PrintHeading1("Importing matches");

    cache_.Setup();

    std::unordered_map<std::string, const Image*> image_name_to_image;
    image_name_to_image.reserve(cache_.GetImageIds().size());
    for (const auto image_id : cache_.GetImageIds()) {
      const auto& image = cache_.GetImage(image_id);
      image_name_to_image.emplace(image.Name(), &image);
    }

    std::ifstream file(options_.match_list_path);
    CHECK(file.is_open()) << options_.match_list_path;

    std::string line;
    while (std::getline(file, line)) {
      if (IsStopped()) {
        GetTimer().PrintMinutes();
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

    GetTimer().PrintMinutes();
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
