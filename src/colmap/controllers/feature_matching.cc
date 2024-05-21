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
#include "colmap/controllers/pairing.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/utils.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <fstream>
#include <numeric>

namespace colmap {
namespace {

void PrintElapsedTime(const Timer& timer) {
  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
}

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
            DerivedPairGenerator::CacheSize(pair_options_), database_)),
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
        database_(new Database(database_path)),
        cache_(new FeatureMatcherCache(options_.batch_size, database_)),
        matcher_(
            matching_options, geometry_options, database_.get(), cache_.get()) {
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

    cache_->Setup();

    const std::vector<image_t> image_ids = cache_->GetImageIds();

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
      database_->ReadTwoViewGeometryNumInliers(&existing_image_pairs,
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
                  DatabaseTransaction database_transaction(database_.get());
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
      DatabaseTransaction database_transaction(database_.get());
      matcher_.Match(image_pairs);
      PrintElapsedTime(timer);
    }

    run_timer.PrintMinutes();
  }

  const TransitiveMatchingOptions options_;
  const SiftMatchingOptions matching_options_;
  const std::shared_ptr<Database> database_;
  const std::shared_ptr<FeatureMatcherCache> cache_;
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
        database_(new Database(database_path)),
        cache_(new FeatureMatcherCache(kCacheSize, database_)) {
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

    cache_->Setup();

    std::unordered_map<std::string, const Image*> image_name_to_image;
    image_name_to_image.reserve(cache_->GetImageIds().size());
    for (const auto image_id : cache_->GetImageIds()) {
      const auto& image = cache_->GetImage(image_id);
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
      if (database_->ExistsInlierMatches(image1.ImageId(), image2.ImageId())) {
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

      const Camera& camera1 = cache_->GetCamera(image1.CameraId());
      const Camera& camera2 = cache_->GetCamera(image2.CameraId());

      if (options_.verify_matches) {
        database_->WriteMatches(image1.ImageId(), image2.ImageId(), matches);

        const auto keypoints1 = cache_->GetKeypoints(image1.ImageId());
        const auto keypoints2 = cache_->GetKeypoints(image2.ImageId());

        TwoViewGeometry two_view_geometry =
            EstimateTwoViewGeometry(camera1,
                                    FeatureKeypointsToPointsVector(*keypoints1),
                                    camera2,
                                    FeatureKeypointsToPointsVector(*keypoints2),
                                    matches,
                                    geometry_options_);

        database_->WriteTwoViewGeometry(
            image1.ImageId(), image2.ImageId(), two_view_geometry);
      } else {
        TwoViewGeometry two_view_geometry;

        if (camera1.has_prior_focal_length && camera2.has_prior_focal_length) {
          two_view_geometry.config = TwoViewGeometry::CALIBRATED;
        } else {
          two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
        }

        two_view_geometry.inlier_matches = matches;

        database_->WriteTwoViewGeometry(
            image1.ImageId(), image2.ImageId(), two_view_geometry);
      }
    }

    run_timer.PrintMinutes();
  }

  const FeaturePairsMatchingOptions options_;
  const SiftMatchingOptions matching_options_;
  const TwoViewGeometryOptions geometry_options_;
  const std::shared_ptr<Database> database_;
  const std::shared_ptr<FeatureMatcherCache> cache_;
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
