// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "base/feature_matching.h"

#include <fstream>
#include <numeric>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include "base/camera_models.h"
#include "base/database.h"
#include "base/gps.h"
#include "base/vocabulary_tree.h"
#include "estimators/essential_matrix.h"
#include "estimators/two_view_geometry.h"
#include "ext/ANNfloat/ANN.h"
#include "optim/ransac.h"
#include "util/misc.h"

namespace colmap {
namespace {

FeatureDescriptors ExtractTopScaleDescriptors(
    const FeatureKeypoints& keypoints, const FeatureDescriptors& descriptors,
    const size_t num_features) {
  FeatureDescriptors top_scale_descriptors;

  if (static_cast<size_t>(descriptors.rows()) <= num_features) {
    top_scale_descriptors = descriptors;
  } else {
    std::vector<std::pair<size_t, float>> scales;
    scales.reserve(static_cast<size_t>(keypoints.size()));
    for (size_t i = 0; i < keypoints.size(); ++i) {
      scales.emplace_back(i, keypoints[i].scale);
    }

    std::partial_sort(scales.begin(), scales.begin() + num_features,
                      scales.end(), [](const std::pair<size_t, float> scale1,
                                       const std::pair<size_t, float> scale2) {
                        return scale1.second > scale2.second;
                      });

    top_scale_descriptors.resize(num_features, descriptors.cols());
    for (size_t i = 0; i < num_features; ++i) {
      top_scale_descriptors.row(i) = descriptors.row(scales[i].first);
    }
  }

  return top_scale_descriptors;
}

}  // namespace

void FeatureMatcher::Options::Check() const {
  CHECK_GE(gpu_index, -1);
  CHECK_GT(max_ratio, 0.0);
  CHECK_GT(max_distance, 0.0);
  CHECK_GT(max_error, 0.0);
  CHECK_GT(max_num_trials, 0);
  CHECK_GE(min_inlier_ratio, 0);
  CHECK_LE(min_inlier_ratio, 1);
  CHECK_GE(min_num_inliers, 0);
}

FeatureMatcher::FeatureMatcher(const Options& options,
                               const std::string& database_path)
    : stop_(false),
      options_(options),
      database_path_(database_path),
      parent_thread_(QThread::currentThread()) {
  options_.Check();
#ifdef CUDA_ENABLED
  if (options_.gpu_index < 0) {
#endif
    surface_ = new QOffscreenSurface();
    surface_->create();
    context_ = new QOpenGLContext();
    context_->create();
    context_->makeCurrent(surface_);
    context_->doneCurrent();
    context_->moveToThread(this);
#ifdef CUDA_ENABLED
  }
#endif
}

FeatureMatcher::~FeatureMatcher() {
#ifdef CUDA_ENABLED
  if (options_.gpu_index < 0) {
#endif
    delete context_;
    surface_->deleteLater();
#ifdef CUDA_ENABLED
  }
#endif
}

void FeatureMatcher::Stop() {
  QMutexLocker locker(&stop_mutex_);
  stop_ = true;
}

void FeatureMatcher::run() {
  total_timer_.Restart();

  SetupData();
  SetupWorkers();
  DoMatching();

  total_timer_.PrintMinutes();

#ifdef CUDA_ENABLED
  if (options_.gpu_index < 0) {
#endif
    context_->doneCurrent();
    context_->moveToThread(parent_thread_);
    database_.Close();
    delete sift_gpu_;
    delete sift_match_gpu_;
    delete verifier_thread_pool_;
#ifdef CUDA_ENABLED
  }
#endif
}

void FeatureMatcher::SetupWorkers() {
#ifdef CUDA_ENABLED
  if (options_.gpu_index < 0) {
#endif
    context_->makeCurrent(surface_);
#ifdef CUDA_ENABLED
  }
#endif

  sift_gpu_ = new SiftGPU();
  sift_match_gpu_ = new SiftMatchGPU(options_.max_num_matches);

  sift_gpu_->SetVerbose(0);
#ifdef CUDA_ENABLED
  if (options_.gpu_index >= 0) {
    sift_match_gpu_->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA_DEVICE0 +
                                 options_.gpu_index);
  } else {
    sift_match_gpu_->SetLanguage(SiftMatchGPU::SIFTMATCH_GLSL);
  }
#else  // CUDA_ENABLED
    sift_match_gpu_->SetLanguage(SiftMatchGPU::SIFTMATCH_GLSL);
#endif  // CUDA_ENABLED

  if (sift_match_gpu_->VerifyContextGL() == 0) {
    std::cerr << "ERROR: SiftMatchGPU not fully supported." << std::endl;
    return;
  }

  // Setup geometric verification workers.
  verifier_thread_pool_ = new ThreadPool(options_.num_threads);
}

void FeatureMatcher::SetupData() {
  database_.Open(database_path_);

  const std::vector<Camera> cameras = database_.ReadAllCameras();
  cameras_.clear();
  cameras_.reserve(cameras.size());
  for (const Camera& camera : cameras) {
    cameras_.emplace(camera.CameraId(), camera);
  }

  const std::vector<Image> images = database_.ReadAllImages();
  images_.clear();
  images_.reserve(images.size());
  for (const Image& image : images) {
    images_.emplace(image.ImageId(), image);
  }
}

bool FeatureMatcher::IsStopped() {
  QMutexLocker locker(&stop_mutex_);
  return stop_;
}

void FeatureMatcher::PrintElapsedTime(const Timer& timer) {
  std::cout << boost::format(" in %.3fs") % timer.ElapsedSeconds() << std::endl;
}

const FeatureKeypoints& FeatureMatcher::CacheKeypoints(const image_t image_id) {
  if (keypoints_cache_.count(image_id) == 0) {
    keypoints_cache_[image_id] = database_.ReadKeypoints(image_id);
  }
  return keypoints_cache_.at(image_id);
}

const FeatureDescriptors& FeatureMatcher::CacheDescriptors(
    const image_t image_id) {
  if (descriptors_cache_.count(image_id) == 0) {
    descriptors_cache_[image_id] = database_.ReadDescriptors(image_id);
  }
  return descriptors_cache_.at(image_id);
}

void FeatureMatcher::CleanCache(
    const std::unordered_set<image_t>& keep_image_ids) {
  for (auto it = keypoints_cache_.begin(); it != keypoints_cache_.end();) {
    if (keep_image_ids.count(it->first) == 0) {
      it = keypoints_cache_.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = descriptors_cache_.begin(); it != descriptors_cache_.end();) {
    if (keep_image_ids.count(it->first) == 0) {
      it = descriptors_cache_.erase(it);
    } else {
      ++it;
    }
  }
}

void FeatureMatcher::MatchImagePairs(
    const std::vector<std::pair<image_t, image_t>>& image_pairs) {
  //////////////////////////////////////////////////////////////////////////////
  // Load data from database
  //////////////////////////////////////////////////////////////////////////////

  std::vector<std::pair<bool, bool>> exists_mask;
  exists_mask.reserve(image_pairs.size());
  std::unordered_set<image_t> image_ids;
  image_ids.reserve(image_pairs.size());
  std::unordered_set<image_pair_t> pair_ids;
  pair_ids.reserve(image_pairs.size());

  bool exists_all = true;

  database_.BeginTransaction();

  for (const auto image_pair : image_pairs) {
    // Avoid self-matches.
    if (image_pair.first == image_pair.second) {
      exists_mask.emplace_back(true, true);
      continue;
    }

    // Avoid duplicate image pairs.
    const image_pair_t pair_id =
        Database::ImagePairToPairId(image_pair.first, image_pair.second);
    if (pair_ids.count(pair_id) > 0) {
      exists_mask.emplace_back(true, true);
      continue;
    }

    pair_ids.insert(pair_id);

    const bool exists_matches =
        database_.ExistsMatches(image_pair.first, image_pair.second);
    const bool exists_inlier_matches =
        database_.ExistsInlierMatches(image_pair.first, image_pair.second);

    exists_all = exists_all && exists_matches && exists_inlier_matches;
    exists_mask.emplace_back(exists_matches, exists_inlier_matches);

    if (!exists_matches || !exists_inlier_matches) {
      image_ids.insert(image_pair.first);
      image_ids.insert(image_pair.second);
    }

    if (!exists_matches) {
      CacheDescriptors(image_pair.first);
      CacheDescriptors(image_pair.second);
    }

    if (!exists_inlier_matches) {
      CacheKeypoints(image_pair.first);
      CacheKeypoints(image_pair.second);
    }
  }

  database_.EndTransaction();

  if (exists_all) {
    return;
  }

  CleanCache(image_ids);

  //////////////////////////////////////////////////////////////////////////////
  // Feature matching and geometric verification
  //////////////////////////////////////////////////////////////////////////////

  image_t prev_image_id1 = kInvalidImageId;
  image_t prev_image_id2 = kInvalidImageId;

  const size_t min_num_inliers = static_cast<size_t>(options_.min_num_inliers);

  std::vector<int> matches_buffer(
      static_cast<size_t>(2 * options_.max_num_matches));

  struct MatchResult {
    image_t image_id1;
    image_t image_id2;
    FeatureMatches matches;
    bool write;
  };

  std::vector<MatchResult> match_results;
  match_results.reserve(image_pairs.size());

  std::vector<std::future<TwoViewGeometry>> verification_results;
  verification_results.reserve(image_pairs.size());
  std::vector<std::pair<image_t, image_t>> verification_image_pairs;
  verification_image_pairs.reserve(image_pairs.size());

  std::vector<std::pair<image_t, image_t>> empty_verification_results;

  TwoViewGeometry::Options two_view_geometry_options;
  two_view_geometry_options.min_num_inliers =
      static_cast<size_t>(options_.min_num_inliers);
  two_view_geometry_options.ransac_options.max_error = options_.max_error;
  two_view_geometry_options.ransac_options.confidence = options_.confidence;
  two_view_geometry_options.ransac_options.max_num_trials =
      static_cast<size_t>(options_.max_num_trials);
  two_view_geometry_options.ransac_options.min_inlier_ratio =
      options_.min_inlier_ratio;

  CHECK_EQ(image_pairs.size(), exists_mask.size());

  for (size_t i = 0; i < image_pairs.size(); ++i) {
    const auto exists = exists_mask[i];

    if (exists.first && exists.second) {
      continue;
    }

    const auto image_pair = image_pairs[i];
    const image_t image_id1 = image_pair.first;
    const image_t image_id2 = image_pair.second;

    ////////////////////////////////////////////////////////////////////////////
    // Feature matching
    ////////////////////////////////////////////////////////////////////////////

    match_results.emplace_back();
    auto& match_result = match_results.back();

    match_result.image_id1 = image_id1;
    match_result.image_id2 = image_id2;

    if (exists.first) {
      match_result.matches = database_.ReadMatches(image_id1, image_id2);
      match_result.write = false;
    } else {
      if (image_id1 != prev_image_id1) {
        const FeatureDescriptors& descriptors1 =
            descriptors_cache_.at(image_id1);
        sift_match_gpu_->SetDescriptors(0, descriptors1.rows(),
                                        descriptors1.data());
      }
      if (image_id2 != prev_image_id2) {
        const FeatureDescriptors& descriptors2 =
            descriptors_cache_.at(image_id2);
        sift_match_gpu_->SetDescriptors(1, descriptors2.rows(),
                                        descriptors2.data());
      }

      prev_image_id1 = image_id1;
      prev_image_id2 = image_id2;

      const int num_matches = sift_match_gpu_->GetSiftMatch(
          options_.max_num_matches,
          reinterpret_cast<int(*)[2]>(matches_buffer.data()),
          static_cast<float>(options_.max_distance),
          static_cast<float>(options_.max_ratio), options_.cross_check);

      if (num_matches >= options_.min_num_inliers) {
        match_result.matches.resize(num_matches);
        for (int j = 0; j < num_matches; ++j) {
          match_result.matches[j].point2D_idx1 =
              static_cast<point2D_t>(matches_buffer[2 * j]);
          match_result.matches[j].point2D_idx2 =
              static_cast<point2D_t>(matches_buffer[2 * j + 1]);
        }
      } else {
        match_result.matches = {};
      }

      match_result.write = true;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Geometric verification
    ////////////////////////////////////////////////////////////////////////////

    if (!exists.second) {
      if (match_result.matches.size() >= min_num_inliers) {
        GeometricVerificationData data;
        data.camera1 = &cameras_[images_[image_id1].CameraId()];
        data.camera2 = &cameras_[images_[image_id2].CameraId()];
        data.keypoints1 = &keypoints_cache_.at(image_id1);
        data.keypoints2 = &keypoints_cache_.at(image_id2);
        data.matches = &match_result.matches;
        data.options = &two_view_geometry_options;
        std::function<TwoViewGeometry(FeatureMatcher::GeometricVerificationData,
                                      bool)>
            verifier_func = FeatureMatcher::VerifyImagePair;
        verification_results.push_back(verifier_thread_pool_->AddTask(
            verifier_func, data, options_.multiple_models));
        verification_image_pairs.push_back(image_pair);
      } else {
        empty_verification_results.push_back(image_pair);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Write results
  //////////////////////////////////////////////////////////////////////////////

  database_.BeginTransaction();

  for (const auto& result : match_results) {
    if (result.write) {
      database_.WriteMatches(result.image_id1, result.image_id2,
                             result.matches);
    }
  }

  for (size_t i = 0; i < verification_results.size(); ++i) {
    const auto& image_pair = verification_image_pairs[i];
    const auto result = verification_results[i].get();
    if (result.inlier_matches.size() >= min_num_inliers) {
      database_.WriteInlierMatches(image_pair.first, image_pair.second, result);
    } else {
      database_.WriteInlierMatches(image_pair.first, image_pair.second,
                                   TwoViewGeometry());
    }
  }

  for (auto& result : empty_verification_results) {
    database_.WriteInlierMatches(result.first, result.second,
                                 TwoViewGeometry());
  }

  database_.EndTransaction();
}

TwoViewGeometry FeatureMatcher::VerifyImagePair(
    const GeometricVerificationData data, const bool multiple_models) {
  TwoViewGeometry two_view_geometry;
  const auto points1 = FeatureKeypointsToPointsVector(*data.keypoints1);
  const auto points2 = FeatureKeypointsToPointsVector(*data.keypoints2);
  if (multiple_models) {
    two_view_geometry.EstimateMultiple(*data.camera1, points1, *data.camera2,
                                       points2, *data.matches, *data.options);
  } else {
    two_view_geometry.Estimate(*data.camera1, points1, *data.camera2, points2,
                               *data.matches, *data.options);
  }
  return two_view_geometry;
}

void ExhaustiveFeatureMatcher::ExhaustiveOptions::Check() const {
  CHECK_GT(block_size, 1);
  CHECK_GT(preemptive_num_features, 0);
  CHECK_GE(preemptive_min_num_matches, 0);
  CHECK_LE(preemptive_min_num_matches, preemptive_num_features);
}

ExhaustiveFeatureMatcher::ExhaustiveFeatureMatcher(
    const Options& options, const ExhaustiveOptions& exhaustive_options,
    const std::string& database_path)
    : FeatureMatcher(options, database_path),
      exhaustive_options_(exhaustive_options) {
  exhaustive_options_.Check();
}

void ExhaustiveFeatureMatcher::DoMatching() {
  PrintHeading1("Exhaustive feature matching");

  std::vector<image_t> image_ids;
  image_ids.reserve(images_.size());

  for (const auto image : images_) {
    image_ids.push_back(image.first);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Matching
  //////////////////////////////////////////////////////////////////////////////

  const size_t block_size = static_cast<size_t>(exhaustive_options_.block_size);
  const size_t num_blocks = static_cast<size_t>(
      std::ceil(static_cast<double>(image_ids.size()) / block_size));

  std::vector<std::pair<image_t, image_t>> image_pairs;

  for (size_t start_idx1 = 0; start_idx1 < image_ids.size();
       start_idx1 += block_size) {
    const size_t end_idx1 =
        std::min(image_ids.size(), start_idx1 + block_size) - 1;
    for (size_t start_idx2 = 0; start_idx2 < image_ids.size();
         start_idx2 += block_size) {
      const size_t end_idx2 =
          std::min(image_ids.size(), start_idx2 + block_size) - 1;

      if (IsStopped()) {
        return;
      }

      Timer timer;
      timer.Start();

      std::cout << boost::format("Matching block [%d/%d, %d/%d]") %
                       (start_idx1 / block_size + 1) % num_blocks %
                       (start_idx2 / block_size + 1) % num_blocks
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

      if (exhaustive_options_.preemptive) {
        image_pairs = PreemptivelyFilterImagePairs(image_pairs);
      }

      MatchImagePairs(image_pairs);
      PrintElapsedTime(timer);
    }
  }
}

std::vector<std::pair<image_t, image_t>>
ExhaustiveFeatureMatcher::PreemptivelyFilterImagePairs(
    const std::vector<std::pair<image_t, image_t>>& image_pairs) {
  const size_t num_features =
      static_cast<size_t>(exhaustive_options_.preemptive_num_features);

  std::unordered_map<image_t, FeatureDescriptors> top_descriptors;

  image_t prev_image_id1 = kInvalidImageId;
  image_t prev_image_id2 = kInvalidImageId;

  FeatureMatches matches_buffer(static_cast<size_t>(options_.max_num_matches));

  std::vector<std::pair<image_t, image_t>> filtered_image_pairs;

  database_.BeginTransaction();

  for (const auto image_pair : image_pairs) {
    if (top_descriptors.count(image_pair.first) == 0) {
      top_descriptors.emplace(
          image_pair.first,
          ExtractTopScaleDescriptors(CacheKeypoints(image_pair.first),
                                     CacheDescriptors(image_pair.first),
                                     num_features));
    }
    if (top_descriptors.count(image_pair.second) == 0) {
      top_descriptors.emplace(
          image_pair.second,
          ExtractTopScaleDescriptors(CacheKeypoints(image_pair.second),
                                     CacheDescriptors(image_pair.second),
                                     num_features));
    }

    if (image_pair.first != prev_image_id1) {
      const auto& descriptors1 = top_descriptors[image_pair.first];
      sift_match_gpu_->SetDescriptors(0, descriptors1.rows(),
                                      descriptors1.data());
      prev_image_id1 = image_pair.first;
    }
    if (image_pair.second != prev_image_id2) {
      const auto& descriptors2 = top_descriptors[image_pair.second];
      sift_match_gpu_->SetDescriptors(1, descriptors2.rows(),
                                      descriptors2.data());
      prev_image_id2 = image_pair.second;
    }

    const int num_matches = sift_match_gpu_->GetSiftMatch(
        options_.max_num_matches, (int(*)[2])matches_buffer.data(),
        options_.max_distance, options_.max_ratio, options_.cross_check);

    if (num_matches >= exhaustive_options_.preemptive_min_num_matches) {
      filtered_image_pairs.push_back(image_pair);
    }
  }

  database_.EndTransaction();

  std::cout << boost::format(" P(%d/%d)") % filtered_image_pairs.size() %
                   image_pairs.size()
            << std::flush;

  return filtered_image_pairs;
}

void SequentialFeatureMatcher::SequentialOptions::Check() const {
  CHECK_GT(overlap, 0);
  CHECK_GT(loop_detection_period, 0);
  CHECK_GT(loop_detection_num_images, 0);
  if (loop_detection) {
    CHECK(boost::filesystem::exists(vocab_tree_path));
  }
}

SequentialFeatureMatcher::SequentialFeatureMatcher(
    const Options& options, const SequentialOptions& sequential_options,
    const std::string& database_path)
    : FeatureMatcher(options, database_path),
      sequential_options_(sequential_options) {
  sequential_options_.Check();
}

void SequentialFeatureMatcher::DoMatching() {
  PrintHeading1("Sequential feature matching");

  //////////////////////////////////////////////////////////////////////////////
  // Sequential matching
  //////////////////////////////////////////////////////////////////////////////

  std::vector<Image> ordered_images;
  ordered_images.reserve(images_.size());

  for (const auto& image : images_) {
    ordered_images.push_back(image.second);
  }

  // Make sure, images are ordered in sequential order
  std::sort(ordered_images.begin(), ordered_images.end(),
            [](const Image& image1, const Image& image2) {
              return image1.Name() < image2.Name();
            });

  std::vector<image_t> image_idxs_nh;

  std::vector<std::pair<image_t, image_t>> image_pairs;

  for (size_t i = 0; i < ordered_images.size(); ++i) {
    if (IsStopped()) {
      return;
    }

    Timer timer;
    timer.Start();

    const auto& image1 = ordered_images[i];

    std::cout << boost::format("Matching image [%d/%d]") % (i + 1) %
                     ordered_images.size()
              << std::flush;

    image_pairs.clear();

    // Iterate through all images in local neighborhood
    for (size_t j = 0; j < image_idxs_nh.size(); ++j) {
      const auto& image2 = ordered_images[image_idxs_nh[j]];
      image_pairs.emplace_back(image1.ImageId(), image2.ImageId());
    }

    MatchImagePairs(image_pairs);
    PrintElapsedTime(timer);

    // Remove "oldest" image in local neighborhood if overlap is exceeded
    if (image_idxs_nh.size() >
        static_cast<size_t>(sequential_options_.overlap)) {
      image_idxs_nh.erase(image_idxs_nh.begin());
    }

    // Add current image to neighborhood for next iteration
    image_idxs_nh.push_back(i);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Loop detection
  //////////////////////////////////////////////////////////////////////////////

  if (!sequential_options_.loop_detection) {
    return;
  }

  VocabularyTree vocab_tree;
  vocab_tree.Read(sequential_options_.vocab_tree_path);

  for (size_t i = 0; i < ordered_images.size(); ++i) {
    if (IsStopped()) {
      return;
    }

    Timer timer;
    timer.Start();

    const auto& image = ordered_images[i];

    std::cout << boost::format("Indexing image [%d/%d]") % (i + 1) %
                     ordered_images.size()
              << std::flush;

    const auto descritpors = database_.ReadDescriptors(image.ImageId());
    vocab_tree.Index(image.ImageId(), descritpors);

    PrintElapsedTime(timer);
  }

  vocab_tree.Prepare();

  const size_t loop_detection_num_images = std::min<size_t>(
      sequential_options_.loop_detection_num_images, ordered_images.size());

  for (size_t i = 0; i < ordered_images.size();
       i += sequential_options_.loop_detection_period) {
    if (IsStopped()) {
      return;
    }

    Timer timer;
    timer.Start();

    const auto& image = ordered_images[i];

    std::cout << boost::format("Detecting loops [%d/%d]") % (i + 1) %
                     ordered_images.size()
              << std::flush;

    auto descriptors = database_.ReadDescriptors(image.ImageId());
    descriptors_cache_[image.ImageId()] = descriptors;

    const auto retrievals =
        vocab_tree.Retrieve(descriptors, loop_detection_num_images);

    image_pairs.clear();
    for (const auto retrieval : retrievals) {
      image_pairs.emplace_back(image.ImageId(), retrieval.first);
    }

    MatchImagePairs(image_pairs);
    PrintElapsedTime(timer);
  }
}

void VocabTreeFeatureMatcher::VocabTreeOptions::Check() const {
  CHECK_GT(num_images, 0);
  CHECK(boost::filesystem::exists(vocab_tree_path));
}

VocabTreeFeatureMatcher::VocabTreeFeatureMatcher(
    const Options& options, const VocabTreeOptions& vocab_tree_options,
    const std::string& database_path)
    : FeatureMatcher(options, database_path),
      vocab_tree_options_(vocab_tree_options) {
  vocab_tree_options_.Check();
}

void VocabTreeFeatureMatcher::DoMatching() {
  PrintHeading1("Vocabulary tree feature matching");

  std::vector<Image> ordered_images;
  ordered_images.reserve(images_.size());

  for (const auto& image : images_) {
    ordered_images.push_back(image.second);
  }

  std::vector<std::pair<image_t, image_t>> image_pairs;

  VocabularyTree vocab_tree;
  vocab_tree.Read(vocab_tree_options_.vocab_tree_path);

  for (size_t i = 0; i < ordered_images.size(); ++i) {
    if (IsStopped()) {
      return;
    }

    Timer timer;
    timer.Start();

    const auto& image = ordered_images[i];

    std::cout << boost::format("Indexing image [%d/%d]") % (i + 1) %
                     ordered_images.size()
              << std::flush;

    const auto descritpors = database_.ReadDescriptors(image.ImageId());
    vocab_tree.Index(image.ImageId(), descritpors);

    PrintElapsedTime(timer);
  }

  vocab_tree.Prepare();

  const size_t num_images =
      std::min<image_t>(vocab_tree_options_.num_images, ordered_images.size());

  for (size_t i = 0; i < ordered_images.size(); ++i) {
    if (IsStopped()) {
      return;
    }

    Timer timer;
    timer.Start();

    const auto& image = ordered_images[i];

    std::cout << boost::format("Matching image [%d/%d]") % (i + 1) %
                     ordered_images.size()
              << std::flush;

    auto descriptors = database_.ReadDescriptors(image.ImageId());
    descriptors_cache_[image.ImageId()] = descriptors;

    const auto retrievals = vocab_tree.Retrieve(descriptors, num_images);

    image_pairs.clear();
    for (const auto retrieval : retrievals) {
      image_pairs.emplace_back(image.ImageId(), retrieval.first);
    }

    MatchImagePairs(image_pairs);
    PrintElapsedTime(timer);
  }
}

void SpatialFeatureMatcher::SpatialOptions::Check() const {
  CHECK_GT(max_num_neighbors, 0);
  CHECK_GT(max_distance, 0.0);
}

SpatialFeatureMatcher::SpatialFeatureMatcher(
    const Options& options, const SpatialOptions& spatial_options,
    const std::string& database_path)
    : FeatureMatcher(options, database_path),
      spatial_options_(spatial_options) {
  spatial_options_.Check();
}

void SpatialFeatureMatcher::DoMatching() {
  using namespace ANNfloat;

  PrintHeading1("Spatial feature matching");

  std::vector<Image> ordered_images;
  ordered_images.reserve(images_.size());

  for (const auto& image : images_) {
    ordered_images.push_back(image.second);
  }

  Timer timer;

  //////////////////////////////////////////////////////////////////////////////
  // Spatial indexing
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();

  std::cout << "Indexing images" << std::flush;

  GPSTransform gps_transform;

  int num_locations = 0;
  ANNpointArray locations =
      annAllocPts(static_cast<int>(ordered_images.size()), 3);
  std::vector<size_t> location_idxs;
  location_idxs.reserve(ordered_images.size());

  std::vector<Eigen::Vector3d> ells(1);

  for (size_t i = 0; i < ordered_images.size(); ++i) {
    const auto& image = ordered_images[i];

    if ((image.TvecPrior(0) == 0 && image.TvecPrior(1) == 0 &&
         spatial_options_.ignore_z) ||
        (image.TvecPrior(0) == 0 && image.TvecPrior(1) == 0 &&
         image.TvecPrior(2) == 0 && !spatial_options_.ignore_z)) {
      continue;
    }

    location_idxs.push_back(i);

    if (spatial_options_.is_gps) {
      ells[0](0) = image.TvecPrior(0);
      ells[0](1) = image.TvecPrior(1);
      ells[0](2) = spatial_options_.ignore_z ? 0 : image.TvecPrior(2);

      const auto xyzs = gps_transform.EllToXYZ(ells);

      locations[num_locations][0] = static_cast<float>(xyzs[0](0));
      locations[num_locations][1] = static_cast<float>(xyzs[0](1));
      locations[num_locations][2] = static_cast<float>(xyzs[0](2));
    } else {
      locations[num_locations][0] = static_cast<float>(image.TvecPrior(0));
      locations[num_locations][1] = static_cast<float>(image.TvecPrior(1));
      locations[num_locations][2] = static_cast<float>(
          spatial_options_.ignore_z ? 0 : image.TvecPrior(2));
    }

    num_locations += 1;
  }

  PrintElapsedTime(timer);

  //////////////////////////////////////////////////////////////////////////////
  // Building spatial index
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();

  std::cout << "Building hierarchical tree" << std::flush;
  ANNkd_tree pose_kdtree(locations, num_locations, 3);

  PrintElapsedTime(timer);

  //////////////////////////////////////////////////////////////////////////////
  // Matching
  //////////////////////////////////////////////////////////////////////////////

  const int k = std::min(spatial_options_.max_num_neighbors, num_locations);

  const float max_distance = static_cast<float>(spatial_options_.max_distance *
                                                spatial_options_.max_distance);

  ANNidxArray nn_idxs = new ANNidx[k];
  ANNdistArray nn_dists = new ANNdist[k];

  std::vector<std::pair<image_t, image_t>> image_pairs;
  image_pairs.reserve(static_cast<size_t>(k));

  for (int i = 0; i < num_locations; ++i) {
    if (IsStopped()) {
      return;
    }

    timer.Restart();

    const size_t idx = location_idxs[i];
    const auto& image = ordered_images[idx];

    pose_kdtree.annkSearch(locations[i], k, nn_idxs, nn_dists);

    std::cout << boost::format("Matching image [%d/%d]") % (i + 1) %
                     num_locations
              << std::flush;

    image_pairs.clear();

    for (int j = 0; j < k; ++j) {
      // Query equals result.
      if (nn_idxs[j] == i) {
        continue;
      }

      if (nn_dists[j] > max_distance) {
        break;
      }

      const size_t nn_idx = location_idxs[nn_idxs[j]];
      image_pairs.emplace_back(image.ImageId(),
                               ordered_images[nn_idx].ImageId());
    }

    MatchImagePairs(image_pairs);
    PrintElapsedTime(timer);
  }

  annDeallocPts(locations);
  delete[] nn_idxs;
  delete[] nn_dists;
}

ImagePairsFeatureMatcher::ImagePairsFeatureMatcher(
    const Options& options, const std::string& database_path,
    const std::string& match_list_path)
    : FeatureMatcher(options, database_path),
      match_list_path_(match_list_path) {}

void ImagePairsFeatureMatcher::DoMatching() {
  PrintHeading1("Custom feature matching");

  Timer timer;

  const auto image_pairs = ReadImagePairsList();

  const size_t kBlockSize = 100;
  const size_t num_match_blocks = image_pairs.size() / kBlockSize + 1;
  std::vector<std::pair<image_t, image_t>> block_image_pairs;
  block_image_pairs.reserve(kBlockSize);

  for (size_t i = 0; i < image_pairs.size(); i += kBlockSize) {
    if (IsStopped()) {
      return;
    }

    timer.Restart();

    std::cout << boost::format("Matching block [%d/%d]") %
                     (i / kBlockSize + 1) % num_match_blocks
              << std::flush;

    block_image_pairs.clear();
    const size_t block_end = i + kBlockSize <= image_pairs.size()
                                 ? i + kBlockSize
                                 : image_pairs.size();
    for (size_t j = i; j < block_end; ++j) {
      block_image_pairs.push_back(image_pairs[j]);
    }

    MatchImagePairs(block_image_pairs);
    PrintElapsedTime(timer);
  }
}

std::vector<std::pair<image_t, image_t>>
ImagePairsFeatureMatcher::ReadImagePairsList() {
  // Lookup table from image name to image
  std::unordered_map<std::string, const Image*> name_to_image;
  for (const auto& image : images_) {
    name_to_image[image.second.Name()] = &image.second;
  }

  std::ifstream file(match_list_path_.c_str());

  std::vector<std::pair<image_t, image_t>> image_pairs;

  std::string line;

  while (std::getline(file, line)) {
    boost::trim(line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream line_stream(line);

    std::string image_name1;
    std::string image_name2;

    std::getline(line_stream, image_name1, ' ');
    boost::trim(image_name1);
    std::getline(line_stream, image_name2, ' ');
    boost::trim(image_name2);

    if (name_to_image.count(image_name1) == 0) {
      std::cerr << "ERROR: Image " << image_name1 << " does not exist."
                << std::endl;
      continue;
    }
    if (name_to_image.count(image_name2) == 0) {
      std::cerr << "ERROR: Image " << image_name2 << " does not exist."
                << std::endl;
      continue;
    }

    const Image& image1 = *name_to_image.at(image_name1);
    const Image& image2 = *name_to_image.at(image_name2);
    image_pairs.emplace_back(image1.ImageId(), image2.ImageId());
  }

  file.close();

  return image_pairs;
}

FeaturePairsFeatureMatcher::FeaturePairsFeatureMatcher(
    const FeatureMatcher::Options& options, const bool compute_inliers,
    const std::string& database_path, const std::string& match_list_path)
    : stop_(false),
      database_path_(database_path),
      match_list_path_(match_list_path),
      options_(options),
      compute_inliers_(compute_inliers) {}

void FeaturePairsFeatureMatcher::Stop() {
  QMutexLocker locker(&mutex_);
  stop_ = true;
}

void FeaturePairsFeatureMatcher::run() {
  PrintHeading1("Importing matches");

  Database database;
  database.Open(database_path_);

  const std::vector<Image> images = database.ReadAllImages();
  const std::vector<Camera> cameras = database.ReadAllCameras();

  // Lookup table from image names
  std::unordered_map<std::string, const Image*> name_to_image;
  for (const Image& image : images) {
    name_to_image[image.Name()] = &image;
  }

  std::ifstream file(match_list_path_.c_str());

  std::string line;
  std::string item;

  FeatureMatches matches;

  database.BeginTransaction();

  while (std::getline(file, line)) {
    boost::trim(line);
    if (line.empty()) {
      continue;
    }

    std::istringstream line_stream(line);

    std::string image_name1, image_name2;
    try {
      line_stream >> image_name1 >> image_name2;
    } catch (...) {
      std::cerr << "ERROR: Could not read image pair" << std::endl;
      break;
    }

    std::cout << image_name1 << " - " << image_name2 << std::endl;

    if (name_to_image.count(image_name1) == 0) {
      std::cout << "SKIP: Image " << image_name1 << " not found in database."
                << std::endl;
      break;
    }
    if (name_to_image.count(image_name2) == 0) {
      std::cout << "SKIP: Image " << image_name2 << " not found in database."
                << std::endl;
      break;
    }

    const Image& image1 = *name_to_image[image_name1];
    const Image& image2 = *name_to_image[image_name2];

    bool skip_pair = false;

    if (database.ExistsInlierMatches(image1.ImageId(), image2.ImageId())) {
      std::cout << "SKIP: Matches for image pair already exist in database."
                << std::endl;
      skip_pair = true;
    }

    matches.clear();

    while (std::getline(file, line)) {
      boost::trim(line);

      if (line.empty()) {
        break;
      }

      std::istringstream line_stream(line);

      FeatureMatch match;
      try {
        line_stream >> match.point2D_idx1 >> match.point2D_idx2;
      } catch (...) {
        std::cerr << "ERROR: Cannot read feature matches." << std::endl;
        break;
      }

      matches.push_back(match);
    }

    if (skip_pair) {
      continue;
    }

    const Camera& camera1 = cameras[image1.CameraId()];
    const Camera& camera2 = cameras[image2.CameraId()];

    if (compute_inliers_) {
      database.WriteMatches(image1.ImageId(), image2.ImageId(), matches);

      const auto keypoints1 = database.ReadKeypoints(image1.ImageId());
      const auto keypoints2 = database.ReadKeypoints(image2.ImageId());

      TwoViewGeometry two_view_geometry;
      TwoViewGeometry::Options two_view_geometry_options;
      two_view_geometry_options.min_num_inliers =
          static_cast<size_t>(options_.min_num_inliers);
      two_view_geometry_options.ransac_options.max_error = options_.max_error;
      two_view_geometry_options.ransac_options.confidence = options_.confidence;
      two_view_geometry_options.ransac_options.max_num_trials =
          static_cast<size_t>(options_.max_num_trials);
      two_view_geometry_options.ransac_options.min_inlier_ratio =
          options_.min_inlier_ratio;

      two_view_geometry.Estimate(
          camera1, FeatureKeypointsToPointsVector(keypoints1), camera2,
          FeatureKeypointsToPointsVector(keypoints2), matches,
          two_view_geometry_options);

      database.WriteInlierMatches(image1.ImageId(), image2.ImageId(),
                                  two_view_geometry);
    } else {
      TwoViewGeometry two_view_geometry;

      if (camera1.HasPriorFocalLength() && camera2.HasPriorFocalLength()) {
        two_view_geometry.config = TwoViewGeometry::CALIBRATED;
      } else {
        two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
      }

      two_view_geometry.inlier_matches = matches;

      database.WriteInlierMatches(image1.ImageId(), image2.ImageId(),
                                  two_view_geometry);
    }
  }

  file.close();

  database.EndTransaction();
}

}  // namespace colmap
