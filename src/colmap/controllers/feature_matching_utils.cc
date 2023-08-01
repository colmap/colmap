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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/controllers/feature_matching_utils.h"

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/utils.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"

#include <fstream>
#include <numeric>
#include <unordered_set>

namespace colmap {

FeatureMatcherCache::FeatureMatcherCache(const size_t cache_size,
                                         const Database* database)
    : cache_size_(cache_size), database_(database) {
  CHECK_NOTNULL(database_);
}

void FeatureMatcherCache::Setup() {
  const std::vector<Camera> cameras = database_->ReadAllCameras();
  cameras_cache_.reserve(cameras.size());
  for (const auto& camera : cameras) {
    cameras_cache_.emplace(camera.CameraId(), camera);
  }

  const std::vector<Image> images = database_->ReadAllImages();
  images_cache_.reserve(images.size());
  for (const auto& image : images) {
    images_cache_.emplace(image.ImageId(), image);
  }

  keypoints_cache_ =
      std::make_unique<LRUCache<image_t, std::shared_ptr<FeatureKeypoints>>>(
          cache_size_, [this](const image_t image_id) {
            return std::make_shared<FeatureKeypoints>(
                database_->ReadKeypoints(image_id));
          });

  descriptors_cache_ =
      std::make_unique<LRUCache<image_t, std::shared_ptr<FeatureDescriptors>>>(
          cache_size_, [this](const image_t image_id) {
            return std::make_shared<FeatureDescriptors>(
                database_->ReadDescriptors(image_id));
          });

  keypoints_exists_cache_ = std::make_unique<LRUCache<image_t, bool>>(
      images.size(), [this](const image_t image_id) {
        return database_->ExistsKeypoints(image_id);
      });

  descriptors_exists_cache_ = std::make_unique<LRUCache<image_t, bool>>(
      images.size(), [this](const image_t image_id) {
        return database_->ExistsDescriptors(image_id);
      });
}

const Camera& FeatureMatcherCache::GetCamera(const camera_t camera_id) const {
  return cameras_cache_.at(camera_id);
}

const Image& FeatureMatcherCache::GetImage(const image_t image_id) const {
  return images_cache_.at(image_id);
}

std::shared_ptr<FeatureKeypoints> FeatureMatcherCache::GetKeypoints(
    const image_t image_id) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return keypoints_cache_->Get(image_id);
}

std::shared_ptr<FeatureDescriptors> FeatureMatcherCache::GetDescriptors(
    const image_t image_id) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return descriptors_cache_->Get(image_id);
}

FeatureMatches FeatureMatcherCache::GetMatches(const image_t image_id1,
                                               const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return database_->ReadMatches(image_id1, image_id2);
}

std::vector<image_t> FeatureMatcherCache::GetImageIds() const {
  std::vector<image_t> image_ids;
  image_ids.reserve(images_cache_.size());
  for (const auto& image : images_cache_) {
    image_ids.push_back(image.first);
  }
  return image_ids;
}

bool FeatureMatcherCache::ExistsKeypoints(const image_t image_id) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return keypoints_exists_cache_->Get(image_id);
}

bool FeatureMatcherCache::ExistsDescriptors(const image_t image_id) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return descriptors_exists_cache_->Get(image_id);
}

bool FeatureMatcherCache::ExistsMatches(const image_t image_id1,
                                        const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return database_->ExistsMatches(image_id1, image_id2);
}

bool FeatureMatcherCache::ExistsInlierMatches(const image_t image_id1,
                                              const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return database_->ExistsInlierMatches(image_id1, image_id2);
}

void FeatureMatcherCache::WriteMatches(const image_t image_id1,
                                       const image_t image_id2,
                                       const FeatureMatches& matches) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  database_->WriteMatches(image_id1, image_id2, matches);
}

void FeatureMatcherCache::WriteTwoViewGeometry(
    const image_t image_id1,
    const image_t image_id2,
    const TwoViewGeometry& two_view_geometry) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  database_->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
}

void FeatureMatcherCache::DeleteMatches(const image_t image_id1,
                                        const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  database_->DeleteMatches(image_id1, image_id2);
}

void FeatureMatcherCache::DeleteInlierMatches(const image_t image_id1,
                                              const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  database_->DeleteInlierMatches(image_id1, image_id2);
}

FeatureMatcherWorker::FeatureMatcherWorker(const SiftMatchingOptions& options,
                                           FeatureMatcherCache* cache,
                                           JobQueue<Input>* input_queue,
                                           JobQueue<Output>* output_queue)
    : options_(options),
      cache_(cache),
      input_queue_(input_queue),
      output_queue_(output_queue) {
  CHECK(options_.Check());

  prev_keypoints_image_ids_[0] = kInvalidImageId;
  prev_keypoints_image_ids_[1] = kInvalidImageId;
  prev_descriptors_image_ids_[0] = kInvalidImageId;
  prev_descriptors_image_ids_[1] = kInvalidImageId;

  if (options_.use_gpu) {
#if !defined(COLMAP_CUDA_ENABLED)
    opengl_context_ = std::make_unique<OpenGLContextManager>();
#endif
  }
}

void FeatureMatcherWorker::SetMaxNumMatches(int max_num_matches) {
  options_.max_num_matches = max_num_matches;
}

void FeatureMatcherWorker::Run() {
  if (options_.use_gpu) {
#if !defined(COLMAP_CUDA_ENABLED)
    CHECK(opengl_context_);
    CHECK(opengl_context_->MakeCurrent());
#endif
  }

  std::unique_ptr<FeatureMatcher> matcher = CreateSiftFeatureMatcher(options_);
  if (matcher == nullptr) {
    std::cerr << "ERROR: Failed to create feature matcher." << std::endl;
    SignalInvalidSetup();
    return;
  }

  SignalValidSetup();

  while (true) {
    if (IsStopped()) {
      break;
    }

    auto input_job = input_queue_->Pop();
    if (input_job.IsValid()) {
      auto& data = input_job.Data();

      if (!cache_->ExistsDescriptors(data.image_id1) ||
          !cache_->ExistsDescriptors(data.image_id2)) {
        CHECK(output_queue_->Push(std::move(data)));
        continue;
      }

      if (options_.guided_matching) {
        matcher->MatchGuided(GetKeypointsPtr(0, data.image_id1),
                             GetKeypointsPtr(1, data.image_id2),
                             GetDescriptorsPtr(0, data.image_id1),
                             GetDescriptorsPtr(1, data.image_id2),
                             &data.two_view_geometry);
      } else {
        matcher->Match(GetDescriptorsPtr(0, data.image_id1),
                       GetDescriptorsPtr(1, data.image_id2),
                       &data.matches);
      }

      CHECK(output_queue_->Push(std::move(data)));
    }
  }
}

std::shared_ptr<FeatureKeypoints> FeatureMatcherWorker::GetKeypointsPtr(
    const int index, const image_t image_id) {
  CHECK_GE(index, 0);
  CHECK_LE(index, 1);
  if (prev_keypoints_image_ids_[index] == image_id) {
    return nullptr;
  } else {
    prev_keypoints_image_ids_[index] = image_id;
    prev_keypoints_[index] = cache_->GetKeypoints(image_id);
    return prev_keypoints_[index];
  }
}

std::shared_ptr<FeatureDescriptors> FeatureMatcherWorker::GetDescriptorsPtr(
    const int index, const image_t image_id) {
  CHECK_GE(index, 0);
  CHECK_LE(index, 1);
  if (prev_descriptors_image_ids_[index] == image_id) {
    return nullptr;
  } else {
    prev_descriptors_image_ids_[index] = image_id;
    prev_descriptors_[index] = cache_->GetDescriptors(image_id);
    return prev_descriptors_[index];
  }
}

namespace {

class VerifierWorker : public Thread {
 public:
  typedef FeatureMatcherData Input;
  typedef FeatureMatcherData Output;

  VerifierWorker(const SiftMatchingOptions& options,
                 FeatureMatcherCache* cache,
                 JobQueue<Input>* input_queue,
                 JobQueue<Output>* output_queue)
      : options_(options),
        cache_(cache),
        input_queue_(input_queue),
        output_queue_(output_queue) {
    CHECK(options_.Check());

    two_view_geometry_options_.min_num_inliers =
        static_cast<size_t>(options_.min_num_inliers);
    two_view_geometry_options_.ransac_options.max_error = options_.max_error;
    two_view_geometry_options_.ransac_options.confidence = options_.confidence;
    two_view_geometry_options_.ransac_options.min_num_trials =
        static_cast<size_t>(options_.min_num_trials);
    two_view_geometry_options_.ransac_options.max_num_trials =
        static_cast<size_t>(options_.max_num_trials);
    two_view_geometry_options_.ransac_options.min_inlier_ratio =
        options_.min_inlier_ratio;
    two_view_geometry_options_.force_H_use = options_.planar_scene;
    two_view_geometry_options_.compute_relative_pose =
        options_.compute_relative_pose;
  }

 protected:
  void Run() override {
    while (true) {
      if (IsStopped()) {
        break;
      }

      auto input_job = input_queue_->Pop();
      if (input_job.IsValid()) {
        auto& data = input_job.Data();

        if (data.matches.size() <
            static_cast<size_t>(options_.min_num_inliers)) {
          CHECK(output_queue_->Push(std::move(data)));
          continue;
        }

        const auto& camera1 =
            cache_->GetCamera(cache_->GetImage(data.image_id1).CameraId());
        const auto& camera2 =
            cache_->GetCamera(cache_->GetImage(data.image_id2).CameraId());
        const auto keypoints1 = cache_->GetKeypoints(data.image_id1);
        const auto keypoints2 = cache_->GetKeypoints(data.image_id2);
        const std::vector<Eigen::Vector2d> points1 =
            FeatureKeypointsToPointsVector(*keypoints1);
        const std::vector<Eigen::Vector2d> points2 =
            FeatureKeypointsToPointsVector(*keypoints2);

        if (options_.multiple_models) {
          data.two_view_geometry =
              EstimateMultipleTwoViewGeometries(camera1,
                                                points1,
                                                camera2,
                                                points2,
                                                data.matches,
                                                two_view_geometry_options_);
        } else {
          data.two_view_geometry =
              EstimateTwoViewGeometry(camera1,
                                      points1,
                                      camera2,
                                      points2,
                                      data.matches,
                                      two_view_geometry_options_);
        }

        CHECK(output_queue_->Push(std::move(data)));
      }
    }
  }

 private:
  const SiftMatchingOptions options_;
  TwoViewGeometryOptions two_view_geometry_options_;
  FeatureMatcherCache* cache_;
  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

}  // namespace

FeatureMatcherController::FeatureMatcherController(
    const SiftMatchingOptions& options,
    Database* database,
    FeatureMatcherCache* cache)
    : options_(options), database_(database), cache_(cache), is_setup_(false) {
  CHECK(options_.Check());

  const int num_threads = GetEffectiveNumThreads(options_.num_threads);
  CHECK_GT(num_threads, 0);

  std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
  CHECK_GT(gpu_indices.size(), 0);

#if defined(COLMAP_CUDA_ENABLED)
  if (options_.use_gpu && gpu_indices.size() == 1 && gpu_indices[0] == -1) {
    const int num_cuda_devices = GetNumCudaDevices();
    CHECK_GT(num_cuda_devices, 0);
    gpu_indices.resize(num_cuda_devices);
    std::iota(gpu_indices.begin(), gpu_indices.end(), 0);
  }
#endif  // COLMAP_CUDA_ENABLED

  if (options_.use_gpu) {
    auto custom_options = options_;
    // The first matching is always without guided matching.
    custom_options.guided_matching = false;
    matchers_.reserve(gpu_indices.size());
    for (const auto& gpu_index : gpu_indices) {
      custom_options.gpu_index = std::to_string(gpu_index);
      matchers_.emplace_back(std::make_unique<FeatureMatcherWorker>(
          custom_options, cache, &matcher_queue_, &verifier_queue_));
    }
  } else {
    auto custom_options = options_;
    // The first matching is always without guided matching.
    custom_options.guided_matching = false;
    matchers_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      matchers_.emplace_back(std::make_unique<FeatureMatcherWorker>(
          custom_options, cache, &matcher_queue_, &verifier_queue_));
    }
  }

  verifiers_.reserve(num_threads);
  if (options_.guided_matching) {
    // Redirect the verification output to final round of guided matching.
    for (int i = 0; i < num_threads; ++i) {
      verifiers_.emplace_back(std::make_unique<VerifierWorker>(
          options_, cache, &verifier_queue_, &guided_matcher_queue_));
    }

    if (options_.use_gpu) {
      auto custom_options = options_;
      guided_matchers_.reserve(gpu_indices.size());
      for (const auto& gpu_index : gpu_indices) {
        custom_options.gpu_index = std::to_string(gpu_index);
        guided_matchers_.emplace_back(std::make_unique<FeatureMatcherWorker>(
            custom_options, cache, &guided_matcher_queue_, &output_queue_));
      }
    } else {
      guided_matchers_.reserve(num_threads);
      for (int i = 0; i < num_threads; ++i) {
        guided_matchers_.emplace_back(std::make_unique<FeatureMatcherWorker>(
            options_, cache, &guided_matcher_queue_, &output_queue_));
      }
    }
  } else {
    for (int i = 0; i < num_threads; ++i) {
      verifiers_.emplace_back(std::make_unique<VerifierWorker>(
          options_, cache, &verifier_queue_, &output_queue_));
    }
  }
}

FeatureMatcherController::~FeatureMatcherController() {
  matcher_queue_.Wait();
  verifier_queue_.Wait();
  guided_matcher_queue_.Wait();
  output_queue_.Wait();

  for (auto& matcher : matchers_) {
    matcher->Stop();
  }

  for (auto& verifier : verifiers_) {
    verifier->Stop();
  }

  for (auto& guided_matcher : guided_matchers_) {
    guided_matcher->Stop();
  }

  matcher_queue_.Stop();
  verifier_queue_.Stop();
  guided_matcher_queue_.Stop();
  output_queue_.Stop();

  for (auto& matcher : matchers_) {
    matcher->Wait();
  }

  for (auto& verifier : verifiers_) {
    verifier->Wait();
  }

  for (auto& guided_matcher : guided_matchers_) {
    guided_matcher->Wait();
  }
}

bool FeatureMatcherController::Setup() {
  // Minimize the amount of allocated GPU memory by computing the maximum number
  // of descriptors for any image over the whole database.
  const int max_num_features = CHECK_NOTNULL(database_)->MaxNumDescriptors();
  options_.max_num_matches =
      std::min(options_.max_num_matches, max_num_features);

  for (auto& matcher : matchers_) {
    matcher->SetMaxNumMatches(options_.max_num_matches);
    matcher->Start();
  }

  for (auto& verifier : verifiers_) {
    verifier->Start();
  }

  for (auto& guided_matcher : guided_matchers_) {
    guided_matcher->SetMaxNumMatches(options_.max_num_matches);
    guided_matcher->Start();
  }

  for (auto& matcher : matchers_) {
    if (!matcher->CheckValidSetup()) {
      return false;
    }
  }

  for (auto& guided_matcher : guided_matchers_) {
    if (!guided_matcher->CheckValidSetup()) {
      return false;
    }
  }

  is_setup_ = true;

  return true;
}

void FeatureMatcherController::Match(
    const std::vector<std::pair<image_t, image_t>>& image_pairs) {
  CHECK_NOTNULL(database_);
  CHECK_NOTNULL(cache_);
  CHECK(is_setup_);

  if (image_pairs.empty()) {
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Match the image pairs
  //////////////////////////////////////////////////////////////////////////////

  std::unordered_set<image_pair_t> image_pair_ids;
  image_pair_ids.reserve(image_pairs.size());

  size_t num_outputs = 0;
  for (const auto& image_pair : image_pairs) {
    // Avoid self-matches.
    if (image_pair.first == image_pair.second) {
      continue;
    }

    // Avoid duplicate image pairs.
    const image_pair_t pair_id =
        Database::ImagePairToPairId(image_pair.first, image_pair.second);
    if (image_pair_ids.count(pair_id) > 0) {
      continue;
    }

    image_pair_ids.insert(pair_id);

    const bool exists_matches =
        cache_->ExistsMatches(image_pair.first, image_pair.second);
    const bool exists_inlier_matches =
        cache_->ExistsInlierMatches(image_pair.first, image_pair.second);

    if (exists_matches && exists_inlier_matches) {
      continue;
    }

    num_outputs += 1;

    // If only one of the matches or inlier matches exist, we recompute them
    // from scratch and delete the existing results. This must be done before
    // pushing the jobs to the queue, otherwise database constraints might fail
    // when writing an existing result into the database.

    if (exists_inlier_matches) {
      cache_->DeleteInlierMatches(image_pair.first, image_pair.second);
    }

    FeatureMatcherData data;
    data.image_id1 = image_pair.first;
    data.image_id2 = image_pair.second;

    if (exists_matches) {
      data.matches = cache_->GetMatches(image_pair.first, image_pair.second);
      cache_->DeleteMatches(image_pair.first, image_pair.second);
      CHECK(verifier_queue_.Push(std::move(data)));
    } else {
      CHECK(matcher_queue_.Push(std::move(data)));
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Write results to database
  //////////////////////////////////////////////////////////////////////////////

  for (size_t i = 0; i < num_outputs; ++i) {
    auto output_job = output_queue_.Pop();
    CHECK(output_job.IsValid());
    auto& output = output_job.Data();

    if (output.matches.size() < static_cast<size_t>(options_.min_num_inliers)) {
      output.matches = {};
    }

    if (output.two_view_geometry.inlier_matches.size() <
        static_cast<size_t>(options_.min_num_inliers)) {
      output.two_view_geometry = TwoViewGeometry();
    }

    cache_->WriteMatches(output.image_id1, output.image_id2, output.matches);
    cache_->WriteTwoViewGeometry(
        output.image_id1, output.image_id2, output.two_view_geometry);
  }

  CHECK_EQ(output_queue_.Size(), 0);
}

}  // namespace colmap
