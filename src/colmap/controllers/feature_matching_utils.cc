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

#include "colmap/controllers/feature_matching_utils.h"

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/utils.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"

#include <fstream>
#include <numeric>
#include <unordered_set>

namespace colmap {

FeatureMatcherWorker::FeatureMatcherWorker(
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    FeatureMatcherCache* cache,
    JobQueue<Input>* input_queue,
    JobQueue<Output>* output_queue)
    : matching_options_(matching_options),
      geometry_options_(geometry_options),
      cache_(cache),
      input_queue_(input_queue),
      output_queue_(output_queue) {
  THROW_CHECK(matching_options_.Check());

  prev_keypoints_image_ids_[0] = kInvalidImageId;
  prev_keypoints_image_ids_[1] = kInvalidImageId;
  prev_descriptors_image_ids_[0] = kInvalidImageId;
  prev_descriptors_image_ids_[1] = kInvalidImageId;

  if (matching_options_.use_gpu) {
#if !defined(COLMAP_CUDA_ENABLED)
    opengl_context_ = std::make_unique<OpenGLContextManager>();
#endif
  }
}

void FeatureMatcherWorker::SetMaxNumMatches(int max_num_matches) {
  matching_options_.max_num_matches = max_num_matches;
}

void FeatureMatcherWorker::Run() {
  if (matching_options_.use_gpu) {
#if !defined(COLMAP_CUDA_ENABLED)
    THROW_CHECK_NOTNULL(opengl_context_);
    THROW_CHECK(opengl_context_->MakeCurrent());
#endif
  }

  std::unique_ptr<FeatureMatcher> matcher =
      CreateSiftFeatureMatcher(matching_options_);
  if (matcher == nullptr) {
    LOG(ERROR) << "Failed to create feature matcher.";
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
        THROW_CHECK(output_queue_->Push(std::move(data)));
        continue;
      }

      if (matching_options_.guided_matching) {
        matcher->MatchGuided(geometry_options_.ransac_options.max_error,
                             GetKeypointsPtr(0, data.image_id1),
                             GetKeypointsPtr(1, data.image_id2),
                             GetDescriptorsPtr(0, data.image_id1),
                             GetDescriptorsPtr(1, data.image_id2),
                             &data.two_view_geometry);
      } else {
        matcher->Match(GetDescriptorsPtr(0, data.image_id1),
                       GetDescriptorsPtr(1, data.image_id2),
                       &data.matches);
      }

      THROW_CHECK(output_queue_->Push(std::move(data)));
    }
  }
}

std::shared_ptr<FeatureKeypoints> FeatureMatcherWorker::GetKeypointsPtr(
    const int index, const image_t image_id) {
  THROW_CHECK_GE(index, 0);
  THROW_CHECK_LE(index, 1);
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
  THROW_CHECK_GE(index, 0);
  THROW_CHECK_LE(index, 1);
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

  VerifierWorker(const TwoViewGeometryOptions& options,
                 FeatureMatcherCache* cache,
                 JobQueue<Input>* input_queue,
                 JobQueue<Output>* output_queue)
      : options_(options),
        cache_(cache),
        input_queue_(input_queue),
        output_queue_(output_queue) {
    THROW_CHECK(options_.Check());
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
          THROW_CHECK(output_queue_->Push(std::move(data)));
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

        data.two_view_geometry = EstimateTwoViewGeometry(
            camera1, points1, camera2, points2, data.matches, options_);

        THROW_CHECK(output_queue_->Push(std::move(data)));
      }
    }
  }

 private:
  const TwoViewGeometryOptions options_;
  FeatureMatcherCache* cache_;
  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

}  // namespace

FeatureMatcherController::FeatureMatcherController(
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    Database* database,
    FeatureMatcherCache* cache)
    : matching_options_(matching_options),
      geometry_options_(geometry_options),
      database_(database),
      cache_(cache),
      is_setup_(false) {
  THROW_CHECK(matching_options_.Check());
  THROW_CHECK(geometry_options_.Check());

  const int num_threads = GetEffectiveNumThreads(matching_options_.num_threads);
  THROW_CHECK_GT(num_threads, 0);

  std::vector<int> gpu_indices = CSVToVector<int>(matching_options_.gpu_index);
  THROW_CHECK_GT(gpu_indices.size(), 0);

#if defined(COLMAP_CUDA_ENABLED)
  if (matching_options_.use_gpu && gpu_indices.size() == 1 &&
      gpu_indices[0] == -1) {
    const int num_cuda_devices = GetNumCudaDevices();
    THROW_CHECK_GT(num_cuda_devices, 0);
    gpu_indices.resize(num_cuda_devices);
    std::iota(gpu_indices.begin(), gpu_indices.end(), 0);
  }
#endif  // COLMAP_CUDA_ENABLED

  if (matching_options_.use_gpu) {
    auto matching_options_copy = matching_options_;
    // The first matching is always without guided matching.
    matching_options_copy.guided_matching = false;
    matchers_.reserve(gpu_indices.size());
    for (const auto& gpu_index : gpu_indices) {
      matching_options_copy.gpu_index = std::to_string(gpu_index);
      matchers_.emplace_back(
          std::make_unique<FeatureMatcherWorker>(matching_options_copy,
                                                 geometry_options_,
                                                 cache,
                                                 &matcher_queue_,
                                                 &verifier_queue_));
    }
  } else {
    auto matching_options_copy = matching_options_;
    // The first matching is always without guided matching.
    matching_options_copy.guided_matching = false;
    matchers_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      matchers_.emplace_back(
          std::make_unique<FeatureMatcherWorker>(matching_options_copy,
                                                 geometry_options_,
                                                 cache,
                                                 &matcher_queue_,
                                                 &verifier_queue_));
    }
  }

  verifiers_.reserve(num_threads);
  if (matching_options_.guided_matching) {
    // Redirect the verification output to final round of guided matching.
    for (int i = 0; i < num_threads; ++i) {
      verifiers_.emplace_back(std::make_unique<VerifierWorker>(
          geometry_options_, cache, &verifier_queue_, &guided_matcher_queue_));
    }

    if (matching_options_.use_gpu) {
      auto matching_options_copy = matching_options_;
      guided_matchers_.reserve(gpu_indices.size());
      for (const auto& gpu_index : gpu_indices) {
        matching_options_copy.gpu_index = std::to_string(gpu_index);
        guided_matchers_.emplace_back(
            std::make_unique<FeatureMatcherWorker>(matching_options_copy,
                                                   geometry_options_,
                                                   cache,
                                                   &guided_matcher_queue_,
                                                   &output_queue_));
      }
    } else {
      guided_matchers_.reserve(num_threads);
      for (int i = 0; i < num_threads; ++i) {
        guided_matchers_.emplace_back(
            std::make_unique<FeatureMatcherWorker>(matching_options_,
                                                   geometry_options_,
                                                   cache,
                                                   &guided_matcher_queue_,
                                                   &output_queue_));
      }
    }
  } else {
    for (int i = 0; i < num_threads; ++i) {
      verifiers_.emplace_back(std::make_unique<VerifierWorker>(
          geometry_options_, cache, &verifier_queue_, &output_queue_));
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
  const int max_num_features =
      THROW_CHECK_NOTNULL(database_)->MaxNumKeypoints();
  matching_options_.max_num_matches =
      std::min(matching_options_.max_num_matches, max_num_features);

  for (auto& matcher : matchers_) {
    matcher->SetMaxNumMatches(matching_options_.max_num_matches);
    matcher->Start();
  }

  for (auto& verifier : verifiers_) {
    verifier->Start();
  }

  for (auto& guided_matcher : guided_matchers_) {
    guided_matcher->SetMaxNumMatches(matching_options_.max_num_matches);
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
  THROW_CHECK_NOTNULL(database_);
  THROW_CHECK_NOTNULL(cache_);
  THROW_CHECK(is_setup_);

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
      THROW_CHECK(verifier_queue_.Push(std::move(data)));
    } else {
      THROW_CHECK(matcher_queue_.Push(std::move(data)));
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Write results to database
  //////////////////////////////////////////////////////////////////////////////

  for (size_t i = 0; i < num_outputs; ++i) {
    auto output_job = output_queue_.Pop();
    THROW_CHECK(output_job.IsValid());
    auto& output = output_job.Data();

    if (output.matches.size() <
        static_cast<size_t>(geometry_options_.min_num_inliers)) {
      output.matches = {};
    }

    if (output.two_view_geometry.inlier_matches.size() <
        static_cast<size_t>(geometry_options_.min_num_inliers)) {
      output.two_view_geometry = TwoViewGeometry();
    }

    cache_->WriteMatches(output.image_id1, output.image_id2, output.matches);
    cache_->WriteTwoViewGeometry(
        output.image_id1, output.image_id2, output.two_view_geometry);
  }

  THROW_CHECK_EQ(output_queue_.Size(), 0);
}

}  // namespace colmap
