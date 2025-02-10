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

#include "colmap/feature/matcher.h"

namespace colmap {

FeatureMatcherCache::FeatureMatcherCache(
    const size_t cache_size, const std::shared_ptr<Database>& database)
    : cache_size_(cache_size),
      database_(THROW_CHECK_NOTNULL(database)),
      descriptor_index_cache_(cache_size_, [this](const image_t image_id) {
        auto descriptors = GetDescriptors(image_id);
        auto index = FeatureDescriptorIndex::Create();
        index->Build(*descriptors);
        return index;
      }) {
  keypoints_cache_ =
      std::make_unique<ThreadSafeLRUCache<image_t, FeatureKeypoints>>(
          cache_size_, [this](const image_t image_id) {
            std::lock_guard<std::mutex> lock(database_mutex_);
            return std::make_shared<FeatureKeypoints>(
                database_->ReadKeypoints(image_id));
          });

  descriptors_cache_ =
      std::make_unique<ThreadSafeLRUCache<image_t, FeatureDescriptors>>(
          cache_size_, [this](const image_t image_id) {
            std::lock_guard<std::mutex> lock(database_mutex_);
            return std::make_shared<FeatureDescriptors>(
                database_->ReadDescriptors(image_id));
          });

  keypoints_exists_cache_ = std::make_unique<ThreadSafeLRUCache<image_t, bool>>(
      cache_size_, [this](const image_t image_id) {
        std::lock_guard<std::mutex> lock(database_mutex_);
        return std::make_shared<bool>(database_->ExistsKeypoints(image_id));
      });

  descriptors_exists_cache_ =
      std::make_unique<ThreadSafeLRUCache<image_t, bool>>(
          cache_size_, [this](const image_t image_id) {
            std::lock_guard<std::mutex> lock(database_mutex_);
            return std::make_shared<bool>(
                database_->ExistsDescriptors(image_id));
          });
}

void FeatureMatcherCache::AccessDatabase(
    const std::function<void(Database& database)>& func) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  func(*database_);
}

const Camera& FeatureMatcherCache::GetCamera(const camera_t camera_id) {
  MaybeLoadCameras();
  return cameras_cache_->at(camera_id);
}

const Image& FeatureMatcherCache::GetImage(const image_t image_id) {
  MaybeLoadImages();
  return images_cache_->at(image_id);
}

const PosePrior* FeatureMatcherCache::GetPosePriorOrNull(
    const image_t image_id) {
  MaybeLoadPosePriors();
  const auto it = pose_priors_cache_->find(image_id);
  if (it == pose_priors_cache_->end()) {
    return nullptr;
  }
  return &it->second;
}

std::shared_ptr<FeatureKeypoints> FeatureMatcherCache::GetKeypoints(
    const image_t image_id) {
  return keypoints_cache_->Get(image_id);
}

std::shared_ptr<FeatureDescriptors> FeatureMatcherCache::GetDescriptors(
    const image_t image_id) {
  return descriptors_cache_->Get(image_id);
}

FeatureMatches FeatureMatcherCache::GetMatches(const image_t image_id1,
                                               const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return database_->ReadMatches(image_id1, image_id2);
}

std::vector<image_t> FeatureMatcherCache::GetImageIds() {
  MaybeLoadImages();

  std::vector<image_t> image_ids;
  image_ids.reserve(images_cache_->size());
  for (const auto& image : *images_cache_) {
    image_ids.push_back(image.first);
  }
  // Sort the images for deterministic behavior. Note that the images_cache_ is
  // an unordered_map, which does not guarantee a deterministic order across
  // different standard library implementations.
  std::sort(image_ids.begin(), image_ids.end());
  return image_ids;
}

ThreadSafeLRUCache<image_t, FeatureDescriptorIndex>&
FeatureMatcherCache::GetFeatureDescriptorIndexCache() {
  return descriptor_index_cache_;
}

bool FeatureMatcherCache::ExistsKeypoints(const image_t image_id) {
  return *keypoints_exists_cache_->Get(image_id);
}

bool FeatureMatcherCache::ExistsDescriptors(const image_t image_id) {
  return *descriptors_exists_cache_->Get(image_id);
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

size_t FeatureMatcherCache::MaxNumKeypoints() {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return database_->MaxNumKeypoints();
}

void FeatureMatcherCache::MaybeLoadCameras() {
  if (cameras_cache_) {
    return;
  }

  std::lock_guard<std::mutex> lock(database_mutex_);
  std::vector<Camera> cameras = database_->ReadAllCameras();
  cameras_cache_ = std::make_unique<std::unordered_map<camera_t, Camera>>();
  cameras_cache_->reserve(cameras.size());
  for (Camera& camera : cameras) {
    cameras_cache_->emplace(camera.camera_id, std::move(camera));
  }
}

void FeatureMatcherCache::MaybeLoadImages() {
  if (images_cache_) {
    return;
  }

  std::lock_guard<std::mutex> lock(database_mutex_);
  std::vector<Image> images = database_->ReadAllImages();
  images_cache_ = std::make_unique<std::unordered_map<image_t, Image>>();
  images_cache_->reserve(images.size());
  for (Image& image : images) {
    images_cache_->emplace(image.ImageId(), std::move(image));
  }
}

void FeatureMatcherCache::MaybeLoadPosePriors() {
  if (pose_priors_cache_) {
    return;
  }

  MaybeLoadImages();

  std::lock_guard<std::mutex> lock(database_mutex_);
  pose_priors_cache_ =
      std::make_unique<std::unordered_map<image_t, PosePrior>>();
  pose_priors_cache_->reserve(database_->NumPosePriors());
  for (const auto& image : *images_cache_) {
    if (database_->ExistsPosePrior(image.first)) {
      pose_priors_cache_->emplace(image.first,
                                  database_->ReadPosePrior(image.first));
    }
  }
}

}  // namespace colmap
