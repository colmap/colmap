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

FeatureMatcherCache::FeatureMatcherCache(const size_t cache_size,
                                         std::shared_ptr<Database> database,
                                         const bool do_setup)
    : cache_size_(cache_size),
      database_(std::move(THROW_CHECK_NOTNULL(database))) {
  if (do_setup) {
    Setup();
  }
}

void FeatureMatcherCache::Setup() {
  std::vector<Camera> cameras = database_->ReadAllCameras();
  cameras_cache_.reserve(cameras.size());
  for (Camera& camera : cameras) {
    cameras_cache_.emplace(camera.camera_id, std::move(camera));
  }

  std::vector<Image> images = database_->ReadAllImages();
  images_cache_.reserve(images.size());
  for (Image& image : images) {
    images_cache_.emplace(image.ImageId(), std::move(image));
  }

  locations_priors_cache_.reserve(database_->NumPosePriors());
  for (const auto& id_and_image : images_cache_) {
    if (database_->ExistsPosePrior(id_and_image.first)) {
      locations_priors_cache_.emplace(
          id_and_image.first, database_->ReadPosePrior(id_and_image.first));
    }
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
      images_cache_.size(), [this](const image_t image_id) {
        return database_->ExistsKeypoints(image_id);
      });

  descriptors_exists_cache_ = std::make_unique<LRUCache<image_t, bool>>(
      images_cache_.size(), [this](const image_t image_id) {
        return database_->ExistsDescriptors(image_id);
      });
}

const Camera& FeatureMatcherCache::GetCamera(const camera_t camera_id) const {
  return cameras_cache_.at(camera_id);
}

const Image& FeatureMatcherCache::GetImage(const image_t image_id) const {
  return images_cache_.at(image_id);
}

const PosePrior& FeatureMatcherCache::GetPosePrior(
    const image_t image_id) const {
  return locations_priors_cache_.at(image_id);
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

bool FeatureMatcherCache::ExistsPosePrior(const image_t image_id) const {
  return locations_priors_cache_.find(image_id) !=
         locations_priors_cache_.end();
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

}  // namespace colmap
