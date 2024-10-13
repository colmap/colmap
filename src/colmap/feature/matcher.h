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

#pragma once

#include "colmap/feature/index.h"
#include "colmap/feature/types.h"
#include "colmap/geometry/gps.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/database.h"
#include "colmap/scene/image.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/cache.h"
#include "colmap/util/types.h"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace colmap {

class FeatureMatcher {
 public:
  virtual ~FeatureMatcher() = default;

  struct Image {
    // Unique identifier for the image. Allows a matcher to cache some
    // computations per image in consecutive calls to matching.
    image_t image_id = kInvalidImageId;
    // Used for both normal and guided matching.
    std::shared_ptr<const FeatureDescriptors> descriptors;
    // Only used for guided matching.
    std::shared_ptr<const FeatureKeypoints> keypoints;
  };

  virtual void Match(const Image& image1,
                     const Image& image2,
                     FeatureMatches* matches) = 0;

  virtual void MatchGuided(double max_error,
                           const Image& image1,
                           const Image& image2,
                           TwoViewGeometry* two_view_geometry) = 0;
};

// Cache for feature matching to minimize database access during matching.
class FeatureMatcherCache {
 public:
  FeatureMatcherCache(size_t cache_size,
                      const std::shared_ptr<Database>& database);

  // Executes a function that accesses the database. This function is thread
  // safe and ensures that only one function can access the database at a time.
  void AccessDatabase(
      const std::function<void(const Database& database)>& func);

  const Camera& GetCamera(camera_t camera_id);
  const Image& GetImage(image_t image_id);
  const PosePrior* GetPosePriorOrNull(image_t image_id);
  std::shared_ptr<FeatureKeypoints> GetKeypoints(image_t image_id);
  std::shared_ptr<FeatureDescriptors> GetDescriptors(image_t image_id);
  FeatureMatches GetMatches(image_t image_id1, image_t image_id2);
  std::vector<image_t> GetImageIds();
  ThreadSafeLRUCache<image_t, FeatureDescriptorIndex>&
  GetFeatureDescriptorIndexCache();

  bool ExistsKeypoints(image_t image_id);
  bool ExistsDescriptors(image_t image_id);

  bool ExistsMatches(image_t image_id1, image_t image_id2);
  bool ExistsInlierMatches(image_t image_id1, image_t image_id2);

  void WriteMatches(image_t image_id1,
                    image_t image_id2,
                    const FeatureMatches& matches);
  void WriteTwoViewGeometry(image_t image_id1,
                            image_t image_id2,
                            const TwoViewGeometry& two_view_geometry);

  void DeleteMatches(image_t image_id1, image_t image_id2);
  void DeleteInlierMatches(image_t image_id1, image_t image_id2);

  size_t MaxNumKeypoints();

 private:
  void MaybeLoadCameras();
  void MaybeLoadImages();
  void MaybeLoadPosePriors();

  const size_t cache_size_;
  const std::shared_ptr<Database> database_;
  std::mutex database_mutex_;
  std::unique_ptr<std::unordered_map<camera_t, Camera>> cameras_cache_;
  std::unique_ptr<std::unordered_map<image_t, Image>> images_cache_;
  std::unique_ptr<std::unordered_map<image_t, PosePrior>> pose_priors_cache_;
  std::unique_ptr<ThreadSafeLRUCache<image_t, FeatureKeypoints>>
      keypoints_cache_;
  std::unique_ptr<ThreadSafeLRUCache<image_t, FeatureDescriptors>>
      descriptors_cache_;
  std::unique_ptr<ThreadSafeLRUCache<image_t, bool>> keypoints_exists_cache_;
  std::unique_ptr<ThreadSafeLRUCache<image_t, bool>> descriptors_exists_cache_;
  ThreadSafeLRUCache<image_t, FeatureDescriptorIndex> descriptor_index_cache_;
};

}  // namespace colmap
