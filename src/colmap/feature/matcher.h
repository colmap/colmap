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

  // If the same matcher is used for matching multiple pairs of feature sets,
  // then the caller may pass a nullptr to one of the keypoint/descriptor
  // arguments to inform the implementation that the keypoints/descriptors are
  // identical to the previous call. This allows the implementation to skip e.g.
  // uploading data to GPU memory or pre-computing search data structures for
  // one of the descriptors.

  virtual void Match(
      const std::shared_ptr<const FeatureDescriptors>& descriptors1,
      const std::shared_ptr<const FeatureDescriptors>& descriptors2,
      FeatureMatches* matches) = 0;

  virtual void MatchGuided(
      double max_error,
      const std::shared_ptr<const FeatureKeypoints>& keypoints1,
      const std::shared_ptr<const FeatureKeypoints>& keypoints2,
      const std::shared_ptr<const FeatureDescriptors>& descriptors1,
      const std::shared_ptr<const FeatureDescriptors>& descriptors2,
      TwoViewGeometry* two_view_geometry) = 0;
};

// Cache for feature matching to minimize database access during matching.
class FeatureMatcherCache {
 public:
  FeatureMatcherCache(size_t cache_size,
                      std::shared_ptr<Database> database,
                      bool do_setup = false);

  void Setup();

  const Camera& GetCamera(camera_t camera_id) const;
  const Image& GetImage(image_t image_id) const;
  const PosePrior& GetPosePrior(image_t image_id) const;
  std::shared_ptr<FeatureKeypoints> GetKeypoints(image_t image_id);
  std::shared_ptr<FeatureDescriptors> GetDescriptors(image_t image_id);
  FeatureMatches GetMatches(image_t image_id1, image_t image_id2);
  std::vector<image_t> GetImageIds() const;

  bool ExistsPosePrior(image_t image_id) const;
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

 private:
  const size_t cache_size_;
  const std::shared_ptr<Database> database_;
  std::mutex database_mutex_;
  std::unordered_map<camera_t, Camera> cameras_cache_;
  std::unordered_map<image_t, Image> images_cache_;
  std::unordered_map<image_t, PosePrior> locations_priors_cache_;
  std::unique_ptr<LRUCache<image_t, std::shared_ptr<FeatureKeypoints>>>
      keypoints_cache_;
  std::unique_ptr<LRUCache<image_t, std::shared_ptr<FeatureDescriptors>>>
      descriptors_cache_;
  std::unique_ptr<LRUCache<image_t, bool>> keypoints_exists_cache_;
  std::unique_ptr<LRUCache<image_t, bool>> descriptors_exists_cache_;
};

}  // namespace colmap
