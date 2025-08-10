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

MAKE_ENUM_CLASS_OVERLOAD_STREAM(FeatureMatcherType, 0, SIFT);

struct SiftMatchingOptions;

struct FeatureMatchingOptions {
  explicit FeatureMatchingOptions(
      FeatureMatcherType type = FeatureMatcherType::SIFT);

  FeatureMatcherType type = FeatureMatcherType::SIFT;

  // Number of threads for feature matching and geometric verification.
  int num_threads = -1;

  // Whether to use the GPU for feature matching.
#ifdef COLMAP_GPU_ENABLED
  bool use_gpu = true;
#else
  bool use_gpu = false;
#endif

  // Index of the GPU used for feature matching. For multi-GPU matching,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  // Maximum number of matches.
  int max_num_matches = 32768;

  // Whether to perform guided matching.
  bool guided_matching = false;

  std::shared_ptr<SiftMatchingOptions> sift;

  bool Check() const;
};

class FeatureMatcher {
 public:
  virtual ~FeatureMatcher() = default;

  struct Image {
    // Unique identifier for the image. Allows a matcher to cache some
    // computations per image in consecutive calls to matching.
    image_t image_id = kInvalidImageId;
    // Sensor dimension in pixels of the image's camera.
    int width = 0;
    int height = 0;
    std::shared_ptr<const FeatureKeypoints> keypoints;
    std::shared_ptr<const FeatureDescriptors> descriptors;
  };

  static std::unique_ptr<FeatureMatcher> Create(
      const FeatureMatchingOptions& options);

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
  void AccessDatabase(const std::function<void(Database& database)>& func);

  const Camera& GetCamera(camera_t camera_id);
  const Frame& GetFrame(frame_t frame_id);
  const Image& GetImage(image_t image_id);
  const PosePrior* GetPosePriorOrNull(image_t image_id);
  std::shared_ptr<FeatureKeypoints> GetKeypoints(image_t image_id);
  std::shared_ptr<FeatureDescriptors> GetDescriptors(image_t image_id);
  FeatureMatches GetMatches(image_t image_id1, image_t image_id2);
  std::vector<frame_t> GetFrameIds();
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
  void MaybeLoadFrames();
  void MaybeLoadImages();
  void MaybeLoadPosePriors();

  const size_t cache_size_;
  const std::shared_ptr<Database> database_;
  std::mutex database_mutex_;
  std::unique_ptr<std::unordered_map<camera_t, Camera>> cameras_cache_;
  std::unique_ptr<std::unordered_map<frame_t, Frame>> frames_cache_;
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
