// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/index.h"
#include "colmap/feature/types.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/database.h"
#include "colmap/scene/image.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/cache.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/types.h"

#include <memory>
#include <mutex>
#include <optional>

namespace colmap {

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
  const PosePrior* FindImagePosePriorOrNull(image_t image_id);
  std::shared_ptr<FeatureKeypoints> GetKeypoints(image_t image_id);
  std::shared_ptr<FeatureDescriptors> GetDescriptors(image_t image_id);
  FeatureMatches GetMatches(image_t image_id1, image_t image_id2);
  TwoViewGeometry GetTwoViewGeometry(image_t image_id1, image_t image_id2);
  std::vector<frame_t> GetFrameIds();
  std::vector<image_t> GetImageIds();
  ThreadSafeLRUCache<image_t, FeatureDescriptorIndex>&
  GetFeatureDescriptorIndexCache();

  bool ExistsKeypoints(image_t image_id);
  bool ExistsDescriptors(image_t image_id);

  bool ExistsMatches(image_t image_id1, image_t image_id2);
  bool ExistsTwoViewGeometry(image_t image_id1, image_t image_id2);
  bool ExistsInlierMatches(image_t image_id1, image_t image_id2);

  void UpdateTwoViewGeometry(image_t image_id1,
                             image_t image_id2,
                             const TwoViewGeometry& two_view_geometry);

  void WriteMatches(image_t image_id1,
                    image_t image_id2,
                    const FeatureMatches& matches);
  void WriteTwoViewGeometry(image_t image_id1,
                            image_t image_id2,
                            const TwoViewGeometry& two_view_geometry);

  void DeleteMatches(image_t image_id1, image_t image_id2);
  void DeleteTwoViewGeometry(image_t image_id1, image_t image_id2);
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
  std::unique_ptr<NodeHashMap<camera_t, Camera>> cameras_cache_;
  std::unique_ptr<NodeHashMap<frame_t, Frame>> frames_cache_;
  std::unique_ptr<NodeHashMap<image_t, Image>> images_cache_;
  std::unique_ptr<NodeHashMap<image_t, PosePrior>> pose_priors_cache_;
  std::unique_ptr<ThreadSafeLRUCache<image_t, FeatureKeypoints>>
      keypoints_cache_;
  std::unique_ptr<ThreadSafeLRUCache<image_t, FeatureDescriptors>>
      descriptors_cache_;
  std::unique_ptr<ThreadSafeLRUCache<image_t, bool>> keypoints_exists_cache_;
  std::unique_ptr<ThreadSafeLRUCache<image_t, bool>> descriptors_exists_cache_;
  ThreadSafeLRUCache<image_t, FeatureDescriptorIndex> descriptor_index_cache_;
  std::optional<size_t> max_num_keypoints_;
};

}  // namespace colmap
