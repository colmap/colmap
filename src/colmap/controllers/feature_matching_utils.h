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

#pragma once

#include "colmap/feature/sift.h"
#include "colmap/scene/database.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/cache.h"
#include "colmap/util/opengl_utils.h"
#include "colmap/util/threading.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace colmap {

struct FeatureMatcherData {
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  FeatureMatches matches;
  TwoViewGeometry two_view_geometry;
};

// Cache for feature matching to minimize database access during matching.
class FeatureMatcherCache {
 public:
  FeatureMatcherCache(size_t cache_size, const Database* database);

  void Setup();

  const Camera& GetCamera(camera_t camera_id) const;
  const Image& GetImage(image_t image_id) const;
  std::shared_ptr<FeatureKeypoints> GetKeypoints(image_t image_id);
  std::shared_ptr<FeatureDescriptors> GetDescriptors(image_t image_id);
  FeatureMatches GetMatches(image_t image_id1, image_t image_id2);
  std::vector<image_t> GetImageIds() const;

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
  const Database* database_;
  std::mutex database_mutex_;
  std::unordered_map<camera_t, Camera> cameras_cache_;
  std::unordered_map<image_t, Image> images_cache_;
  std::unique_ptr<LRUCache<image_t, std::shared_ptr<FeatureKeypoints>>>
      keypoints_cache_;
  std::unique_ptr<LRUCache<image_t, std::shared_ptr<FeatureDescriptors>>>
      descriptors_cache_;
  std::unique_ptr<LRUCache<image_t, bool>> keypoints_exists_cache_;
  std::unique_ptr<LRUCache<image_t, bool>> descriptors_exists_cache_;
};

class FeatureMatcherWorker : public Thread {
 public:
  typedef FeatureMatcherData Input;
  typedef FeatureMatcherData Output;

  FeatureMatcherWorker(const SiftMatchingOptions& options,
                       FeatureMatcherCache* cache,
                       JobQueue<Input>* input_queue,
                       JobQueue<Output>* output_queue);

  void SetMaxNumMatches(int max_num_matches);

 private:
  void Run() override;

  std::shared_ptr<FeatureKeypoints> GetKeypointsPtr(int index,
                                                    image_t image_id);
  std::shared_ptr<FeatureDescriptors> GetDescriptorsPtr(int index,
                                                        image_t image_id);

  SiftMatchingOptions options_;
  FeatureMatcherCache* cache_;
  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;

  std::unique_ptr<OpenGLContextManager> opengl_context_;

  std::array<image_t, 2> prev_keypoints_image_ids_;
  std::array<std::shared_ptr<FeatureKeypoints>, 2> prev_keypoints_;
  std::array<image_t, 2> prev_descriptors_image_ids_;
  std::array<std::shared_ptr<FeatureDescriptors>, 2> prev_descriptors_;
};

// Multi-threaded and multi-GPU SIFT feature matcher, which writes the computed
// results to the database and skips already matched image pairs. To improve
// performance of the matching by taking advantage of caching and database
// transactions, pass multiple images to the `Match` function. Note that the
// database should be in an active transaction while calling `Match`.
class FeatureMatcherController {
 public:
  FeatureMatcherController(const SiftMatchingOptions& options,
                           Database* database,
                           FeatureMatcherCache* cache);

  ~FeatureMatcherController();

  // Setup the matchers and return if successful.
  bool Setup();

  // Match one batch of multiple image pairs.
  void Match(const std::vector<std::pair<image_t, image_t>>& image_pairs);

 private:
  SiftMatchingOptions options_;
  Database* database_;
  FeatureMatcherCache* cache_;

  bool is_setup_;

  std::vector<std::unique_ptr<FeatureMatcherWorker>> matchers_;
  std::vector<std::unique_ptr<FeatureMatcherWorker>> guided_matchers_;
  std::vector<std::unique_ptr<Thread>> verifiers_;
  std::unique_ptr<ThreadPool> thread_pool_;

  JobQueue<FeatureMatcherData> matcher_queue_;
  JobQueue<FeatureMatcherData> verifier_queue_;
  JobQueue<FeatureMatcherData> guided_matcher_queue_;
  JobQueue<FeatureMatcherData> output_queue_;
};

}  // namespace colmap
