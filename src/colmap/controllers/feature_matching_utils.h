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

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/scene/database.h"
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

class FeatureMatcherWorker : public Thread {
 public:
  typedef FeatureMatcherData Input;
  typedef FeatureMatcherData Output;

  FeatureMatcherWorker(const SiftMatchingOptions& matching_options,
                       const TwoViewGeometryOptions& geometry_options,
                       const std::shared_ptr<FeatureMatcherCache>& cache,
                       JobQueue<Input>* input_queue,
                       JobQueue<Output>* output_queue);

  void SetMaxNumMatches(int max_num_matches);

 private:
  void Run() override;

  SiftMatchingOptions matching_options_;
  TwoViewGeometryOptions geometry_options_;
  std::shared_ptr<FeatureMatcherCache> cache_;
  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;

  std::unique_ptr<OpenGLContextManager> opengl_context_;
};

// Multi-threaded and multi-GPU SIFT feature matcher, which writes the computed
// results to the database and skips already matched image pairs. To improve
// performance of the matching by taking advantage of caching and database
// transactions, pass multiple images to the `Match` function. Note that the
// database should be in an active transaction while calling `Match`.
class FeatureMatcherController {
 public:
  FeatureMatcherController(
      const SiftMatchingOptions& matching_options,
      const TwoViewGeometryOptions& two_view_geometry_options,
      std::shared_ptr<FeatureMatcherCache> cache);

  ~FeatureMatcherController();

  // Setup the matchers and return if successful.
  bool Setup();

  // Match one batch of multiple image pairs.
  void Match(const std::vector<std::pair<image_t, image_t>>& image_pairs);

 private:
  SiftMatchingOptions matching_options_;
  TwoViewGeometryOptions geometry_options_;
  std::shared_ptr<FeatureMatcherCache> cache_;

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
