// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/feature_matching.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/matcher.h"
#include "colmap/util/opengl_utils.h"
#include "colmap/util/threading.h"

#include <memory>
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
  using Input = FeatureMatcherData;
  using Output = FeatureMatcherData;

  FeatureMatcherWorker(const FeatureMatchingOptions& matching_options,
                       const TwoViewGeometryOptions& geometry_options,
                       const std::shared_ptr<FeatureMatcherCache>& cache,
                       JobQueue<Input>* input_queue,
                       JobQueue<Output>* output_queue);

 private:
  void Run() override;

  FeatureMatchingOptions matching_options_;
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
  FeatureMatcherController(const FeatureMatchingOptions& matching_options,
                           const TwoViewGeometryOptions& geometry_options,
                           std::shared_ptr<FeatureMatcherCache> cache);

  ~FeatureMatcherController();

  // Setup the matchers and return if successful.
  bool Setup();

  // Match one batch of multiple image pairs.
  void Match(const std::vector<std::pair<image_t, image_t>>& image_pairs);

 private:
  FeatureMatchingOptions matching_options_;
  TwoViewGeometryOptions geometry_options_;
  std::shared_ptr<FeatureMatcherCache> cache_;

  bool is_setup_;

  std::vector<std::unique_ptr<FeatureMatcherWorker>> matchers_;
  std::vector<std::unique_ptr<FeatureMatcherWorker>> guided_matchers_;
  std::vector<std::unique_ptr<Thread>> verifiers_;

  JobQueue<FeatureMatcherData> matcher_queue_;
  JobQueue<FeatureMatcherData> verifier_queue_;
  JobQueue<FeatureMatcherData> guided_matcher_queue_;
  JobQueue<FeatureMatcherData> output_queue_;
};

class GeometricVerifierController {
 public:
  GeometricVerifierController(const GeometricVerifierOptions& verifier_options,
                              const TwoViewGeometryOptions& geometry_options,
                              std::shared_ptr<FeatureMatcherCache> cache);

  const GeometricVerifierOptions& Options() const;
  GeometricVerifierOptions& Options();

  ~GeometricVerifierController();

  // Setup the verifiers and return if successful.
  bool Setup();

  // Verify one batch of multiple image pairs.
  void Verify(const std::vector<std::pair<image_t, image_t>>& image_pairs);

 private:
  TwoViewGeometryOptions geometry_options_;
  std::shared_ptr<FeatureMatcherCache> cache_;
  GeometricVerifierOptions options_;

  bool is_setup_;

  std::vector<std::unique_ptr<Thread>> verifiers_;

  JobQueue<FeatureMatcherData> verifier_queue_;
  JobQueue<FeatureMatcherData> output_queue_;
};

}  // namespace colmap
