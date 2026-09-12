// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/gravity_refinement.h"
#include "colmap/estimators/rotation_averaging.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/base_controller.h"

#include <memory>
#include <vector>

namespace colmap {

struct RotationAveragingPipelineOptions {
  // The minimum number of matches for inlier matches to be considered.
  int min_num_matches = 0;

  // Whether to ignore the inlier matches of watermark image pairs.
  bool ignore_watermarks = false;

  // Names of images to reconstruct. If empty, all images are used.
  std::vector<std::string> image_names;

  // Number of threads.
  int num_threads = -1;

  // PRNG seed for all stochastic methods during reconstruction.
  // If -1 (default), the seed is derived from the current time
  // (non-deterministic). If >= 0, the pipeline is deterministic with the given
  // seed.
  int random_seed = -1;

  // Whether to decompose relative poses from two-view geometries.
  bool decompose_relative_pose = true;

  // Whether to refine gravity priors before rotation averaging.
  bool refine_gravity = false;

  // Options for gravity refinement.
  GravityRefinerOptions gravity_refiner;

  // Options for rotation averaging.
  RotationEstimatorOptions rotation_estimation;
};

class RotationAveragingPipeline : public BaseController {
 public:
  RotationAveragingPipeline(const RotationAveragingPipelineOptions& options,
                            std::shared_ptr<Database> database,
                            std::shared_ptr<Reconstruction> reconstruction);

  void Run() override;

 private:
  const RotationAveragingPipelineOptions options_;
  std::shared_ptr<DatabaseCache> database_cache_;
  std::shared_ptr<Reconstruction> reconstruction_;
};

}  // namespace colmap
