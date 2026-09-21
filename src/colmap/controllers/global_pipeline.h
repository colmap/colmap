// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/reconstruction_manager.h"
#include "colmap/sfm/global_mapper.h"
#include "colmap/util/base_controller.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace colmap {

struct GlobalPipelineOptions {
  // The minimum number of matches for inlier matches to be considered.
  int min_num_matches = 15;

  // Whether to ignore the inlier matches of watermark image pairs.
  bool ignore_watermarks = false;

  // Names of images to reconstruct. If empty, all images are used.
  std::vector<std::string> image_names;

  // The image path at which to find the images to extract point colors.
  std::filesystem::path image_path;

  // Number of threads for parallel processing.
  int num_threads = -1;

  // Random seed for reproducibility.
  int random_seed = -1;

  // Whether to decompose relative poses from two-view geometries.
  bool decompose_relative_pose = true;

  // If true (default), reconstruct every connected component of the view graph
  // (one model per component). If false, reconstruct only the largest connected
  // component.
  bool multiple_models = true;

  // Minimum number of registered frames for a reconstruction to be kept.
  // Reconstructions with fewer registered frames are discarded.
  int min_model_size = 3;

  // Options for the global mapper.
  GlobalMapperOptions mapper;
};

class GlobalPipeline : public BaseController {
 public:
  enum CallbackType {
    // Triggered after global positioning, after each global refinement
    // iteration, and after retriangulation, so the in-progress reconstruction
    // can be rendered.
    MODEL_UPDATE_CALLBACK,
  };

  GlobalPipeline(GlobalPipelineOptions options,
                 std::shared_ptr<Database> database,
                 std::shared_ptr<ReconstructionManager> reconstruction_manager);

  void Run() override;

 private:
  struct ReconstructionStats {
    // Number of components that failed during rotation averaging or mapping.
    size_t num_failed = 0;

    // Number of components discarded for having too few registered frames.
    size_t num_too_small = 0;
  };

  // Run the full global SfM pipeline on the given database cache and return
  // the resulting reconstruction, or nullopt if mapping fails. The in-progress
  // reconstruction is added to the manager so callbacks can render it. The
  // caller decides whether to keep it.
  std::optional<std::shared_ptr<Reconstruction>> ReconstructSingleComponent(
      const std::shared_ptr<const DatabaseCache>& database_cache,
      const GlobalMapperOptions& mapper_options);

  // Partition the input view graph once using rotation averaging and
  // reconstruct each resulting component at most once.
  ReconstructionStats ReconstructMultiComponents(
      const GlobalMapperOptions& mapper_options);

  const GlobalPipelineOptions options_;
  std::shared_ptr<DatabaseCache> database_cache_;
  std::shared_ptr<ReconstructionManager> reconstruction_manager_;
};

}  // namespace colmap
