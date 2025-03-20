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

#include "colmap/scene/reconstruction_manager.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/util/base_controller.h"

namespace colmap {

// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct IncrementalPipelineOptions {
  // The minimum number of matches for inlier matches to be considered.
  int min_num_matches = 15;

  // Whether to ignore the inlier matches of watermark image pairs.
  bool ignore_watermarks = false;

  // Whether to reconstruct multiple sub-models.
  bool multiple_models = true;

  // The number of sub-models to reconstruct.
  int max_num_models = 50;

  // The maximum number of overlapping images between sub-models. If the
  // current sub-models shares more than this number of images with another
  // model, then the reconstruction is stopped.
  int max_model_overlap = 20;

  // The minimum number of registered images of a sub-model, otherwise the
  // sub-model is discarded. Note that the first sub-model is always kept
  // independent of size.
  int min_model_size = 10;

  // The image identifiers used to initialize the reconstruction. Note that
  // only one or both image identifiers can be specified. In the former case,
  // the second image is automatically determined.
  int init_image_id1 = -1;
  int init_image_id2 = -1;

  // The number of trials to initialize the reconstruction.
  int init_num_trials = 200;

  // Whether to extract colors for reconstructed points.
  bool extract_colors = true;

  // The number of threads to use during reconstruction.
  int num_threads = -1;

  // Thresholds for filtering images with degenerate intrinsics.
  double min_focal_length_ratio = 0.1;
  double max_focal_length_ratio = 10.0;
  double max_extra_param = 1.0;

  // Which intrinsic parameters to optimize during the reconstruction.
  bool ba_refine_focal_length = true;
  bool ba_refine_principal_point = false;
  bool ba_refine_extra_params = true;

  // The minimum number of residuals per bundle adjustment problem to
  // enable multi-threading solving of the problems.
  int ba_min_num_residuals_for_cpu_multi_threading = 50000;

  // The number of images to optimize in local bundle adjustment.
  int ba_local_num_images = 6;

  // Ceres solver function tolerance for local bundle adjustment
  double ba_local_function_tolerance = 0.0;

  // The maximum number of local bundle adjustment iterations.
  int ba_local_max_num_iterations = 25;

  // The growth rates after which to perform global bundle adjustment.
  double ba_global_images_ratio = 1.1;
  double ba_global_points_ratio = 1.1;
  int ba_global_images_freq = 500;
  int ba_global_points_freq = 250000;

  // Ceres solver function tolerance for global bundle adjustment
  double ba_global_function_tolerance = 0.0;

  // The maximum number of global bundle adjustment iterations.
  int ba_global_max_num_iterations = 50;

  // The thresholds for iterative bundle adjustment refinements.
  int ba_local_max_refinements = 2;
  double ba_local_max_refinement_change = 0.001;
  int ba_global_max_refinements = 5;
  double ba_global_max_refinement_change = 0.0005;

  // Whether to use Ceres' CUDA sparse linear algebra library, if available.
  bool ba_use_gpu = false;
  std::string ba_gpu_index = "-1";

  // Whether to use priors on the camera positions.
  bool use_prior_position = false;

  // Whether to use a robust loss on prior camera positions.
  bool use_robust_loss_on_prior_position = false;

  // Threshold on the residual for the robust position prior loss
  // (chi2 for 3DOF at 95% = 7.815).
  double prior_position_loss_scale = 7.815;

  // Path to a folder with reconstruction snapshots during incremental
  // reconstruction. Snapshots will be saved according to the specified
  // frequency of registered images.
  std::string snapshot_path = "";
  int snapshot_images_freq = 0;

  // Optional list of image names to reconstruct. If no images are specified,
  // all images will be reconstructed by default.
  std::vector<std::string> image_names;

  // If reconstruction is provided as input, fix the existing image poses.
  bool fix_existing_images = false;

  IncrementalMapper::Options mapper;
  IncrementalTriangulator::Options triangulation;

  IncrementalMapper::Options Mapper() const;
  IncrementalTriangulator::Options Triangulation() const;
  BundleAdjustmentOptions LocalBundleAdjustment() const;
  BundleAdjustmentOptions GlobalBundleAdjustment() const;

  inline bool IsInitialPairProvided() const {
    return init_image_id1 != -1 && init_image_id2 != -1;
  }

  bool Check() const;
};

// Class that controls the incremental mapping procedure by iteratively
// initializing reconstructions from the same scene graph.
class IncrementalPipeline : public BaseController {
 public:
  enum CallbackType {
    INITIAL_IMAGE_PAIR_REG_CALLBACK,
    NEXT_IMAGE_REG_CALLBACK,
    LAST_IMAGE_REG_CALLBACK,
  };

  enum class Status { NO_INITIAL_PAIR, BAD_INITIAL_PAIR, SUCCESS, INTERRUPTED };

  IncrementalPipeline(
      std::shared_ptr<const IncrementalPipelineOptions> options,
      const std::string& image_path,
      const std::string& database_path,
      std::shared_ptr<class ReconstructionManager> reconstruction_manager);

  void Run();

  bool LoadDatabase();

  // getter functions for python pipelines
  const std::string& ImagePath() const { return image_path_; }
  const std::string& DatabasePath() const { return database_path_; }
  const std::shared_ptr<const IncrementalPipelineOptions>& Options() const {
    return options_;
  }
  const std::shared_ptr<class ReconstructionManager>& ReconstructionManager()
      const {
    return reconstruction_manager_;
  }
  const std::shared_ptr<class DatabaseCache>& DatabaseCache() const {
    return database_cache_;
  }

  void Reconstruct(IncrementalMapper& mapper,
                   const IncrementalMapper::Options& mapper_options,
                   bool continue_reconstruction);

  Status ReconstructSubModel(
      IncrementalMapper& mapper,
      const IncrementalMapper::Options& mapper_options,
      const std::shared_ptr<Reconstruction>& reconstruction);

  Status InitializeReconstruction(
      IncrementalMapper& mapper,
      const IncrementalMapper::Options& mapper_options,
      Reconstruction& reconstruction);

  void TriangulateReconstruction(
      const std::shared_ptr<Reconstruction>& reconstruction);

  bool CheckRunGlobalRefinement(const Reconstruction& reconstruction,
                                size_t ba_prev_num_reg_images,
                                size_t ba_prev_num_points);

 private:
  const std::shared_ptr<const IncrementalPipelineOptions> options_;
  const std::string image_path_;
  const std::string database_path_;
  std::shared_ptr<class ReconstructionManager> reconstruction_manager_;
  std::shared_ptr<class DatabaseCache> database_cache_;
};

}  // namespace colmap
