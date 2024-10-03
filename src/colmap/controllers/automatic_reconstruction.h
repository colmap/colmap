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

#include "colmap/controllers/option_manager.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/util/threading.h"

#include <memory>
#include <string>

namespace colmap {

class AutomaticReconstructionController : public Thread {
 public:
  enum class DataType { INDIVIDUAL, VIDEO, INTERNET };
  enum class Quality { LOW, MEDIUM, HIGH, EXTREME };
  enum class Mesher { POISSON, DELAUNAY };

  struct Options {
    // The path to the workspace folder in which all results are stored.
    std::string workspace_path;

    // The path to the image folder which are used as input.
    std::string image_path;

    // The path to the mask folder which are used as input.
    std::string mask_path;

    // The path to the vocabulary tree for feature matching.
    std::string vocab_tree_path;

    // The type of input data used to choose optimal mapper settings.
    DataType data_type = DataType::INDIVIDUAL;

    // Whether to perform low- or high-quality reconstruction.
    Quality quality = Quality::HIGH;

    // Whether to use shared intrinsics or not.
    bool single_camera = false;

    // Whether to use shared intrinsics or not for all images in the same
    // sub-folder.
    bool single_camera_per_folder = false;

    // Which camera model to use for images.
    std::string camera_model = "SIMPLE_RADIAL";

    // Initial camera params for all images.
    std::string camera_params;

    // Whether to perform sparse mapping.
    bool sparse = true;

// Whether to perform dense mapping.
#if defined(COLMAP_CUDA_ENABLED)
    bool dense = true;
#else
    bool dense = false;
#endif

    // The meshing algorithm to be used.
    Mesher mesher = Mesher::POISSON;

    // The number of threads to use in all stages.
    int num_threads = -1;

    // Whether to use the GPU in feature extraction, feature matching, and
    // bundle adjustment.
    bool use_gpu = true;

    // Index of the GPU used for GPU stages. For multi-GPU computation in
    // feature extraction/matching, you should separate multiple GPU indices by
    // comma, e.g., "0,1,2,3". For single-GPU stages only the first GPU will be
    // used. By default, all available GPUs will be used in all stages.
    std::string gpu_index = "-1";
  };

  AutomaticReconstructionController(
      const Options& options,
      std::shared_ptr<ReconstructionManager> reconstruction_manager);

  void Stop() override;

 private:
  void Run() override;
  void RunFeatureExtraction();
  void RunFeatureMatching();
  void RunSparseMapper();
  void RunDenseMapper();

  const Options options_;
  OptionManager option_manager_;
  std::shared_ptr<ReconstructionManager> reconstruction_manager_;
  Thread* active_thread_;
  std::unique_ptr<Thread> feature_extractor_;
  std::unique_ptr<Thread> exhaustive_matcher_;
  std::unique_ptr<Thread> sequential_matcher_;
  std::unique_ptr<Thread> vocab_tree_matcher_;
};

}  // namespace colmap
