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
#include "colmap/sfm/global_mapper.h"
#include "colmap/util/base_controller.h"

#include <filesystem>
#include <memory>
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

  // Options for the global mapper.
  GlobalMapperOptions mapper;
};

class GlobalPipeline : public BaseController {
 public:
  GlobalPipeline(GlobalPipelineOptions options,
                 std::shared_ptr<Database> database,
                 std::shared_ptr<ReconstructionManager> reconstruction_manager);

  void Run() override;

 private:
  const GlobalPipelineOptions options_;
  std::shared_ptr<DatabaseCache> database_cache_;
  std::shared_ptr<ReconstructionManager> reconstruction_manager_;
};

}  // namespace colmap
