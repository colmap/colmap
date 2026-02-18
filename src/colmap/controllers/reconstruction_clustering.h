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

#include "colmap/scene/reconstruction.h"
#include "colmap/scene/reconstruction_clustering.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/util/base_controller.h"

#include <memory>

namespace colmap {

// Controller that clusters frames from a reconstruction
// and splits it into multiple reconstructions based on clustering.
// Note: this module is experimental and should be verified carefully
// before use in production pipelines.
class ReconstructionClustererController : public BaseController {
 public:
  ReconstructionClustererController(
      const ReconstructionClusteringOptions& options,
      std::shared_ptr<Reconstruction> reconstruction,
      std::shared_ptr<ReconstructionManager> reconstruction_manager);

  // Runs the pruning and clustering algorithm.
  // Results are stored in the reconstruction manager passed to the constructor.
  void Run() override;

 private:
  const ReconstructionClusteringOptions options_;
  std::shared_ptr<Reconstruction> reconstruction_;
  std::shared_ptr<ReconstructionManager> reconstruction_manager_;
};

}  // namespace colmap
