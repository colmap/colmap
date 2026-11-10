// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/mat.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match.h"

#include <memory>
#include <vector>

namespace colmap {
namespace mvs {

// Returns the number of available Metal devices.
int GetNumMetalDevices();

// Metal implementation of the patch match stereo algorithm for Apple GPUs. The
// interface and the algorithm are the same as for PatchMatchCuda (see
// patch_match_cuda.cu); the kernels are in patch_match_metal.metal. The
// shaders are compiled at runtime when the first problem is processed.
//
// The class hides all Objective-C types from the header, so that it can be
// included from plain C++ code.
class PatchMatchMetal {
 public:
  PatchMatchMetal(const PatchMatchOptions& options,
                  const PatchMatch::Problem& problem);
  ~PatchMatchMetal();

  void Run();

  DepthMap GetDepthMap() const;
  NormalMap GetNormalMap() const;
  Mat<float> GetSelProbMap() const;
  std::vector<int> GetConsistentImageIdxs() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mvs
}  // namespace colmap
