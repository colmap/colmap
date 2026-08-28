// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in COPYING.txt
// are met.

#pragma once

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match.h"

#include <memory>
#include <vector>

namespace colmap {
namespace mvs {

// Apple Metal implementation of dense PatchMatch stereo. The implementation
// is isolated in an Objective-C++ translation unit so the rest of COLMAP stays
// portable C++17 and does not expose Metal framework types.
class PatchMatchMetal {
 public:
  PatchMatchMetal(const PatchMatchOptions& options,
                  const PatchMatch::Problem& problem);
  ~PatchMatchMetal();

  PatchMatchMetal(const PatchMatchMetal&) = delete;
  PatchMatchMetal& operator=(const PatchMatchMetal&) = delete;

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
