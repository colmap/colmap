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

#include "colmap/estimators/bundle_adjustment.h"

#include <array>
#include <cstddef>
#include <vector>
#ifdef CASPAR_ENABLED
#include <solver.h>
#endif

#ifdef CASPAR_USE_DOUBLE
typedef double StorageType;
#else
typedef float StorageType;
#endif

#define CASPAR_NUM_VARIANTS 4
enum class FactorVariant {
  BASE,
  FIXED_POSE,
  FIXED_POINT,
  FIXED_POSE_FIXED_POINT
};

struct VariantData {
  // Indexed args, tunable nodes (empty when param is fixed)
  std::vector<unsigned int> pose_indices;
  std::vector<unsigned int> calib_indices;
  std::vector<unsigned int> point_indices;

  // Constant data
  std::vector<StorageType> const_poses;   // 7 entries per factor
  std::vector<StorageType> const_points;  // 3 entries per factor

  // Always present
  std::vector<StorageType> pixels;  // 2 entries per factor
  size_t num_factors = 0;
};

struct ModelData {
  std::vector<StorageType> calib_data;
  std::array<VariantData, CASPAR_NUM_VARIANTS>
      variants{};  // Indexed by FactorVariant
};

namespace colmap {
std::unique_ptr<BundleAdjuster> CreateDefaultCasparBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction);

#ifdef CASPAR_ENABLED
struct CasparBundleAdjustmentSummary : public BundleAdjustmentSummary {
  static std::shared_ptr<CasparBundleAdjustmentSummary> Create(
      const caspar::SolveResult& caspar_summary);
};
#endif
}  // namespace colmap
