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

#define CASPAR_NUM_VARIANTS 15  // 2^4 - 1: all combinations with at least one variable param.
enum class FactorVariant {
  // r=0
  BASE,
  // r=1
  FIXED_POSE,
  FIXED_FOCAL_AND_EXTRA,
  FIXED_PRINCIPAL_POINT,
  FIXED_POINT,
  // r=2
  FIXED_POSE_FIXED_FOCAL_AND_EXTRA,
  FIXED_POSE_FIXED_PRINCIPAL_POINT,
  FIXED_POSE_FIXED_POINT,
  FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT,   // calibrated camera
  FIXED_FOCAL_AND_EXTRA_FIXED_POINT,
  FIXED_PRINCIPAL_POINT_FIXED_POINT,
  // r=3
  FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT,
  FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT,
  FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT,
  FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT,
};

struct VariantData {
  std::vector<unsigned int> pose_indices;
  std::vector<unsigned int> focal_and_extra_indices;
  std::vector<unsigned int> principal_point_indices;
  std::vector<unsigned int> point_indices;

  std::vector<StorageType> const_poses;             // 7 floats per factor
  std::vector<StorageType> const_focal_and_extra;   // FocalAndExtraSize() per factor
  std::vector<StorageType> const_principal_point;   // PrincipalPointSize() per factor
  std::vector<StorageType> const_points;            // 3 floats per factor

  std::vector<StorageType> pixels;  // 2 floats per factor
  size_t num_factors = 0;
};

struct ModelData {
  std::vector<StorageType> focal_and_extra_data;   // FocalAndExtraSize() entries per camera
  std::vector<StorageType> principal_point_data;   // PrincipalPointSize() per camera
  std::array<VariantData, CASPAR_NUM_VARIANTS>
      variants{};  // Indexed by FactorVariant
};

namespace colmap {

// Caspar-specific solver options. Field names and defaults mirror
// caspar::SolverParams; stored as double so they round-trip through COLMAP's
// OptionManager regardless of the StorageType (float or double) build.
struct CasparBundleAdjustmentOptions {
  int solver_iter_max = 100;
  int pcg_iter_max = 20;
  double diag_init = 1.0;
  double diag_min = 1e-12;
  double diag_scaling_up = 2.0;
  double diag_scaling_down = 0.333333;
  double diag_exit_value = 1e3;
  double score_exit_value = 0.0;
  double pcg_rel_error_exit = 1e-4;
  // Negative value disables the corresponding early-exit criterion.
  double pcg_rel_score_exit = -1.0;
  double pcg_rel_decrease_min = -1.0;
  double solver_rel_decrease_min = 1.0;
};

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
