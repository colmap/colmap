// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/enum_utils.h"

#include <memory>

#include <ceres/loss_function.h>

namespace colmap {

MAKE_ENUM_CLASS_OVERLOAD_STREAM(
    CeresLossFunctionType, 0, TRIVIAL, SOFT_L1, CAUCHY, HUBER);

// Standard construction accepts a non-negative `robust_scale` and finite
// positive `weight`.
bool IsValidCeresLossFunction(CeresLossFunctionType type,
                              double robust_scale,
                              double weight = 1.0);

// Create a standard Ceres loss function. For robust losses, `robust_scale`
// determines the residual at which robustification takes place. The weight
// multiplies the loss function output.
std::unique_ptr<ceres::LossFunction> CreateCeresLossFunction(
    CeresLossFunctionType type, double robust_scale, double weight = 1.0);

}  // namespace colmap
