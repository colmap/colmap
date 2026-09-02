// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#pragma once

#include <memory>

#include <ceres/loss_function.h>

namespace colmap {

enum class CeresLossFunctionType { TRIVIAL, SOFT_L1, CAUCHY, HUBER };

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
