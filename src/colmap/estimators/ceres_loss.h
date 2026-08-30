// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#pragma once

#include <memory>

#include <ceres/loss_function.h>

namespace colmap {

enum class CeresLossFunctionType { TRIVIAL, SOFT_L1, CAUCHY, HUBER };

// Standard construction accepts non-negative scale (including positive
// infinity for compatibility) and finite non-negative weight. Callers may
// enforce narrower policy without duplicating the common checks.
bool IsValidCeresLossFunction(CeresLossFunctionType type,
                              double scale,
                              double weight = 1.0);

// Constructs a standard Ceres loss and wraps it in an owning ScaledLoss when
// weight differs from one.
std::unique_ptr<ceres::LossFunction> CreateCeresLossFunction(
    CeresLossFunctionType type, double scale, double weight = 1.0);

struct CeresLossFunctionConfig {
  CeresLossFunctionType type = CeresLossFunctionType::TRIVIAL;
  double scale = 1.0;
  double weight = 1.0;

  bool Check() const;
  std::unique_ptr<ceres::LossFunction> Create() const;
};

}  // namespace colmap
