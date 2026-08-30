// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "colmap/estimators/ceres_loss.h"

#include <cmath>
#include <stdexcept>

#include <ceres/ceres.h>

namespace colmap {

bool IsValidCeresLossFunction(const CeresLossFunctionType type,
                              const double scale,
                              const double weight) {
  // Preserve the pre-centralization bundle-adjustment contract: non-negative
  // infinite scales were accepted, while NaN and negative scales were not.
  if (std::isnan(scale) || !std::isfinite(weight)) {
    return false;
  }
  switch (type) {
    case CeresLossFunctionType::TRIVIAL:
    case CeresLossFunctionType::SOFT_L1:
    case CeresLossFunctionType::CAUCHY:
    case CeresLossFunctionType::HUBER:
      return scale >= 0.0 && weight > 0.0;
  }
  return false;
}

std::unique_ptr<ceres::LossFunction> CreateCeresLossFunction(
    const CeresLossFunctionType type, const double scale, const double weight) {
  if (!IsValidCeresLossFunction(type, scale, weight)) {
    throw std::invalid_argument("invalid Ceres loss configuration");
  }

  std::unique_ptr<ceres::LossFunction> loss;
  switch (type) {
    case CeresLossFunctionType::TRIVIAL:
      loss = std::make_unique<ceres::TrivialLoss>();
      break;
    case CeresLossFunctionType::SOFT_L1:
      loss = std::make_unique<ceres::SoftLOneLoss>(scale);
      break;
    case CeresLossFunctionType::CAUCHY:
      loss = std::make_unique<ceres::CauchyLoss>(scale);
      break;
    case CeresLossFunctionType::HUBER:
      loss = std::make_unique<ceres::HuberLoss>(scale);
      break;
  }
  if (weight == 1.0) return loss;
  auto scaled_loss = std::make_unique<ceres::ScaledLoss>(
      loss.get(), weight, ceres::TAKE_OWNERSHIP);
  loss.release();
  return scaled_loss;
}

bool CeresLossFunctionConfig::Check() const {
  return IsValidCeresLossFunction(type, scale, weight);
}

std::unique_ptr<ceres::LossFunction> CeresLossFunctionConfig::Create() const {
  return CreateCeresLossFunction(type, scale, weight);
}

}  // namespace colmap
