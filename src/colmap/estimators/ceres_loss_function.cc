// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/ceres_loss_function.h"

#include <cmath>
#include <stdexcept>

#include <ceres/ceres.h>

namespace colmap {

namespace {

// Redescending Geman-McClure loss function:
//   rho(s) = (c * s) / (c + s), where c = a^2 and s = ||r||^2.
// Scaled so rho'(0) = 1, matching Ceres's CauchyLoss convention.
class GemanMcClureLoss final : public ceres::LossFunction {
 public:
  explicit GemanMcClureLoss(const double a) : c_(a * a), inv_c_(1.0 / c_) {}

  void Evaluate(const double s, double rho[3]) const override {
    if (s > c_) {
      const double t = c_ / s;
      const double inv = 1.0 / (1.0 + t);
      const double u = t * inv;
      rho[0] = c_ * inv;
      rho[1] = u * u;
      rho[2] = -2.0 * rho[1] * (inv / s);
    } else {
      const double inv = 1.0 / (1.0 + s * inv_c_);
      rho[0] = s * inv;
      rho[1] = inv * inv;
      rho[2] = -2.0 * rho[1] * inv * inv_c_;
    }
  }

 private:
  const double c_;
  const double inv_c_;
};

}  // namespace

bool IsValidCeresLossFunction(const CeresLossFunctionType type,
                              const double robust_scale,
                              const double weight) {
  // Preserve the pre-centralization bundle-adjustment contract: non-negative
  // infinite scales were accepted, while NaN and negative scales were not.
  if (std::isnan(robust_scale) || !std::isfinite(weight)) {
    return false;
  }
  switch (type) {
    case CeresLossFunctionType::TRIVIAL:
    case CeresLossFunctionType::SOFT_L1:
    case CeresLossFunctionType::CAUCHY:
    case CeresLossFunctionType::HUBER:
      return robust_scale >= 0.0 && weight > 0.0;
    case CeresLossFunctionType::GEMAN_MCCLURE: {
      const double c = robust_scale * robust_scale;
      return robust_scale > 0.0 && c > 0.0 && std::isfinite(c) &&
             std::isfinite(2.0 / c) && weight > 0.0;
    }
  }
  return false;
}

std::unique_ptr<ceres::LossFunction> CreateCeresLossFunction(
    const CeresLossFunctionType type,
    const double robust_scale,
    const double weight) {
  if (!IsValidCeresLossFunction(type, robust_scale, weight)) {
    throw std::invalid_argument("invalid Ceres loss configuration");
  }

  std::unique_ptr<ceres::LossFunction> loss;
  switch (type) {
    case CeresLossFunctionType::TRIVIAL:
      loss = std::make_unique<ceres::TrivialLoss>();
      break;
    case CeresLossFunctionType::SOFT_L1:
      loss = std::make_unique<ceres::SoftLOneLoss>(robust_scale);
      break;
    case CeresLossFunctionType::CAUCHY:
      loss = std::make_unique<ceres::CauchyLoss>(robust_scale);
      break;
    case CeresLossFunctionType::HUBER:
      loss = std::make_unique<ceres::HuberLoss>(robust_scale);
      break;
    case CeresLossFunctionType::GEMAN_MCCLURE:
      loss = std::make_unique<GemanMcClureLoss>(robust_scale);
      break;
  }
  if (weight == 1.0) return loss;
  auto scaled_loss = std::make_unique<ceres::ScaledLoss>(
      loss.release(), weight, ceres::TAKE_OWNERSHIP);
  return scaled_loss;
}

}  // namespace colmap
