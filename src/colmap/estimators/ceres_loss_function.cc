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

#include "colmap/estimators/ceres_loss_function.h"

#include <cmath>
#include <stdexcept>

#include <ceres/ceres.h>

namespace colmap {

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
  }
  if (weight == 1.0) return loss;
  auto scaled_loss = std::make_unique<ceres::ScaledLoss>(
      loss.release(), weight, ceres::TAKE_OWNERSHIP);
  return scaled_loss;
}

}  // namespace colmap
