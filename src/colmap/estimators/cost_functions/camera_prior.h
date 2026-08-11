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

#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

// Weighted prior on the parameters of a camera. Computes
// residuals = weights .* (params - priors), such that parameters with a zero
// weight are unconstrained.
template <typename CameraModel>
class CameraParamsPriorCostFunctor
    : public AutoDiffCostFunctor<CameraParamsPriorCostFunctor<CameraModel>,
                                 CameraModel::num_params,
                                 CameraModel::num_params> {
 public:
  using VectorN = Eigen::Matrix<double, CameraModel::num_params, 1>;

  CameraParamsPriorCostFunctor(const std::vector<double>& weights,
                               const std::vector<double>& priors) {
    THROW_CHECK_EQ(weights.size(), CameraModel::num_params);
    THROW_CHECK_EQ(priors.size(), CameraModel::num_params);
    weights_ = Eigen::Map<const VectorN>(weights.data());
    priors_ = Eigen::Map<const VectorN>(priors.data());
  }

  template <typename T>
  bool operator()(const T* const params, T* residuals_ptr) const {
    Eigen::Map<Eigen::Matrix<T, CameraModel::num_params, 1>> residuals(
        residuals_ptr);
    residuals = weights_.template cast<T>().cwiseProduct(
        Eigen::Map<const Eigen::Matrix<T, CameraModel::num_params, 1>>(params) -
        priors_.template cast<T>());
    return true;
  }

 private:
  VectorN weights_;
  VectorN priors_;
};

}  // namespace colmap
