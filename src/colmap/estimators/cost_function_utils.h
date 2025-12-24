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

#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

template <typename T>
using EigenVector3Map = Eigen::Map<const Eigen::Matrix<T, 3, 1>>;
template <typename T>
using EigenQuaternionMap = Eigen::Map<const Eigen::Quaternion<T>>;

template <typename CostFunctor, int kNumResiduals, int... kParameterDims>
ceres::CostFunction* CreateAutoDiffCostFunction(
    CostFunctor* functor, std::integer_sequence<int, kParameterDims...>) {
  return new ceres::AutoDiffCostFunction<CostFunctor,
                                         kNumResiduals,
                                         kParameterDims...>(functor);
}

template <typename CostFunctor>
ceres::CostFunction* CreateAutoDiffCostFunction(CostFunctor* functor) {
  return CreateAutoDiffCostFunction<CostFunctor, CostFunctor::kNumResiduals>(
      functor, typename CostFunctor::kParameterDims{});
}

template <class DerivedCostFunctor, int NumResiduals, int... ParamDims>
class AutoDiffCostFunctor {
 public:
  static constexpr int kNumResiduals = NumResiduals;
  using kParameterDims = std::integer_sequence<int, ParamDims...>;

  template <typename... Args>
  static ceres::CostFunction* Create(Args&&... args) {
    return CreateAutoDiffCostFunction<DerivedCostFunctor>(
        new DerivedCostFunctor(std::forward<Args>(args)...));
  }

 private:
  AutoDiffCostFunctor() = default;
  friend DerivedCostFunctor;
};

// Cost functor for a single parameter against a fixed prior.
// Computes residual = param - prior.
template <int N>
class NormalPriorCostFunctor
    : public AutoDiffCostFunctor<NormalPriorCostFunctor<N>, N, N> {
 public:
  using VectorN = Eigen::Matrix<double, N, 1>;

  explicit NormalPriorCostFunctor(const VectorN& prior) : prior_(prior) {}

  template <typename T>
  bool operator()(const T* const param, T* residuals_ptr) const {
    Eigen::Map<Eigen::Matrix<T, N, 1>> residuals(residuals_ptr);
    residuals = Eigen::Map<const Eigen::Matrix<T, N, 1>>(param) -
                prior_.template cast<T>();
    return true;
  }

 private:
  const VectorN prior_;
};

// Cost functor for the difference between two parameters.
// Computes residual = param0 - param1.
template <int N>
class NormalErrorCostFunctor
    : public AutoDiffCostFunctor<NormalErrorCostFunctor<N>, N, N, N> {
 public:
  NormalErrorCostFunctor() = default;

  template <typename T>
  bool operator()(const T* const param0,
                  const T* const param1,
                  T* residuals_ptr) const {
    Eigen::Map<Eigen::Matrix<T, N, 1>> residuals(residuals_ptr);
    residuals = Eigen::Map<const Eigen::Matrix<T, N, 1>>(param0) -
                Eigen::Map<const Eigen::Matrix<T, N, 1>>(param1);
    return true;
  }
};

template <typename... Args>
auto LastValueParameterPack(Args&&... args) {
  return std::get<sizeof...(Args) - 1>(std::forward_as_tuple(args...));
}

// A cost function wrapper that whitens residuals with a given covariance.
// For example, to weight the reprojection error with an image measurement
// covariance, one can wrap it as:
//
//    using ReprojCostFunctor = ReprojErrorCostFunctor<PinholeCameraModel>;
//    ceres::CostFunction* cost_function =
//        CovarianceWeightedCostFunctor<ReprojCostFunctor>::Create(
//            point2D_cov, point2D));
template <class CostFunctor>
class CovarianceWeightedCostFunctor {
 public:
  static constexpr int kNumResiduals = CostFunctor::kNumResiduals;
  using kParameterDims = typename CostFunctor::kParameterDims;

  // Covariance or sqrt information matrix type.
  using CovMat = Eigen::Matrix<double, kNumResiduals, kNumResiduals>;

  template <typename... Args>
  explicit CovarianceWeightedCostFunctor(const CovMat& cov, Args&&... args)
      : left_sqrt_info_(LeftSqrtInformation(cov)),
        cost_(std::forward<Args>(args)...) {}

  template <typename... Args>
  static ceres::CostFunction* Create(const CovMat& cov, Args&&... args) {
    return CreateAutoDiffCostFunction(
        new CovarianceWeightedCostFunctor<CostFunctor>(
            cov, std::forward<Args>(args)...));
  }

  template <typename... Args>
  bool operator()(Args... args) const {
    if (!cost_(args...)) {
      return false;
    }

    auto residuals_ptr = LastValueParameterPack(args...);
    typedef typename std::remove_reference<decltype(*residuals_ptr)>::type T;
    Eigen::Map<Eigen::Matrix<T, kNumResiduals, 1>> residuals(residuals_ptr);
    residuals.applyOnTheLeft(left_sqrt_info_.template cast<T>());
    return true;
  }

 private:
  CovMat LeftSqrtInformation(const CovMat& cov) {
    return cov.inverse().llt().matrixL().transpose();
  }

  const CovMat left_sqrt_info_;
  const CostFunctor cost_;
};

}  // namespace colmap
