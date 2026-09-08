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
#include "colmap/util/logging.h"

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

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

// Whitens the residuals and jacobians of an inner cost function with a given
// covariance. Whitening is linear, so applying it to the evaluated jacobians
// is equivalent to, and cheaper than, propagating it through autodiff.
template <class CostFunctor,
          class ParameterDims = typename CostFunctor::kParameterDims>
class CovarianceWeightedCostFunction;

template <class CostFunctor, int... ParameterDims>
class CovarianceWeightedCostFunction<
    CostFunctor,
    std::integer_sequence<int, ParameterDims...>>
    : public ceres::SizedCostFunction<CostFunctor::kNumResiduals,
                                      ParameterDims...> {
 public:
  static constexpr int kNumResiduals = CostFunctor::kNumResiduals;
  using CovMat = Eigen::Matrix<double, kNumResiduals, kNumResiduals>;

  CovarianceWeightedCostFunction(const CovMat& cov, ceres::CostFunction* cost)
      : left_sqrt_info_(cov.inverse().llt().matrixL().transpose()),
        cost_(cost) {
    // The wrapped cost function need not come from CostFunctor, so a shape
    // mismatch would run the jacobian maps past the buffers Ceres allocates.
    THROW_CHECK_EQ(cost_->num_residuals(), kNumResiduals);
    const std::vector<int32_t> expected_parameter_block_sizes = {
        ParameterDims...};
    THROW_CHECK(cost_->parameter_block_sizes() ==
                expected_parameter_block_sizes);
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    if (!cost_->Evaluate(parameters, residuals, jacobians)) {
      return false;
    }
    Eigen::Map<Eigen::Matrix<double, kNumResiduals, 1>>(residuals)
        .applyOnTheLeft(left_sqrt_info_);
    if (jacobians != nullptr) {
      WhitenJacobians(jacobians,
                      std::make_index_sequence<sizeof...(ParameterDims)>{});
    }
    return true;
  }

 private:
  template <size_t kIndex, int kDim>
  void WhitenJacobian(double** jacobians) const {
    if (jacobians[kIndex] == nullptr) {
      return;
    }
    // Eigen requires single-column matrices to be column major.
    constexpr int kOptions = kDim == 1 ? Eigen::ColMajor : Eigen::RowMajor;
    Eigen::Map<Eigen::Matrix<double, kNumResiduals, kDim, kOptions>>(
        jacobians[kIndex])
        .applyOnTheLeft(left_sqrt_info_);
  }

  // Expands the index and dimension packs in parallel. Indexing a static
  // constexpr array here instead is rejected by clang.
  template <size_t... kIndices>
  void WhitenJacobians(double** jacobians,
                       std::index_sequence<kIndices...>) const {
    (WhitenJacobian<kIndices, ParameterDims>(jacobians), ...);
  }

  const CovMat left_sqrt_info_;
  const std::unique_ptr<ceres::CostFunction> cost_;
};

// Whitens the given cost functor with a covariance. For example, to weight
// the reprojection error with an image measurement covariance:
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
  static ceres::CostFunction* Create(const CovMat& cov, Args&&... args) {
    return new CovarianceWeightedCostFunction<CostFunctor>(
        cov,
        CreateAutoDiffCostFunction(
            new CostFunctor(std::forward<Args>(args)...)));
  }
};

// Whitens residuals with per-residual standard deviations, broadcasting a
// single one over all residuals. Equivalent to CovarianceWeightedCostFunctor
// with a diagonal covariance, but avoids its inverse, Cholesky factorization,
// and dense product. A diagonal is as cheap through autodiff as applied to the
// evaluated jacobians, so this stays a functor. For example, to weight the
// reprojection error with isotropic image measurement noise:
//
//    using ReprojCostFunctor = ReprojErrorCostFunctor<PinholeCameraModel>;
//    ceres::CostFunction* cost_function =
//        ScaleWeightedCostFunctor<ReprojCostFunctor>::Create(stddev, point2D);
template <class CostFunctor>
class ScaleWeightedCostFunctor {
 public:
  static constexpr int kNumResiduals = CostFunctor::kNumResiduals;
  using kParameterDims = typename CostFunctor::kParameterDims;

  using StddevVec = Eigen::Matrix<double, kNumResiduals, 1>;

  template <typename... Args>
  explicit ScaleWeightedCostFunctor(const StddevVec& stddevs, Args&&... args)
      : sqrt_info_(stddevs.cwiseInverse()),
        cost_(std::forward<Args>(args)...) {}

  template <typename... Args>
  explicit ScaleWeightedCostFunctor(double stddev, Args&&... args)
      : sqrt_info_(StddevVec::Constant(1.0 / stddev)),
        cost_(std::forward<Args>(args)...) {}

  template <typename... Args>
  static ceres::CostFunction* Create(const StddevVec& stddevs, Args&&... args) {
    return CreateAutoDiffCostFunction(new ScaleWeightedCostFunctor<CostFunctor>(
        stddevs, std::forward<Args>(args)...));
  }

  template <typename... Args>
  static ceres::CostFunction* Create(double stddev, Args&&... args) {
    return CreateAutoDiffCostFunction(new ScaleWeightedCostFunctor<CostFunctor>(
        stddev, std::forward<Args>(args)...));
  }

  template <typename... Args>
  bool operator()(Args... args) const {
    if (!cost_(args...)) {
      return false;
    }

    auto residuals_ptr = LastValueParameterPack(args...);
    using T = typename std::remove_reference<decltype(*residuals_ptr)>::type;
    Eigen::Map<Eigen::Matrix<T, kNumResiduals, 1>> residuals(residuals_ptr);
    residuals.array() *= sqrt_info_.array();
    return true;
  }

 private:
  // Inverse stddevs, so that evaluation multiplies rather than divides.
  const StddevVec sqrt_info_;
  const CostFunctor cost_;
};

}  // namespace colmap
