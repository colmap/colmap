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

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

#include <cmath>

#include "ceres/ceres.h"

inline void SetQuaternionManifold(ceres::Problem* problem, double* quat_xyzw) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  problem->SetManifold(quat_xyzw, new ceres::EigenQuaternionManifold);
#else
  problem->SetParameterization(quat_xyzw,
                               new ceres::EigenQuaternionParameterization);
#endif
}

inline void SetSubsetManifold(int size,
                              const std::vector<int>& constant_params,
                              ceres::Problem* problem,
                              double* params) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  problem->SetManifold(params,
                       new ceres::SubsetManifold(size, constant_params));
#else
  problem->SetParameterization(
      params, new ceres::SubsetParameterization(size, constant_params));
#endif
}

template <int size>
inline void SetSphereManifold(ceres::Problem* problem, double* params) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  problem->SetManifold(params, new ceres::SphereManifold<size>);
#else
  problem->SetParameterization(
      params, new ceres::HomogeneousVectorParameterization(size));
#endif
}

// Use an exponential function to ensure the variable to be strictly positive
// Generally applicable for scale parameters (e.g. in colmap::Sim3d)
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
template <int AmbientSpaceDimension>
class PositiveExponentialManifold : public ceres::Manifold {
 public:
  static_assert(ceres::DYNAMIC == Eigen::Dynamic,
                "ceres::DYNAMIC needs to be the same as Eigen::Dynamic.");

  PositiveExponentialManifold() : size_{AmbientSpaceDimension} {}
  explicit PositiveExponentialManifold(int size) : size_{size} {
    if (AmbientSpaceDimension != Eigen::Dynamic) {
      CHECK_EQ(AmbientSpaceDimension, size)
          << "Specified size by template parameter differs from the supplied "
             "one.";
    } else {
      CHECK_GT(size_, 0)
          << "The size of the manifold needs to be a positive integer.";
    }
  }

  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override {
    for (int i = 0; i < size_; ++i) {
      x_plus_delta[i] = x[i] * std::exp(delta[i]);
    }
    return true;
  }

  bool PlusJacobian(const double* x, double* jacobian) const override {
    for (int i = 0; i < size_; ++i) {
      jacobian[size_ * i + i] = x[i];
    }
    return true;
  }

  virtual bool Minus(const double* y,
                     const double* x,
                     double* y_minus_x) const override {
    for (int i = 0; i < size_; ++i) {
      y_minus_x[i] = std::log(y[i] / x[i]);
    }
    return true;
  }

  virtual bool MinusJacobian(const double* x, double* jacobian) const override {
    for (int i = 0; i < size_; ++i) {
      jacobian[size_ * i + i] = 1.0 / x[i];
    }
    return true;
  }

  int AmbientSize() const override {
    return AmbientSpaceDimension == ceres::DYNAMIC ? size_
                                                   : AmbientSpaceDimension;
  }
  int TangentSize() const override { return AmbientSize(); }

 private:
  const int size_{};
};
#else
class PositiveExponentialParameterization
    : public ceres::LocalParameterization {
 public:
  explicit PositiveExponentialParameterization(int size) : size_{size} {
    CHECK_GT(size_, 0)
        << "The size of the manifold needs to be a positive integer.";
  }
  ~PositiveExponentialParameterization() {}

  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override {
    for (int i = 0; i < size_; ++i) {
      x_plus_delta[i] = x[i] * std::exp(delta[i]);
    }
    return true;
  }

  bool ComputeJacobian(const double* x, double* jacobian) const override {
    for (int i = 0; i < size_; ++i) {
      jacobian[size_ * i + i] = x[i];
    }
    return true;
  }

  int GlobalSize() const override { return size_; }
  int LocalSize() const override { return size_; }

 private:
  const int size_{};
};

#endif

template <int size>
inline void SetPositiveExponentialManifold(ceres::Problem* problem,
                                           double* params) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  problem->SetManifold(params, new PositiveExponentialManifold<size>);
#else
  problem->SetParameterization(params,
                               new PositiveExponentialParameterization(size));
#endif
}

inline int ParameterBlockTangentSize(const ceres::Problem& problem,
                                     const double* param) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  return problem.ParameterBlockTangentSize(param);
#else
  return problem.ParameterBlockLocalSize(param);
#endif
}

}  // namespace colmap
