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

#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)

inline void SetManifold(ceres::Problem* problem,
                        double* params,
                        ceres::Manifold* manifold) {
  problem->SetManifold(params, manifold);
}

template <int size>
inline ceres::Manifold* CreateEuclideanManifold() {
  return new ceres::EuclideanManifold<size>();
}

inline ceres::Manifold* CreateEigenQuaternionManifold() {
  return new ceres::EigenQuaternionManifold();
}

inline ceres::Manifold* CreateSubsetManifold(
    int size, const std::vector<int>& constant_params) {
  return new ceres::SubsetManifold(size, constant_params);
}

template <int size>
inline ceres::Manifold* CreateSphereManifold() {
  return new ceres::SphereManifold<size>();
}

template <typename... Args>
inline ceres::Manifold* CreateProductManifold(Args&&... manifolds) {
  return new ceres::ProductManifold(std::forward<Args>(manifolds)...);
}

inline int ParameterBlockTangentSize(const ceres::Problem& problem,
                                     const double* param) {
  return problem.ParameterBlockTangentSize(param);
}

#else  // CERES_VERSION_MAJOR < 2.1.0

inline void SetManifold(ceres::Problem* problem,
                        double* params,
                        ceres::LocalParameterization* parameterization) {
  problem->SetParameterization(params, parameterization);
}

template <int size>
inline ceres::LocalParameterization* CreateEuclideanManifold() {
  return new ceres::IdentityParameterization<size>();
}

inline ceres::LocalParameterization* CreateEigenQuaternionManifold() {
  return new ceres::EigenQuaternionParameterization();
}

inline ceres::LocalParameterization* CreateSubsetManifold(
    int size, const std::vector<int>& constant_params) {
  return new ceres::SubsetParameterization(size, constant_params);
}

template <int size>
inline ceres::LocalParameterization* CreateSphereManifold() {
  return new ceres::HomogeneousVectorParameterization(size);
}

template <typename... Args>
inline ceres::LocalParameterization* CreateProductManifold(
    Args&&... parameterizations) {
  return new ceres::ProductParameterization(
      std::forward<Args>(parameterizations)...);
}

inline int ParameterBlockTangentSize(const ceres::Problem& problem,
                                     const double* param) {
  return problem.ParameterBlockLocalSize(param);
}

#endif

}  // namespace colmap
