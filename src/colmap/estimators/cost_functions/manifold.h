// SPDX-License-Identifier: BSD-3-Clause

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

inline void SetManifold(ceres::Problem* problem,
                        double* params,
                        std::unique_ptr<ceres::Manifold> manifold) {
  problem->SetManifold(params, manifold.release());
}

template <int size>
inline std::unique_ptr<ceres::Manifold> CreateEuclideanManifold() {
  return std::make_unique<ceres::EuclideanManifold<size>>();
}

inline std::unique_ptr<ceres::Manifold> CreateEigenQuaternionManifold() {
  return std::make_unique<ceres::EigenQuaternionManifold>();
}

inline std::unique_ptr<ceres::Manifold> CreateSubsetManifold(
    int size, const std::vector<int>& constant_params) {
  return std::make_unique<ceres::SubsetManifold>(size, constant_params);
}

template <int size>
inline std::unique_ptr<ceres::Manifold> CreateSphereManifold() {
  return std::make_unique<ceres::SphereManifold<size>>();
}

template <typename... Args>
inline std::unique_ptr<ceres::Manifold> CreateProductManifold(
    Args&&... manifolds) {
  // Note: Does not support make_unique due to template constructor.
  return std::unique_ptr<ceres::Manifold>(
      new ceres::ProductManifold(std::forward<Args>(manifolds)...));
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

inline void SetManifold(
    ceres::Problem* problem,
    double* params,
    std::unique_ptr<ceres::LocalParameterization> parameterization) {
  problem->SetParameterization(params, parameterization.release());
}

template <int size>
inline std::unique_ptr<ceres::LocalParameterization> CreateEuclideanManifold() {
  return std::make_unique<ceres::IdentityParameterization>(size);
}

inline std::unique_ptr<ceres::LocalParameterization>
CreateEigenQuaternionManifold() {
  return std::make_unique<ceres::EigenQuaternionParameterization>();
}

inline std::unique_ptr<ceres::LocalParameterization> CreateSubsetManifold(
    int size, const std::vector<int>& constant_params) {
  return std::make_unique<ceres::SubsetParameterization>(size, constant_params);
}

template <int size>
inline std::unique_ptr<ceres::LocalParameterization> CreateSphereManifold() {
  return std::make_unique<ceres::HomogeneousVectorParameterization>(size);
}

template <typename... Args>
inline std::unique_ptr<ceres::LocalParameterization> CreateProductManifold(
    Args&&... parameterizations) {
  // Note: Does not support make_unique due to template constructor.
  return std::unique_ptr<ceres::ProductParameterization>(
      new ceres::ProductParameterization(parameterizations.release()...));
}

inline int ParameterBlockTangentSize(const ceres::Problem& problem,
                                     const double* param) {
  return problem.ParameterBlockLocalSize(param);
}

#endif

}  // namespace colmap
