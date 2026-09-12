// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/math/random.h"
#include "colmap/util/eigen_alignment.h"

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

// Random Eigen quantities that are drawn from COLMAP's deterministic,
// thread-safe PRNG (see math/random.h) instead of Eigen's built-in
// `Random()` / `UnitRandom()`, which rely on the platform's `rand()` and
// therefore produce different sequences across platforms even for the same
// seed. Prefer these helpers over `Eigen::*::Random()` and
// `Eigen::Quaterniond::UnitRandom()` so that seeded results are reproducible.

// Random matrix with each entry uniformly distributed in [-1, 1], matching the
// value range of `Eigen::Matrix<...>::Random()`.
template <int Rows, int Cols>
Eigen::Matrix<double, Rows, Cols> RandomEigenMatrixd();
template <int Rows, int Cols>
Eigen::Matrix<float, Rows, Cols> RandomEigenMatrixf();

// Dynamically sized variants of the above.
Eigen::MatrixXd RandomEigenMatrixXd(Eigen::Index rows, Eigen::Index cols);
Eigen::MatrixXf RandomEigenMatrixXf(Eigen::Index rows, Eigen::Index cols);

// Random column vector with each entry uniformly distributed in [-1, 1].
template <int N>
Eigen::Matrix<double, N, 1> RandomEigenVectord();
template <int N>
Eigen::Matrix<float, N, 1> RandomEigenVectorf();

// Dynamically sized variants of the above.
Eigen::VectorXd RandomEigenVectorXd(Eigen::Index size);
Eigen::VectorXf RandomEigenVectorXf(Eigen::Index size);

// Uniformly distributed random unit quaternion, matching
// `Eigen::Quaterniond::UnitRandom()` (Shoemake's method).
Eigen::Quaterniond RandomEigenQuaterniond();

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

namespace internal {

template <typename Derived>
Derived& SetRandomEigen(Eigen::MatrixBase<Derived>& matrix) {
  using Scalar = typename Derived::Scalar;
  for (Eigen::Index i = 0; i < matrix.size(); ++i) {
    matrix(i) = RandomUniformReal<Scalar>(Scalar(-1), Scalar(1));
  }
  return matrix.derived();
}

}  // namespace internal

template <int Rows, int Cols>
Eigen::Matrix<double, Rows, Cols> RandomEigenMatrixd() {
  Eigen::Matrix<double, Rows, Cols> matrix;
  return internal::SetRandomEigen(matrix);
}

template <int Rows, int Cols>
Eigen::Matrix<float, Rows, Cols> RandomEigenMatrixf() {
  Eigen::Matrix<float, Rows, Cols> matrix;
  return internal::SetRandomEigen(matrix);
}

inline Eigen::MatrixXd RandomEigenMatrixXd(const Eigen::Index rows,
                                           const Eigen::Index cols) {
  Eigen::MatrixXd matrix(rows, cols);
  return internal::SetRandomEigen(matrix);
}

inline Eigen::MatrixXf RandomEigenMatrixXf(const Eigen::Index rows,
                                           const Eigen::Index cols) {
  Eigen::MatrixXf matrix(rows, cols);
  return internal::SetRandomEigen(matrix);
}

template <int N>
Eigen::Matrix<double, N, 1> RandomEigenVectord() {
  Eigen::Matrix<double, N, 1> vector;
  return internal::SetRandomEigen(vector);
}

template <int N>
Eigen::Matrix<float, N, 1> RandomEigenVectorf() {
  Eigen::Matrix<float, N, 1> vector;
  return internal::SetRandomEigen(vector);
}

inline Eigen::VectorXd RandomEigenVectorXd(const Eigen::Index size) {
  Eigen::VectorXd vector(size);
  return internal::SetRandomEigen(vector);
}

inline Eigen::VectorXf RandomEigenVectorXf(const Eigen::Index size) {
  Eigen::VectorXf vector(size);
  return internal::SetRandomEigen(vector);
}

inline Eigen::Quaterniond RandomEigenQuaterniond() {
  const double u1 = RandomUniformReal<double>(0.0, 1.0);
  const double u2 = RandomUniformReal<double>(0.0, 2.0 * EIGEN_PI);
  const double u3 = RandomUniformReal<double>(0.0, 2.0 * EIGEN_PI);
  const double a = std::sqrt(1.0 - u1);
  const double b = std::sqrt(u1);
  return Eigen::Quaterniond(
      a * std::sin(u2), a * std::cos(u2), b * std::sin(u3), b * std::cos(u3));
}

}  // namespace colmap
