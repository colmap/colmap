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
