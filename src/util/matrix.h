// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_UTIL_MATRIX_H_
#define COLMAP_SRC_UTIL_MATRIX_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>

namespace colmap {

// Check if the given floating point array contains a NaN value.
template <typename Derived>
inline bool IsNaN(const Eigen::MatrixBase<Derived>& x);

// Check if the given floating point array contains infinity.
template <typename Derived>
inline bool IsInf(const Eigen::MatrixBase<Derived>& x);

// Perform RQ decomposition on matrix. The RQ decomposition transforms a matrix
// A into the product of an upper triangular matrix R (also known as
// right-triangular) and an orthogonal matrix Q.
template <typename MatrixType>
void DecomposeMatrixRQ(const MatrixType& A, MatrixType* R, MatrixType* Q);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename Derived>
bool IsNaN(const Eigen::MatrixBase<Derived>& x) {
  return !(x.array() == x.array()).all();
}

template <typename Derived>
bool IsInf(const Eigen::MatrixBase<Derived>& x) {
  return !((x - x).array() == (x - x).array()).all();
}

template <typename MatrixType>
void DecomposeMatrixRQ(const MatrixType& A, MatrixType* R, MatrixType* Q) {
  const MatrixType A_flipud_transpose =
      A.transpose().rowwise().reverse().eval();

  const Eigen::HouseholderQR<MatrixType> QR(A_flipud_transpose);
  const MatrixType& Q0 = QR.householderQ();
  const MatrixType& R0 = QR.matrixQR();

  *R = R0.transpose().colwise().reverse().eval();
  *R = R->rowwise().reverse().eval();
  for (int i = 0; i < R->rows(); ++i) {
    for (int j = 0; j < R->cols() && (R->cols() - j) > (R->rows() - i); ++j) {
      (*R)(i, j) = 0;
    }
  }

  *Q = Q0.transpose().colwise().reverse().eval();

  // Make the decomposition unique by requiring that det(Q) > 0.
  if (Q->determinant() < 0) {
    Q->row(1) *= -1.0;
    R->col(1) *= -1.0;
  }
}

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_MATRIX_H_
