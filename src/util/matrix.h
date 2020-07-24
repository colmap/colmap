// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

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
