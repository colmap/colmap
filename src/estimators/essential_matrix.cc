// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#include "estimators/essential_matrix.h"

#include <complex>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "estimators/utils.h"
#include "util/logging.h"
#include "util/math.h"

namespace colmap {

std::vector<EssentialMatrixFivePointEstimator::M_t>
EssentialMatrixFivePointEstimator::Estimate(const std::vector<X_t>& points1,
                                            const std::vector<Y_t>& points2) {
  CHECK_EQ(points1.size(), points2.size());

  // Step 1: Extraction of the nullspace x, y, z, w

  Eigen::Matrix<double, Eigen::Dynamic, 9> Q(points1.size(), 9);
  for (size_t i = 0; i < points1.size(); ++i) {
    const double x1_0 = points1[i](0);
    const double x1_1 = points1[i](1);
    const double x1_2 = points1[i](2);
    const double x2_0 = points2[i](0);
    const double x2_1 = points2[i](1);
    const double x2_2 = points2[i](2);

    Q(i, 0) = x1_0 * x2_0;
    Q(i, 1) = x1_1 * x2_0;
    Q(i, 2) = x1_2 * x2_0;
    Q(i, 3) = x1_0 * x2_1;
    Q(i, 4) = x1_1 * x2_1;
    Q(i, 5) = x1_2 * x2_1;
    Q(i, 6) = x1_0 * x2_2;
    Q(i, 7) = x1_1 * x2_2;
    Q(i, 8) = x1_2 * x2_2;
  }

  // Extract the 4 Eigen vectors corresponding to the smallest singular values
  Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
      Q, Eigen::ComputeFullV);
  Eigen::Matrix<double, 4, 9, Eigen::RowMajor> E =
      svd.matrixV().block<9, 4>(0, 5).transpose();

  // Step 3: Gauss-Jordan elimination with partial pivoting on the
  //         10x20 matrix A

  Eigen::Matrix<double, 10, 20, Eigen::ColMajor> A;
#include "estimators/essential_matrix_poly.h"
  Eigen::Matrix<double, 10, 10> AA =
      A.block<10, 10>(0, 0).partialPivLu().solve(A.block<10, 10>(0, 10));

  // Step 4: Expansion of the determinant polynomial of the 3x3 polynomial
  //         matrix B to obtain the tenth degree polynomial

  Eigen::Matrix<double, 13, 3> B;
  Eigen::Matrix<double, 1, 13> B_row1, B_row2;
  B_row1(0, 0) = 0;
  B_row1(0, 4) = 0;
  B_row1(0, 8) = 0;
  B_row2(0, 3) = 0;
  B_row2(0, 7) = 0;
  B_row2(0, 12) = 0;
  for (size_t i = 0; i < 3; ++i) {
    B_row1.block<1, 3>(0, 1) = AA.block<1, 3>(i * 2 + 4, 0);
    B_row1.block<1, 3>(0, 5) = AA.block<1, 3>(i * 2 + 4, 3);
    B_row1.block<1, 4>(0, 9) = AA.block<1, 4>(i * 2 + 4, 6);
    B_row2.block<1, 3>(0, 0) = AA.block<1, 3>(i * 2 + 5, 0);
    B_row2.block<1, 3>(0, 4) = AA.block<1, 3>(i * 2 + 5, 3);
    B_row2.block<1, 4>(0, 8) = AA.block<1, 4>(i * 2 + 5, 6);
    B.col(i) = B_row1 - B_row2;
  }

  // Step 5: Extraction of roots from the degree 10 polynomial
  std::vector<double> coeffs(11);
#include "estimators/essential_matrix_coeff.h"

  std::vector<std::complex<double>> roots = SolvePolynomialN(coeffs);

  std::vector<M_t> models;

  const double kEps = 1e-10;

  for (size_t i = 0; i < roots.size(); ++i) {
    if (std::abs(roots[i].imag()) > kEps) {
      continue;
    }

    const double z1 = roots[i].real();
    const double z2 = z1 * z1;
    const double z3 = z2 * z1;
    const double z4 = z3 * z1;

    Eigen::Matrix3d Bz;
    for (size_t j = 0; j < 3; ++j) {
      const double* br = b + j * 13;
      Bz(j, 0) = br[0] * z3 + br[1] * z2 + br[2] * z1 + br[3];
      Bz(j, 1) = br[4] * z3 + br[5] * z2 + br[6] * z1 + br[7];
      Bz(j, 2) = br[8] * z4 + br[9] * z3 + br[10] * z2 + br[11] * z1 + br[12];
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Bz, Eigen::ComputeFullV);
    const Eigen::Vector3d X = svd.matrixV().block<3, 1>(0, 2);

    if (std::abs(X(2)) < kEps) {
      continue;
    }

    Eigen::MatrixXd essential_vec = E.row(0) * (X(0) / X(2)) +
                                    E.row(1) * (X(1) / X(2)) + E.row(2) * z1 +
                                    E.row(3);

    const double inv_norm = 1.0 / essential_vec.norm();
    essential_vec *= inv_norm;

    essential_vec.resize(3, 3);
    const Eigen::Matrix3d essential_matrix = essential_vec.transpose();

    models.push_back(essential_matrix);
  }

  return models;
}

void EssentialMatrixFivePointEstimator::Residuals(
    const std::vector<X_t>& points1, const std::vector<Y_t>& points2,
    const M_t& E, std::vector<double>* residuals) {
  ComputeSquaredSampsonError(points1, points2, E, residuals);
}

// std::vector<EssentialMatrixEightPointEstimator::M_t>
// EssentialMatrixEightPointEstimator::Estimate(const std::vector<X_t>& points1,
//                                              const std::vector<Y_t>& points2) {
//   CHECK_EQ(points1.size(), points2.size());

//   // Center and normalize image points for better numerical stability.
//   std::vector<X_t> normed_points1;
//   std::vector<Y_t> normed_points2;
//   Eigen::Matrix3d points1_norm_matrix;
//   Eigen::Matrix3d points2_norm_matrix;
//   CenterAndNormalizeImagePoints(points1, &normed_points1, &points1_norm_matrix);
//   CenterAndNormalizeImagePoints(points2, &normed_points2, &points2_norm_matrix);

//   // Setup homogeneous linear equation as x2' * F * x1 = 0.
//   Eigen::Matrix<double, Eigen::Dynamic, 9> cmatrix(points1.size(), 9);
//   for (size_t i = 0; i < points1.size(); ++i) {
//     cmatrix.block<1, 3>(i, 0) = normed_points1[i] * normed_points2[i].x();
//     cmatrix.block<1, 3>(i, 3) = normed_points1[i] * normed_points2[i].y();
//     cmatrix.block<1, 3>(i, 6) = normed_points1[i] * normed_points2[i].z();
//   }

//   // Solve for the nullspace of the constraint matrix.
//   Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> cmatrix_svd(
//       cmatrix, Eigen::ComputeFullV);
//   const Eigen::VectorXd ematrix_nullspace = cmatrix_svd.matrixV().col(8);
//   const Eigen::Map<const Eigen::Matrix3d> ematrix_t(ematrix_nullspace.data());

//   // Enforcing the internal constraint that two singular values must be equal
//   // and one must be zero.
//   Eigen::JacobiSVD<Eigen::Matrix3d> ematrix_svd(
//       ematrix_t.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
//   Eigen::Vector3d singular_values = ematrix_svd.singularValues();
//   singular_values(0) = (singular_values(0) + singular_values(1)) / 2.0;
//   singular_values(1) = singular_values(0);
//   singular_values(2) = 0.0;
//   const Eigen::Matrix3d E = ematrix_svd.matrixU() *
//                             singular_values.asDiagonal() *
//                             ematrix_svd.matrixV().transpose();

//   const std::vector<M_t> models = {points2_norm_matrix.transpose() * E *
//                                    points1_norm_matrix};
//   return models;
// }

// void EssentialMatrixEightPointEstimator::Residuals(
//     const std::vector<X_t>& points1, const std::vector<Y_t>& points2,
//     const M_t& E, std::vector<double>* residuals) {
//   ComputeSquaredSampsonError(points1, points2, E, residuals);
// }

}  // namespace colmap
