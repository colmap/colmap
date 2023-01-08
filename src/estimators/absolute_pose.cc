// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "estimators/absolute_pose.h"

#include "base/polynomial.h"
#include "estimators/utils.h"
#include "util/logging.h"

namespace colmap {
namespace {

Eigen::Vector3d LiftImagePoint(const Eigen::Vector2d& point) {
  return point.homogeneous() / std::sqrt(point.squaredNorm() + 1);
}

}  // namespace

std::vector<P3PEstimator::M_t> P3PEstimator::Estimate(
    const std::vector<X_t>& points2D, const std::vector<Y_t>& points3D) {
  CHECK_EQ(points2D.size(), 3);
  CHECK_EQ(points3D.size(), 3);

  Eigen::Matrix3d points3D_world;
  points3D_world.col(0) = points3D[0];
  points3D_world.col(1) = points3D[1];
  points3D_world.col(2) = points3D[2];

  const Eigen::Vector3d u = LiftImagePoint(points2D[0]);
  const Eigen::Vector3d v = LiftImagePoint(points2D[1]);
  const Eigen::Vector3d w = LiftImagePoint(points2D[2]);

  // Angles between 2D points.
  const double cos_uv = u.transpose() * v;
  const double cos_uw = u.transpose() * w;
  const double cos_vw = v.transpose() * w;

  // Distances between 2D points.
  const double dist_AB_2 = (points3D[0] - points3D[1]).squaredNorm();
  const double dist_AC_2 = (points3D[0] - points3D[2]).squaredNorm();
  const double dist_BC_2 = (points3D[1] - points3D[2]).squaredNorm();

  const double dist_AB = std::sqrt(dist_AB_2);

  const double a = dist_BC_2 / dist_AB_2;
  const double b = dist_AC_2 / dist_AB_2;

  // Helper variables for calculation of coefficients.
  const double a2 = a * a;
  const double b2 = b * b;
  const double p = 2 * cos_vw;
  const double q = 2 * cos_uw;
  const double r = 2 * cos_uv;
  const double p2 = p * p;
  const double p3 = p2 * p;
  const double q2 = q * q;
  const double r2 = r * r;
  const double r3 = r2 * r;
  const double r4 = r3 * r;
  const double r5 = r4 * r;

  // Build polynomial coefficients: a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0.
  Eigen::Matrix<double, 5, 1> coeffs;
  coeffs(0) = -2 * b + b2 + a2 + 1 + a * b * (2 - r2) - 2 * a;
  coeffs(1) = -2 * q * a2 - r * p * b2 + 4 * q * a + (2 * q + p * r) * b +
              (r2 * q - 2 * q + r * p) * a * b - 2 * q;
  coeffs(2) = (2 + q2) * a2 + (p2 + r2 - 2) * b2 - (4 + 2 * q2) * a -
              (p * q * r + p2) * b - (p * q * r + r2) * a * b + q2 + 2;
  coeffs(3) = -2 * q * a2 - r * p * b2 + 4 * q * a +
              (p * r + q * p2 - 2 * q) * b + (r * p + 2 * q) * a * b - 2 * q;
  coeffs(4) = a2 + b2 - 2 * a + (2 - p2) * b - 2 * a * b + 1;

  Eigen::VectorXd roots_real;
  Eigen::VectorXd roots_imag;
  if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag)) {
    return std::vector<P3PEstimator::M_t>({});
  }

  std::vector<M_t> models;
  models.reserve(roots_real.size());

  for (Eigen::VectorXd::Index i = 0; i < roots_real.size(); ++i) {
    const double kMaxRootImag = 1e-10;
    if (std::abs(roots_imag(i)) > kMaxRootImag) {
      continue;
    }

    const double x = roots_real(i);
    if (x < 0) {
      continue;
    }

    const double x2 = x * x;
    const double x3 = x2 * x;

    // Build polynomial coefficients: b1*y + b0 = 0.
    const double bb1 =
        (p2 - p * q * r + r2) * a + (p2 - r2) * b - p2 + p * q * r - r2;
    const double b1 = b * bb1 * bb1;
    const double b0 =
        ((1 - a - b) * x2 + (a - 1) * q * x - a + b + 1) *
        (r3 * (a2 + b2 - 2 * a - 2 * b + (2 - r2) * a * b + 1) * x3 +
         r2 *
             (p + p * a2 - 2 * r * q * a * b + 2 * r * q * b - 2 * r * q -
              2 * p * a - 2 * p * b + p * r2 * b + 4 * r * q * a +
              q * r3 * a * b - 2 * r * q * a2 + 2 * p * a * b + p * b2 -
              r2 * p * b2) *
             x2 +
         (r5 * (b2 - a * b) - r4 * p * q * b +
          r3 * (q2 - 4 * a - 2 * q2 * a + q2 * a2 + 2 * a2 - 2 * b2 + 2) +
          r2 * (4 * p * q * a - 2 * p * q * a * b + 2 * p * q * b - 2 * p * q -
                2 * p * q * a2) +
          r * (p2 * b2 - 2 * p2 * b + 2 * p2 * a * b - 2 * p2 * a + p2 +
               p2 * a2)) *
             x +
         (2 * p * r2 - 2 * r3 * q + p3 - 2 * p2 * q * r + p * q2 * r2) * a2 +
         (p3 - 2 * p * r2) * b2 +
         (4 * q * r3 - 4 * p * r2 - 2 * p3 + 4 * p2 * q * r - 2 * p * q2 * r2) *
             a +
         (-2 * q * r3 + p * r4 + 2 * p2 * q * r - 2 * p3) * b +
         (2 * p3 + 2 * q * r3 - 2 * p2 * q * r) * a * b + p * q2 * r2 -
         2 * p2 * q * r + 2 * p * r2 + p3 - 2 * r3 * q);

    // Solve for y.
    const double y = b0 / b1;
    const double y2 = y * y;

    const double nu = x2 + y2 - 2 * x * y * cos_uv;

    const double dist_PC = dist_AB / std::sqrt(nu);
    const double dist_PB = y * dist_PC;
    const double dist_PA = x * dist_PC;

    Eigen::Matrix3d points3D_camera;
    points3D_camera.col(0) = u * dist_PA;  // A'
    points3D_camera.col(1) = v * dist_PB;  // B'
    points3D_camera.col(2) = w * dist_PC;  // C'

    // Find transformation from the world to the camera system.
    const Eigen::Matrix4d transform =
        Eigen::umeyama(points3D_world, points3D_camera, false);
    models.push_back(transform.topLeftCorner<3, 4>());
  }

  return models;
}

void P3PEstimator::Residuals(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             const M_t& proj_matrix,
                             std::vector<double>* residuals) {
  ComputeSquaredReprojectionError(points2D, points3D, proj_matrix, residuals);
}

std::vector<EPNPEstimator::M_t> EPNPEstimator::Estimate(
    const std::vector<X_t>& points2D, const std::vector<Y_t>& points3D) {
  CHECK_GE(points2D.size(), 4);
  CHECK_EQ(points2D.size(), points3D.size());

  EPNPEstimator epnp;
  M_t proj_matrix;
  if (!epnp.ComputePose(points2D, points3D, &proj_matrix)) {
    return std::vector<EPNPEstimator::M_t>({});
  }

  return std::vector<EPNPEstimator::M_t>({proj_matrix});
}

void EPNPEstimator::Residuals(const std::vector<X_t>& points2D,
                              const std::vector<Y_t>& points3D,
                              const M_t& proj_matrix,
                              std::vector<double>* residuals) {
  ComputeSquaredReprojectionError(points2D, points3D, proj_matrix, residuals);
}

bool EPNPEstimator::ComputePose(const std::vector<Eigen::Vector2d>& points2D,
                                const std::vector<Eigen::Vector3d>& points3D,
                                Eigen::Matrix3x4d* proj_matrix) {
  points2D_ = &points2D;
  points3D_ = &points3D;

  ChooseControlPoints();

  if (!ComputeBarycentricCoordinates()) {
    return false;
  }

  const Eigen::Matrix<double, Eigen::Dynamic, 12> M = ComputeM();
  const Eigen::Matrix<double, 12, 12> MtM = M.transpose() * M;

  Eigen::JacobiSVD<Eigen::Matrix<double, 12, 12>> svd(
      MtM, Eigen::ComputeFullV | Eigen::ComputeFullU);
  const Eigen::Matrix<double, 12, 12> Ut = svd.matrixU().transpose();

  const Eigen::Matrix<double, 6, 10> L6x10 = ComputeL6x10(Ut);
  const Eigen::Matrix<double, 6, 1> rho = ComputeRho();

  Eigen::Vector4d betas[4];
  std::array<double, 4> reproj_errors;
  std::array<Eigen::Matrix3d, 4> Rs;
  std::array<Eigen::Vector3d, 4> ts;

  FindBetasApprox1(L6x10, rho, &betas[1]);
  RunGaussNewton(L6x10, rho, &betas[1]);
  reproj_errors[1] = ComputeRT(Ut, betas[1], &Rs[1], &ts[1]);

  FindBetasApprox2(L6x10, rho, &betas[2]);
  RunGaussNewton(L6x10, rho, &betas[2]);
  reproj_errors[2] = ComputeRT(Ut, betas[2], &Rs[2], &ts[2]);

  FindBetasApprox3(L6x10, rho, &betas[3]);
  RunGaussNewton(L6x10, rho, &betas[3]);
  reproj_errors[3] = ComputeRT(Ut, betas[3], &Rs[3], &ts[3]);

  int best_idx = 1;
  if (reproj_errors[2] < reproj_errors[1]) {
    best_idx = 2;
  }
  if (reproj_errors[3] < reproj_errors[best_idx]) {
    best_idx = 3;
  }

  proj_matrix->leftCols<3>() = Rs[best_idx];
  proj_matrix->rightCols<1>() = ts[best_idx];

  return true;
}

void EPNPEstimator::ChooseControlPoints() {
  // Take C0 as the reference points centroid:
  cws_[0].setZero();
  for (size_t i = 0; i < points3D_->size(); ++i) {
    cws_[0] += (*points3D_)[i];
  }
  cws_[0] /= points3D_->size();

  Eigen::Matrix<double, Eigen::Dynamic, 3> PW0(points3D_->size(), 3);
  for (size_t i = 0; i < points3D_->size(); ++i) {
    PW0.row(i) = (*points3D_)[i] - cws_[0];
  }

  const Eigen::Matrix3d PW0tPW0 = PW0.transpose() * PW0;
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      PW0tPW0, Eigen::ComputeFullV | Eigen::ComputeFullU);
  const Eigen::Vector3d D = svd.singularValues();
  const Eigen::Matrix3d Ut = svd.matrixU().transpose();

  for (int i = 1; i < 4; ++i) {
    const double k = std::sqrt(D(i - 1) / points3D_->size());
    cws_[i] = cws_[0] + k * Ut.row(i - 1).transpose();
  }
}

bool EPNPEstimator::ComputeBarycentricCoordinates() {
  Eigen::Matrix3d CC;
  for (int i = 0; i < 3; ++i) {
    for (int j = 1; j < 4; ++j) {
      CC(i, j - 1) = cws_[j][i] - cws_[0][i];
    }
  }

  if (CC.colPivHouseholderQr().rank() < 3) {
    return false;
  }

  const Eigen::Matrix3d CC_inv = CC.inverse();

  alphas_.resize(points2D_->size());
  for (size_t i = 0; i < points3D_->size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      alphas_[i][1 + j] = CC_inv(j, 0) * ((*points3D_)[i][0] - cws_[0][0]) +
                          CC_inv(j, 1) * ((*points3D_)[i][1] - cws_[0][1]) +
                          CC_inv(j, 2) * ((*points3D_)[i][2] - cws_[0][2]);
    }
    alphas_[i][0] = 1.0 - alphas_[i][1] - alphas_[i][2] - alphas_[i][3];
  }

  return true;
}

Eigen::Matrix<double, Eigen::Dynamic, 12> EPNPEstimator::ComputeM() {
  Eigen::Matrix<double, Eigen::Dynamic, 12> M(2 * points2D_->size(), 12);
  for (size_t i = 0; i < points3D_->size(); ++i) {
    for (size_t j = 0; j < 4; ++j) {
      M(2 * i, 3 * j) = alphas_[i][j];
      M(2 * i, 3 * j + 1) = 0.0;
      M(2 * i, 3 * j + 2) = -alphas_[i][j] * (*points2D_)[i].x();

      M(2 * i + 1, 3 * j) = 0.0;
      M(2 * i + 1, 3 * j + 1) = alphas_[i][j];
      M(2 * i + 1, 3 * j + 2) = -alphas_[i][j] * (*points2D_)[i].y();
    }
  }
  return M;
}

Eigen::Matrix<double, 6, 10> EPNPEstimator::ComputeL6x10(
    const Eigen::Matrix<double, 12, 12>& Ut) {
  Eigen::Matrix<double, 6, 10> L6x10;

  std::array<std::array<Eigen::Vector3d, 6>, 4> dv;
  for (int i = 0; i < 4; ++i) {
    int a = 0, b = 1;
    for (int j = 0; j < 6; ++j) {
      dv[i][j][0] = Ut(11 - i, 3 * a) - Ut(11 - i, 3 * b);
      dv[i][j][1] = Ut(11 - i, 3 * a + 1) - Ut(11 - i, 3 * b + 1);
      dv[i][j][2] = Ut(11 - i, 3 * a + 2) - Ut(11 - i, 3 * b + 2);

      b += 1;
      if (b > 3) {
        a += 1;
        b = a + 1;
      }
    }
  }

  for (int i = 0; i < 6; ++i) {
    L6x10(i, 0) = dv[0][i].transpose() * dv[0][i];
    L6x10(i, 1) = 2.0 * dv[0][i].transpose() * dv[1][i];
    L6x10(i, 2) = dv[1][i].transpose() * dv[1][i];
    L6x10(i, 3) = 2.0 * dv[0][i].transpose() * dv[2][i];
    L6x10(i, 4) = 2.0 * dv[1][i].transpose() * dv[2][i];
    L6x10(i, 5) = dv[2][i].transpose() * dv[2][i];
    L6x10(i, 6) = 2.0 * dv[0][i].transpose() * dv[3][i];
    L6x10(i, 7) = 2.0 * dv[1][i].transpose() * dv[3][i];
    L6x10(i, 8) = 2.0 * dv[2][i].transpose() * dv[3][i];
    L6x10(i, 9) = dv[3][i].transpose() * dv[3][i];
  }

  return L6x10;
}

Eigen::Matrix<double, 6, 1> EPNPEstimator::ComputeRho() {
  Eigen::Matrix<double, 6, 1> rho;
  rho[0] = (cws_[0] - cws_[1]).squaredNorm();
  rho[1] = (cws_[0] - cws_[2]).squaredNorm();
  rho[2] = (cws_[0] - cws_[3]).squaredNorm();
  rho[3] = (cws_[1] - cws_[2]).squaredNorm();
  rho[4] = (cws_[1] - cws_[3]).squaredNorm();
  rho[5] = (cws_[2] - cws_[3]).squaredNorm();
  return rho;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

void EPNPEstimator::FindBetasApprox1(const Eigen::Matrix<double, 6, 10>& L6x10,
                                     const Eigen::Matrix<double, 6, 1>& rho,
                                     Eigen::Vector4d* betas) {
  Eigen::Matrix<double, 6, 4> L_6x4;
  for (int i = 0; i < 6; ++i) {
    L_6x4(i, 0) = L6x10(i, 0);
    L_6x4(i, 1) = L6x10(i, 1);
    L_6x4(i, 2) = L6x10(i, 3);
    L_6x4(i, 3) = L6x10(i, 6);
  }

  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 4>> svd(
      L_6x4, Eigen::ComputeFullV | Eigen::ComputeFullU);
  Eigen::Matrix<double, 6, 1> Rho_temp = rho;
  const Eigen::Matrix<double, 4, 1> b4 = svd.solve(Rho_temp);

  if (b4[0] < 0) {
    (*betas)[0] = std::sqrt(-b4[0]);
    (*betas)[1] = -b4[1] / (*betas)[0];
    (*betas)[2] = -b4[2] / (*betas)[0];
    (*betas)[3] = -b4[3] / (*betas)[0];
  } else {
    (*betas)[0] = std::sqrt(b4[0]);
    (*betas)[1] = b4[1] / (*betas)[0];
    (*betas)[2] = b4[2] / (*betas)[0];
    (*betas)[3] = b4[3] / (*betas)[0];
  }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

void EPNPEstimator::FindBetasApprox2(const Eigen::Matrix<double, 6, 10>& L6x10,
                                     const Eigen::Matrix<double, 6, 1>& rho,
                                     Eigen::Vector4d* betas) {
  Eigen::Matrix<double, 6, 3> L_6x3(6, 3);

  for (int i = 0; i < 6; ++i) {
    L_6x3(i, 0) = L6x10(i, 0);
    L_6x3(i, 1) = L6x10(i, 1);
    L_6x3(i, 2) = L6x10(i, 2);
  }

  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 3>> svd(
      L_6x3, Eigen::ComputeFullV | Eigen::ComputeFullU);
  Eigen::Matrix<double, 6, 1> Rho_temp = rho;
  const Eigen::Matrix<double, 3, 1> b3 = svd.solve(Rho_temp);

  if (b3[0] < 0) {
    (*betas)[0] = std::sqrt(-b3[0]);
    (*betas)[1] = (b3[2] < 0) ? std::sqrt(-b3[2]) : 0.0;
  } else {
    (*betas)[0] = std::sqrt(b3[0]);
    (*betas)[1] = (b3[2] > 0) ? std::sqrt(b3[2]) : 0.0;
  }

  if (b3[1] < 0) {
    (*betas)[0] = -(*betas)[0];
  }

  (*betas)[2] = 0.0;
  (*betas)[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

void EPNPEstimator::FindBetasApprox3(const Eigen::Matrix<double, 6, 10>& L6x10,
                                     const Eigen::Matrix<double, 6, 1>& rho,
                                     Eigen::Vector4d* betas) {
  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 5>> svd(
      L6x10.leftCols<5>(), Eigen::ComputeFullV | Eigen::ComputeFullU);
  Eigen::Matrix<double, 6, 1> Rho_temp = rho;
  const Eigen::Matrix<double, 5, 1> b5 = svd.solve(Rho_temp);

  if (b5[0] < 0) {
    (*betas)[0] = std::sqrt(-b5[0]);
    (*betas)[1] = (b5[2] < 0) ? std::sqrt(-b5[2]) : 0.0;
  } else {
    (*betas)[0] = std::sqrt(b5[0]);
    (*betas)[1] = (b5[2] > 0) ? std::sqrt(b5[2]) : 0.0;
  }
  if (b5[1] < 0) {
    (*betas)[0] = -(*betas)[0];
  }
  (*betas)[2] = b5[3] / (*betas)[0];
  (*betas)[3] = 0.0;
}

void EPNPEstimator::RunGaussNewton(const Eigen::Matrix<double, 6, 10>& L6x10,
                                   const Eigen::Matrix<double, 6, 1>& rho,
                                   Eigen::Vector4d* betas) {
  Eigen::Matrix<double, 6, 4> A;
  Eigen::Matrix<double, 6, 1> b;

  const int kNumIterations = 5;
  for (int k = 0; k < kNumIterations; ++k) {
    for (int i = 0; i < 6; ++i) {
      A(i, 0) = 2 * L6x10(i, 0) * (*betas)[0] + L6x10(i, 1) * (*betas)[1] +
                L6x10(i, 3) * (*betas)[2] + L6x10(i, 6) * (*betas)[3];
      A(i, 1) = L6x10(i, 1) * (*betas)[0] + 2 * L6x10(i, 2) * (*betas)[1] +
                L6x10(i, 4) * (*betas)[2] + L6x10(i, 7) * (*betas)[3];
      A(i, 2) = L6x10(i, 3) * (*betas)[0] + L6x10(i, 4) * (*betas)[1] +
                2 * L6x10(i, 5) * (*betas)[2] + L6x10(i, 8) * (*betas)[3];
      A(i, 3) = L6x10(i, 6) * (*betas)[0] + L6x10(i, 7) * (*betas)[1] +
                L6x10(i, 8) * (*betas)[2] + 2 * L6x10(i, 9) * (*betas)[3];

      b(i) = rho[i] - (L6x10(i, 0) * (*betas)[0] * (*betas)[0] +
                       L6x10(i, 1) * (*betas)[0] * (*betas)[1] +
                       L6x10(i, 2) * (*betas)[1] * (*betas)[1] +
                       L6x10(i, 3) * (*betas)[0] * (*betas)[2] +
                       L6x10(i, 4) * (*betas)[1] * (*betas)[2] +
                       L6x10(i, 5) * (*betas)[2] * (*betas)[2] +
                       L6x10(i, 6) * (*betas)[0] * (*betas)[3] +
                       L6x10(i, 7) * (*betas)[1] * (*betas)[3] +
                       L6x10(i, 8) * (*betas)[2] * (*betas)[3] +
                       L6x10(i, 9) * (*betas)[3] * (*betas)[3]);
    }

    const Eigen::Vector4d x = A.colPivHouseholderQr().solve(b);

    (*betas) += x;
  }
}

double EPNPEstimator::ComputeRT(const Eigen::Matrix<double, 12, 12>& Ut,
                                const Eigen::Vector4d& betas,
                                Eigen::Matrix3d* R, Eigen::Vector3d* t) {
  ComputeCcs(betas, Ut);
  ComputePcs();

  SolveForSign();

  EstimateRT(R, t);

  return ComputeTotalReprojectionError(*R, *t);
}

void EPNPEstimator::ComputeCcs(const Eigen::Vector4d& betas,
                               const Eigen::Matrix<double, 12, 12>& Ut) {
  for (int i = 0; i < 4; ++i) {
    ccs_[i][0] = ccs_[i][1] = ccs_[i][2] = 0.0;
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 3; ++k) {
        ccs_[j][k] += betas[i] * Ut(11 - i, 3 * j + k);
      }
    }
  }
}

void EPNPEstimator::ComputePcs() {
  pcs_.resize(points2D_->size());
  for (size_t i = 0; i < points3D_->size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      pcs_[i][j] = alphas_[i][0] * ccs_[0][j] + alphas_[i][1] * ccs_[1][j] +
                   alphas_[i][2] * ccs_[2][j] + alphas_[i][3] * ccs_[3][j];
    }
  }
}

void EPNPEstimator::SolveForSign() {
  if (pcs_[0][2] < 0.0) {
    for (int i = 0; i < 4; ++i) {
      ccs_[i] = -ccs_[i];
    }
    for (size_t i = 0; i < points3D_->size(); ++i) {
      pcs_[i] = -pcs_[i];
    }
  }
}

void EPNPEstimator::EstimateRT(Eigen::Matrix3d* R, Eigen::Vector3d* t) {
  Eigen::Vector3d pc0 = Eigen::Vector3d::Zero();
  Eigen::Vector3d pw0 = Eigen::Vector3d::Zero();

  for (size_t i = 0; i < points3D_->size(); ++i) {
    pc0 += pcs_[i];
    pw0 += (*points3D_)[i];
  }
  pc0 /= points3D_->size();
  pw0 /= points3D_->size();

  Eigen::Matrix3d abt = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < points3D_->size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      abt(j, 0) += (pcs_[i][j] - pc0[j]) * ((*points3D_)[i][0] - pw0[0]);
      abt(j, 1) += (pcs_[i][j] - pc0[j]) * ((*points3D_)[i][1] - pw0[1]);
      abt(j, 2) += (pcs_[i][j] - pc0[j]) * ((*points3D_)[i][2] - pw0[2]);
    }
  }

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      abt, Eigen::ComputeFullV | Eigen::ComputeFullU);
  const Eigen::Matrix3d abt_U = svd.matrixU();
  const Eigen::Matrix3d abt_V = svd.matrixV();

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      (*R)(i, j) = abt_U.row(i) * abt_V.row(j).transpose();
    }
  }

  if (R->determinant() < 0) {
    Eigen::Matrix3d Abt_v_prime = abt_V;
    Abt_v_prime.col(2) = -abt_V.col(2);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        (*R)(i, j) = abt_U.row(i) * Abt_v_prime.row(j).transpose();
      }
    }
  }

  *t = pc0 - *R * pw0;
}

double EPNPEstimator::ComputeTotalReprojectionError(const Eigen::Matrix3d& R,
                                                    const Eigen::Vector3d& t) {
  Eigen::Matrix3x4d proj_matrix;
  proj_matrix.leftCols<3>() = R;
  proj_matrix.rightCols<1>() = t;

  std::vector<double> residuals;
  ComputeSquaredReprojectionError(*points2D_, *points3D_, proj_matrix,
                                  &residuals);

  double reproj_error = 0.0;
  for (const double residual : residuals) {
    reproj_error += std::sqrt(residual);
  }

  return reproj_error;
}

}  // namespace colmap
