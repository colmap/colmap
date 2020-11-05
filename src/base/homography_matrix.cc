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

#include "base/homography_matrix.h"

#include <array>

#include <Eigen/Dense>

#include "base/pose.h"
#include "util/logging.h"
#include "util/math.h"

namespace colmap {
namespace {

double ComputeOppositeOfMinor(const Eigen::Matrix3d& matrix, const size_t row,
                              const size_t col) {
  const size_t col1 = col == 0 ? 1 : 0;
  const size_t col2 = col == 2 ? 1 : 2;
  const size_t row1 = row == 0 ? 1 : 0;
  const size_t row2 = row == 2 ? 1 : 2;
  return (matrix(row1, col2) * matrix(row2, col1) -
          matrix(row1, col1) * matrix(row2, col2));
}

Eigen::Matrix3d ComputeHomographyRotation(const Eigen::Matrix3d& H_normalized,
                                          const Eigen::Vector3d& tstar,
                                          const Eigen::Vector3d& n,
                                          const double v) {
  return H_normalized *
         (Eigen::Matrix3d::Identity() - (2.0 / v) * tstar * n.transpose());
}

}  // namespace

void DecomposeHomographyMatrix(const Eigen::Matrix3d& H,
                               const Eigen::Matrix3d& K1,
                               const Eigen::Matrix3d& K2,
                               std::vector<Eigen::Matrix3d>* R,
                               std::vector<Eigen::Vector3d>* t,
                               std::vector<Eigen::Vector3d>* n) {
  // Remove calibration from homography.
  Eigen::Matrix3d H_normalized = K2.inverse() * H * K1;

  // Remove scale from normalized homography.
  Eigen::JacobiSVD<Eigen::Matrix3d> hmatrix_norm_svd(H_normalized);
  H_normalized.array() /= hmatrix_norm_svd.singularValues()[1];

  // Ensure that we always return rotations, and never reflections.
  //
  // It's enough to take det(H_normalized) > 0.
  //
  // To see this:
  // - In the paper: R := H_normalized * (Id + x y^t)^{-1} (page 32).
  // - Can check that this implies that R is orthogonal: RR^t = Id.
  // - To return a rotation, we also need det(R) > 0.
  // - By Sylvester's idenitity: det(Id + x y^t) = (1 + x^t y), which
  //   is positive by choice of x and y (page 24).
  // - So det(R) and det(H_normalized) have the same sign.
  if (H_normalized.determinant() < 0) {
    H_normalized.array() *= -1.0;
  }

  const Eigen::Matrix3d S =
      H_normalized.transpose() * H_normalized - Eigen::Matrix3d::Identity();

  // Check if H is rotation matrix.
  const double kMinInfinityNorm = 1e-3;
  if (S.lpNorm<Eigen::Infinity>() < kMinInfinityNorm) {
    *R = {H_normalized};
    *t = {Eigen::Vector3d::Zero()};
    *n = {Eigen::Vector3d::Zero()};
    return;
  }

  const double M00 = ComputeOppositeOfMinor(S, 0, 0);
  const double M11 = ComputeOppositeOfMinor(S, 1, 1);
  const double M22 = ComputeOppositeOfMinor(S, 2, 2);

  const double rtM00 = std::sqrt(M00);
  const double rtM11 = std::sqrt(M11);
  const double rtM22 = std::sqrt(M22);

  const double M01 = ComputeOppositeOfMinor(S, 0, 1);
  const double M12 = ComputeOppositeOfMinor(S, 1, 2);
  const double M02 = ComputeOppositeOfMinor(S, 0, 2);

  const int e12 = SignOfNumber(M12);
  const int e02 = SignOfNumber(M02);
  const int e01 = SignOfNumber(M01);

  const double nS00 = std::abs(S(0, 0));
  const double nS11 = std::abs(S(1, 1));
  const double nS22 = std::abs(S(2, 2));

  const std::array<double, 3> nS{{nS00, nS11, nS22}};
  const size_t idx =
      std::distance(nS.begin(), std::max_element(nS.begin(), nS.end()));

  Eigen::Vector3d np1;
  Eigen::Vector3d np2;
  if (idx == 0) {
    np1[0] = S(0, 0);
    np2[0] = S(0, 0);
    np1[1] = S(0, 1) + rtM22;
    np2[1] = S(0, 1) - rtM22;
    np1[2] = S(0, 2) + e12 * rtM11;
    np2[2] = S(0, 2) - e12 * rtM11;
  } else if (idx == 1) {
    np1[0] = S(0, 1) + rtM22;
    np2[0] = S(0, 1) - rtM22;
    np1[1] = S(1, 1);
    np2[1] = S(1, 1);
    np1[2] = S(1, 2) - e02 * rtM00;
    np2[2] = S(1, 2) + e02 * rtM00;
  } else if (idx == 2) {
    np1[0] = S(0, 2) + e01 * rtM11;
    np2[0] = S(0, 2) - e01 * rtM11;
    np1[1] = S(1, 2) + rtM00;
    np2[1] = S(1, 2) - rtM00;
    np1[2] = S(2, 2);
    np2[2] = S(2, 2);
  }

  const double traceS = S.trace();
  const double v = 2.0 * std::sqrt(1.0 + traceS - M00 - M11 - M22);

  const double ESii = SignOfNumber(S(idx, idx));
  const double r_2 = 2 + traceS + v;
  const double nt_2 = 2 + traceS - v;

  const double r = std::sqrt(r_2);
  const double n_t = std::sqrt(nt_2);

  const Eigen::Vector3d n1 = np1.normalized();
  const Eigen::Vector3d n2 = np2.normalized();

  const double half_nt = 0.5 * n_t;
  const double esii_t_r = ESii * r;

  const Eigen::Vector3d t1_star = half_nt * (esii_t_r * n2 - n_t * n1);
  const Eigen::Vector3d t2_star = half_nt * (esii_t_r * n1 - n_t * n2);

  const Eigen::Matrix3d R1 =
      ComputeHomographyRotation(H_normalized, t1_star, n1, v);
  const Eigen::Vector3d t1 = R1 * t1_star;

  const Eigen::Matrix3d R2 =
      ComputeHomographyRotation(H_normalized, t2_star, n2, v);
  const Eigen::Vector3d t2 = R2 * t2_star;

  *R = {R1, R1, R2, R2};
  *t = {t1, -t1, t2, -t2};
  *n = {-n1, n1, -n2, n2};
}

void PoseFromHomographyMatrix(const Eigen::Matrix3d& H,
                              const Eigen::Matrix3d& K1,
                              const Eigen::Matrix3d& K2,
                              const std::vector<Eigen::Vector2d>& points1,
                              const std::vector<Eigen::Vector2d>& points2,
                              Eigen::Matrix3d* R, Eigen::Vector3d* t,
                              Eigen::Vector3d* n,
                              std::vector<Eigen::Vector3d>* points3D) {
  CHECK_EQ(points1.size(), points2.size());

  std::vector<Eigen::Matrix3d> R_cmbs;
  std::vector<Eigen::Vector3d> t_cmbs;
  std::vector<Eigen::Vector3d> n_cmbs;
  DecomposeHomographyMatrix(H, K1, K2, &R_cmbs, &t_cmbs, &n_cmbs);

  points3D->clear();
  for (size_t i = 0; i < R_cmbs.size(); ++i) {
    std::vector<Eigen::Vector3d> points3D_cmb;
    CheckCheirality(R_cmbs[i], t_cmbs[i], points1, points2, &points3D_cmb);
    if (points3D_cmb.size() >= points3D->size()) {
      *R = R_cmbs[i];
      *t = t_cmbs[i];
      *n = n_cmbs[i];
      *points3D = points3D_cmb;
    }
  }
}

Eigen::Matrix3d HomographyMatrixFromPose(const Eigen::Matrix3d& K1,
                                         const Eigen::Matrix3d& K2,
                                         const Eigen::Matrix3d& R,
                                         const Eigen::Vector3d& t,
                                         const Eigen::Vector3d& n,
                                         const double d) {
  CHECK_GT(d, 0);
  return K2 * (R - t * n.normalized().transpose() / d) * K1.inverse();
}

}  // namespace colmap
