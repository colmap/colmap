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

#include "colmap/estimators/generalized_relative_pose.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/math/random.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Dense>

namespace colmap {
namespace {

void ComposePlueckerData(const Rigid3d& rig_from_cam,
                         const Eigen::Vector3d& ray_in_cam,
                         Eigen::Vector3d* proj_center,
                         Eigen::Vector6d* pluecker) {
  const Eigen::Vector3d ray_in_rig =
      (rig_from_cam.rotation * ray_in_cam).normalized();
  *proj_center = rig_from_cam.translation;
  *pluecker << ray_in_rig, rig_from_cam.translation.cross(ray_in_rig);
}

Eigen::Matrix3d CayleyToRotationMatrix(const Eigen::Vector3d& cayley) {
  const double cayley0_sqr = cayley[0] * cayley[0];
  const double cayley1_sqr = cayley[1] * cayley[1];
  const double cayley2_sqr = cayley[2] * cayley[2];
  const double cayley01 = cayley[0] * cayley[1];
  const double cayley12 = cayley[1] * cayley[2];
  const double cayley02 = cayley[0] * cayley[2];

  const double scale = 1 + cayley0_sqr + cayley1_sqr + cayley2_sqr;
  const double inv_scale = 1.0 / scale;

  Eigen::Matrix3d R;

  R(0, 0) = inv_scale * (1 + cayley0_sqr - cayley1_sqr - cayley2_sqr);
  R(0, 1) = inv_scale * (2 * (cayley01 - cayley[2]));
  R(0, 2) = inv_scale * (2 * (cayley02 + cayley[1]));
  R(1, 0) = inv_scale * (2 * (cayley01 + cayley[2]));
  R(1, 1) = inv_scale * (1 - cayley0_sqr + cayley1_sqr - cayley2_sqr);
  R(1, 2) = inv_scale * (2 * (cayley12 - cayley[0]));
  R(2, 0) = inv_scale * (2 * (cayley02 - cayley[1]));
  R(2, 1) = inv_scale * (2 * (cayley12 + cayley[0]));
  R(2, 2) = inv_scale * (1 - cayley0_sqr - cayley1_sqr + cayley2_sqr);

  return R;
}

Eigen::Vector3d RotationMatrixToCaley(const Eigen::Matrix3d& R) {
  const Eigen::Matrix3d C1 = R - Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d C2 = R + Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d C = C1 * C2.inverse();
  return Eigen::Vector3d(-C(1, 2), C(0, 2), -C(0, 1));
}

Eigen::Vector3d ComputeRotationBetweenPoints(
    const std::vector<Eigen::Vector6d>& plueckers1,
    const std::vector<Eigen::Vector6d>& plueckers2) {
  THROW_CHECK_EQ(plueckers1.size(), plueckers2.size());

  // Compute the center of all observed points.
  Eigen::Vector3d points_center1 = Eigen::Vector3d::Zero();
  Eigen::Vector3d points_center2 = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < plueckers1.size(); i++) {
    points_center1 += plueckers1[i].head<3>();
    points_center2 += plueckers2[i].head<3>();
  }
  points_center1 = points_center1 / plueckers1.size();
  points_center2 = points_center2 / plueckers1.size();

  Eigen::Matrix3d Hcross = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < plueckers1.size(); i++) {
    const Eigen::Vector3d f1 = plueckers1[i].head<3>() - points_center1;
    const Eigen::Vector3d f2 = plueckers2[i].head<3>() - points_center2;
    Hcross += f2 * f1.transpose();
  }

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      Hcross, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Matrix3d& V = svd.matrixV();
  const Eigen::Matrix3d& U = svd.matrixU();

  Eigen::Matrix3d R = V * U.transpose();
  if (R.determinant() < 0) {
    Eigen::Matrix3d V_prime;
    V_prime.col(0) = V.col(0);
    V_prime.col(1) = V.col(1);
    V_prime.col(2) = -V.col(2);
    R = V_prime * U.transpose();
  }

  return RotationMatrixToCaley(R);
}

Eigen::Matrix4d ComposeG(const Eigen::Matrix3d& xxF,
                         const Eigen::Matrix3d& yyF,
                         const Eigen::Matrix3d& zzF,
                         const Eigen::Matrix3d& xyF,
                         const Eigen::Matrix3d& yzF,
                         const Eigen::Matrix3d& zxF,
                         const Eigen::Matrix<double, 3, 9>& x1P,
                         const Eigen::Matrix<double, 3, 9>& y1P,
                         const Eigen::Matrix<double, 3, 9>& z1P,
                         const Eigen::Matrix<double, 3, 9>& x2P,
                         const Eigen::Matrix<double, 3, 9>& y2P,
                         const Eigen::Matrix<double, 3, 9>& z2P,
                         const Eigen::Matrix<double, 9, 9>& m11P,
                         const Eigen::Matrix<double, 9, 9>& m12P,
                         const Eigen::Matrix<double, 9, 9>& m22P,
                         const Eigen::Vector3d& rotation) {
  const Eigen::Matrix3d R = CayleyToRotationMatrix(rotation);

  Eigen::Matrix<double, 1, 9> R_rows;
  R_rows << R.row(0), R.row(1), R.row(2);

  Eigen::Matrix<double, 9, 1> R_cols;
  R_cols << R.col(0), R.col(1), R.col(2);

  const Eigen::Vector3d xxFr1t = xxF * R.row(1).transpose();
  const Eigen::Vector3d yyFr0t = yyF * R.row(0).transpose();
  const Eigen::Vector3d zzFr0t = zzF * R.row(0).transpose();
  const Eigen::Vector3d yzFr0t = yzF * R.row(0).transpose();
  const Eigen::Vector3d xyFr1t = xyF * R.row(1).transpose();
  const Eigen::Vector3d xyFr2t = xyF * R.row(2).transpose();
  const Eigen::Vector3d zxFr1t = zxF * R.row(1).transpose();
  const Eigen::Vector3d zxFr2t = zxF * R.row(2).transpose();

  const Eigen::Vector3d x1PC = x1P * R_cols;
  const Eigen::Vector3d y1PC = y1P * R_cols;
  const Eigen::Vector3d z1PC = z1P * R_cols;

  const Eigen::Vector3d x2PR = x2P * R_rows.transpose();
  const Eigen::Vector3d y2PR = y2P * R_rows.transpose();
  const Eigen::Vector3d z2PR = z2P * R_rows.transpose();

  Eigen::Matrix4d G;

  G(0, 0) = R.row(2) * yyF * R.row(2).transpose();
  G(0, 0) += -2.0 * R.row(2) * yzF * R.row(1).transpose();
  G(0, 0) += R.row(1) * zzF * R.row(1).transpose();

  G(0, 1) = R.row(2) * yzFr0t;
  G(0, 1) += -1.0 * R.row(2) * xyFr2t;
  G(0, 1) += -1.0 * R.row(1) * zzFr0t;
  G(0, 1) += R.row(1) * zxFr2t;

  G(0, 2) = R.row(2) * xyFr1t;
  G(0, 2) += -1.0 * R.row(2) * yyFr0t;
  G(0, 2) += -1.0 * R.row(1) * zxFr1t;
  G(0, 2) += R.row(1) * yzFr0t;

  G(1, 1) = R.row(0) * zzFr0t;
  G(1, 1) += -2.0 * R.row(0) * zxFr2t;
  G(1, 1) += R.row(2) * xxF * R.row(2).transpose();

  G(1, 2) = R.row(0) * zxFr1t;
  G(1, 2) += -1.0 * R.row(0) * yzFr0t;
  G(1, 2) += -1.0 * R.row(2) * xxFr1t;
  G(1, 2) += R.row(0) * xyFr2t;

  G(2, 2) = R.row(1) * xxFr1t;
  G(2, 2) += -2.0 * R.row(0) * xyFr1t;
  G(2, 2) += R.row(0) * yyFr0t;

  G(1, 0) = G(0, 1);
  G(2, 0) = G(0, 2);
  G(2, 1) = G(1, 2);

  G(0, 3) = R.row(2) * y1PC;
  G(0, 3) += R.row(2) * y2PR;
  G(0, 3) += -1.0 * R.row(1) * z1PC;
  G(0, 3) += -1.0 * R.row(1) * z2PR;

  G(1, 3) = R.row(0) * z1PC;
  G(1, 3) += R.row(0) * z2PR;
  G(1, 3) += -1.0 * R.row(2) * x1PC;
  G(1, 3) += -1.0 * R.row(2) * x2PR;

  G(2, 3) = R.row(1) * x1PC;
  G(2, 3) += R.row(1) * x2PR;
  G(2, 3) += -1.0 * R.row(0) * y1PC;
  G(2, 3) += -1.0 * R.row(0) * y2PR;

  G(3, 3) = -1.0 * R_cols.transpose() * m11P * R_cols;
  G(3, 3) += -1.0 * R_rows * m22P * R_rows.transpose();
  G(3, 3) += -2.0 * R_rows * m12P * R_cols;

  G(3, 0) = G(0, 3);
  G(3, 1) = G(1, 3);
  G(3, 2) = G(2, 3);

  return G;
}

Eigen::Vector4d ComputeEigenValue(const Eigen::Matrix3d& xxF,
                                  const Eigen::Matrix3d& yyF,
                                  const Eigen::Matrix3d& zzF,
                                  const Eigen::Matrix3d& xyF,
                                  const Eigen::Matrix3d& yzF,
                                  const Eigen::Matrix3d& zxF,
                                  const Eigen::Matrix<double, 3, 9>& x1P,
                                  const Eigen::Matrix<double, 3, 9>& y1P,
                                  const Eigen::Matrix<double, 3, 9>& z1P,
                                  const Eigen::Matrix<double, 3, 9>& x2P,
                                  const Eigen::Matrix<double, 3, 9>& y2P,
                                  const Eigen::Matrix<double, 3, 9>& z2P,
                                  const Eigen::Matrix<double, 9, 9>& m11P,
                                  const Eigen::Matrix<double, 9, 9>& m12P,
                                  const Eigen::Matrix<double, 9, 9>& m22P,
                                  const Eigen::Vector3d& rotation) {
  const Eigen::Matrix4d G = ComposeG(xxF,
                                     yyF,
                                     zzF,
                                     xyF,
                                     yzF,
                                     zxF,
                                     x1P,
                                     y1P,
                                     z1P,
                                     x2P,
                                     y2P,
                                     z2P,
                                     m11P,
                                     m12P,
                                     m22P,
                                     rotation);

  // Compute the roots in closed-form.
  // const double G00_2 = G(0,0) * G(0,0);
  const double G01_2 = G(0, 1) * G(0, 1);
  const double G02_2 = G(0, 2) * G(0, 2);
  const double G03_2 = G(0, 3) * G(0, 3);
  // const double G11_2 = G(1,1) * G(1,1);
  const double G12_2 = G(1, 2) * G(1, 2);
  const double G13_2 = G(1, 3) * G(1, 3);
  // const double G22_2 = G(2,2) * G(2,2);
  const double G23_2 = G(2, 3) * G(2, 3);
  // const double G33_2 = G(3,3) * G(3,3);

  const double B = -G(3, 3) - G(2, 2) - G(1, 1) - G(0, 0);
  const double C = -G23_2 + G(2, 2) * G(3, 3) - G13_2 - G12_2 +
                   G(1, 1) * G(3, 3) + G(1, 1) * G(2, 2) - G03_2 - G02_2 -
                   G01_2 + G(0, 0) * G(3, 3) + G(0, 0) * G(2, 2) +
                   G(0, 0) * G(1, 1);
  const double D =
      G13_2 * G(2, 2) - 2.0 * G(1, 2) * G(1, 3) * G(2, 3) + G12_2 * G(3, 3) +
      G(1, 1) * G23_2 - G(1, 1) * G(2, 2) * G(3, 3) + G03_2 * G(2, 2) +
      G03_2 * G(1, 1) - 2.0 * G(0, 2) * G(0, 3) * G(2, 3) + G02_2 * G(3, 3) +
      G02_2 * G(1, 1) - 2.0 * G(0, 1) * G(0, 3) * G(1, 3) -
      2.0 * G(0, 1) * G(0, 2) * G(1, 2) + G01_2 * G(3, 3) + G01_2 * G(2, 2) +
      G(0, 0) * G23_2 - G(0, 0) * G(2, 2) * G(3, 3) + G(0, 0) * G13_2 +
      G(0, 0) * G12_2 - G(0, 0) * G(1, 1) * G(3, 3) -
      G(0, 0) * G(1, 1) * G(2, 2);
  const double E =
      G03_2 * G12_2 - G03_2 * G(1, 1) * G(2, 2) -
      2.0 * G(0, 2) * G(0, 3) * G(1, 2) * G(1, 3) +
      2.0 * G(0, 2) * G(0, 3) * G(1, 1) * G(2, 3) + G02_2 * G13_2 -
      G02_2 * G(1, 1) * G(3, 3) + 2.0 * G(0, 1) * G(0, 3) * G(1, 3) * G(2, 2) -
      2.0 * G(0, 1) * G(0, 3) * G(1, 2) * G(2, 3) -
      2.0 * G(0, 1) * G(0, 2) * G(1, 3) * G(2, 3) +
      2.0 * G(0, 1) * G(0, 2) * G(1, 2) * G(3, 3) + G01_2 * G23_2 -
      G01_2 * G(2, 2) * G(3, 3) - G(0, 0) * G13_2 * G(2, 2) +
      2.0 * G(0, 0) * G(1, 2) * G(1, 3) * G(2, 3) - G(0, 0) * G12_2 * G(3, 3) -
      G(0, 0) * G(1, 1) * G23_2 + G(0, 0) * G(1, 1) * G(2, 2) * G(3, 3);

  const double B_pw2 = B * B;
  const double B_pw3 = B_pw2 * B;
  const double B_pw4 = B_pw3 * B;
  const double alpha = -0.375 * B_pw2 + C;
  const double beta = B_pw3 / 8.0 - B * C / 2.0 + D;
  const double gamma = -0.01171875 * B_pw4 + B_pw2 * C / 16.0 - B * D / 4.0 + E;
  const double alpha_pw2 = alpha * alpha;
  const double alpha_pw3 = alpha_pw2 * alpha;
  const double p = -alpha_pw2 / 12.0 - gamma;
  const double q = -alpha_pw3 / 108.0 + alpha * gamma / 3.0 - beta * beta / 8.0;
  const double helper1 = -p * p * p / 27.0;
  const double theta2 = std::pow(helper1, (1.0 / 3.0));
  const double theta1 =
      std::sqrt(theta2) *
      std::cos((1.0 / 3.0) * std::acos((-q / 2.0) / std::sqrt(helper1)));
  const double y = -(5.0 / 6.0) * alpha -
                   ((1.0 / 3.0) * p * theta1 - theta1 * theta2) / theta2;
  const double w = std::sqrt(alpha + 2.0 * y);

  Eigen::Vector4d roots;
  roots(0) = -B / 4.0 + 0.5 * w +
             0.5 * std::sqrt(-3.0 * alpha - 2.0 * y - 2.0 * beta / w);
  roots(1) = -B / 4.0 + 0.5 * w -
             0.5 * std::sqrt(-3.0 * alpha - 2.0 * y - 2.0 * beta / w);
  roots(2) = -B / 4.0 - 0.5 * w +
             0.5 * std::sqrt(-3.0 * alpha - 2.0 * y + 2.0 * beta / w);
  roots(3) = -B / 4.0 - 0.5 * w -
             0.5 * std::sqrt(-3.0 * alpha - 2.0 * y + 2.0 * beta / w);
  return roots;
}

double ComputeCost(const Eigen::Matrix3d& xxF,
                   const Eigen::Matrix3d& yyF,
                   const Eigen::Matrix3d& zzF,
                   const Eigen::Matrix3d& xyF,
                   const Eigen::Matrix3d& yzF,
                   const Eigen::Matrix3d& zxF,
                   const Eigen::Matrix<double, 3, 9>& x1P,
                   const Eigen::Matrix<double, 3, 9>& y1P,
                   const Eigen::Matrix<double, 3, 9>& z1P,
                   const Eigen::Matrix<double, 3, 9>& x2P,
                   const Eigen::Matrix<double, 3, 9>& y2P,
                   const Eigen::Matrix<double, 3, 9>& z2P,
                   const Eigen::Matrix<double, 9, 9>& m11P,
                   const Eigen::Matrix<double, 9, 9>& m12P,
                   const Eigen::Matrix<double, 9, 9>& m22P,
                   const Eigen::Vector3d& rotation,
                   const int step) {
  THROW_CHECK_GE(step, 0);
  THROW_CHECK_LE(step, 1);

  const Eigen::Vector4d roots = ComputeEigenValue(xxF,
                                                  yyF,
                                                  zzF,
                                                  xyF,
                                                  yzF,
                                                  zxF,
                                                  x1P,
                                                  y1P,
                                                  z1P,
                                                  x2P,
                                                  y2P,
                                                  z2P,
                                                  m11P,
                                                  m12P,
                                                  m22P,
                                                  rotation);

  if (step == 0) {
    return roots[2];
  } else if (step == 1) {
    return roots[3];
  }

  return 0;
}

Eigen::Vector3d ComputeJacobian(const Eigen::Matrix3d& xxF,
                                const Eigen::Matrix3d& yyF,
                                const Eigen::Matrix3d& zzF,
                                const Eigen::Matrix3d& xyF,
                                const Eigen::Matrix3d& yzF,
                                const Eigen::Matrix3d& zxF,
                                const Eigen::Matrix<double, 3, 9>& x1P,
                                const Eigen::Matrix<double, 3, 9>& y1P,
                                const Eigen::Matrix<double, 3, 9>& z1P,
                                const Eigen::Matrix<double, 3, 9>& x2P,
                                const Eigen::Matrix<double, 3, 9>& y2P,
                                const Eigen::Matrix<double, 3, 9>& z2P,
                                const Eigen::Matrix<double, 9, 9>& m11P,
                                const Eigen::Matrix<double, 9, 9>& m12P,
                                const Eigen::Matrix<double, 9, 9>& m22P,
                                const Eigen::Vector3d& rotation,
                                const double current_cost,
                                const int step) {
  Eigen::Vector3d jacobian;
  constexpr double kStepSize = 1e-8;
  for (int j = 0; j < 3; j++) {
    Eigen::Vector3d cayley_j = rotation;
    cayley_j[j] += kStepSize;
    const double cost_j = ComputeCost(xxF,
                                      yyF,
                                      zzF,
                                      xyF,
                                      yzF,
                                      zxF,
                                      x1P,
                                      y1P,
                                      z1P,
                                      x2P,
                                      y2P,
                                      z2P,
                                      m11P,
                                      m12P,
                                      m22P,
                                      cayley_j,
                                      step);
    jacobian(j) = cost_j - current_cost;
  }
  return jacobian;
}

}  // namespace

void GR8PEstimator::Estimate(const std::vector<X_t>& points1,
                             const std::vector<Y_t>& points2,
                             std::vector<M_t>* rigs2_from_rigs1) {
  THROW_CHECK_GE(points1.size(), 6);
  THROW_CHECK_EQ(points1.size(), points2.size());
  THROW_CHECK(rigs2_from_rigs1 != nullptr);

  rigs2_from_rigs1->clear();

  std::vector<Eigen::Vector3d> proj_centers1(points1.size());
  std::vector<Eigen::Vector3d> proj_centers2(points1.size());
  std::vector<Eigen::Vector6d> plueckers1(points1.size());
  std::vector<Eigen::Vector6d> plueckers2(points1.size());
  for (size_t i = 0; i < points1.size(); ++i) {
    ComposePlueckerData(Inverse(points1[i].cam_from_rig),
                        points1[i].ray_in_cam,
                        &proj_centers1[i],
                        &plueckers1[i]);
    ComposePlueckerData(Inverse(points2[i].cam_from_rig),
                        points2[i].ray_in_cam,
                        &proj_centers2[i],
                        &plueckers2[i]);
  }

  Eigen::Matrix3d xxF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d yyF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d zzF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d xyF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d yzF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d zxF = Eigen::Matrix3d::Zero();

  Eigen::Matrix<double, 3, 9> x1P = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> y1P = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> z1P = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> x2P = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> y2P = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> z2P = Eigen::Matrix<double, 3, 9>::Zero();

  Eigen::Matrix<double, 9, 9> m11P = Eigen::Matrix<double, 9, 9>::Zero();
  Eigen::Matrix<double, 9, 9> m12P = Eigen::Matrix<double, 9, 9>::Zero();
  Eigen::Matrix<double, 9, 9> m22P = Eigen::Matrix<double, 9, 9>::Zero();

  for (size_t i = 0; i < points1.size(); ++i) {
    const Eigen::Vector3d f1 = plueckers1[i].head<3>();
    const Eigen::Vector3d f2 = plueckers2[i].head<3>();
    const Eigen::Vector3d t1 = proj_centers1[i];
    const Eigen::Vector3d t2 = proj_centers2[i];

    const Eigen::Matrix3d F = f2 * f2.transpose();
    xxF += f1[0] * f1[0] * F;
    yyF += f1[1] * f1[1] * F;
    zzF += f1[2] * f1[2] * F;
    xyF += f1[0] * f1[1] * F;
    yzF += f1[1] * f1[2] * F;
    zxF += f1[2] * f1[0] * F;

    Eigen::Matrix<double, 9, 1> ff1;
    ff1(0) = f1[0] * (f2[1] * t2[2] - f2[2] * t2[1]);
    ff1(1) = f1[1] * (f2[1] * t2[2] - f2[2] * t2[1]);
    ff1(2) = f1[2] * (f2[1] * t2[2] - f2[2] * t2[1]);
    ff1(3) = f1[0] * (f2[2] * t2[0] - f2[0] * t2[2]);
    ff1(4) = f1[1] * (f2[2] * t2[0] - f2[0] * t2[2]);
    ff1(5) = f1[2] * (f2[2] * t2[0] - f2[0] * t2[2]);
    ff1(6) = f1[0] * (f2[0] * t2[1] - f2[1] * t2[0]);
    ff1(7) = f1[1] * (f2[0] * t2[1] - f2[1] * t2[0]);
    ff1(8) = f1[2] * (f2[0] * t2[1] - f2[1] * t2[0]);

    x1P += f1[0] * f2 * ff1.transpose();
    y1P += f1[1] * f2 * ff1.transpose();
    z1P += f1[2] * f2 * ff1.transpose();

    Eigen::Matrix<double, 9, 1> ff2;
    ff2(0) = f2[0] * (f1[1] * t1[2] - f1[2] * t1[1]);
    ff2(1) = f2[1] * (f1[1] * t1[2] - f1[2] * t1[1]);
    ff2(2) = f2[2] * (f1[1] * t1[2] - f1[2] * t1[1]);
    ff2(3) = f2[0] * (f1[2] * t1[0] - f1[0] * t1[2]);
    ff2(4) = f2[1] * (f1[2] * t1[0] - f1[0] * t1[2]);
    ff2(5) = f2[2] * (f1[2] * t1[0] - f1[0] * t1[2]);
    ff2(6) = f2[0] * (f1[0] * t1[1] - f1[1] * t1[0]);
    ff2(7) = f2[1] * (f1[0] * t1[1] - f1[1] * t1[0]);
    ff2(8) = f2[2] * (f1[0] * t1[1] - f1[1] * t1[0]);

    x2P += f1[0] * f2 * ff2.transpose();
    y2P += f1[1] * f2 * ff2.transpose();
    z2P += f1[2] * f2 * ff2.transpose();

    m11P -= ff1 * ff1.transpose();
    m22P -= ff2 * ff2.transpose();
    m12P -= ff2 * ff1.transpose();
  }

  const Eigen::Vector3d initial_rotation =
      ComputeRotationBetweenPoints(plueckers1, plueckers2);

  const double kMinLambda = 0.00001;
  const double kMaxLambda = 0.08;
  const double kLambdaModifier = 2.0;
  const int kMaxNumIterations = 50;
  const bool kDisableIncrements = true;

  double perturbation_amplitude = 0.3;
  int num_random_trials = 0;

  Eigen::Vector3d rotation;
  while (num_random_trials < 5) {
    if (num_random_trials > 2) {
      perturbation_amplitude = 0.6;
    }

    if (num_random_trials == 0) {
      rotation = initial_rotation;
    } else {
      const Eigen::Vector3d perturbation(
          RandomUniformReal<double>(-perturbation_amplitude,
                                    perturbation_amplitude),
          RandomUniformReal<double>(-perturbation_amplitude,
                                    perturbation_amplitude),
          RandomUniformReal<double>(-perturbation_amplitude,
                                    perturbation_amplitude));
      rotation = initial_rotation + perturbation;
    }

    double lambda = 0.01;
    int num_iterations = 0;
    double smallest_eigen_value = ComputeCost(xxF,
                                              yyF,
                                              zzF,
                                              xyF,
                                              yzF,
                                              zxF,
                                              x1P,
                                              y1P,
                                              z1P,
                                              x2P,
                                              y2P,
                                              z2P,
                                              m11P,
                                              m12P,
                                              m22P,
                                              rotation,
                                              1);

    for (int iter = 0; iter < kMaxNumIterations; ++iter) {
      const Eigen::Vector3d jacobian = ComputeJacobian(xxF,
                                                       yyF,
                                                       zzF,
                                                       xyF,
                                                       yzF,
                                                       zxF,
                                                       x1P,
                                                       y1P,
                                                       z1P,
                                                       x2P,
                                                       y2P,
                                                       z2P,
                                                       m11P,
                                                       m12P,
                                                       m22P,
                                                       rotation,
                                                       smallest_eigen_value,
                                                       1);

      const Eigen::Vector3d normalized_jacobian = jacobian.normalized();

      Eigen::Vector3d sampling_point = rotation - lambda * normalized_jacobian;
      double sampling_eigen_value = ComputeCost(xxF,
                                                yyF,
                                                zzF,
                                                xyF,
                                                yzF,
                                                zxF,
                                                x1P,
                                                y1P,
                                                z1P,
                                                x2P,
                                                y2P,
                                                z2P,
                                                m11P,
                                                m12P,
                                                m22P,
                                                sampling_point,
                                                1);

      if (num_iterations == 0 || !kDisableIncrements) {
        while (sampling_eigen_value < smallest_eigen_value) {
          smallest_eigen_value = sampling_eigen_value;
          if (lambda * kLambdaModifier > kMaxLambda) {
            break;
          }
          lambda *= kLambdaModifier;
          sampling_point = rotation - lambda * normalized_jacobian;
          sampling_eigen_value = ComputeCost(xxF,
                                             yyF,
                                             zzF,
                                             xyF,
                                             yzF,
                                             zxF,
                                             x1P,
                                             y1P,
                                             z1P,
                                             x2P,
                                             y2P,
                                             z2P,
                                             m11P,
                                             m12P,
                                             m22P,
                                             sampling_point,
                                             1);
        }
      }

      while (sampling_eigen_value > smallest_eigen_value) {
        lambda /= kLambdaModifier;
        sampling_point = rotation - lambda * normalized_jacobian;
        sampling_eigen_value = ComputeCost(xxF,
                                           yyF,
                                           zzF,
                                           xyF,
                                           yzF,
                                           zxF,
                                           x1P,
                                           y1P,
                                           z1P,
                                           x2P,
                                           y2P,
                                           z2P,
                                           m11P,
                                           m12P,
                                           m22P,
                                           sampling_point,
                                           1);
      }

      rotation = sampling_point;
      smallest_eigen_value = sampling_eigen_value;

      if (lambda < kMinLambda) {
        break;
      }
    }

    if (rotation.norm() < 0.01) {
      const double eigen_value2 = ComputeCost(xxF,
                                              yyF,
                                              zzF,
                                              xyF,
                                              yzF,
                                              zxF,
                                              x1P,
                                              y1P,
                                              z1P,
                                              x2P,
                                              y2P,
                                              z2P,
                                              m11P,
                                              m12P,
                                              m22P,
                                              rotation,
                                              0);
      if (eigen_value2 > 0.001) {
        num_random_trials += 1;
      } else {
        break;
      }
    } else {
      break;
    }
  }

  const Eigen::Matrix3d R = CayleyToRotationMatrix(rotation).transpose();

  const Eigen::Matrix4d G = ComposeG(xxF,
                                     yyF,
                                     zzF,
                                     xyF,
                                     yzF,
                                     zxF,
                                     x1P,
                                     y1P,
                                     z1P,
                                     x2P,
                                     y2P,
                                     z2P,
                                     m11P,
                                     m12P,
                                     m22P,
                                     rotation);

  const Eigen::EigenSolver<Eigen::Matrix4d> eigen_solver_G(G, true);
  const Eigen::Matrix4cd V = eigen_solver_G.eigenvectors();
  const Eigen::Matrix3x4d VV = V.real().colwise().hnormalized();

  rigs2_from_rigs1->resize(4);
  for (int i = 0; i < 4; ++i) {
    (*rigs2_from_rigs1)[i].rotation = Eigen::Quaterniond(R);
    (*rigs2_from_rigs1)[i].translation = -R * VV.col(i);
  }
}

void GR8PEstimator::Residuals(const std::vector<X_t>& points1,
                              const std::vector<Y_t>& points2,
                              const M_t& rig2_from_rig1,
                              std::vector<double>* residuals) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  residuals->resize(points1.size(), 0);
  for (size_t i = 0; i < points1.size(); ++i) {
    const Rigid3d cam2_from_cam1 = points2[i].cam_from_rig * rig2_from_rig1 *
                                   Inverse(points1[i].cam_from_rig);
    const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);
    const Eigen::Vector3d Ex1 =
        E * points1[i].ray_in_cam.hnormalized().homogeneous();
    const Eigen::Vector3d x2 =
        points2[i].ray_in_cam.hnormalized().homogeneous();
    const Eigen::Vector3d Etx2 = E.transpose() * x2;
    const double x2tEx1 = x2.transpose() * Ex1;
    (*residuals)[i] = x2tEx1 * x2tEx1 /
                      (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) +
                       Etx2(1) * Etx2(1));
  }
}

// Copyright (c) 2020, Viktor Larsson
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
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

namespace poselib {
namespace {

#define USE_FAST_EIGENVECTOR_SOLVER

inline Eigen::Matrix3d quat_to_rotmat(const Eigen::Vector4d& q) {
  return Eigen::Quaterniond(q(0), q(1), q(2), q(3)).toRotationMatrix();
}

inline Eigen::Vector3d quat_rotate(const Eigen::Vector4d& q,
                                   const Eigen::Vector3d& p) {
  const double q1 = q(0), q2 = q(1), q3 = q(2), q4 = q(3);
  const double p1 = p(0), p2 = p(1), p3 = p(2);
  const double px1 = -p1 * q2 - p2 * q3 - p3 * q4;
  const double px2 = p1 * q1 - p2 * q4 + p3 * q3;
  const double px3 = p2 * q1 + p1 * q4 - p3 * q2;
  const double px4 = p2 * q2 - p1 * q3 + p3 * q1;
  return Eigen::Vector3d(px2 * q1 - px1 * q2 - px3 * q4 + px4 * q3,
                         px3 * q1 - px1 * q3 + px2 * q4 - px4 * q2,
                         px3 * q2 - px2 * q3 - px1 * q4 + px4 * q1);
}

inline Eigen::Vector4d quat_conj(const Eigen::Vector4d& q) {
  return Eigen::Vector4d(q(0), -q(1), -q(2), -q(3));
}

inline Eigen::Vector4d quat_multiply(const Eigen::Vector4d& qa,
                                     const Eigen::Vector4d& qb) {
  const double qa1 = qa(0), qa2 = qa(1), qa3 = qa(2), qa4 = qa(3);
  const double qb1 = qb(0), qb2 = qb(1), qb3 = qb(2), qb4 = qb(3);

  return Eigen::Vector4d(qa1 * qb1 - qa2 * qb2 - qa3 * qb3 - qa4 * qb4,
                         qa1 * qb2 + qa2 * qb1 + qa3 * qb4 - qa4 * qb3,
                         qa1 * qb3 + qa3 * qb1 - qa2 * qb4 + qa4 * qb2,
                         qa1 * qb4 + qa2 * qb3 - qa3 * qb2 + qa4 * qb1);
}

inline Eigen::Vector4d quat_exp(const Eigen::Vector3d& w) {
  const double theta2 = w.squaredNorm();
  const double theta = std::sqrt(theta2);
  const double theta_half = 0.5 * theta;

  double re, im;
  if (theta > 1e-6) {
    re = std::cos(theta_half);
    im = std::sin(theta_half) / theta;
  } else {
    // we are close to zero, use taylor expansion to avoid problems
    // with zero divisors in sin(theta/2)/theta
    const double theta4 = theta2 * theta2;
    re = 1.0 - (1.0 / 8.0) * theta2 + (1.0 / 384.0) * theta4;
    im = 0.5 - (1.0 / 48.0) * theta2 + (1.0 / 3840.0) * theta4;

    // for the linearized part we re-normalize to ensure unit length
    // here s should be roughly 1.0 anyways, so no problem with zero div
    const double s = std::sqrt(re * re + im * im * theta2);
    re /= s;
    im /= s;
  }
  return Eigen::Vector4d(re, im * w(0), im * w(1), im * w(2));
}

inline Eigen::Vector4d quat_step_pre(const Eigen::Vector4d& q,
                                     const Eigen::Vector3d& w_delta) {
  return quat_multiply(quat_exp(w_delta), q);
}

struct alignas(32) CameraPose {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Rotation is represented as a unit quaternion
  // with real part first, i.e. QW, QX, QY, QZ
  Eigen::Vector4d q;
  Eigen::Vector3d t;

  // Constructors (Defaults to identity camera)
  CameraPose() : q(1.0, 0.0, 0.0, 0.0), t(0.0, 0.0, 0.0) {}
  CameraPose(const Eigen::Vector4d& qq, const Eigen::Vector3d& tt)
      : q(qq), t(tt) {}

  // Helper functions
  inline Eigen::Matrix3d R() const { return quat_to_rotmat(q); }
  inline Eigen::Matrix<double, 3, 4> Rt() const {
    Eigen::Matrix<double, 3, 4> tmp;
    tmp.block<3, 3>(0, 0) = quat_to_rotmat(q);
    tmp.col(3) = t;
    return tmp;
  }
  inline Eigen::Vector3d rotate(const Eigen::Vector3d& p) const {
    return quat_rotate(q, p);
  }
  inline Eigen::Vector3d derotate(const Eigen::Vector3d& p) const {
    return quat_rotate(quat_conj(q), p);
  }
  inline Eigen::Vector3d apply(const Eigen::Vector3d& p) const {
    return rotate(p) + t;
  }

  inline Eigen::Vector3d center() const { return -derotate(t); }
};

bool check_cheirality(const CameraPose& pose,
                      const Eigen::Vector3d& p1,
                      const Eigen::Vector3d& x1,
                      const Eigen::Vector3d& p2,
                      const Eigen::Vector3d& x2,
                      double min_depth = 0.0) {
  // This code assumes that x1 and x2 are unit vectors
  const Eigen::Vector3d Rx1 = pose.rotate(x1);

  // [1 a; a 1] * [lambda1; lambda2] = [b1; b2]
  // [lambda1; lambda2] = [1 -a; -a 1] * [b1; b2] / (1 - a*a)
  const Eigen::Vector3d rhs = pose.t + pose.rotate(p1) - p2;
  const double a = -Rx1.dot(x2);
  const double b1 = -Rx1.dot(rhs);
  const double b2 = x2.dot(rhs);

  // Note that we drop the factor 1.0/(1-a*a) since it is always positive.
  const double lambda1 = b1 - a * b2;
  const double lambda2 = -a * b1 + b2;

  min_depth = min_depth * (1 - a * a);
  return lambda1 > min_depth && lambda2 > min_depth;
}

static const int coeffs0_ind[] = {
    0,    84,   168,  252,  336,  420,  504,  588,  1,    85,   169,  253,
    337,  421,  505,  589,  2,    86,   170,  254,  338,  422,  506,  590,
    3,    87,   171,  255,  339,  423,  507,  591,  4,    88,   172,  256,
    340,  424,  508,  592,  5,    89,   173,  257,  341,  425,  509,  593,
    6,    90,   174,  258,  342,  426,  510,  594,  0,    84,   168,  252,
    336,  420,  504,  1,    84,   85,   168,  169,  252,  253,  336,  0,
    337,  420,  421,  504,  505,  588,  672,  756,  840,  924,  1008, 7,
    91,   175,  259,  343,  427,  511,  2,    85,   86,   169,  170,  253,
    254,  337,  1,    338,  421,  422,  505,  506,  589,  595,  673,  757,
    841,  925,  1009, 8,    92,   176,  260,  344,  428,  512,  3,    86,
    87,   170,  171,  254,  255,  338,  2,    339,  422,  423,  506,  507,
    590,  596,  674,  758,  842,  926,  1010, 9,    93,   177,  261,  345,
    429,  513,  4,    87,   88,   171,  172,  255,  256,  339,  3,    340,
    423,  424,  507,  508,  591,  597,  675,  759,  843,  927,  1011, 10,
    94,   178,  262,  346,  430,  514,  5,    88,   89,   172,  173,  256,
    257,  340,  4,    341,  424,  425,  508,  509,  592,  598,  676,  760,
    844,  928,  1012, 11,   95,   179,  263,  347,  431,  515,  6,    89,
    90,   173,  174,  257,  258,  341,  5,    342,  425,  426,  509,  510,
    593,  599,  677,  761,  845,  929,  1013, 12,   96,   180,  264,  348,
    432,  516,  90,   174,  258,  342,  6,    426,  510,  594,  600,  678,
    762,  846,  930,  1014, 7,    91,   175,  259,  168,  84,   0,    252,
    336,  343,  420,  427,  504,  511,  588,  672,  756,  840,  924,  1008,
    1092, 1176, 8,    91,   92,   175,  176,  259,  260,  343,  169,  85,
    1,    253,  337,  7,    344,  421,  427,  428,  505,  511,  512,  589,
    595,  673,  679,  757,  763,  841,  847,  925,  931,  1009, 1015, 1093,
    1177, 13,   97,   181,  265,  349,  433,  517,  9,    92,   93,   176,
    177,  260,  261,  344,  170,  86,   2,    254,  338,  8,    345,  422,
    428,  429,  506,  512,  513,  590,  596,  601,  674,  680,  758,  764,
    842,  848,  926,  932,  1010, 1016, 1094, 1178, 14,   98,   182,  266,
    350,  434,  518,  10,   93,   94,   177,  178,  261,  262,  345,  171,
    87,   3,    255,  339,  9,    346,  423,  429,  430,  507,  513,  514,
    591,  597,  602,  675,  681,  759,  765,  843,  849,  927,  933,  1011,
    1017, 1095, 1179, 15,   99,   183,  267,  351,  435,  519,  11,   94,
    95,   178,  179,  262,  263,  346,  172,  88,   4,    256,  340,  10,
    347,  424,  430,  431,  508,  514,  515,  592,  598,  603,  676,  682,
    760,  766,  844,  850,  928,  934,  1012, 1018, 1096, 1180, 16,   100,
    184,  268,  352,  436,  520,  12,   95,   96,   179,  180,  263,  264,
    347,  173,  89,   5,    257,  341,  11,   348,  425,  431,  432,  509,
    515,  516,  593,  599,  604,  677,  683,  761,  767,  845,  851,  929,
    935,  1013, 1019, 1097, 1181, 17,   101,  185,  269,  353,  437,  521,
    96,   180,  264,  348,  174,  90,   6,    258,  342,  12,   426,  432,
    510,  516,  594,  600,  605,  678,  684,  762,  768,  846,  852,  930,
    936,  1014, 1020, 1098, 1182, 13,   97,   181,  265,  175,  91,   7,
    259,  343,  349,  427,  433,  511,  517,  595,  679,  763,  847,  931,
    1015, 1099, 1183, 14,   97,   98,   181,  182,  265,  266,  349,  176,
    92,   8,    260,  344,  13,   350,  428,  433,  434,  512,  517,  518,
    596,  601,  680,  685,  764,  769,  848,  853,  932,  937,  1016, 1021,
    1100, 1184, 18,   102,  186,  270,  354,  438,  522,  15,   98,   99,
    182,  183,  266,  267,  350,  177,  93,   9,    261,  345,  14,   351,
    429,  434,  435,  513,  518,  519,  597,  602,  606,  681,  686,  765,
    770,  849,  854,  933,  938,  1017, 1022, 1101, 1185, 19,   103,  187,
    271,  355,  439,  523,  16,   99,   100,  183,  184,  267,  268,  351,
    178,  94,   10,   262,  346,  15,   352,  430,  435,  436,  514,  519,
    520,  598,  603,  607,  682,  687,  766,  771,  850,  855,  934,  939,
    1018, 1023, 1102, 1186, 20,   104,  188,  272,  356,  440,  524,  17,
    100,  101,  184,  185,  268,  269,  352,  179,  95,   11,   263,  347,
    16,   353,  431,  436,  437,  515,  520,  521,  599,  604,  608,  683,
    688,  767,  772,  851,  856,  935,  940,  1019, 1024, 1103, 1187, 21,
    105,  189,  273,  357,  441,  525,  101,  185,  269,  353,  180,  96,
    12,   264,  348,  17,   432,  437,  516,  521,  600,  605,  609,  684,
    689,  768,  773,  852,  857,  936,  941,  1020, 1025, 1104, 1188, 18,
    102,  186,  270,  181,  97,   13,   265,  349,  354,  433,  438,  517,
    522,  601,  685,  769,  853,  937,  1021, 1105, 1189, 19,   102,  103,
    186,  187,  270,  271,  354,  182,  98,   14,   266,  350,  18,   355,
    434,  438,  439,  518,  522,  523,  602,  606,  686,  690,  770,  774,
    854,  858,  938,  942,  1022, 1026, 1106, 1190, 22,   106,  190,  274,
    358,  442,  526,  20,   103,  104,  187,  188,  271,  272,  355,  183,
    99,   15,   267,  351,  19,   356,  435,  439,  440,  519,  523,  524,
    603,  607,  610,  687,  691,  771,  775,  855,  859,  939,  943,  1023,
    1027, 1107, 1191, 23,   107,  191,  275,  359,  443,  527,  21,   104,
    105,  188,  189,  272,  273,  356,  184,  100,  16,   268,  352,  20,
    357,  436,  440,  441,  520,  524,  525,  604,  608,  611,  688,  692,
    772,  776,  856,  860,  940,  944,  1024, 1028, 1108, 1192, 24,   108,
    192,  276,  360,  444,  528,  105,  189,  273,  357,  185,  101,  17,
    269,  353,  21,   437,  441,  521,  525,  605,  609,  612,  689,  693,
    773,  777,  857,  861,  941,  945,  1025, 1029, 1109, 1193, 22,   106,
    190,  274,  186,  102,  18,   270,  354,  358,  438,  442,  522,  526,
    606,  690,  774,  858,  942,  1026, 1110, 1194, 23,   106,  107,  190,
    191,  274,  275,  358,  187,  103,  19,   271,  355,  22,   359,  439,
    442,  443,  523,  526,  527,  607,  610,  691,  694,  775,  778,  859,
    862,  943,  946,  1027, 1030, 1111, 1195, 25,   109,  193,  277,  361,
    445,  529,  24,   107,  108,  191,  192,  275,  276,  359,  188,  104,
    20,   272,  356,  23,   360,  440,  443,  444,  524,  527,  528,  608,
    611,  613,  692,  695,  776,  779,  860,  863,  944,  947,  1028, 1031,
    1112, 1196, 26,   110,  194,  278,  362,  446,  530,  108,  192,  276,
    360,  189,  105,  21,   273,  357,  24,   441,  444,  525,  528,  609,
    612,  614,  693,  696,  777,  780,  861,  864,  945,  948,  1029, 1032,
    1113, 1197, 25,   109,  193,  277,  190,  106,  22,   274,  358,  361,
    442,  445,  526,  529,  610,  694,  778,  862,  946,  1030, 1114, 1198,
    26,   109,  110,  193,  194,  277,  278,  361,  191,  107,  23,   275,
    359,  25,   362,  443,  445,  446,  527,  529,  530,  611,  613,  695,
    697,  779,  781,  863,  865,  947,  949,  1031, 1033, 1115, 1199, 27,
    111,  195,  279,  363,  447,  531,  110,  194,  278,  362,  192,  108,
    24,   276,  360,  26,   444,  446,  528,  530,  612,  614,  615,  696,
    698,  780,  782,  864,  866,  948,  950,  1032, 1034, 1116, 1200, 27,
    111,  195,  279,  193,  109,  25,   277,  361,  363,  445,  447,  529,
    531,  613,  697,  781,  865,  949,  1033, 1117, 1201, 111,  195,  279,
    363,  194,  110,  26,   278,  362,  27,   446,  447,  530,  531,  614,
    615,  698,  699,  782,  783,  866,  867,  950,  951,  1034, 1035, 1118,
    1202, 168,  420,  0,    504,  252,  588,  672,  756,  84,   840,  336,
    924,  169,  420,  84,   421,  504,  1,    505,  252,  588,  0,    253,
    589,  672,  673,  756,  168,  757,  336,  840,  85,   841,  924,  337,
    925,  1008, 1092, 28,   112,  196,  280,  364,  448,  532,  616,  170,
    421,  85,   422,  505,  2,    506,  253,  589,  1,    254,  590,  673,
    674,  757,  169,  758,  337,  841,  86,   842,  925,  338,  926,  1009,
    1093, 29,   113,  197,  281,  365,  449,  533,  617,  171,  422,  86,
    423,  506,  3,    507,  254,  590,  2,    255,  591,  674,  675,  758,
    170,  759,  338,  842,  87,   843,  926,  339,  927,  1010, 1094, 30,
    114,  198,  282,  366,  450,  534,  618,  172,  423,  87,   424,  507,
    4,    508,  255,  591,  3,    256,  592,  675,  676,  759,  171,  760,
    339,  843,  88,   844,  927,  340,  928,  1011, 1095, 31,   115,  199,
    283,  367,  451,  535,  619,  173,  424,  88,   425,  508,  5,    509,
    256,  592,  4,    257,  593,  676,  677,  760,  172,  761,  340,  844,
    89,   845,  928,  341,  929,  1012, 1096, 32,   116,  200,  284,  368,
    452,  536,  620,  174,  425,  89,   426,  509,  6,    510,  257,  593,
    5,    258,  594,  677,  678,  761,  173,  762,  341,  845,  90,   846,
    929,  342,  930,  1013, 1097, 33,   117,  201,  285,  369,  453,  537,
    621,  426,  90,   510,  258,  594,  6,    678,  762,  174,  342,  846,
    930,  1014, 1098, 28,   112,  196,  280,  364,  448,  532,  175,  427,
    7,    511,  252,  84,   504,  588,  259,  595,  672,  0,    679,  756,
    336,  763,  840,  91,   847,  924,  343,  931,  1008, 168,  1092, 420,
    1176, 29,   112,  113,  196,  197,  280,  281,  364,  28,   365,  448,
    449,  532,  533,  616,  700,  784,  868,  952,  1036, 176,  427,  91,
    428,  511,  8,    512,  259,  253,  85,   505,  589,  595,  7,    260,
    596,  673,  1,    679,  680,  757,  337,  763,  175,  764,  841,  343,
    847,  92,   848,  925,  931,  344,  932,  1009, 169,  1015, 1093, 421,
    1099, 1177, 34,   118,  202,  286,  370,  454,  538,  30,   113,  114,
    197,  198,  281,  282,  365,  29,   366,  449,  450,  533,  534,  617,
    622,  701,  785,  869,  953,  1037, 177,  428,  92,   429,  512,  9,
    513,  260,  254,  86,   506,  590,  596,  8,    261,  597,  674,  2,
    680,  681,  758,  338,  764,  176,  765,  842,  344,  848,  93,   849,
    926,  932,  345,  933,  1010, 170,  1016, 1094, 422,  1100, 1178, 35,
    119,  203,  287,  371,  455,  539,  31,   114,  115,  198,  199,  282,
    283,  366,  30,   367,  450,  451,  534,  535,  618,  623,  702,  786,
    870,  954,  1038, 178,  429,  93,   430,  513,  10,   514,  261,  255,
    87,   507,  591,  597,  9,    262,  598,  675,  3,    681,  682,  759,
    339,  765,  177,  766,  843,  345,  849,  94,   850,  927,  933,  346,
    934,  1011, 171,  1017, 1095, 423,  1101, 1179, 36,   120,  204,  288,
    372,  456,  540,  32,   115,  116,  199,  200,  283,  284,  367,  31,
    368,  451,  452,  535,  536,  619,  624,  703,  787,  871,  955,  1039,
    179,  430,  94,   431,  514,  11,   515,  262,  256,  88,   508,  592,
    598,  10,   263,  599,  676,  4,    682,  683,  760,  340,  766,  178,
    767,  844,  346,  850,  95,   851,  928,  934,  347,  935,  1012, 172,
    1018, 1096, 424,  1102, 1180, 37,   121,  205,  289,  373,  457,  541,
    33,   116,  117,  200,  201,  284,  285,  368,  32,   369,  452,  453,
    536,  537,  620,  625,  704,  788,  872,  956,  1040, 180,  431,  95,
    432,  515,  12,   516,  263,  257,  89,   509,  593,  599,  11,   264,
    600,  677,  5,    683,  684,  761,  341,  767,  179,  768,  845,  347,
    851,  96,   852,  929,  935,  348,  936,  1013, 173,  1019, 1097, 425,
    1103, 1181, 38,   122,  206,  290,  374,  458,  542,  117,  201,  285,
    369,  33,   453,  537,  621,  626,  705,  789,  873,  957,  1041, 432,
    96,   516,  264,  258,  90,   510,  594,  600,  12,   678,  6,    684,
    762,  342,  768,  180,  846,  348,  852,  930,  936,  1014, 174,  1020,
    1098, 426,  1104, 1182, 34,   118,  202,  286,  196,  112,  28,   280,
    364,  370,  448,  454,  532,  538,  616,  700,  784,  868,  952,  1036,
    1120, 181,  433,  13,   517,  259,  91,   511,  595,  265,  601,  679,
    7,    685,  763,  343,  769,  847,  97,   853,  931,  349,  937,  1015,
    175,  1099, 427,  1183, 1204, 35,   118,  119,  202,  203,  286,  287,
    370,  197,  113,  29,   281,  365,  34,   371,  449,  454,  455,  533,
    538,  539,  617,  622,  701,  706,  785,  790,  869,  874,  953,  958,
    1037, 1042, 1121, 182,  433,  97,   434,  517,  14,   518,  265,  260,
    92,   512,  596,  601,  13,   266,  602,  680,  8,    685,  686,  764,
    344,  769,  181,  770,  848,  349,  853,  98,   854,  932,  937,  350,
    938,  1016, 176,  1021, 1100, 428,  1105, 1184, 1205, 39,   123,  207,
    291,  375,  459,  543,  36,   119,  120,  203,  204,  287,  288,  371,
    198,  114,  30,   282,  366,  35,   372,  450,  455,  456,  534,  539,
    540,  618,  623,  627,  702,  707,  786,  791,  870,  875,  954,  959,
    1038, 1043, 1122, 183,  434,  98,   435,  518,  15,   519,  266,  261,
    93,   513,  597,  602,  14,   267,  603,  681,  9,    686,  687,  765,
    345,  770,  182,  771,  849,  350,  854,  99,   855,  933,  938,  351,
    939,  1017, 177,  1022, 1101, 429,  1106, 1185, 1206, 40,   124,  208,
    292,  376,  460,  544,  37,   120,  121,  204,  205,  288,  289,  372,
    199,  115,  31,   283,  367,  36,   373,  451,  456,  457,  535,  540,
    541,  619,  624,  628,  703,  708,  787,  792,  871,  876,  955,  960,
    1039, 1044, 1123, 184,  435,  99,   436,  519,  16,   520,  267,  262,
    94,   514,  598,  603,  15,   268,  604,  682,  10,   687,  688,  766,
    346,  771,  183,  772,  850,  351,  855,  100,  856,  934,  939,  352,
    940,  1018, 178,  1023, 1102, 430,  1107, 1186, 1207, 41,   125,  209,
    293,  377,  461,  545,  38,   121,  122,  205,  206,  289,  290,  373,
    200,  116,  32,   284,  368,  37,   374,  452,  457,  458,  536,  541,
    542,  620,  625,  629,  704,  709,  788,  793,  872,  877,  956,  961,
    1040, 1045, 1124, 185,  436,  100,  437,  520,  17,   521,  268,  263,
    95,   515,  599,  604,  16,   269,  605,  683,  11,   688,  689,  767,
    347,  772,  184,  773,  851,  352,  856,  101,  857,  935,  940,  353,
    941,  1019, 179,  1024, 1103, 431,  1108, 1187, 1208, 42,   126,  210,
    294,  378,  462,  546,  122,  206,  290,  374,  201,  117,  33,   285,
    369,  38,   453,  458,  537,  542,  621,  626,  630,  705,  710,  789,
    794,  873,  878,  957,  962,  1041, 1046, 1125, 437,  101,  521,  269,
    264,  96,   516,  600,  605,  17,   684,  12,   689,  768,  348,  773,
    185,  852,  353,  857,  936,  941,  1020, 180,  1025, 1104, 432,  1109,
    1188, 1209, 39,   123,  207,  291,  202,  118,  34,   286,  370,  375,
    454,  459,  538,  543,  622,  706,  790,  874,  958,  1042, 1126, 186,
    438,  18,   522,  265,  97,   517,  601,  270,  606,  685,  13,   690,
    769,  349,  774,  853,  102,  858,  937,  354,  942,  1021, 181,  1105,
    433,  1189, 1210, 40,   123,  124,  207,  208,  291,  292,  375,  203,
    119,  35,   287,  371,  39,   376,  455,  459,  460,  539,  543,  544,
    623,  627,  707,  711,  791,  795,  875,  879,  959,  963,  1043, 1047,
    1127, 187,  438,  102,  439,  522,  19,   523,  270,  266,  98,   518,
    602,  606,  18,   271,  607,  686,  14,   690,  691,  770,  350,  774,
    186,  775,  854,  354,  858,  103,  859,  938,  942,  355,  943,  1022,
    182,  1026, 1106, 434,  1110, 1190, 1211, 43,   127,  211,  295,  379,
    463,  547,  41,   124,  125,  208,  209,  292,  293,  376,  204,  120,
    36,   288,  372,  40,   377,  456,  460,  461,  540,  544,  545,  624,
    628,  631,  708,  712,  792,  796,  876,  880,  960,  964,  1044, 1048,
    1128, 188,  439,  103,  440,  523,  20,   524,  271,  267,  99,   519,
    603,  607,  19,   272,  608,  687,  15,   691,  692,  771,  351,  775,
    187,  776,  855,  355,  859,  104,  860,  939,  943,  356,  944,  1023,
    183,  1027, 1107, 435,  1111, 1191, 1212, 44,   128,  212,  296,  380,
    464,  548,  42,   125,  126,  209,  210,  293,  294,  377,  205,  121,
    37,   289,  373,  41,   378,  457,  461,  462,  541,  545,  546,  625,
    629,  632,  709,  713,  793,  797,  877,  881,  961,  965,  1045, 1049,
    1129, 189,  440,  104,  441,  524,  21,   525,  272,  268,  100,  520,
    604,  608,  20,   273,  609,  688,  16,   692,  693,  772,  352,  776,
    188,  777,  856,  356,  860,  105,  861,  940,  944,  357,  945,  1024,
    184,  1028, 1108, 436,  1112, 1192, 1213, 45,   129,  213,  297,  381,
    465,  549,  126,  210,  294,  378,  206,  122,  38,   290,  374,  42,
    458,  462,  542,  546,  626,  630,  633,  710,  714,  794,  798,  878,
    882,  962,  966,  1046, 1050, 1130, 441,  105,  525,  273,  269,  101,
    521,  605,  609,  21,   689,  17,   693,  773,  353,  777,  189,  857,
    357,  861,  941,  945,  1025, 185,  1029, 1109, 437,  1113, 1193, 1214,
    43,   127,  211,  295,  207,  123,  39,   291,  375,  379,  459,  463,
    543,  547,  627,  711,  795,  879,  963,  1047, 1131, 190,  442,  22,
    526,  270,  102,  522,  606,  274,  610,  690,  18,   694,  774,  354,
    778,  858,  106,  862,  942,  358,  946,  1026, 186,  1110, 438,  1194,
    1215, 44,   127,  128,  211,  212,  295,  296,  379,  208,  124,  40,
    292,  376,  43,   380,  460,  463,  464,  544,  547,  548,  628,  631,
    712,  715,  796,  799,  880,  883,  964,  967,  1048, 1051, 1132, 191,
    442,  106,  443,  526,  23,   527,  274,  271,  103,  523,  607,  610,
    22,   275,  611,  691,  19,   694,  695,  775,  355,  778,  190,  779,
    859,  358,  862,  107,  863,  943,  946,  359,  947,  1027, 187,  1030,
    1111, 439,  1114, 1195, 1216, 46,   130,  214,  298,  382,  466,  550,
    45,   128,  129,  212,  213,  296,  297,  380,  209,  125,  41,   293,
    377,  44,   381,  461,  464,  465,  545,  548,  549,  629,  632,  634,
    713,  716,  797,  800,  881,  884,  965,  968,  1049, 1052, 1133, 192,
    443,  107,  444,  527,  24,   528,  275,  272,  104,  524,  608,  611,
    23,   276,  612,  692,  20,   695,  696,  776,  356,  779,  191,  780,
    860,  359,  863,  108,  864,  944,  947,  360,  948,  1028, 188,  1031,
    1112, 440,  1115, 1196, 1217, 196,  448,  28,   532,  280,  616,  700,
    784,  112,  672,  84,   504,  0,    840,  336,  588,  756,  252,  868,
    924,  168,  364,  952,  1008, 420,  1092, 1176, 197,  448,  112,  449,
    532,  29,   533,  280,  616,  28,   281,  617,  700,  701,  784,  196,
    785,  364,  868,  113,  673,  85,   505,  1,    841,  337,  589,  757,
    253,  869,  925,  169,  952,  365,  953,  1009, 1036, 421,  1093, 1120,
    1177, 49,   133,  217,  301,  385,  469,  553,  637,  198,  449,  113,
    450,  533,  30,   534,  281,  617,  29,   282,  618,  701,  702,  785,
    197,  786,  365,  869,  114,  674,  86,   506,  2,    842,  338,  590,
    758,  254,  870,  926,  170,  953,  366,  954,  1010, 1037, 422,  1094,
    1121, 1178, 50,   134,  218,  302,  386,  470,  554,  638,  199,  450,
    114,  451,  534,  31,   535,  282,  618,  30,   283,  619,  702,  703,
    786,  198,  787,  366,  870,  115,  675,  87,   507,  3,    843,  339,
    591,  759,  255,  871,  927,  171,  954,  367,  955,  1011, 1038, 423,
    1095, 1122, 1179, 51,   135,  219,  303,  387,  471,  555,  639,  200,
    451,  115,  452,  535,  32,   536,  283,  619,  31,   284,  620,  703,
    704,  787,  199,  788,  367,  871,  116,  676,  88,   508,  4,    844,
    340,  592,  760,  256,  872,  928,  172,  955,  368,  956,  1012, 1039,
    424,  1096, 1123, 1180, 52,   136,  220,  304,  388,  472,  556,  640,
    201,  452,  116,  453,  536,  33,   537,  284,  620,  32,   285,  621,
    704,  705,  788,  200,  789,  368,  872,  117,  677,  89,   509,  5,
    845,  341,  593,  761,  257,  873,  929,  173,  956,  369,  957,  1013,
    1040, 425,  1097, 1124, 1181, 53,   137,  221,  305,  389,  473,  557,
    641,  453,  117,  537,  285,  621,  33,   705,  789,  201,  369,  873,
    678,  90,   510,  6,    846,  342,  594,  762,  258,  930,  174,  957,
    1014, 1041, 426,  1098, 1125, 1182, 49,   133,  217,  301,  385,  469,
    553,  202,  454,  34,   538,  280,  112,  532,  616,  286,  622,  700,
    28,   706,  784,  364,  790,  868,  118,  679,  91,   511,  7,    847,
    343,  595,  763,  259,  874,  931,  952,  175,  370,  958,  1015, 1036,
    196,  427,  1099, 1120, 448,  1183, 1204, 50,   133,  134,  217,  218,
    301,  302,  385,  49,   386,  469,  470,  553,  554,  637,  721,  805,
    889,  973,  1057, 203,  454,  118,  455,  538,  35,   539,  286,  281,
    113,  533,  617,  622,  34,   287,  623,  701,  29,   706,  707,  785,
    365,  790,  202,  791,  869,  370,  874,  119,  680,  92,   512,  8,
    848,  344,  596,  764,  260,  875,  932,  953,  176,  958,  371,  959,
    1016, 1037, 197,  1042, 428,  1100, 1121, 449,  1126, 1184, 1205, 54,
    138,  222,  306,  217,  133,  49,   301,  385,  390,  469,  474,  553,
    558,  637,  721,  805,  889,  973,  1057, 1141, 207,  459,  39,   543,
    286,  118,  538,  622,  291,  627,  706,  34,   711,  790,  370,  795,
    874,  123,  685,  97,   517,  13,   853,  349,  601,  769,  265,  879,
    937,  958,  181,  375,  963,  1021, 1042, 202,  433,  1105, 1126, 454,
    1189, 1210, 1225, 54,   138,  222,  306,  390,  474,  558,  51,   134,
    135,  218,  219,  302,  303,  386,  50,   387,  470,  471,  554,  555,
    638,  642,  722,  806,  890,  974,  1058, 204,  455,  119,  456,  539,
    36,   540,  287,  282,  114,  534,  618,  623,  35,   288,  624,  702,
    30,   707,  708,  786,  366,  791,  203,  792,  870,  371,  875,  120,
    681,  93,   513,  9,    849,  345,  597,  765,  261,  876,  933,  954,
    177,  959,  372,  960,  1017, 1038, 198,  1043, 429,  1101, 1122, 450,
    1127, 1185, 1206, 55,   138,  139,  222,  223,  306,  307,  390,  218,
    134,  50,   302,  386,  54,   391,  470,  474,  475,  554,  558,  559,
    638,  642,  722,  726,  806,  810,  890,  894,  974,  978,  1058, 1062,
    1142, 208,  459,  123,  460,  543,  40,   544,  291,  287,  119,  539,
    623,  627,  39,   292,  628,  707,  35,   711,  712,  791,  371,  795,
    207,  796,  875,  375,  879,  124,  686,  98,   518,  14,   854,  350,
    602,  770,  266,  880,  938,  959,  182,  963,  376,  964,  1022, 1043,
    203,  1047, 434,  1106, 1127, 455,  1131, 1190, 1211, 1226, 58,   142,
    226,  310,  222,  138,  54,   306,  390,  394,  474,  478,  558,  562,
    642,  726,  810,  894,  978,  1062, 1146, 211,  463,  43,   547,  291,
    123,  543,  627,  295,  631,  711,  39,   715,  795,  375,  799,  879,
    127,  690,  102,  522,  18,   858,  354,  606,  774,  270,  883,  942,
    963,  186,  379,  967,  1026, 1047, 207,  438,  1110, 1131, 459,  1194,
    1215, 1230, 55,   139,  223,  307,  391,  475,  559,  52,   135,  136,
    219,  220,  303,  304,  387,  51,   388,  471,  472,  555,  556,  639,
    643,  723,  807,  891,  975,  1059, 205,  456,  120,  457,  540,  37,
    541,  288,  283,  115,  535,  619,  624,  36,   289,  625,  703,  31,
    708,  709,  787,  367,  792,  204,  793,  871,  372,  876,  121,  682,
    94,   514,  10,   850,  346,  598,  766,  262,  877,  934,  955,  178,
    960,  373,  961,  1018, 1039, 199,  1044, 430,  1102, 1123, 451,  1128,
    1186, 1207, 58,   142,  226,  310,  394,  478,  562,  56,   139,  140,
    223,  224,  307,  308,  391,  219,  135,  51,   303,  387,  55,   392,
    471,  475,  476,  555,  559,  560,  639,  643,  646,  723,  727,  807,
    811,  891,  895,  975,  979,  1059, 1063, 1143, 209,  460,  124,  461,
    544,  41,   545,  292,  288,  120,  540,  624,  628,  40,   293,  629,
    708,  36,   712,  713,  792,  372,  796,  208,  797,  876,  376,  880,
    125,  687,  99,   519,  15,   855,  351,  603,  771,  267,  881,  939,
    960,  183,  964,  377,  965,  1023, 1044, 204,  1048, 435,  1107, 1128,
    456,  1132, 1191, 1212, 1227, 59,   142,  143,  226,  227,  310,  311,
    394,  223,  139,  55,   307,  391,  58,   395,  475,  478,  479,  559,
    562,  563,  643,  646,  727,  730,  811,  814,  895,  898,  979,  982,
    1063, 1066, 1147, 212,  463,  127,  464,  547,  44,   548,  295,  292,
    124,  544,  628,  631,  43,   296,  632,  712,  40,   715,  716,  796,
    376,  799,  211,  800,  880,  379,  883,  128,  691,  103,  523,  19,
    859,  355,  607,  775,  271,  884,  943,  964,  187,  967,  380,  968,
    1027, 1048, 208,  1051, 439,  1111, 1132, 460,  1135, 1195, 1216, 1231,
    46,   130,  214,  298,  211,  127,  43,   295,  379,  382,  463,  466,
    547,  550,  631,  715,  799,  883,  967,  1051, 1135, 193,  445,  25,
    529,  274,  106,  526,  610,  277,  613,  694,  22,   697,  778,  358,
    781,  862,  109,  865,  946,  361,  949,  1030, 190,  1114, 442,  1198,
    1219, 56,   140,  224,  308,  392,  476,  560,  53,   136,  137,  220,
    221,  304,  305,  388,  52,   389,  472,  473,  556,  557,  640,  644,
    724,  808,  892,  976,  1060, 206,  457,  121,  458,  541,  38,   542,
    289,  284,  116,  536,  620,  625,  37,   290,  626,  704,  32,   709,
    710,  788,  368,  793,  205,  794,  872,  373,  877,  122,  683,  95,
    515,  11,   851,  347,  599,  767,  263,  878,  935,  956,  179,  961,
    374,  962,  1019, 1040, 200,  1045, 431,  1103, 1124, 452,  1129, 1187,
    1208, 59,   143,  227,  311,  395,  479,  563,  57,   140,  141,  224,
    225,  308,  309,  392,  220,  136,  52,   304,  388,  56,   393,  472,
    476,  477,  556,  560,  561,  640,  644,  647,  724,  728,  808,  812,
    892,  896,  976,  980,  1060, 1064, 1144, 210,  461,  125,  462,  545,
    42,   546,  293,  289,  121,  541,  625,  629,  41,   294,  630,  709,
    37,   713,  714,  793,  373,  797,  209,  798,  877,  377,  881,  126,
    688,  100,  520,  16,   856,  352,  604,  772,  268,  882,  940,  961,
    184,  965,  378,  966,  1024, 1045, 205,  1049, 436,  1108, 1129, 457,
    1133, 1192, 1213, 1228, 61,   145,  229,  313,  397,  481,  565,  60,
    143,  144,  227,  228,  311,  312,  395,  224,  140,  56,   308,  392,
    59,   396,  476,  479,  480,  560,  563,  564,  644,  647,  649,  728,
    731,  812,  815,  896,  899,  980,  983,  1064, 1067, 1148, 213,  464,
    128,  465,  548,  45,   549,  296,  293,  125,  545,  629,  632,  44,
    297,  633,  713,  41,   716,  717,  797,  377,  800,  212,  801,  881,
    380,  884,  129,  692,  104,  524,  20,   860,  356,  608,  776,  272,
    885,  944,  965,  188,  968,  381,  969,  1028, 1049, 209,  1052, 440,
    1112, 1133, 461,  1136, 1196, 1217, 1232, 47,   130,  131,  214,  215,
    298,  299,  382,  212,  128,  44,   296,  380,  46,   383,  464,  466,
    467,  548,  550,  551,  632,  634,  716,  718,  800,  802,  884,  886,
    968,  970,  1052, 1054, 1136, 194,  445,  109,  446,  529,  26,   530,
    277,  275,  107,  527,  611,  613,  25,   278,  614,  695,  23,   697,
    698,  779,  359,  781,  193,  782,  863,  361,  865,  110,  866,  947,
    949,  362,  950,  1031, 191,  1033, 1115, 443,  1117, 1199, 1220, 48,
    132,  216,  300,  214,  130,  46,   298,  382,  384,  466,  468,  550,
    552,  634,  718,  802,  886,  970,  1054, 1138, 195,  447,  27,   531,
    277,  109,  529,  613,  279,  615,  697,  25,   699,  781,  361,  783,
    865,  111,  867,  949,  363,  951,  1033, 193,  1117, 445,  1201, 1222,
    57,   141,  225,  309,  393,  477,  561,  137,  221,  305,  389,  53,
    473,  557,  641,  645,  725,  809,  893,  977,  1061, 458,  122,  542,
    290,  285,  117,  537,  621,  626,  38,   705,  33,   710,  789,  369,
    794,  206,  873,  374,  878,  684,  96,   516,  12,   852,  348,  600,
    768,  264,  936,  957,  180,  962,  1020, 1041, 201,  1046, 432,  1104,
    1125, 453,  1130, 1188, 1209, 60,   144,  228,  312,  396,  480,  564,
    141,  225,  309,  393,  221,  137,  53,   305,  389,  57,   473,  477,
    557,  561,  641,  645,  648,  725,  729,  809,  813,  893,  897,  977,
    981,  1061, 1065, 1145, 462,  126,  546,  294,  290,  122,  542,  626,
    630,  42,   710,  38,   714,  794,  374,  798,  210,  878,  378,  882,
    689,  101,  521,  17,   857,  353,  605,  773,  269,  941,  962,  185,
    966,  1025, 1046, 206,  1050, 437,  1109, 1130, 458,  1134, 1193, 1214,
    1229, 47,   131,  215,  299,  383,  467,  551,  129,  213,  297,  381,
    210,  126,  42,   294,  378,  45,   462,  465,  546,  549,  630,  633,
    635,  714,  717,  798,  801,  882,  885,  966,  969,  1050, 1053, 1134,
    444,  108,  528,  276,  273,  105,  525,  609,  612,  24,   693,  21,
    696,  777,  357,  780,  192,  861,  360,  864,  945,  948,  1029, 189,
    1032, 1113, 441,  1116, 1197, 1218, 48,   132,  216,  300,  384,  468,
    552,  131,  215,  299,  383,  213,  129,  45,   297,  381,  47,   465,
    467,  549,  551,  633,  635,  636,  717,  719,  801,  803,  885,  887,
    969,  971,  1053, 1055, 1137, 446,  110,  530,  278,  276,  108,  528,
    612,  614,  26,   696,  24,   698,  780,  360,  782,  194,  864,  362,
    866,  948,  950,  1032, 192,  1034, 1116, 444,  1118, 1200, 1221, 132,
    216,  300,  384,  215,  131,  47,   299,  383,  48,   467,  468,  551,
    552,  635,  636,  719,  720,  803,  804,  887,  888,  971,  972,  1055,
    1056, 1139, 447,  111,  531,  279,  278,  110,  530,  614,  615,  27,
    698,  26,   699,  782,  362,  783,  195,  866,  363,  867,  950,  951,
    1034, 194,  1035, 1118, 446,  1119, 1202, 1223, 195,  111,  27,   279,
    363,  447,  531,  615,  699,  783,  867,  951,  1035, 1119, 1203};
static const int coeffs1_ind[] = {
    755,  167,  587,  83,   923,  419,  671,  839,  335,  1007, 251,  1091,
    503,  1175, 1259, 251,  503,  83,   587,  335,  671,  755,  839,  167,
    752,  164,  584,  80,   920,  416,  668,  836,  332,  923,  1004, 248,
    419,  1007, 1088, 500,  1172, 1256, 248,  500,  80,   584,  332,  668,
    752,  836,  164,  746,  158,  578,  74,   914,  410,  662,  830,  326,
    920,  998,  242,  416,  1004, 1082, 494,  1166, 1250, 242,  494,  74,
    578,  326,  662,  746,  830,  158,  736,  148,  568,  64,   904,  400,
    652,  820,  316,  914,  988,  232,  410,  998,  1072, 484,  1156, 1240,
    232,  484,  64,   568,  316,  652,  736,  820,  148,  721,  133,  553,
    49,   889,  385,  637,  805,  301,  904,  973,  217,  400,  988,  1057,
    469,  1141, 1225, 217,  469,  49,   553,  301,  637,  721,  805,  133,
    700,  112,  532,  28,   868,  364,  616,  784,  280,  889,  952,  196,
    385,  973,  1036, 448,  1120, 1204, 218,  469,  133,  470,  553,  50,
    554,  301,  637,  49,   302,  638,  721,  722,  805,  217,  806,  385,
    889,  134,  701,  113,  533,  29,   869,  365,  617,  785,  281,  890,
    953,  197,  973,  386,  974,  1037, 1057, 449,  1121, 1141, 1205, 64,
    148,  232,  316,  400,  484,  568,  222,  474,  54,   558,  301,  133,
    553,  637,  306,  642,  721,  49,   726,  805,  385,  810,  889,  138,
    706,  118,  538,  34,   874,  370,  622,  790,  286,  894,  958,  973,
    202,  390,  978,  1042, 1057, 217,  454,  1126, 1141, 469,  1210, 1225,
    233,  484,  148,  485,  568,  65,   569,  316,  652,  64,   317,  653,
    736,  737,  820,  232,  821,  400,  904,  149,  722,  134,  554,  50,
    890,  386,  638,  806,  302,  905,  974,  218,  988,  401,  989,  1058,
    1072, 470,  1142, 1156, 1226, 64,   148,  232,  316,  400,  484,  568,
    652,  219,  470,  134,  471,  554,  51,   555,  302,  638,  50,   303,
    639,  722,  723,  806,  218,  807,  386,  890,  135,  702,  114,  534,
    30,   870,  366,  618,  786,  282,  891,  954,  198,  974,  387,  975,
    1038, 1058, 450,  1122, 1142, 1206, 65,   148,  149,  232,  233,  316,
    317,  400,  64,   401,  484,  485,  568,  569,  652,  736,  820,  904,
    988,  1072, 223,  474,  138,  475,  558,  55,   559,  306,  302,  134,
    554,  638,  642,  54,   307,  643,  722,  50,   726,  727,  806,  386,
    810,  222,  811,  890,  390,  894,  139,  707,  119,  539,  35,   875,
    371,  623,  791,  287,  895,  959,  974,  203,  978,  391,  979,  1043,
    1058, 218,  1062, 455,  1127, 1142, 470,  1146, 1211, 1226, 74,   158,
    242,  326,  410,  494,  578,  236,  488,  68,   572,  316,  148,  568,
    652,  320,  656,  736,  64,   740,  820,  400,  824,  904,  152,  726,
    138,  558,  54,   894,  390,  642,  810,  306,  908,  978,  988,  222,
    404,  992,  1062, 1072, 232,  474,  1146, 1156, 484,  1230, 1240, 68,
    152,  236,  320,  232,  148,  64,   316,  400,  404,  484,  488,  568,
    572,  652,  736,  820,  904,  988,  1072, 1156, 226,  478,  58,   562,
    306,  138,  558,  642,  310,  646,  726,  54,   730,  810,  390,  814,
    894,  142,  711,  123,  543,  39,   879,  375,  627,  795,  291,  898,
    963,  978,  207,  394,  982,  1047, 1062, 222,  459,  1131, 1146, 474,
    1215, 1230, 1240, 243,  494,  158,  495,  578,  75,   579,  326,  662,
    74,   327,  663,  746,  747,  830,  242,  831,  410,  914,  159,  737,
    149,  569,  65,   905,  401,  653,  821,  317,  915,  989,  233,  998,
    411,  999,  1073, 1082, 485,  1157, 1166, 1241, 74,   158,  242,  326,
    410,  494,  578,  662,  234,  485,  149,  486,  569,  66,   570,  317,
    653,  65,   318,  654,  737,  738,  821,  233,  822,  401,  905,  150,
    723,  135,  555,  51,   891,  387,  639,  807,  303,  906,  975,  219,
    989,  402,  990,  1059, 1073, 471,  1143, 1157, 1227, 65,   149,  233,
    317,  401,  485,  569,  653,  220,  471,  135,  472,  555,  52,   556,
    303,  639,  51,   304,  640,  723,  724,  807,  219,  808,  387,  891,
    136,  703,  115,  535,  31,   871,  367,  619,  787,  283,  892,  955,
    199,  975,  388,  976,  1039, 1059, 451,  1123, 1143, 1207, 68,   152,
    236,  320,  404,  488,  572,  66,   149,  150,  233,  234,  317,  318,
    401,  65,   402,  485,  486,  569,  570,  653,  656,  737,  821,  905,
    989,  1073, 224,  475,  139,  476,  559,  56,   560,  307,  303,  135,
    555,  639,  643,  55,   308,  644,  723,  51,   727,  728,  807,  387,
    811,  223,  812,  891,  391,  895,  140,  708,  120,  540,  36,   876,
    372,  624,  792,  288,  896,  960,  975,  204,  979,  392,  980,  1044,
    1059, 219,  1063, 456,  1128, 1143, 471,  1147, 1212, 1227, 75,   158,
    159,  242,  243,  326,  327,  410,  74,   411,  494,  495,  578,  579,
    662,  746,  830,  914,  998,  1082, 237,  488,  152,  489,  572,  69,
    573,  320,  317,  149,  569,  653,  656,  68,   321,  657,  737,  65,
    740,  741,  821,  401,  824,  236,  825,  905,  404,  908,  153,  727,
    139,  559,  55,   895,  391,  643,  811,  307,  909,  979,  989,  223,
    992,  405,  993,  1063, 1073, 233,  1076, 475,  1147, 1157, 485,  1160,
    1231, 1241, 69,   152,  153,  236,  237,  320,  321,  404,  233,  149,
    65,   317,  401,  68,   405,  485,  488,  489,  569,  572,  573,  653,
    656,  737,  740,  821,  824,  905,  908,  989,  992,  1073, 1076, 1157,
    227,  478,  142,  479,  562,  59,   563,  310,  307,  139,  559,  643,
    646,  58,   311,  647,  727,  55,   730,  731,  811,  391,  814,  226,
    815,  895,  394,  898,  143,  712,  124,  544,  40,   880,  376,  628,
    796,  292,  899,  964,  979,  208,  982,  395,  983,  1048, 1063, 223,
    1066, 460,  1132, 1147, 475,  1150, 1216, 1231, 1241, 80,   164,  248,
    332,  416,  500,  584,  245,  497,  77,   581,  326,  158,  578,  662,
    329,  665,  746,  74,   749,  830,  410,  833,  914,  161,  740,  152,
    572,  68,   908,  404,  656,  824,  320,  917,  992,  998,  236,  413,
    1001, 1076, 1082, 242,  488,  1160, 1166, 494,  1244, 1250, 77,   161,
    245,  329,  242,  158,  74,   326,  410,  413,  494,  497,  578,  581,
    662,  746,  830,  914,  998,  1082, 1166, 239,  491,  71,   575,  320,
    152,  572,  656,  323,  659,  740,  68,   743,  824,  404,  827,  908,
    155,  730,  142,  562,  58,   898,  394,  646,  814,  310,  911,  982,
    992,  226,  407,  995,  1066, 1076, 236,  478,  1150, 1160, 488,  1234,
    1244, 1250, 71,   155,  239,  323,  236,  152,  68,   320,  404,  407,
    488,  491,  572,  575,  656,  740,  824,  908,  992,  1076, 1160, 229,
    481,  61,   565,  310,  142,  562,  646,  313,  649,  730,  58,   733,
    814,  394,  817,  898,  145,  715,  127,  547,  43,   883,  379,  631,
    799,  295,  901,  967,  982,  211,  397,  985,  1051, 1066, 226,  463,
    1135, 1150, 478,  1219, 1234, 1244, 61,   145,  229,  313,  226,  142,
    58,   310,  394,  397,  478,  481,  562,  565,  646,  730,  814,  898,
    982,  1066, 1150, 214,  466,  46,   550,  295,  127,  547,  631,  298,
    634,  715,  43,   718,  799,  379,  802,  883,  130,  694,  106,  526,
    22,   862,  358,  610,  778,  274,  886,  946,  967,  190,  382,  970,
    1030, 1051, 211,  442,  1114, 1135, 463,  1198, 1219, 1234, 249,  500,
    164,  501,  584,  81,   585,  332,  668,  80,   333,  669,  752,  753,
    836,  248,  837,  416,  920,  165,  747,  159,  579,  75,   915,  411,
    663,  831,  327,  921,  999,  243,  1004, 417,  1005, 1083, 1088, 495,
    1167, 1172, 1251, 80,   164,  248,  332,  416,  500,  584,  668,  244,
    495,  159,  496,  579,  76,   580,  327,  663,  75,   328,  664,  747,
    748,  831,  243,  832,  411,  915,  160,  738,  150,  570,  66,   906,
    402,  654,  822,  318,  916,  990,  234,  999,  412,  1000, 1074, 1083,
    486,  1158, 1167, 1242, 75,   159,  243,  327,  411,  495,  579,  663,
    235,  486,  150,  487,  570,  67,   571,  318,  654,  66,   319,  655,
    738,  739,  822,  234,  823,  402,  906,  151,  724,  136,  556,  52,
    892,  388,  640,  808,  304,  907,  976,  220,  990,  403,  991,  1060,
    1074, 472,  1144, 1158, 1228, 66,   150,  234,  318,  402,  486,  570,
    654,  221,  472,  136,  473,  556,  53,   557,  304,  640,  52,   305,
    641,  724,  725,  808,  220,  809,  388,  892,  137,  704,  116,  536,
    32,   872,  368,  620,  788,  284,  893,  956,  200,  976,  389,  977,
    1040, 1060, 452,  1124, 1144, 1208, 69,   153,  237,  321,  405,  489,
    573,  67,   150,  151,  234,  235,  318,  319,  402,  66,   403,  486,
    487,  570,  571,  654,  657,  738,  822,  906,  990,  1074, 225,  476,
    140,  477,  560,  57,   561,  308,  304,  136,  556,  640,  644,  56,
    309,  645,  724,  52,   728,  729,  808,  388,  812,  224,  813,  892,
    392,  896,  141,  709,  121,  541,  37,   877,  373,  625,  793,  289,
    897,  961,  976,  205,  980,  393,  981,  1045, 1060, 220,  1064, 457,
    1129, 1144, 472,  1148, 1213, 1228, 77,   161,  245,  329,  413,  497,
    581,  76,   159,  160,  243,  244,  327,  328,  411,  75,   412,  495,
    496,  579,  580,  663,  665,  747,  831,  915,  999,  1083, 238,  489,
    153,  490,  573,  70,   574,  321,  318,  150,  570,  654,  657,  69,
    322,  658,  738,  66,   741,  742,  822,  402,  825,  237,  826,  906,
    405,  909,  154,  728,  140,  560,  56,   896,  392,  644,  812,  308,
    910,  980,  990,  224,  993,  406,  994,  1064, 1074, 234,  1077, 476,
    1148, 1158, 486,  1161, 1232, 1242, 71,   155,  239,  323,  407,  491,
    575,  70,   153,  154,  237,  238,  321,  322,  405,  234,  150,  66,
    318,  402,  69,   406,  486,  489,  490,  570,  573,  574,  654,  657,
    659,  738,  741,  822,  825,  906,  909,  990,  993,  1074, 1077, 1158,
    228,  479,  143,  480,  563,  60,   564,  311,  308,  140,  560,  644,
    647,  59,   312,  648,  728,  56,   731,  732,  812,  392,  815,  227,
    816,  896,  395,  899,  144,  713,  125,  545,  41,   881,  377,  629,
    797,  293,  900,  965,  980,  209,  983,  396,  984,  1049, 1064, 224,
    1067, 461,  1133, 1148, 476,  1151, 1217, 1232, 1242, 81,   164,  165,
    248,  249,  332,  333,  416,  80,   417,  500,  501,  584,  585,  668,
    752,  836,  920,  1004, 1088, 246,  497,  161,  498,  581,  78,   582,
    329,  327,  159,  579,  663,  665,  77,   330,  666,  747,  75,   749,
    750,  831,  411,  833,  245,  834,  915,  413,  917,  162,  741,  153,
    573,  69,   909,  405,  657,  825,  321,  918,  993,  999,  237,  1001,
    414,  1002, 1077, 1083, 243,  1085, 489,  1161, 1167, 495,  1169, 1245,
    1251, 78,   161,  162,  245,  246,  329,  330,  413,  243,  159,  75,
    327,  411,  77,   414,  495,  497,  498,  579,  581,  582,  663,  665,
    747,  749,  831,  833,  915,  917,  999,  1001, 1083, 1085, 1167, 240,
    491,  155,  492,  575,  72,   576,  323,  321,  153,  573,  657,  659,
    71,   324,  660,  741,  69,   743,  744,  825,  405,  827,  239,  828,
    909,  407,  911,  156,  731,  143,  563,  59,   899,  395,  647,  815,
    311,  912,  983,  993,  227,  995,  408,  996,  1067, 1077, 237,  1079,
    479,  1151, 1161, 489,  1163, 1235, 1245, 1251, 72,   155,  156,  239,
    240,  323,  324,  407,  237,  153,  69,   321,  405,  71,   408,  489,
    491,  492,  573,  575,  576,  657,  659,  741,  743,  825,  827,  909,
    911,  993,  995,  1077, 1079, 1161, 230,  481,  145,  482,  565,  62,
    566,  313,  311,  143,  563,  647,  649,  61,   314,  650,  731,  59,
    733,  734,  815,  395,  817,  229,  818,  899,  397,  901,  146,  716,
    128,  548,  44,   884,  380,  632,  800,  296,  902,  968,  983,  212,
    985,  398,  986,  1052, 1067, 227,  1069, 464,  1136, 1151, 479,  1153,
    1220, 1235, 1245, 62,   145,  146,  229,  230,  313,  314,  397,  227,
    143,  59,   311,  395,  61,   398,  479,  481,  482,  563,  565,  566,
    647,  649,  731,  733,  815,  817,  899,  901,  983,  985,  1067, 1069,
    1151, 215,  466,  130,  467,  550,  47,   551,  298,  296,  128,  548,
    632,  634,  46,   299,  635,  716,  44,   718,  719,  800,  380,  802,
    214,  803,  884,  382,  886,  131,  695,  107,  527,  23,   863,  359,
    611,  779,  275,  887,  947,  968,  191,  970,  383,  971,  1031, 1052,
    212,  1054, 443,  1115, 1136, 464,  1138, 1199, 1220, 1235, 83,   167,
    251,  335,  419,  503,  587,  250,  502,  82,   586,  332,  164,  584,
    668,  334,  670,  752,  80,   754,  836,  416,  838,  920,  166,  749,
    161,  581,  77,   917,  413,  665,  833,  329,  922,  1001, 1004, 245,
    418,  1006, 1085, 1088, 248,  497,  1169, 1172, 500,  1253, 1256, 82,
    166,  250,  334,  248,  164,  80,   332,  416,  418,  500,  502,  584,
    586,  668,  752,  836,  920,  1004, 1088, 1172, 247,  499,  79,   583,
    329,  161,  581,  665,  331,  667,  749,  77,   751,  833,  413,  835,
    917,  163,  743,  155,  575,  71,   911,  407,  659,  827,  323,  919,
    995,  1001, 239,  415,  1003, 1079, 1085, 245,  491,  1163, 1169, 497,
    1247, 1253, 1256, 79,   163,  247,  331,  245,  161,  77,   329,  413,
    415,  497,  499,  581,  583,  665,  749,  833,  917,  1001, 1085, 1169,
    241,  493,  73,   577,  323,  155,  575,  659,  325,  661,  743,  71,
    745,  827,  407,  829,  911,  157,  733,  145,  565,  61,   901,  397,
    649,  817,  313,  913,  985,  995,  229,  409,  997,  1069, 1079, 239,
    481,  1153, 1163, 491,  1237, 1247, 1253, 73,   157,  241,  325,  239,
    155,  71,   323,  407,  409,  491,  493,  575,  577,  659,  743,  827,
    911,  995,  1079, 1163, 231,  483,  63,   567,  313,  145,  565,  649,
    315,  651,  733,  61,   735,  817,  397,  819,  901,  147,  718,  130,
    550,  46,   886,  382,  634,  802,  298,  903,  970,  985,  214,  399,
    987,  1054, 1069, 229,  466,  1138, 1153, 481,  1222, 1237, 1247, 63,
    147,  231,  315,  229,  145,  61,   313,  397,  399,  481,  483,  565,
    567,  649,  733,  817,  901,  985,  1069, 1153, 216,  468,  48,   552,
    298,  130,  550,  634,  300,  636,  718,  46,   720,  802,  382,  804,
    886,  132,  697,  109,  529,  25,   865,  361,  613,  781,  277,  888,
    949,  970,  193,  384,  972,  1033, 1054, 214,  445,  1117, 1138, 466,
    1201, 1222, 1237, 503,  167,  587,  335,  671,  83,   755,  839,  251,
    419,  923,  753,  165,  585,  81,   921,  417,  669,  837,  333,  1005,
    249,  1007, 1089, 1091, 501,  1173, 1175, 1257, 83,   167,  251,  335,
    419,  503,  587,  671,  501,  165,  585,  333,  669,  81,   753,  837,
    249,  417,  921,  748,  160,  580,  76,   916,  412,  664,  832,  328,
    1000, 244,  1005, 1084, 1089, 496,  1168, 1173, 1252, 81,   165,  249,
    333,  417,  501,  585,  669,  496,  160,  580,  328,  664,  76,   748,
    832,  244,  412,  916,  739,  151,  571,  67,   907,  403,  655,  823,
    319,  991,  235,  1000, 1075, 1084, 487,  1159, 1168, 1243, 76,   160,
    244,  328,  412,  496,  580,  664,  487,  151,  571,  319,  655,  67,
    739,  823,  235,  403,  907,  725,  137,  557,  53,   893,  389,  641,
    809,  305,  977,  221,  991,  1061, 1075, 473,  1145, 1159, 1229, 67,
    151,  235,  319,  403,  487,  571,  655,  473,  137,  557,  305,  641,
    53,   725,  809,  221,  389,  893,  705,  117,  537,  33,   873,  369,
    621,  789,  285,  957,  201,  977,  1041, 1061, 453,  1125, 1145, 1209,
    70,   154,  238,  322,  406,  490,  574,  151,  235,  319,  403,  67,
    487,  571,  655,  658,  739,  823,  907,  991,  1075, 477,  141,  561,
    309,  305,  137,  557,  641,  645,  57,   725,  53,   729,  809,  389,
    813,  225,  893,  393,  897,  710,  122,  542,  38,   878,  374,  626,
    794,  290,  962,  977,  206,  981,  1046, 1061, 221,  1065, 458,  1130,
    1145, 473,  1149, 1214, 1229, 78,   162,  246,  330,  414,  498,  582,
    160,  244,  328,  412,  76,   496,  580,  664,  666,  748,  832,  916,
    1000, 1084, 490,  154,  574,  322,  319,  151,  571,  655,  658,  70,
    739,  67,   742,  823,  403,  826,  238,  907,  406,  910,  729,  141,
    561,  57,   897,  393,  645,  813,  309,  981,  991,  225,  994,  1065,
    1075, 235,  1078, 477,  1149, 1159, 487,  1162, 1233, 1243, 72,   156,
    240,  324,  408,  492,  576,  154,  238,  322,  406,  235,  151,  67,
    319,  403,  70,   487,  490,  571,  574,  655,  658,  660,  739,  742,
    823,  826,  907,  910,  991,  994,  1075, 1078, 1159, 480,  144,  564,
    312,  309,  141,  561,  645,  648,  60,   729,  57,   732,  813,  393,
    816,  228,  897,  396,  900,  714,  126,  546,  42,   882,  378,  630,
    798,  294,  966,  981,  210,  984,  1050, 1065, 225,  1068, 462,  1134,
    1149, 477,  1152, 1218, 1233, 1243, 62,   146,  230,  314,  398,  482,
    566,  144,  228,  312,  396,  225,  141,  57,   309,  393,  60,   477,
    480,  561,  564,  645,  648,  650,  729,  732,  813,  816,  897,  900,
    981,  984,  1065, 1068, 1149, 465,  129,  549,  297,  294,  126,  546,
    630,  633,  45,   714,  42,   717,  798,  378,  801,  213,  882,  381,
    885,  693,  105,  525,  21,   861,  357,  609,  777,  273,  945,  966,
    189,  969,  1029, 1050, 210,  1053, 441,  1113, 1134, 462,  1137, 1197,
    1218, 1233, 82,   166,  250,  334,  418,  502,  586,  165,  249,  333,
    417,  81,   501,  585,  669,  670,  753,  837,  921,  1005, 1089, 498,
    162,  582,  330,  328,  160,  580,  664,  666,  78,   748,  76,   750,
    832,  412,  834,  246,  916,  414,  918,  742,  154,  574,  70,   910,
    406,  658,  826,  322,  994,  1000, 238,  1002, 1078, 1084, 244,  1086,
    490,  1162, 1168, 496,  1170, 1246, 1252, 79,   163,  247,  331,  415,
    499,  583,  162,  246,  330,  414,  244,  160,  76,   328,  412,  78,
    496,  498,  580,  582,  664,  666,  667,  748,  750,  832,  834,  916,
    918,  1000, 1002, 1084, 1086, 1168, 492,  156,  576,  324,  322,  154,
    574,  658,  660,  72,   742,  70,   744,  826,  406,  828,  240,  910,
    408,  912,  732,  144,  564,  60,   900,  396,  648,  816,  312,  984,
    994,  228,  996,  1068, 1078, 238,  1080, 480,  1152, 1162, 490,  1164,
    1236, 1246, 1252, 73,   157,  241,  325,  409,  493,  577,  156,  240,
    324,  408,  238,  154,  70,   322,  406,  72,   490,  492,  574,  576,
    658,  660,  661,  742,  744,  826,  828,  910,  912,  994,  996,  1078,
    1080, 1162, 482,  146,  566,  314,  312,  144,  564,  648,  650,  62,
    732,  60,   734,  816,  396,  818,  230,  900,  398,  902,  717,  129,
    549,  45,   885,  381,  633,  801,  297,  969,  984,  213,  986,  1053,
    1068, 228,  1070, 465,  1137, 1152, 480,  1154, 1221, 1236, 1246, 63,
    147,  231,  315,  399,  483,  567,  146,  230,  314,  398,  228,  144,
    60,   312,  396,  62,   480,  482,  564,  566,  648,  650,  651,  732,
    734,  816,  818,  900,  902,  984,  986,  1068, 1070, 1152, 467,  131,
    551,  299,  297,  129,  549,  633,  635,  47,   717,  45,   719,  801,
    381,  803,  215,  885,  383,  887,  696,  108,  528,  24,   864,  360,
    612,  780,  276,  948,  969,  192,  971,  1032, 1053, 213,  1055, 444,
    1116, 1137, 465,  1139, 1200, 1221, 1236, 167,  251,  335,  419,  83,
    503,  587,  671,  755,  839,  923,  1007, 1091, 502,  166,  586,  334,
    333,  165,  585,  669,  670,  82,   753,  81,   754,  837,  417,  838,
    250,  921,  418,  922,  750,  162,  582,  78,   918,  414,  666,  834,
    330,  1002, 1005, 246,  1006, 1086, 1089, 249,  1090, 498,  1170, 1173,
    501,  1174, 1254, 1257, 166,  250,  334,  418,  249,  165,  81,   333,
    417,  82,   501,  502,  585,  586,  669,  670,  753,  754,  837,  838,
    921,  922,  1005, 1006, 1089, 1090, 1173, 499,  163,  583,  331,  330,
    162,  582,  666,  667,  79,   750,  78,   751,  834,  414,  835,  247,
    918,  415,  919,  744,  156,  576,  72,   912,  408,  660,  828,  324,
    996,  1002, 240,  1003, 1080, 1086, 246,  1087, 492,  1164, 1170, 498,
    1171, 1248, 1254, 1257, 163,  247,  331,  415,  246,  162,  78,   330,
    414,  79,   498,  499,  582,  583,  666,  667,  750,  751,  834,  835,
    918,  919,  1002, 1003, 1086, 1087, 1170, 493,  157,  577,  325,  324,
    156,  576,  660,  661,  73,   744,  72,   745,  828,  408,  829,  241,
    912,  409,  913,  734,  146,  566,  62,   902,  398,  650,  818,  314,
    986,  996,  230,  997,  1070, 1080, 240,  1081, 482,  1154, 1164, 492,
    1165, 1238, 1248, 1254, 157,  241,  325,  409,  240,  156,  72,   324,
    408,  73,   492,  493,  576,  577,  660,  661,  744,  745,  828,  829,
    912,  913,  996,  997,  1080, 1081, 1164, 483,  147,  567,  315,  314,
    146,  566,  650,  651,  63,   734,  62,   735,  818,  398,  819,  231,
    902,  399,  903,  719,  131,  551,  47,   887,  383,  635,  803,  299,
    971,  986,  215,  987,  1055, 1070, 230,  1071, 467,  1139, 1154, 482,
    1155, 1223, 1238, 1248, 147,  231,  315,  399,  230,  146,  62,   314,
    398,  63,   482,  483,  566,  567,  650,  651,  734,  735,  818,  819,
    902,  903,  986,  987,  1070, 1071, 1154, 468,  132,  552,  300,  299,
    131,  551,  635,  636,  48,   719,  47,   720,  803,  383,  804,  216,
    887,  384,  888,  698,  110,  530,  26,   866,  362,  614,  782,  278,
    950,  971,  194,  972,  1034, 1055, 215,  1056, 446,  1118, 1139, 467,
    1140, 1202, 1223, 1238, 335,  167,  587,  671,  755,  83,   839,  419,
    923,  754,  166,  586,  82,   922,  418,  670,  838,  334,  1006, 1007,
    250,  1090, 1091, 251,  502,  1174, 1175, 503,  1258, 1259, 251,  167,
    83,   335,  419,  503,  587,  671,  755,  839,  923,  1007, 1091, 1175,
    334,  166,  586,  670,  754,  82,   838,  418,  922,  751,  163,  583,
    79,   919,  415,  667,  835,  331,  1003, 1006, 247,  1087, 1090, 250,
    499,  1171, 1174, 502,  1255, 1258, 1259, 250,  166,  82,   334,  418,
    502,  586,  670,  754,  838,  922,  1006, 1090, 1174, 331,  163,  583,
    667,  751,  79,   835,  415,  919,  745,  157,  577,  73,   913,  409,
    661,  829,  325,  997,  1003, 241,  1081, 1087, 247,  493,  1165, 1171,
    499,  1249, 1255, 1258, 247,  163,  79,   331,  415,  499,  583,  667,
    751,  835,  919,  1003, 1087, 1171, 325,  157,  577,  661,  745,  73,
    829,  409,  913,  735,  147,  567,  63,   903,  399,  651,  819,  315,
    987,  997,  231,  1071, 1081, 241,  483,  1155, 1165, 493,  1239, 1249,
    1255, 241,  157,  73,   325,  409,  493,  577,  661,  745,  829,  913,
    997,  1081, 1165, 315,  147,  567,  651,  735,  63,   819,  399,  903,
    720,  132,  552,  48,   888,  384,  636,  804,  300,  972,  987,  216,
    1056, 1071, 231,  468,  1140, 1155, 483,  1224, 1239, 1249, 231,  147,
    63,   315,  399,  483,  567,  651,  735,  819,  903,  987,  1071, 1155,
    300,  132,  552,  636,  720,  48,   804,  384,  888,  699,  111,  531,
    27,   867,  363,  615,  783,  279,  951,  972,  195,  1035, 1056, 216,
    447,  1119, 1140, 468,  1203, 1224, 1239, 216,  132,  48,   300,  384,
    468,  552,  636,  720,  804,  888,  972,  1056, 1140, 279,  111,  531,
    615,  699,  27,   783,  363,  867,  951,  1035, 195,  1119, 447,  1203,
    1224};
static const int C0_ind[] = {
    0,    1,    2,    3,    4,    5,    6,    30,   99,   100,  101,  102,
    103,  104,  105,  129,  198,  199,  200,  201,  202,  203,  204,  228,
    297,  298,  299,  300,  301,  302,  303,  327,  396,  397,  398,  399,
    400,  401,  402,  426,  495,  496,  497,  498,  499,  500,  501,  525,
    594,  595,  596,  597,  598,  599,  600,  624,  700,  702,  704,  706,
    714,  717,  720,  799,  800,  801,  802,  803,  804,  805,  806,  812,
    813,  815,  816,  818,  819,  821,  824,  826,  828,  830,  832,  891,
    892,  893,  894,  895,  896,  897,  898,  899,  900,  901,  902,  903,
    904,  905,  911,  912,  914,  915,  917,  918,  920,  921,  923,  925,
    927,  929,  931,  990,  991,  992,  993,  994,  995,  996,  997,  998,
    999,  1000, 1001, 1002, 1003, 1004, 1010, 1011, 1013, 1014, 1016, 1017,
    1019, 1020, 1022, 1024, 1026, 1028, 1030, 1089, 1090, 1091, 1092, 1093,
    1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1109, 1110,
    1112, 1113, 1115, 1116, 1118, 1119, 1121, 1123, 1125, 1127, 1129, 1188,
    1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200,
    1201, 1202, 1208, 1209, 1211, 1212, 1214, 1215, 1217, 1218, 1220, 1222,
    1224, 1226, 1228, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295,
    1296, 1297, 1298, 1299, 1300, 1301, 1307, 1308, 1310, 1311, 1313, 1314,
    1316, 1317, 1319, 1321, 1323, 1325, 1327, 1386, 1387, 1388, 1389, 1390,
    1391, 1392, 1394, 1396, 1398, 1400, 1406, 1409, 1412, 1415, 1416, 1418,
    1420, 1422, 1424, 1426, 1492, 1494, 1496, 1498, 1500, 1501, 1502, 1503,
    1504, 1506, 1507, 1509, 1510, 1512, 1513, 1516, 1518, 1520, 1522, 1524,
    1526, 1583, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600,
    1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612,
    1613, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625,
    1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693,
    1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705,
    1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717,
    1718, 1719, 1720, 1721, 1722, 1723, 1724, 1781, 1782, 1783, 1784, 1785,
    1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797,
    1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809,
    1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821,
    1822, 1823, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889,
    1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901,
    1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913,
    1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1979, 1980, 1981,
    1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993,
    1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005,
    2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
    2018, 2019, 2020, 2021, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085,
    2087, 2089, 2091, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2101, 2102,
    2104, 2105, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116,
    2117, 2118, 2119, 2120, 2177, 2185, 2187, 2189, 2191, 2193, 2194, 2195,
    2196, 2197, 2199, 2200, 2202, 2203, 2205, 2206, 2209, 2211, 2213, 2215,
    2217, 2219, 2276, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292,
    2293, 2294, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304,
    2305, 2306, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317,
    2318, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385,
    2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397,
    2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409,
    2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2474, 2475, 2476, 2477,
    2478, 2479, 2480, 2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489,
    2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501,
    2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512, 2513,
    2514, 2515, 2516, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2581,
    2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590, 2591, 2592, 2593,
    2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605,
    2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2672, 2673,
    2674, 2675, 2676, 2677, 2678, 2679, 2681, 2683, 2685, 2687, 2688, 2689,
    2690, 2691, 2692, 2693, 2695, 2696, 2698, 2699, 2701, 2702, 2703, 2704,
    2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2771, 2779,
    2781, 2783, 2785, 2787, 2788, 2789, 2790, 2791, 2793, 2794, 2796, 2797,
    2799, 2800, 2803, 2805, 2807, 2809, 2811, 2813, 2870, 2878, 2879, 2880,
    2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892,
    2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2902, 2903, 2904, 2905,
    2906, 2907, 2908, 2909, 2910, 2911, 2912, 2969, 2970, 2971, 2972, 2973,
    2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985,
    2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997,
    2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009,
    3010, 3011, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077,
    3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089,
    3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101,
    3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3167, 3168, 3169,
    3170, 3171, 3172, 3173, 3174, 3176, 3178, 3180, 3182, 3183, 3184, 3185,
    3186, 3187, 3188, 3190, 3191, 3193, 3194, 3196, 3197, 3198, 3199, 3200,
    3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3266, 3274, 3276,
    3278, 3280, 3282, 3283, 3284, 3285, 3286, 3288, 3289, 3291, 3292, 3294,
    3295, 3298, 3300, 3302, 3304, 3306, 3308, 3365, 3373, 3374, 3375, 3376,
    3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388,
    3389, 3390, 3391, 3392, 3393, 3394, 3395, 3397, 3398, 3399, 3400, 3401,
    3402, 3403, 3404, 3405, 3406, 3407, 3464, 3465, 3466, 3467, 3468, 3469,
    3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481,
    3482, 3483, 3484, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3492, 3493,
    3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505,
    3506, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3572, 3574, 3576,
    3578, 3579, 3580, 3581, 3582, 3583, 3584, 3586, 3587, 3589, 3590, 3592,
    3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604,
    3605, 3662, 3670, 3672, 3674, 3676, 3678, 3679, 3680, 3681, 3682, 3684,
    3685, 3687, 3688, 3690, 3691, 3694, 3696, 3698, 3700, 3702, 3704, 3761,
    3769, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777, 3778, 3779, 3780,
    3781, 3782, 3783, 3784, 3785, 3786, 3787, 3788, 3789, 3790, 3791, 3793,
    3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3860, 3861,
    3862, 3863, 3864, 3865, 3866, 3867, 3869, 3871, 3873, 3875, 3876, 3877,
    3878, 3879, 3880, 3881, 3883, 3884, 3886, 3887, 3889, 3890, 3891, 3892,
    3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901, 3902, 3959, 3967,
    3969, 3971, 3973, 3975, 3976, 3977, 3978, 3979, 3981, 3982, 3984, 3985,
    3987, 3988, 3991, 3993, 3995, 3997, 3999, 4001, 4058, 4067, 4069, 4071,
    4073, 4074, 4075, 4076, 4077, 4078, 4079, 4081, 4082, 4084, 4085, 4087,
    4088, 4090, 4091, 4092, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4100,
    4157, 4200, 4203, 4205, 4206, 4214, 4215, 4219, 4224, 4228, 4238, 4243,
    4244, 4299, 4300, 4301, 4302, 4303, 4304, 4305, 4306, 4311, 4312, 4313,
    4314, 4317, 4318, 4321, 4322, 4323, 4325, 4326, 4327, 4337, 4341, 4342,
    4343, 4347, 4352, 4356, 4357, 4358, 4359, 4360, 4361, 4362, 4386, 4398,
    4399, 4400, 4401, 4402, 4403, 4404, 4405, 4410, 4411, 4412, 4413, 4416,
    4417, 4420, 4421, 4422, 4424, 4425, 4426, 4436, 4440, 4441, 4442, 4446,
    4451, 4455, 4456, 4457, 4458, 4459, 4460, 4461, 4485, 4497, 4498, 4499,
    4500, 4501, 4502, 4503, 4504, 4509, 4510, 4511, 4512, 4515, 4516, 4519,
    4520, 4521, 4523, 4524, 4525, 4535, 4539, 4540, 4541, 4545, 4550, 4554,
    4555, 4556, 4557, 4558, 4559, 4560, 4584, 4596, 4597, 4598, 4599, 4600,
    4601, 4602, 4603, 4608, 4609, 4610, 4611, 4614, 4615, 4618, 4619, 4620,
    4622, 4623, 4624, 4634, 4638, 4639, 4640, 4644, 4649, 4653, 4654, 4655,
    4656, 4657, 4658, 4659, 4683, 4695, 4696, 4697, 4698, 4699, 4700, 4701,
    4702, 4707, 4708, 4709, 4710, 4713, 4714, 4717, 4718, 4719, 4721, 4722,
    4723, 4733, 4737, 4738, 4739, 4743, 4748, 4752, 4753, 4754, 4755, 4756,
    4757, 4758, 4782, 4794, 4795, 4796, 4797, 4798, 4799, 4800, 4801, 4806,
    4807, 4808, 4809, 4812, 4813, 4816, 4817, 4818, 4820, 4821, 4822, 4832,
    4836, 4837, 4838, 4842, 4847, 4851, 4852, 4853, 4854, 4855, 4856, 4857,
    4881, 4894, 4895, 4897, 4900, 4905, 4906, 4911, 4915, 4916, 4919, 4920,
    4935, 4941, 4946, 4957, 4959, 4961, 4963, 4971, 4974, 4977, 4992, 4995,
    4997, 4998, 5000, 5001, 5002, 5003, 5006, 5007, 5008, 5009, 5011, 5012,
    5013, 5016, 5017, 5020, 5030, 5032, 5035, 5036, 5038, 5039, 5043, 5044,
    5047, 5056, 5057, 5058, 5059, 5060, 5061, 5062, 5063, 5069, 5070, 5072,
    5073, 5075, 5076, 5078, 5081, 5083, 5085, 5087, 5089, 5091, 5092, 5093,
    5094, 5095, 5096, 5097, 5098, 5099, 5100, 5101, 5102, 5103, 5104, 5105,
    5106, 5107, 5108, 5109, 5110, 5111, 5112, 5113, 5114, 5115, 5116, 5117,
    5118, 5119, 5129, 5131, 5133, 5134, 5135, 5137, 5138, 5139, 5142, 5143,
    5144, 5146, 5148, 5149, 5150, 5151, 5152, 5153, 5154, 5155, 5156, 5157,
    5158, 5159, 5160, 5161, 5162, 5168, 5169, 5171, 5172, 5174, 5175, 5177,
    5178, 5180, 5182, 5184, 5186, 5188, 5190, 5191, 5192, 5193, 5194, 5195,
    5196, 5197, 5198, 5199, 5200, 5201, 5202, 5203, 5204, 5205, 5206, 5207,
    5208, 5209, 5210, 5211, 5212, 5213, 5214, 5215, 5216, 5217, 5218, 5228,
    5230, 5232, 5233, 5234, 5236, 5237, 5238, 5241, 5242, 5243, 5245, 5247,
    5248, 5249, 5250, 5251, 5252, 5253, 5254, 5255, 5256, 5257, 5258, 5259,
    5260, 5261, 5267, 5268, 5270, 5271, 5273, 5274, 5276, 5277, 5279, 5281,
    5283, 5285, 5287, 5289, 5290, 5291, 5292, 5293, 5294, 5295, 5296, 5297,
    5298, 5299, 5300, 5301, 5302, 5303, 5304, 5305, 5306, 5307, 5308, 5309,
    5310, 5311, 5312, 5313, 5314, 5315, 5316, 5317, 5327, 5329, 5331, 5332,
    5333, 5335, 5336, 5337, 5340, 5341, 5342, 5344, 5346, 5347, 5348, 5349,
    5350, 5351, 5352, 5353, 5354, 5355, 5356, 5357, 5358, 5359, 5360, 5366,
    5367, 5369, 5370, 5372, 5373, 5375, 5376, 5378, 5380, 5382, 5384, 5386,
    5388, 5389, 5390, 5391, 5392, 5393, 5394, 5395, 5396, 5397, 5398, 5399,
    5400, 5401, 5402, 5403, 5404, 5405, 5406, 5407, 5408, 5409, 5410, 5411,
    5412, 5413, 5414, 5415, 5416, 5426, 5428, 5430, 5431, 5432, 5434, 5435,
    5436, 5439, 5440, 5441, 5443, 5445, 5446, 5447, 5448, 5449, 5450, 5451,
    5452, 5453, 5454, 5455, 5456, 5457, 5458, 5459, 5465, 5466, 5468, 5469,
    5471, 5472, 5474, 5475, 5477, 5479, 5481, 5483, 5485, 5487, 5488, 5489,
    5490, 5491, 5492, 5493, 5494, 5495, 5496, 5497, 5498, 5499, 5500, 5501,
    5502, 5503, 5504, 5505, 5506, 5507, 5508, 5509, 5510, 5511, 5512, 5513,
    5514, 5515, 5525, 5527, 5529, 5530, 5531, 5533, 5534, 5535, 5538, 5539,
    5540, 5542, 5544, 5545, 5546, 5547, 5548, 5549, 5550, 5552, 5554, 5556,
    5558, 5564, 5567, 5570, 5573, 5574, 5576, 5578, 5580, 5582, 5584, 5587,
    5588, 5590, 5593, 5594, 5595, 5596, 5597, 5598, 5599, 5602, 5603, 5604,
    5606, 5607, 5608, 5609, 5611, 5612, 5613, 5626, 5628, 5632, 5633, 5634,
    5637, 5638, 5639, 5641, 5650, 5652, 5654, 5656, 5658, 5659, 5660, 5661,
    5662, 5664, 5665, 5667, 5668, 5670, 5671, 5674, 5676, 5678, 5680, 5682,
    5684, 5685, 5688, 5690, 5691, 5693, 5694, 5695, 5696, 5699, 5700, 5701,
    5702, 5704, 5705, 5706, 5709, 5710, 5713, 5723, 5725, 5728, 5729, 5731,
    5732, 5736, 5737, 5740, 5741, 5749, 5750, 5751, 5752, 5753, 5754, 5755,
    5756, 5757, 5758, 5759, 5760, 5761, 5762, 5763, 5764, 5765, 5766, 5767,
    5768, 5769, 5770, 5771, 5773, 5774, 5775, 5776, 5777, 5778, 5779, 5780,
    5781, 5782, 5783, 5784, 5785, 5786, 5787, 5788, 5789, 5790, 5791, 5792,
    5793, 5794, 5795, 5796, 5797, 5798, 5799, 5800, 5801, 5802, 5803, 5804,
    5805, 5806, 5807, 5808, 5809, 5810, 5811, 5812, 5822, 5824, 5826, 5827,
    5828, 5830, 5831, 5832, 5835, 5836, 5837, 5839, 5840, 5841, 5842, 5843,
    5844, 5845, 5846, 5847, 5848, 5849, 5850, 5851, 5852, 5853, 5854, 5855,
    5856, 5857, 5858, 5859, 5860, 5861, 5862, 5863, 5864, 5865, 5866, 5867,
    5868, 5869, 5870, 5871, 5872, 5873, 5874, 5875, 5876, 5877, 5878, 5879,
    5880, 5881, 5882, 5883, 5884, 5885, 5886, 5887, 5888, 5889, 5890, 5891,
    5892, 5893, 5894, 5895, 5896, 5897, 5898, 5899, 5900, 5901, 5902, 5903,
    5904, 5905, 5906, 5907, 5908, 5909, 5910, 5911, 5921, 5923, 5925, 5926,
    5927, 5929, 5930, 5931, 5934, 5935, 5936, 5938, 5939, 5940, 5941, 5942,
    5943, 5944, 5945, 5946, 5947, 5948, 5949, 5950, 5951, 5952, 5953, 5954,
    5955, 5956, 5957, 5958, 5959, 5960, 5961, 5962, 5963, 5964, 5965, 5966,
    5967, 5968, 5969, 5970, 5971, 5972, 5973, 5974, 5975, 5976, 5977, 5978,
    5979, 5980, 5981, 5982, 5983, 5984, 5985, 5986, 5987, 5988, 5989, 5990,
    5991, 5992, 5993, 5994, 5995, 5996, 5997, 5998, 5999, 6000, 6001, 6002,
    6003, 6004, 6005, 6006, 6007, 6008, 6009, 6010, 6020, 6022, 6024, 6025,
    6026, 6028, 6029, 6030, 6033, 6034, 6035, 6037, 6038, 6039, 6040, 6041,
    6042, 6043, 6044, 6045, 6046, 6047, 6048, 6049, 6050, 6051, 6052, 6053,
    6054, 6055, 6056, 6057, 6058, 6059, 6060, 6061, 6062, 6063, 6064, 6065,
    6066, 6067, 6068, 6069, 6070, 6071, 6072, 6073, 6074, 6075, 6076, 6077,
    6078, 6079, 6080, 6081, 6082, 6083, 6084, 6085, 6086, 6087, 6088, 6089,
    6090, 6091, 6092, 6093, 6094, 6095, 6096, 6097, 6098, 6099, 6100, 6101,
    6102, 6103, 6104, 6105, 6106, 6107, 6108, 6109, 6119, 6121, 6123, 6124,
    6125, 6127, 6128, 6129, 6132, 6133, 6134, 6136, 6137, 6138, 6139, 6140,
    6141, 6142, 6143, 6144, 6146, 6148, 6150, 6152, 6153, 6154, 6155, 6156,
    6157, 6158, 6160, 6161, 6163, 6164, 6166, 6167, 6168, 6169, 6170, 6171,
    6172, 6173, 6174, 6175, 6176, 6177, 6178, 6179, 6181, 6182, 6184, 6187,
    6188, 6189, 6190, 6191, 6192, 6193, 6196, 6197, 6198, 6200, 6201, 6202,
    6203, 6205, 6206, 6207, 6220, 6222, 6226, 6227, 6228, 6231, 6232, 6233,
    6235, 6236, 6244, 6246, 6248, 6250, 6252, 6253, 6254, 6255, 6256, 6258,
    6259, 6261, 6262, 6264, 6265, 6268, 6270, 6272, 6274, 6276, 6278, 6279,
    6282, 6284, 6285, 6287, 6288, 6289, 6290, 6293, 6294, 6295, 6296, 6298,
    6299, 6300, 6303, 6304, 6307, 6317, 6319, 6322, 6323, 6325, 6326, 6330,
    6331, 6334, 6335, 6343, 6344, 6345, 6346, 6347, 6348, 6349, 6350, 6351,
    6352, 6353, 6354, 6355, 6356, 6357, 6358, 6359, 6360, 6361, 6362, 6363,
    6364, 6365, 6367, 6368, 6369, 6370, 6371, 6372, 6373, 6374, 6375, 6376,
    6377, 6378, 6379, 6380, 6381, 6382, 6383, 6384, 6385, 6386, 6387, 6388,
    6389, 6390, 6391, 6392, 6393, 6394, 6395, 6396, 6397, 6398, 6399, 6400,
    6401, 6402, 6403, 6404, 6405, 6406, 6416, 6418, 6420, 6421, 6422, 6424,
    6425, 6426, 6429, 6430, 6431, 6433, 6434, 6435, 6436, 6437, 6438, 6439,
    6440, 6441, 6442, 6443, 6444, 6445, 6446, 6447, 6448, 6449, 6450, 6451,
    6452, 6453, 6454, 6455, 6456, 6457, 6458, 6459, 6460, 6461, 6462, 6463,
    6464, 6465, 6466, 6467, 6468, 6469, 6470, 6471, 6472, 6473, 6474, 6475,
    6476, 6477, 6478, 6479, 6480, 6481, 6482, 6483, 6484, 6485, 6486, 6487,
    6488, 6489, 6490, 6491, 6492, 6493, 6494, 6495, 6496, 6497, 6498, 6499,
    6500, 6501, 6502, 6503, 6504, 6505, 6515, 6517, 6519, 6520, 6521, 6523,
    6524, 6525, 6528, 6529, 6530, 6532, 6533, 6534, 6535, 6536, 6537, 6538,
    6539, 6540, 6541, 6542, 6543, 6544, 6545, 6546, 6547, 6548, 6549, 6550,
    6551, 6552, 6553, 6554, 6555, 6556, 6557, 6558, 6559, 6560, 6561, 6562,
    6563, 6564, 6565, 6566, 6567, 6568, 6569, 6570, 6571, 6572, 6573, 6574,
    6575, 6576, 6577, 6578, 6579, 6580, 6581, 6582, 6583, 6584, 6585, 6586,
    6587, 6588, 6589, 6590, 6591, 6592, 6593, 6594, 6595, 6596, 6597, 6598,
    6599, 6600, 6601, 6602, 6603, 6604, 6614, 6616, 6618, 6619, 6620, 6622,
    6623, 6624, 6627, 6628, 6629, 6631, 6632, 6633, 6634, 6635, 6636, 6637,
    6638, 6639, 6641, 6643, 6645, 6647, 6648, 6649, 6650, 6651, 6652, 6653,
    6655, 6656, 6658, 6659, 6661, 6662, 6663, 6664, 6665, 6666, 6667, 6668,
    6669, 6670, 6671, 6672, 6673, 6674, 6676, 6677, 6679, 6682, 6683, 6684,
    6685, 6686, 6687, 6688, 6691, 6692, 6693, 6695, 6696, 6697, 6698, 6700,
    6701, 6702, 6715, 6717, 6721, 6722, 6723, 6726, 6727, 6728, 6730, 6731,
    6739, 6741, 6743, 6745, 6747, 6748, 6749, 6750, 6751, 6753, 6754, 6756,
    6757, 6759, 6760, 6763, 6765, 6767, 6769, 6771, 6773, 6774, 6777, 6779,
    6780, 6782, 6783, 6784, 6785, 6788, 6789, 6790, 6791, 6793, 6794, 6795,
    6798, 6799, 6802, 6812, 6814, 6817, 6818, 6820, 6821, 6825, 6826, 6829,
    6830, 6838, 6839, 6840, 6841, 6842, 6843, 6844, 6845, 6846, 6847, 6848,
    6849, 6850, 6851, 6852, 6853, 6854, 6855, 6856, 6857, 6858, 6859, 6860,
    6862, 6863, 6864, 6865, 6866, 6867, 6868, 6869, 6870, 6871, 6872, 6873,
    6874, 6875, 6876, 6877, 6878, 6879, 6880, 6881, 6882, 6883, 6884, 6885,
    6886, 6887, 6888, 6889, 6890, 6891, 6892, 6893, 6894, 6895, 6896, 6897,
    6898, 6899, 6900, 6901, 6911, 6913, 6915, 6916, 6917, 6919, 6920, 6921,
    6924, 6925, 6926, 6928, 6929, 6930, 6931, 6932, 6933, 6934, 6935, 6936,
    6937, 6938, 6939, 6940, 6941, 6942, 6943, 6944, 6945, 6946, 6947, 6948,
    6949, 6950, 6951, 6952, 6953, 6954, 6955, 6956, 6957, 6958, 6959, 6960,
    6961, 6962, 6963, 6964, 6965, 6966, 6967, 6968, 6969, 6970, 6971, 6972,
    6973, 6974, 6975, 6976, 6977, 6978, 6979, 6980, 6981, 6982, 6983, 6984,
    6985, 6986, 6987, 6988, 6989, 6990, 6991, 6992, 6993, 6994, 6995, 6996,
    6997, 6998, 6999, 7000, 7010, 7012, 7014, 7015, 7016, 7018, 7019, 7020,
    7023, 7024, 7025, 7027, 7028, 7071, 7074, 7076, 7077, 7085, 7086, 7090,
    7095, 7099, 7100, 7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108, 7109,
    7110, 7112, 7114, 7115, 7116, 7120, 7121, 7125, 7170, 7171, 7172, 7173,
    7174, 7175, 7176, 7177, 7182, 7183, 7184, 7185, 7188, 7189, 7192, 7193,
    7194, 7196, 7197, 7198, 7199, 7200, 7201, 7202, 7203, 7204, 7205, 7206,
    7207, 7208, 7209, 7211, 7212, 7213, 7214, 7215, 7218, 7219, 7220, 7223,
    7224, 7227, 7228, 7229, 7230, 7231, 7232, 7233, 7257, 7269, 7270, 7271,
    7272, 7273, 7274, 7275, 7276, 7281, 7282, 7283, 7284, 7287, 7288, 7291,
    7292, 7293, 7295, 7296, 7297, 7298, 7299, 7300, 7301, 7302, 7303, 7304,
    7305, 7306, 7307, 7308, 7310, 7311, 7312, 7313, 7314, 7317, 7318, 7319,
    7322, 7323, 7326, 7327, 7328, 7329, 7330, 7331, 7332, 7356, 7368, 7369,
    7370, 7371, 7372, 7373, 7374, 7375, 7380, 7381, 7382, 7383, 7386, 7387,
    7390, 7391, 7392, 7394, 7395, 7396, 7397, 7398, 7399, 7400, 7401, 7402,
    7403, 7404, 7405, 7406, 7407, 7409, 7410, 7411, 7412, 7413, 7416, 7417,
    7418, 7421, 7422, 7425, 7426, 7427, 7428, 7429, 7430, 7431, 7455, 7467,
    7468, 7469, 7470, 7471, 7472, 7473, 7474, 7479, 7480, 7481, 7482, 7485,
    7486, 7489, 7490, 7491, 7493, 7494, 7495, 7496, 7497, 7498, 7499, 7500,
    7501, 7502, 7503, 7504, 7505, 7506, 7508, 7509, 7510, 7511, 7512, 7515,
    7516, 7517, 7520, 7521, 7524, 7525, 7526, 7527, 7528, 7529, 7530, 7554,
    7566, 7567, 7568, 7569, 7570, 7571, 7572, 7573, 7578, 7579, 7580, 7581,
    7584, 7585, 7588, 7589, 7590, 7592, 7593, 7594, 7595, 7596, 7597, 7598,
    7599, 7600, 7601, 7602, 7603, 7604, 7605, 7607, 7608, 7609, 7610, 7611,
    7614, 7615, 7616, 7619, 7620, 7623, 7624, 7625, 7626, 7627, 7628, 7629,
    7653, 7666, 7667, 7669, 7672, 7677, 7678, 7683, 7687, 7688, 7691, 7692,
    7694, 7695, 7696, 7697, 7698, 7699, 7700, 7701, 7702, 7704, 7706, 7707,
    7710, 7713, 7714, 7715, 7718, 7719, 7729, 7731, 7733, 7735, 7743, 7746,
    7749, 7764, 7767, 7769, 7770, 7772, 7773, 7774, 7775, 7778, 7779, 7780,
    7781, 7783, 7784, 7785, 7788, 7789, 7792, 7793, 7794, 7795, 7796, 7797,
    7798, 7799, 7800, 7801, 7802, 7803, 7804, 7805, 7807, 7808, 7809, 7810,
    7811, 7813, 7814, 7815, 7816, 7818, 7819, 7828, 7829, 7830, 7831, 7832,
    7833, 7834, 7835, 7841, 7842, 7844, 7845, 7847, 7848, 7850, 7853, 7855,
    7857, 7859, 7861, 7863, 7864, 7865, 7866, 7867, 7868, 7869, 7870, 7871,
    7872, 7873, 7874, 7875, 7876, 7877, 7878, 7879, 7880, 7881, 7882, 7883,
    7884, 7885, 7886, 7887, 7888, 7889, 7890, 7891, 7892, 7893, 7894, 7895,
    7896, 7897, 7898, 7899, 7900, 7901, 7902, 7903, 7904, 7905, 7906, 7907,
    7908, 7909, 7910, 7911, 7912, 7913, 7914, 7915, 7916, 7917, 7918, 7927,
    7929, 7931, 7933, 7935, 7936, 7937, 7938, 7939, 7941, 7942, 7944, 7945,
    7947, 7948, 7951, 7953, 7955, 7957, 7959, 7961, 7962, 7965, 7967, 7968,
    7970, 7971, 7972, 7973, 7976, 7977, 7978, 7979, 7981, 7982, 7983, 7986,
    7987, 7990, 7991, 7992, 7993, 7994, 7995, 7996, 7997, 7998, 7999, 8000,
    8001, 8002, 8003, 8005, 8006, 8007, 8008, 8009, 8011, 8012, 8013, 8014,
    8016, 8017, 8018, 8019, 8020, 8021, 8022, 8023, 8024, 8025, 8026, 8027,
    8028, 8029, 8030, 8031, 8032, 8033, 8039, 8040, 8042, 8043, 8045, 8046,
    8048, 8049, 8051, 8053, 8055, 8057, 8059, 8061, 8062, 8063, 8064, 8065,
    8066, 8067, 8068, 8069, 8070, 8071, 8072, 8073, 8074, 8075, 8076, 8077,
    8078, 8079, 8080, 8081, 8082, 8083, 8084, 8085, 8086, 8087, 8088, 8089,
    8090, 8091, 8092, 8093, 8094, 8095, 8096, 8097, 8098, 8099, 8100, 8101,
    8102, 8103, 8104, 8105, 8106, 8107, 8108, 8109, 8110, 8111, 8112, 8113,
    8114, 8115, 8116, 8125, 8126, 8127, 8128, 8129, 8130, 8131, 8132, 8133,
    8134, 8135, 8136, 8137, 8138, 8139, 8140, 8141, 8142, 8143, 8144, 8145,
    8146, 8147, 8149, 8150, 8151, 8152, 8153, 8154, 8155, 8156, 8157, 8158,
    8159, 8160, 8161, 8162, 8163, 8164, 8165, 8166, 8167, 8168, 8169, 8170,
    8171, 8172, 8173, 8174, 8175, 8176, 8177, 8178, 8179, 8180, 8181, 8182,
    8183, 8184, 8185, 8186, 8187, 8188, 8189, 8190, 8191, 8192, 8193, 8194,
    8195, 8196, 8197, 8198, 8199, 8200, 8201, 8202, 8203, 8204, 8205, 8206,
    8207, 8208, 8209, 8210, 8211, 8212, 8213, 8214, 8215, 8216, 8224, 8226,
    8228, 8230, 8232, 8233, 8234, 8235, 8236, 8238, 8239, 8241, 8242, 8244,
    8245, 8248, 8250, 8252, 8254, 8256, 8258, 8259, 8262, 8264, 8265, 8267,
    8268, 8269, 8270, 8273, 8274, 8275, 8276, 8278, 8279, 8280, 8283, 8284,
    8287, 8288, 8289, 8290, 8291, 8292, 8293, 8294, 8295, 8296, 8297, 8298,
    8299, 8300, 8302, 8303, 8304, 8305, 8306, 8308, 8309, 8310, 8311, 8313,
    8314, 8315, 8316, 8317, 8318, 8319, 8320, 8321, 8322, 8323, 8324, 8325,
    8326, 8327, 8328, 8329, 8330, 8336, 8337, 8339, 8340, 8342, 8343, 8345,
    8346, 8348, 8350, 8352, 8354, 8356, 8358, 8359, 8360, 8361, 8362, 8363,
    8364, 8365, 8366, 8367, 8368, 8369, 8370, 8371, 8372, 8373, 8374, 8375,
    8376, 8377, 8378, 8379, 8380, 8381, 8382, 8383, 8384, 8385, 8386, 8387,
    8388, 8389, 8390, 8391, 8392, 8393, 8394, 8395, 8396, 8397, 8398, 8399,
    8400, 8401, 8402, 8403, 8404, 8405, 8406, 8407, 8408, 8409, 8410, 8411,
    8412, 8413, 8415, 8416, 8417, 8418, 8419, 8420, 8421, 8422, 8423, 8424,
    8425, 8426, 8427, 8428, 8429, 8430, 8431, 8432, 8433, 8434, 8435, 8436,
    8437, 8438, 8439, 8440, 8441, 8442, 8443, 8444, 8445, 8446, 8447, 8448,
    8449, 8450, 8451, 8452, 8453, 8454, 8455, 8456, 8457, 8458, 8459, 8460,
    8461, 8462, 8463, 8464, 8465, 8466, 8467, 8468, 8469, 8470, 8471, 8472,
    8473, 8474, 8475, 8476, 8477, 8478, 8479, 8480, 8481, 8482, 8483, 8484,
    8485, 8486, 8487, 8488, 8489, 8490, 8491, 8492, 8493, 8494, 8495, 8496,
    8497, 8498, 8499, 8500, 8501, 8502, 8503, 8504, 8505, 8506, 8507, 8508,
    8509, 8510, 8511, 8512, 8513, 8521, 8522, 8523, 8524, 8525, 8526, 8527,
    8528, 8529, 8530, 8531, 8532, 8533, 8534, 8535, 8536, 8537, 8538, 8539,
    8540, 8541, 8542, 8543, 8545, 8546, 8547, 8548, 8549, 8550, 8551, 8552,
    8553, 8554, 8555, 8556, 8557, 8558, 8559, 8560, 8561, 8562, 8563, 8564,
    8565, 8566, 8567, 8568, 8569, 8570, 8571, 8572, 8573, 8574, 8575, 8576,
    8577, 8578, 8579, 8580, 8581, 8582, 8583, 8584, 8585, 8586, 8587, 8588,
    8589, 8590, 8591, 8592, 8593, 8594, 8595, 8596, 8597, 8598, 8599, 8600,
    8601, 8602, 8603, 8604, 8605, 8606, 8607, 8608, 8609, 8610, 8611, 8612,
    8620, 8622, 8624, 8626, 8628, 8629, 8630, 8631, 8632, 8634, 8635, 8637,
    8638, 8640, 8641, 8644, 8646, 8648, 8650, 8652, 8654, 8655, 8658, 8660,
    8661, 8663, 8664, 8665, 8666, 8669, 8670, 8671, 8672, 8674, 8675, 8676,
    8679, 8680, 8683, 8693, 8695, 8698, 8699, 8701, 8702, 8706, 8707, 8710,
    8711, 8712, 8713, 8714, 8715, 8716, 8717, 8718, 8719, 8720, 8721, 8722,
    8723, 8724, 8725, 8726, 8732, 8733, 8735, 8736, 8738, 8739, 8741, 8742,
    8744, 8746, 8748, 8750, 8752, 8754, 8755, 8756, 8757, 8758, 8759, 8760,
    8761, 8762, 8763, 8764, 8765, 8766, 8767, 8768, 8769, 8770, 8771, 8772,
    8773, 8774, 8775, 8776, 8777, 8778, 8779, 8780, 8781, 8782, 8783, 8784,
    8785, 8786, 8787, 8788, 8789, 8790, 8791, 8792, 8793, 8794, 8795, 8796,
    8797, 8798, 8799, 8800, 8801, 8802, 8803, 8804, 8805, 8806, 8807, 8808,
    8809, 8811, 8812, 8813, 8814, 8815, 8816, 8817, 8818, 8819, 8820, 8821,
    8822, 8823, 8824, 8825, 8826, 8827, 8828, 8829, 8830, 8831, 8832, 8833,
    8834, 8835, 8836, 8837, 8838, 8839, 8840, 8841, 8842, 8843, 8844, 8845,
    8846, 8847, 8848, 8849, 8850, 8851, 8852, 8853, 8854, 8855, 8856, 8857,
    8858, 8859, 8860, 8861, 8862, 8863, 8864, 8865, 8866, 8867, 8868, 8869,
    8870, 8871, 8872, 8873, 8874, 8875, 8876, 8877, 8878, 8879, 8880, 8881,
    8882, 8883, 8884, 8885, 8886, 8887, 8888, 8889, 8890, 8891, 8892, 8893,
    8894, 8895, 8896, 8897, 8898, 8899, 8900, 8901, 8902, 8903, 8904, 8905,
    8906, 8907, 8908, 8909, 8910, 8911, 8912, 8913, 8914, 8915, 8916, 8917,
    8918, 8919, 8920, 8921, 8922, 8923, 8924, 8925, 8926, 8927, 8928, 8929,
    8930, 8931, 8932, 8933, 8934, 8935, 8936, 8937, 8938, 8939, 8940, 8941,
    8942, 8943, 8944, 8945, 8946, 8947, 8948, 8949, 8950, 8951, 8952, 8953,
    8954, 8955, 8956, 8957, 8958, 8959, 8960, 8961, 8962, 8963, 8964, 8965,
    8966, 8967, 8968, 8969, 8970, 8971, 8972, 8973, 8974, 8975, 8976, 8977,
    8978, 8979, 8980, 8981, 8982, 8983, 8984, 8985, 8986, 8987, 8988, 8989,
    8990, 8991, 8992, 8993, 8994, 8995, 8996, 8997, 8998, 8999, 9000, 9001,
    9002, 9003, 9004, 9005, 9006, 9007, 9008, 9016, 9017, 9018, 9019, 9020,
    9021, 9022, 9023, 9024, 9025, 9026, 9027, 9028, 9029, 9030, 9031, 9032,
    9033, 9034, 9035, 9036, 9037, 9038, 9040, 9041, 9042, 9043, 9044, 9045,
    9046, 9047, 9048, 9049, 9050, 9051, 9052, 9053, 9054, 9055, 9056, 9057,
    9058, 9059, 9060, 9061, 9062, 9063, 9064, 9065, 9066, 9067, 9068, 9069,
    9070, 9071, 9072, 9073, 9074, 9075, 9076, 9077, 9078, 9079, 9089, 9091,
    9093, 9094, 9095, 9097, 9098, 9099, 9102, 9103, 9104, 9106, 9107, 9115,
    9117, 9119, 9121, 9123, 9124, 9125, 9126, 9127, 9129, 9130, 9132, 9133,
    9135, 9136, 9139, 9141, 9143, 9145, 9147, 9149, 9150, 9153, 9155, 9156,
    9158, 9159, 9160, 9161, 9164, 9165, 9166, 9167, 9169, 9170, 9171, 9174,
    9175, 9178, 9188, 9190, 9193, 9194, 9196, 9197, 9201, 9202, 9205, 9206,
    9207, 9208, 9209, 9210, 9211, 9212, 9213, 9215, 9217, 9219, 9221, 9227,
    9230, 9233, 9236, 9237, 9239, 9241, 9243, 9245, 9247, 9250, 9251, 9253,
    9256, 9257, 9258, 9259, 9260, 9261, 9262, 9265, 9266, 9267, 9269, 9270,
    9271, 9272, 9274, 9275, 9276, 9278, 9279, 9280, 9281, 9282, 9283, 9284,
    9285, 9286, 9288, 9289, 9290, 9291, 9294, 9295, 9296, 9297, 9298, 9299,
    9300, 9301, 9302, 9303, 9304, 9306, 9307, 9308, 9309, 9310, 9311, 9312,
    9314, 9316, 9318, 9320, 9321, 9322, 9323, 9324, 9325, 9326, 9328, 9329,
    9331, 9332, 9334, 9335, 9336, 9337, 9338, 9339, 9340, 9341, 9342, 9343,
    9344, 9345, 9346, 9347, 9349, 9350, 9352, 9355, 9356, 9357, 9358, 9359,
    9360, 9361, 9364, 9365, 9366, 9368, 9369, 9370, 9371, 9373, 9374, 9375,
    9377, 9378, 9379, 9380, 9381, 9382, 9383, 9384, 9385, 9387, 9388, 9389,
    9390, 9393, 9394, 9395, 9396, 9397, 9398, 9399, 9400, 9401, 9402, 9403,
    9404, 9405, 9406, 9407, 9408, 9409, 9410, 9411, 9413, 9415, 9417, 9419,
    9420, 9421, 9422, 9423, 9424, 9425, 9427, 9428, 9430, 9431, 9433, 9434,
    9435, 9436, 9437, 9438, 9439, 9440, 9441, 9442, 9443, 9444, 9445, 9446,
    9448, 9449, 9451, 9454, 9455, 9456, 9457, 9458, 9459, 9460, 9463, 9464,
    9465, 9467, 9468, 9469, 9470, 9472, 9473, 9474, 9487, 9489, 9493, 9494,
    9495, 9498, 9499, 9500, 9502, 9503, 9504, 9505, 9506, 9507, 9508, 9509,
    9510, 9512, 9514, 9516, 9518, 9519, 9520, 9521, 9522, 9523, 9524, 9526,
    9527, 9529, 9530, 9532, 9533, 9534, 9535, 9536, 9537, 9538, 9539, 9540,
    9541, 9542, 9543, 9544, 9545, 9547, 9548, 9550, 9553, 9554, 9555, 9556,
    9557, 9558, 9559, 9562, 9563, 9564, 9566, 9567, 9568, 9569, 9571, 9572,
    9573, 9586, 9588, 9592, 9593, 9594, 9597, 9598, 9599, 9601, 9602, 9611,
    9613, 9615, 9617, 9618, 9619, 9620, 9621, 9622, 9623, 9625, 9626, 9628,
    9629, 9631, 9632, 9634, 9635, 9636, 9637, 9638, 9639, 9640, 9641, 9642,
    9643, 9644, 9646, 9647, 9649, 9652, 9653, 9654, 9655, 9656, 9657, 9658,
    9661, 9662, 9663, 9665, 9666, 9667, 9668, 9670, 9671, 9672, 9685, 9687,
    9691, 9692, 9693, 9696, 9697, 9698, 9700, 9701, 9717, 9718, 9719, 9720,
    9721, 9724, 9727, 9730, 9733, 9735, 9737, 9739, 9741, 9743, 9800};
static const int C1_ind[] = {
    71,   72,   73,   74,   75,   76,   77,   78,   79,   81,   83,   87,
    91,   92,   96,   141,  144,  146,  147,  155,  156,  160,  165,  169,
    170,  171,  172,  173,  174,  175,  176,  177,  178,  179,  180,  182,
    184,  185,  186,  190,  191,  195,  240,  243,  245,  246,  254,  255,
    259,  264,  268,  269,  270,  271,  272,  273,  274,  275,  276,  277,
    278,  279,  281,  283,  284,  285,  289,  290,  294,  339,  342,  344,
    345,  353,  354,  358,  363,  367,  368,  369,  370,  371,  372,  373,
    374,  375,  376,  377,  378,  380,  382,  383,  384,  388,  389,  393,
    438,  441,  443,  444,  452,  453,  457,  462,  466,  467,  468,  469,
    470,  471,  472,  473,  474,  475,  476,  477,  479,  481,  482,  483,
    487,  488,  492,  537,  540,  542,  543,  551,  552,  556,  561,  565,
    566,  567,  568,  569,  570,  571,  572,  573,  574,  575,  576,  578,
    580,  581,  582,  586,  587,  591,  636,  637,  638,  639,  640,  641,
    642,  643,  648,  649,  650,  651,  654,  655,  658,  659,  660,  662,
    663,  664,  665,  666,  667,  668,  669,  670,  671,  672,  673,  674,
    675,  677,  678,  679,  680,  681,  684,  685,  686,  689,  690,  700,
    702,  704,  706,  714,  717,  720,  735,  738,  740,  741,  743,  744,
    745,  746,  749,  750,  751,  752,  754,  755,  756,  759,  760,  763,
    764,  765,  766,  767,  768,  769,  770,  771,  772,  773,  774,  775,
    776,  778,  779,  780,  781,  782,  784,  785,  786,  787,  789,  790,
    834,  835,  836,  837,  838,  839,  840,  841,  846,  847,  848,  849,
    852,  853,  856,  857,  858,  860,  861,  862,  863,  864,  865,  866,
    867,  868,  869,  870,  871,  872,  873,  875,  876,  877,  878,  879,
    882,  883,  884,  887,  888,  891,  892,  893,  894,  895,  896,  897,
    921,  933,  934,  935,  936,  937,  938,  939,  940,  945,  946,  947,
    948,  951,  952,  955,  956,  957,  959,  960,  961,  962,  963,  964,
    965,  966,  967,  968,  969,  970,  971,  972,  974,  975,  976,  977,
    978,  981,  982,  983,  986,  987,  997,  998,  999,  1000, 1001, 1002,
    1003, 1004, 1010, 1011, 1013, 1014, 1016, 1017, 1019, 1022, 1024, 1026,
    1028, 1030, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041,
    1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053,
    1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065,
    1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077,
    1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1096, 1098,
    1100, 1102, 1110, 1113, 1116, 1131, 1134, 1136, 1137, 1139, 1140, 1141,
    1142, 1145, 1146, 1147, 1148, 1150, 1151, 1152, 1155, 1156, 1159, 1160,
    1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172,
    1174, 1175, 1176, 1177, 1178, 1180, 1181, 1182, 1183, 1185, 1186, 1195,
    1197, 1199, 1201, 1203, 1204, 1205, 1206, 1207, 1209, 1210, 1212, 1213,
    1215, 1216, 1219, 1221, 1223, 1225, 1227, 1229, 1230, 1233, 1235, 1236,
    1238, 1239, 1240, 1241, 1244, 1245, 1246, 1247, 1249, 1250, 1251, 1254,
    1255, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268,
    1269, 1270, 1271, 1273, 1274, 1275, 1276, 1277, 1279, 1280, 1281, 1282,
    1284, 1285, 1286, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1341,
    1342, 1343, 1344, 1347, 1348, 1351, 1352, 1353, 1355, 1356, 1357, 1358,
    1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1370, 1371,
    1372, 1373, 1374, 1377, 1378, 1379, 1382, 1383, 1386, 1387, 1388, 1389,
    1390, 1391, 1392, 1416, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435,
    1440, 1441, 1442, 1443, 1446, 1447, 1450, 1451, 1452, 1454, 1455, 1456,
    1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1469,
    1470, 1471, 1472, 1473, 1476, 1477, 1478, 1481, 1482, 1485, 1486, 1487,
    1488, 1489, 1490, 1491, 1515, 1527, 1528, 1529, 1530, 1531, 1532, 1533,
    1534, 1539, 1540, 1541, 1542, 1545, 1546, 1549, 1550, 1551, 1553, 1554,
    1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566,
    1568, 1569, 1570, 1571, 1572, 1575, 1576, 1577, 1580, 1581, 1584, 1585,
    1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597,
    1598, 1604, 1605, 1607, 1608, 1610, 1611, 1613, 1614, 1616, 1618, 1620,
    1622, 1624, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635,
    1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647,
    1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659,
    1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671,
    1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1690, 1691,
    1692, 1693, 1694, 1695, 1696, 1697, 1703, 1704, 1706, 1707, 1709, 1710,
    1712, 1715, 1717, 1719, 1721, 1723, 1725, 1726, 1727, 1728, 1729, 1730,
    1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742,
    1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754,
    1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766,
    1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778,
    1779, 1780, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798,
    1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810,
    1811, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823,
    1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835,
    1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847,
    1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859,
    1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1871,
    1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1888, 1890, 1892,
    1894, 1902, 1905, 1908, 1923, 1926, 1928, 1929, 1931, 1932, 1933, 1934,
    1937, 1938, 1939, 1940, 1942, 1943, 1944, 1947, 1948, 1951, 1952, 1953,
    1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1966,
    1967, 1968, 1969, 1970, 1972, 1973, 1974, 1975, 1977, 1978, 1987, 1989,
    1991, 1993, 1995, 1996, 1997, 1998, 1999, 2001, 2002, 2004, 2005, 2007,
    2008, 2011, 2013, 2015, 2017, 2019, 2021, 2022, 2025, 2027, 2028, 2030,
    2031, 2032, 2033, 2036, 2037, 2038, 2039, 2041, 2042, 2043, 2046, 2047,
    2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061,
    2062, 2063, 2065, 2066, 2067, 2068, 2069, 2071, 2072, 2073, 2074, 2076,
    2077, 2078, 2086, 2088, 2090, 2092, 2094, 2095, 2096, 2097, 2098, 2100,
    2101, 2103, 2104, 2106, 2107, 2110, 2112, 2114, 2116, 2118, 2120, 2121,
    2124, 2126, 2127, 2129, 2130, 2131, 2132, 2135, 2136, 2137, 2138, 2140,
    2141, 2142, 2145, 2146, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156,
    2157, 2158, 2159, 2160, 2161, 2162, 2164, 2165, 2166, 2167, 2168, 2170,
    2171, 2172, 2173, 2175, 2176, 2177, 2185, 2187, 2189, 2191, 2193, 2194,
    2195, 2196, 2197, 2199, 2200, 2202, 2203, 2205, 2206, 2209, 2211, 2213,
    2215, 2217, 2219, 2220, 2223, 2225, 2226, 2228, 2229, 2230, 2231, 2234,
    2235, 2236, 2237, 2239, 2240, 2241, 2244, 2245, 2248, 2249, 2250, 2251,
    2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2263, 2264,
    2265, 2266, 2267, 2269, 2270, 2271, 2272, 2274, 2275, 2276, 2319, 2320,
    2321, 2322, 2323, 2324, 2325, 2326, 2331, 2332, 2333, 2334, 2337, 2338,
    2341, 2342, 2343, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353,
    2354, 2355, 2356, 2357, 2358, 2360, 2361, 2362, 2363, 2364, 2367, 2368,
    2369, 2372, 2373, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2406, 2418,
    2419, 2420, 2421, 2422, 2423, 2424, 2425, 2430, 2431, 2432, 2433, 2436,
    2437, 2440, 2441, 2442, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451,
    2452, 2453, 2454, 2455, 2456, 2457, 2459, 2460, 2461, 2462, 2463, 2466,
    2467, 2468, 2471, 2472, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2505,
    2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2529, 2530, 2531, 2532,
    2535, 2536, 2539, 2540, 2541, 2543, 2544, 2545, 2546, 2547, 2548, 2549,
    2550, 2551, 2552, 2553, 2554, 2555, 2556, 2558, 2559, 2560, 2561, 2562,
    2565, 2566, 2567, 2570, 2571, 2574, 2575, 2576, 2577, 2578, 2579, 2580,
    2604, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2628, 2629, 2630,
    2631, 2634, 2635, 2638, 2639, 2640, 2642, 2643, 2644, 2645, 2646, 2647,
    2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2657, 2658, 2659, 2660,
    2661, 2664, 2665, 2666, 2669, 2670, 2673, 2674, 2675, 2676, 2677, 2678,
    2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2693, 2694, 2696,
    2697, 2699, 2700, 2702, 2703, 2705, 2707, 2709, 2711, 2713, 2715, 2716,
    2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728,
    2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740,
    2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752,
    2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764,
    2765, 2766, 2767, 2768, 2769, 2770, 2772, 2773, 2774, 2775, 2776, 2777,
    2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2792, 2793, 2795,
    2796, 2798, 2799, 2801, 2802, 2804, 2806, 2808, 2810, 2812, 2814, 2815,
    2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827,
    2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839,
    2840, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851,
    2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863,
    2864, 2865, 2866, 2867, 2868, 2869, 2871, 2872, 2873, 2874, 2875, 2876,
    2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888,
    2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900,
    2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912,
    2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924,
    2925, 2926, 2927, 2928, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936,
    2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948,
    2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960,
    2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2977, 2978, 2979,
    2980, 2981, 2982, 2983, 2984, 2990, 2991, 2993, 2994, 2996, 2997, 2999,
    3002, 3004, 3006, 3008, 3010, 3012, 3013, 3014, 3015, 3016, 3017, 3018,
    3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030,
    3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042,
    3043, 3044, 3045, 3046, 3047, 3048, 3049, 3050, 3051, 3052, 3053, 3054,
    3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066,
    3067, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086,
    3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098,
    3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111,
    3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123,
    3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135,
    3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147,
    3148, 3149, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159,
    3160, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 3175, 3176, 3177, 3178,
    3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190,
    3191, 3192, 3193, 3194, 3195, 3196, 3197, 3199, 3200, 3201, 3202, 3203,
    3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215,
    3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225, 3226, 3227,
    3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239,
    3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251,
    3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263,
    3264, 3265, 3266, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282,
    3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3292, 3293, 3294,
    3295, 3296, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307,
    3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319,
    3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331,
    3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343,
    3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355,
    3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3373, 3375,
    3377, 3379, 3387, 3390, 3393, 3408, 3411, 3413, 3414, 3416, 3417, 3418,
    3419, 3422, 3423, 3424, 3425, 3427, 3428, 3429, 3432, 3433, 3436, 3437,
    3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449,
    3451, 3452, 3453, 3454, 3455, 3457, 3458, 3459, 3460, 3462, 3463, 3472,
    3474, 3476, 3478, 3480, 3481, 3482, 3483, 3484, 3486, 3487, 3489, 3490,
    3492, 3493, 3496, 3498, 3500, 3502, 3504, 3506, 3507, 3510, 3512, 3513,
    3515, 3516, 3517, 3518, 3521, 3522, 3523, 3524, 3526, 3527, 3528, 3531,
    3532, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542, 3543, 3544, 3545,
    3546, 3547, 3548, 3550, 3551, 3552, 3553, 3554, 3556, 3557, 3558, 3559,
    3561, 3562, 3563, 3571, 3573, 3575, 3577, 3579, 3580, 3581, 3582, 3583,
    3585, 3586, 3588, 3589, 3591, 3592, 3595, 3597, 3599, 3601, 3603, 3605,
    3606, 3609, 3611, 3612, 3614, 3615, 3616, 3617, 3620, 3621, 3622, 3623,
    3625, 3626, 3627, 3630, 3631, 3634, 3635, 3636, 3637, 3638, 3639, 3640,
    3641, 3642, 3643, 3644, 3645, 3646, 3647, 3649, 3650, 3651, 3652, 3653,
    3655, 3656, 3657, 3658, 3660, 3661, 3662, 3670, 3672, 3674, 3676, 3678,
    3679, 3680, 3681, 3682, 3684, 3685, 3687, 3688, 3690, 3691, 3694, 3696,
    3698, 3700, 3702, 3704, 3705, 3708, 3710, 3711, 3713, 3714, 3715, 3716,
    3719, 3720, 3721, 3722, 3724, 3725, 3726, 3729, 3730, 3733, 3734, 3735,
    3736, 3737, 3738, 3739, 3740, 3741, 3742, 3743, 3744, 3745, 3746, 3748,
    3749, 3750, 3751, 3752, 3754, 3755, 3756, 3757, 3759, 3760, 3761, 3769,
    3771, 3773, 3775, 3777, 3778, 3779, 3780, 3781, 3783, 3784, 3786, 3787,
    3789, 3790, 3793, 3795, 3797, 3799, 3801, 3803, 3804, 3807, 3809, 3810,
    3812, 3813, 3814, 3815, 3818, 3819, 3820, 3821, 3823, 3824, 3825, 3828,
    3829, 3832, 3833, 3834, 3835, 3836, 3837, 3838, 3839, 3840, 3841, 3842,
    3843, 3844, 3845, 3847, 3848, 3849, 3850, 3851, 3853, 3854, 3855, 3856,
    3858, 3859, 3860, 3904, 3905, 3907, 3910, 3915, 3916, 3921, 3925, 3926,
    3929, 3930, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3942,
    3944, 3945, 3948, 3951, 3952, 3953, 3956, 3957, 3960, 3961, 3962, 3963,
    3964, 3965, 3966, 3990, 4003, 4004, 4006, 4009, 4014, 4015, 4020, 4024,
    4025, 4028, 4029, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039,
    4041, 4043, 4044, 4047, 4050, 4051, 4052, 4055, 4056, 4059, 4060, 4061,
    4062, 4063, 4064, 4065, 4089, 4102, 4103, 4105, 4108, 4113, 4114, 4119,
    4123, 4124, 4127, 4128, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137,
    4138, 4140, 4142, 4143, 4146, 4149, 4150, 4151, 4154, 4155, 4158, 4159,
    4160, 4161, 4162, 4163, 4164, 4188, 4201, 4202, 4204, 4207, 4212, 4213,
    4218, 4222, 4223, 4226, 4227, 4229, 4230, 4231, 4232, 4233, 4234, 4235,
    4236, 4237, 4239, 4241, 4242, 4245, 4248, 4249, 4250, 4253, 4254, 4257,
    4258, 4259, 4260, 4261, 4262, 4263, 4287, 4300, 4301, 4303, 4306, 4311,
    4312, 4317, 4321, 4322, 4325, 4326, 4328, 4329, 4330, 4331, 4332, 4333,
    4334, 4335, 4336, 4338, 4340, 4341, 4344, 4347, 4348, 4349, 4352, 4353,
    4356, 4357, 4358, 4359, 4360, 4361, 4362, 4364, 4366, 4368, 4370, 4376,
    4379, 4382, 4385, 4386, 4388, 4390, 4392, 4394, 4396, 4399, 4400, 4402,
    4405, 4406, 4407, 4408, 4409, 4410, 4411, 4414, 4415, 4416, 4418, 4419,
    4420, 4421, 4423, 4424, 4425, 4427, 4428, 4429, 4430, 4431, 4432, 4433,
    4434, 4435, 4437, 4438, 4439, 4440, 4443, 4444, 4445, 4446, 4447, 4448,
    4449, 4450, 4451, 4452, 4453, 4455, 4456, 4457, 4458, 4459, 4460, 4461,
    4463, 4465, 4467, 4469, 4475, 4478, 4481, 4484, 4485, 4487, 4489, 4491,
    4493, 4495, 4498, 4499, 4501, 4504, 4505, 4506, 4507, 4508, 4509, 4510,
    4513, 4514, 4515, 4517, 4518, 4519, 4520, 4522, 4523, 4524, 4526, 4527,
    4528, 4529, 4530, 4531, 4532, 4533, 4534, 4536, 4537, 4538, 4539, 4542,
    4543, 4544, 4545, 4546, 4547, 4548, 4549, 4550, 4551, 4552, 4554, 4555,
    4556, 4557, 4558, 4559, 4560, 4562, 4564, 4566, 4568, 4569, 4570, 4571,
    4572, 4573, 4574, 4576, 4577, 4579, 4580, 4582, 4583, 4584, 4585, 4586,
    4587, 4588, 4589, 4590, 4591, 4592, 4593, 4594, 4595, 4597, 4598, 4600,
    4603, 4604, 4605, 4606, 4607, 4608, 4609, 4612, 4613, 4614, 4616, 4617,
    4618, 4619, 4621, 4622, 4623, 4625, 4626, 4627, 4628, 4629, 4630, 4631,
    4632, 4633, 4635, 4636, 4637, 4638, 4641, 4642, 4643, 4644, 4645, 4646,
    4647, 4648, 4649, 4650, 4651, 4652, 4653, 4654, 4655, 4656, 4657, 4658,
    4659, 4661, 4663, 4665, 4667, 4668, 4669, 4670, 4671, 4672, 4673, 4675,
    4676, 4678, 4679, 4681, 4682, 4683, 4684, 4685, 4686, 4687, 4688, 4689,
    4690, 4691, 4692, 4693, 4694, 4696, 4697, 4699, 4702, 4703, 4704, 4705,
    4706, 4707, 4708, 4711, 4712, 4713, 4715, 4716, 4717, 4718, 4720, 4721,
    4722, 4724, 4725, 4726, 4727, 4728, 4729, 4730, 4731, 4732, 4734, 4735,
    4736, 4737, 4740, 4741, 4742, 4743, 4744, 4745, 4746, 4747, 4748, 4749,
    4750, 4751, 4752, 4753, 4754, 4755, 4756, 4757, 4758, 4760, 4762, 4764,
    4766, 4772, 4775, 4778, 4781, 4782, 4784, 4786, 4788, 4790, 4792, 4795,
    4796, 4798, 4801, 4802, 4803, 4804, 4805, 4806, 4807, 4810, 4811, 4812,
    4814, 4815, 4816, 4817, 4819, 4820, 4821, 4823, 4824, 4825, 4826, 4827,
    4828, 4829, 4830, 4831, 4833, 4834, 4835, 4836, 4839, 4840, 4841, 4842,
    4843, 4844, 4845, 4846, 4847, 4848, 4849, 4851, 4852, 4853, 4854, 4855,
    4856, 4857, 4859, 4861, 4863, 4865, 4866, 4867, 4868, 4869, 4870, 4871,
    4873, 4874, 4876, 4877, 4879, 4880, 4881, 4882, 4883, 4884, 4885, 4886,
    4887, 4888, 4889, 4890, 4891, 4892, 4894, 4895, 4897, 4900, 4901, 4902,
    4903, 4904, 4905, 4906, 4909, 4910, 4911, 4913, 4914, 4915, 4916, 4918,
    4919, 4920, 4922, 4923, 4924, 4925, 4926, 4927, 4928, 4929, 4930, 4932,
    4933, 4934, 4935, 4938, 4939, 4940, 4941, 4942, 4943, 4944, 4945, 4946,
    4947, 4948, 4949, 4950, 4951, 4952, 4953, 4954, 4955, 4956, 4958, 4960,
    4962, 4964, 4965, 4966, 4967, 4968, 4969, 4970, 4972, 4973, 4975, 4976,
    4978, 4979, 4980, 4981, 4982, 4983, 4984, 4985, 4986, 4987, 4988, 4989,
    4990, 4991, 4993, 4994, 4996, 4999, 5000, 5001, 5002, 5003, 5004, 5005,
    5008, 5009, 5010, 5012, 5013, 5014, 5015, 5017, 5018, 5019, 5021, 5022,
    5023, 5024, 5025, 5026, 5027, 5028, 5029, 5031, 5032, 5033, 5034, 5037,
    5038, 5039, 5040, 5041, 5042, 5043, 5044, 5045, 5046, 5047, 5048, 5049,
    5050, 5051, 5052, 5053, 5054, 5055, 5057, 5059, 5061, 5063, 5064, 5065,
    5066, 5067, 5068, 5069, 5071, 5072, 5074, 5075, 5077, 5078, 5079, 5080,
    5081, 5082, 5083, 5084, 5085, 5086, 5087, 5088, 5089, 5090, 5092, 5093,
    5095, 5098, 5099, 5100, 5101, 5102, 5103, 5104, 5107, 5108, 5109, 5111,
    5112, 5113, 5114, 5116, 5117, 5118, 5120, 5121, 5122, 5123, 5124, 5125,
    5126, 5127, 5128, 5130, 5131, 5132, 5133, 5136, 5137, 5138, 5139, 5140,
    5141, 5142, 5143, 5144, 5145, 5146, 5147, 5156, 5158, 5160, 5162, 5168,
    5171, 5174, 5177, 5180, 5182, 5184, 5186, 5188, 5191, 5192, 5194, 5197,
    5198, 5199, 5200, 5201, 5202, 5203, 5206, 5207, 5208, 5210, 5211, 5212,
    5213, 5215, 5216, 5217, 5219, 5220, 5221, 5222, 5223, 5224, 5225, 5226,
    5227, 5229, 5230, 5231, 5232, 5235, 5236, 5237, 5238, 5239, 5240, 5241,
    5242, 5243, 5244, 5245, 5255, 5257, 5259, 5261, 5262, 5263, 5264, 5265,
    5266, 5267, 5269, 5270, 5272, 5273, 5275, 5276, 5278, 5279, 5280, 5281,
    5282, 5283, 5284, 5285, 5286, 5287, 5288, 5290, 5291, 5293, 5296, 5297,
    5298, 5299, 5300, 5301, 5302, 5305, 5306, 5307, 5309, 5310, 5311, 5312,
    5314, 5315, 5316, 5318, 5319, 5320, 5321, 5322, 5323, 5324, 5325, 5326,
    5328, 5329, 5330, 5331, 5334, 5335, 5336, 5337, 5338, 5339, 5340, 5341,
    5342, 5343, 5344, 5345, 5354, 5356, 5358, 5360, 5361, 5362, 5363, 5364,
    5365, 5366, 5368, 5369, 5371, 5372, 5374, 5375, 5377, 5378, 5379, 5380,
    5381, 5382, 5383, 5384, 5385, 5386, 5387, 5389, 5390, 5392, 5395, 5396,
    5397, 5398, 5399, 5400, 5401, 5404, 5405, 5406, 5408, 5409, 5410, 5411,
    5413, 5414, 5415, 5417, 5418, 5419, 5420, 5421, 5422, 5423, 5424, 5425,
    5427, 5428, 5429, 5430, 5433, 5434, 5435, 5436, 5437, 5438, 5439, 5440,
    5441, 5442, 5443, 5444, 5453, 5455, 5457, 5459, 5460, 5461, 5462, 5463,
    5464, 5465, 5467, 5468, 5470, 5471, 5473, 5474, 5476, 5477, 5478, 5479,
    5480, 5481, 5482, 5483, 5484, 5485, 5486, 5488, 5489, 5491, 5494, 5495,
    5496, 5497, 5498, 5499, 5500, 5503, 5504, 5505, 5507, 5508, 5509, 5510,
    5512, 5513, 5514, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524,
    5526, 5527, 5528, 5529, 5532, 5533, 5534, 5535, 5536, 5537, 5538, 5539,
    5540, 5541, 5542, 5543, 5552, 5554, 5556, 5558, 5559, 5560, 5561, 5562,
    5563, 5564, 5566, 5567, 5569, 5570, 5572, 5573, 5575, 5576, 5577, 5578,
    5579, 5580, 5581, 5582, 5583, 5584, 5585, 5587, 5588, 5590, 5593, 5594,
    5595, 5596, 5597, 5598, 5599, 5602, 5603, 5604, 5606, 5607, 5608, 5609,
    5611, 5612, 5613, 5615, 5616, 5617, 5618, 5619, 5620, 5621, 5622, 5623,
    5625, 5626, 5627, 5628, 5631, 5632, 5633, 5634, 5635, 5636, 5637, 5638,
    5639, 5640, 5641, 5642, 5693, 5694, 5695, 5696, 5701, 5702, 5705, 5706,
    5710, 5714, 5715, 5716, 5717, 5718, 5719, 5720, 5721, 5722, 5724, 5725,
    5726, 5730, 5731, 5732, 5734, 5735, 5736, 5737, 5739, 5740, 5757, 5758,
    5759, 5760, 5761, 5764, 5767, 5770, 5773, 5775, 5777, 5779, 5781, 5783,
    5792, 5793, 5794, 5795, 5800, 5801, 5804, 5805, 5809, 5813, 5814, 5815,
    5816, 5817, 5818, 5819, 5820, 5821, 5823, 5824, 5825, 5829, 5830, 5831,
    5833, 5834, 5835, 5836, 5838, 5839, 5840, 5856, 5857, 5858, 5859, 5860,
    5863, 5866, 5869, 5872, 5874, 5876, 5878, 5880, 5882, 5891, 5892, 5893,
    5894, 5899, 5900, 5903, 5904, 5908, 5912, 5913, 5914, 5915, 5916, 5917,
    5918, 5919, 5920, 5922, 5923, 5924, 5928, 5929, 5930, 5932, 5933, 5934,
    5935, 5937, 5938, 5939, 5955, 5956, 5957, 5958, 5959, 5962, 5965, 5968,
    5971, 5973, 5975, 5977, 5979, 5981, 5990, 5991, 5992, 5993, 5998, 5999,
    6002, 6003, 6007, 6011, 6012, 6013, 6014, 6015, 6016, 6017, 6018, 6019,
    6021, 6022, 6023, 6027, 6028, 6029, 6031, 6032, 6033, 6034, 6036, 6037,
    6038, 6054, 6055, 6056, 6057, 6058, 6061, 6064, 6067, 6070, 6072, 6074,
    6076, 6078, 6080, 6089, 6090, 6091, 6092, 6097, 6098, 6101, 6102, 6106,
    6110, 6111, 6112, 6113, 6114, 6115, 6116, 6117, 6118, 6120, 6121, 6122,
    6126, 6127, 6128, 6130, 6131, 6132, 6133, 6135, 6136, 6137, 6153, 6154,
    6155, 6156, 6157, 6160, 6163, 6166, 6169, 6171, 6173, 6175, 6177, 6179,
    6188, 6189, 6190, 6191, 6196, 6197, 6200, 6201, 6205, 6209, 6210, 6211,
    6212, 6213, 6214, 6215, 6216, 6217, 6219, 6220, 6221, 6225, 6226, 6227,
    6229, 6230, 6231, 6232, 6234, 6235, 6236, 6252, 6253, 6254, 6255, 6256,
    6259, 6262, 6265, 6268, 6270, 6272, 6274, 6276, 6278, 6287, 6288, 6289,
    6290, 6295, 6296, 6299, 6300, 6304, 6319, 6325, 6326, 6330, 6331, 6334,
    6335};

// which combinations of 4 points used to generate the 15 equations used in the
// solver
// clang-format off
static const int pt_index[] = {
    0,1,2,3,
    0,1,2,4,
    0,1,2,5,
    0,1,3,4,
    0,1,3,5,
    0,1,4,5,
    0,2,3,4,
    0,2,3,5,
    0,2,4,5,
    0,3,4,5,
    1,2,3,4,
    1,2,3,5,
    1,2,4,5,
    1,3,4,5,
    2,3,4,5,
};
// clang-format on

// Multiplies a deg 2 poly with a deg 2 poly
void mul2_2(double* a, double* b, double* c) {
  c[0] = a[0] * b[0];
  c[1] = a[0] * b[1] + a[1] * b[0];
  c[2] = a[0] * b[3] + a[3] * b[0];
  c[3] = a[0] * b[6] + a[6] * b[0];
  c[4] = a[0] * b[2] + a[1] * b[1] + a[2] * b[0];
  c[5] = a[0] * b[4] + a[1] * b[3] + a[3] * b[1] + a[4] * b[0];
  c[6] = a[0] * b[7] + a[1] * b[6] + a[6] * b[1] + a[7] * b[0];
  c[7] = a[0] * b[5] + a[5] * b[0] + a[3] * b[3];
  c[8] = a[0] * b[8] + a[8] * b[0] + a[3] * b[6] + a[6] * b[3];
  c[9] = a[0] * b[9] + a[9] * b[0] + a[6] * b[6];
  c[10] = a[1] * b[2] + a[2] * b[1];
  c[11] = a[1] * b[4] + a[2] * b[3] + a[3] * b[2] + a[4] * b[1];
  c[12] = a[1] * b[7] + a[2] * b[6] + a[6] * b[2] + a[7] * b[1];
  c[13] = a[1] * b[5] + a[5] * b[1] + a[3] * b[4] + a[4] * b[3];
  c[14] = a[1] * b[8] + a[8] * b[1] + a[3] * b[7] + a[4] * b[6] + a[6] * b[4] +
          a[7] * b[3];
  c[15] = a[1] * b[9] + a[9] * b[1] + a[6] * b[7] + a[7] * b[6];
  c[16] = a[3] * b[5] + a[5] * b[3];
  c[17] = a[3] * b[8] + a[5] * b[6] + a[6] * b[5] + a[8] * b[3];
  c[18] = a[3] * b[9] + a[9] * b[3] + a[6] * b[8] + a[8] * b[6];
  c[19] = a[6] * b[9] + a[9] * b[6];
  c[20] = a[2] * b[2];
  c[21] = a[2] * b[4] + a[4] * b[2];
  c[22] = a[2] * b[7] + a[7] * b[2];
  c[23] = a[2] * b[5] + a[5] * b[2] + a[4] * b[4];
  c[24] = a[2] * b[8] + a[8] * b[2] + a[4] * b[7] + a[7] * b[4];
  c[25] = a[2] * b[9] + a[9] * b[2] + a[7] * b[7];
  c[26] = a[4] * b[5] + a[5] * b[4];
  c[27] = a[4] * b[8] + a[5] * b[7] + a[7] * b[5] + a[8] * b[4];
  c[28] = a[4] * b[9] + a[9] * b[4] + a[7] * b[8] + a[8] * b[7];
  c[29] = a[7] * b[9] + a[9] * b[7];
  c[30] = a[5] * b[5];
  c[31] = a[5] * b[8] + a[8] * b[5];
  c[32] = a[5] * b[9] + a[9] * b[5] + a[8] * b[8];
  c[33] = a[8] * b[9] + a[9] * b[8];
  c[34] = a[9] * b[9];
}

// Multiplies a deg 2 poly with a deg 2 poly and subtracts it from c
void mul2_2m(double* a, double* b, double* c) {
  c[0] -= a[0] * b[0];
  c[1] -= a[0] * b[1] + a[1] * b[0];
  c[2] -= a[0] * b[3] + a[3] * b[0];
  c[3] -= a[0] * b[6] + a[6] * b[0];
  c[4] -= a[0] * b[2] + a[1] * b[1] + a[2] * b[0];
  c[5] -= a[0] * b[4] + a[1] * b[3] + a[3] * b[1] + a[4] * b[0];
  c[6] -= a[0] * b[7] + a[1] * b[6] + a[6] * b[1] + a[7] * b[0];
  c[7] -= a[0] * b[5] + a[5] * b[0] + a[3] * b[3];
  c[8] -= a[0] * b[8] + a[8] * b[0] + a[3] * b[6] + a[6] * b[3];
  c[9] -= a[0] * b[9] + a[9] * b[0] + a[6] * b[6];
  c[10] -= a[1] * b[2] + a[2] * b[1];
  c[11] -= a[1] * b[4] + a[2] * b[3] + a[3] * b[2] + a[4] * b[1];
  c[12] -= a[1] * b[7] + a[2] * b[6] + a[6] * b[2] + a[7] * b[1];
  c[13] -= a[1] * b[5] + a[5] * b[1] + a[3] * b[4] + a[4] * b[3];
  c[14] -= a[1] * b[8] + a[8] * b[1] + a[3] * b[7] + a[4] * b[6] + a[6] * b[4] +
           a[7] * b[3];
  c[15] -= a[1] * b[9] + a[9] * b[1] + a[6] * b[7] + a[7] * b[6];
  c[16] -= a[3] * b[5] + a[5] * b[3];
  c[17] -= a[3] * b[8] + a[5] * b[6] + a[6] * b[5] + a[8] * b[3];
  c[18] -= a[3] * b[9] + a[9] * b[3] + a[6] * b[8] + a[8] * b[6];
  c[19] -= a[6] * b[9] + a[9] * b[6];
  c[20] -= a[2] * b[2];
  c[21] -= a[2] * b[4] + a[4] * b[2];
  c[22] -= a[2] * b[7] + a[7] * b[2];
  c[23] -= a[2] * b[5] + a[5] * b[2] + a[4] * b[4];
  c[24] -= a[2] * b[8] + a[8] * b[2] + a[4] * b[7] + a[7] * b[4];
  c[25] -= a[2] * b[9] + a[9] * b[2] + a[7] * b[7];
  c[26] -= a[4] * b[5] + a[5] * b[4];
  c[27] -= a[4] * b[8] + a[5] * b[7] + a[7] * b[5] + a[8] * b[4];
  c[28] -= a[4] * b[9] + a[9] * b[4] + a[7] * b[8] + a[8] * b[7];
  c[29] -= a[7] * b[9] + a[9] * b[7];
  c[30] -= a[5] * b[5];
  c[31] -= a[5] * b[8] + a[8] * b[5];
  c[32] -= a[5] * b[9] + a[9] * b[5] + a[8] * b[8];
  c[33] -= a[8] * b[9] + a[9] * b[8];
  c[34] -= a[9] * b[9];
}

// Multiplies a deg 2 poly with a deg 4 poly and adds it to c
void mul2_4p(double* a, double* b, double* c) {
  c[0] += a[0] * b[0];
  c[1] += a[0] * b[1] + a[1] * b[0];
  c[2] += a[1] * b[1] + a[2] * b[0] + a[0] * b[4];
  c[3] += a[2] * b[1] + a[1] * b[4] + a[0] * b[10];
  c[4] += a[2] * b[4] + a[1] * b[10] + a[0] * b[20];
  c[5] += a[2] * b[10] + a[1] * b[20];
  c[6] += a[2] * b[20];
  c[7] += a[0] * b[2] + a[3] * b[0];
  c[8] += a[1] * b[2] + a[3] * b[1] + a[4] * b[0] + a[0] * b[5];
  c[9] += a[2] * b[2] + a[4] * b[1] + a[1] * b[5] + a[3] * b[4] + a[0] * b[11];
  c[10] +=
      a[2] * b[5] + a[4] * b[4] + a[1] * b[11] + a[3] * b[10] + a[0] * b[21];
  c[11] += a[2] * b[11] + a[4] * b[10] + a[1] * b[21] + a[3] * b[20];
  c[12] += a[2] * b[21] + a[4] * b[20];
  c[13] += a[3] * b[2] + a[5] * b[0] + a[0] * b[7];
  c[14] += a[4] * b[2] + a[5] * b[1] + a[1] * b[7] + a[3] * b[5] + a[0] * b[13];
  c[15] += a[2] * b[7] + a[4] * b[5] + a[5] * b[4] + a[1] * b[13] +
           a[3] * b[11] + a[0] * b[23];
  c[16] +=
      a[2] * b[13] + a[4] * b[11] + a[5] * b[10] + a[1] * b[23] + a[3] * b[21];
  c[17] += a[2] * b[23] + a[4] * b[21] + a[5] * b[20];
  c[18] += a[5] * b[2] + a[3] * b[7] + a[0] * b[16];
  c[19] +=
      a[5] * b[5] + a[4] * b[7] + a[3] * b[13] + a[1] * b[16] + a[0] * b[26];
  c[20] +=
      a[5] * b[11] + a[4] * b[13] + a[2] * b[16] + a[3] * b[23] + a[1] * b[26];
  c[21] += a[5] * b[21] + a[4] * b[23] + a[2] * b[26];
  c[22] += a[5] * b[7] + a[3] * b[16] + a[0] * b[30];
  c[23] += a[5] * b[13] + a[4] * b[16] + a[3] * b[26] + a[1] * b[30];
  c[24] += a[5] * b[23] + a[4] * b[26] + a[2] * b[30];
  c[25] += a[5] * b[16] + a[3] * b[30];
  c[26] += a[5] * b[26] + a[4] * b[30];
  c[27] += a[5] * b[30];
  c[28] += a[0] * b[3] + a[6] * b[0];
  c[29] += a[1] * b[3] + a[0] * b[6] + a[6] * b[1] + a[7] * b[0];
  c[30] += a[2] * b[3] + a[1] * b[6] + a[7] * b[1] + a[6] * b[4] + a[0] * b[12];
  c[31] +=
      a[2] * b[6] + a[7] * b[4] + a[1] * b[12] + a[6] * b[10] + a[0] * b[22];
  c[32] += a[2] * b[12] + a[7] * b[10] + a[1] * b[22] + a[6] * b[20];
  c[33] += a[2] * b[22] + a[7] * b[20];
  c[34] += a[3] * b[3] + a[0] * b[8] + a[6] * b[2] + a[8] * b[0];
  c[35] += a[4] * b[3] + a[1] * b[8] + a[3] * b[6] + a[7] * b[2] + a[8] * b[1] +
           a[6] * b[5] + a[0] * b[14];
  c[36] += a[2] * b[8] + a[4] * b[6] + a[7] * b[5] + a[8] * b[4] +
           a[1] * b[14] + a[3] * b[12] + a[6] * b[11] + a[0] * b[24];
  c[37] += a[2] * b[14] + a[4] * b[12] + a[7] * b[11] + a[8] * b[10] +
           a[1] * b[24] + a[3] * b[22] + a[6] * b[21];
  c[38] += a[2] * b[24] + a[4] * b[22] + a[7] * b[21] + a[8] * b[20];
  c[39] += a[5] * b[3] + a[8] * b[2] + a[3] * b[8] + a[6] * b[7] + a[0] * b[17];
  c[40] += a[5] * b[6] + a[4] * b[8] + a[8] * b[5] + a[7] * b[7] +
           a[3] * b[14] + a[1] * b[17] + a[6] * b[13] + a[0] * b[27];
  c[41] += a[5] * b[12] + a[4] * b[14] + a[2] * b[17] + a[8] * b[11] +
           a[7] * b[13] + a[3] * b[24] + a[1] * b[27] + a[6] * b[23];
  c[42] +=
      a[5] * b[22] + a[4] * b[24] + a[2] * b[27] + a[8] * b[21] + a[7] * b[23];
  c[43] +=
      a[5] * b[8] + a[8] * b[7] + a[3] * b[17] + a[6] * b[16] + a[0] * b[31];
  c[44] += a[5] * b[14] + a[4] * b[17] + a[8] * b[13] + a[7] * b[16] +
           a[3] * b[27] + a[1] * b[31] + a[6] * b[26];
  c[45] +=
      a[5] * b[24] + a[4] * b[27] + a[8] * b[23] + a[2] * b[31] + a[7] * b[26];
  c[46] += a[5] * b[17] + a[8] * b[16] + a[3] * b[31] + a[6] * b[30];
  c[47] += a[5] * b[27] + a[8] * b[26] + a[4] * b[31] + a[7] * b[30];
  c[48] += a[5] * b[31] + a[8] * b[30];
  c[49] += a[0] * b[9] + a[6] * b[3] + a[9] * b[0];
  c[50] += a[1] * b[9] + a[7] * b[3] + a[9] * b[1] + a[6] * b[6] + a[0] * b[15];
  c[51] += a[2] * b[9] + a[7] * b[6] + a[9] * b[4] + a[1] * b[15] +
           a[6] * b[12] + a[0] * b[25];
  c[52] +=
      a[2] * b[15] + a[7] * b[12] + a[9] * b[10] + a[1] * b[25] + a[6] * b[22];
  c[53] += a[2] * b[25] + a[7] * b[22] + a[9] * b[20];
  c[54] += a[8] * b[3] + a[9] * b[2] + a[3] * b[9] + a[6] * b[8] + a[0] * b[18];
  c[55] += a[4] * b[9] + a[8] * b[6] + a[9] * b[5] + a[7] * b[8] +
           a[3] * b[15] + a[1] * b[18] + a[6] * b[14] + a[0] * b[28];
  c[56] += a[4] * b[15] + a[2] * b[18] + a[8] * b[12] + a[9] * b[11] +
           a[7] * b[14] + a[3] * b[25] + a[1] * b[28] + a[6] * b[24];
  c[57] +=
      a[4] * b[25] + a[2] * b[28] + a[8] * b[22] + a[9] * b[21] + a[7] * b[24];
  c[58] += a[5] * b[9] + a[8] * b[8] + a[9] * b[7] + a[3] * b[18] +
           a[6] * b[17] + a[0] * b[32];
  c[59] += a[5] * b[15] + a[4] * b[18] + a[8] * b[14] + a[9] * b[13] +
           a[7] * b[17] + a[3] * b[28] + a[1] * b[32] + a[6] * b[27];
  c[60] += a[5] * b[25] + a[4] * b[28] + a[8] * b[24] + a[9] * b[23] +
           a[2] * b[32] + a[7] * b[27];
  c[61] +=
      a[5] * b[18] + a[8] * b[17] + a[9] * b[16] + a[3] * b[32] + a[6] * b[31];
  c[62] +=
      a[5] * b[28] + a[8] * b[27] + a[9] * b[26] + a[4] * b[32] + a[7] * b[31];
  c[63] += a[5] * b[32] + a[8] * b[31] + a[9] * b[30];
  c[64] += a[9] * b[3] + a[6] * b[9] + a[0] * b[19];
  c[65] +=
      a[9] * b[6] + a[7] * b[9] + a[1] * b[19] + a[6] * b[15] + a[0] * b[29];
  c[66] +=
      a[2] * b[19] + a[9] * b[12] + a[7] * b[15] + a[1] * b[29] + a[6] * b[25];
  c[67] += a[2] * b[29] + a[9] * b[22] + a[7] * b[25];
  c[68] +=
      a[8] * b[9] + a[9] * b[8] + a[3] * b[19] + a[6] * b[18] + a[0] * b[33];
  c[69] += a[4] * b[19] + a[8] * b[15] + a[9] * b[14] + a[7] * b[18] +
           a[3] * b[29] + a[1] * b[33] + a[6] * b[28];
  c[70] +=
      a[4] * b[29] + a[8] * b[25] + a[9] * b[24] + a[2] * b[33] + a[7] * b[28];
  c[71] +=
      a[5] * b[19] + a[8] * b[18] + a[9] * b[17] + a[3] * b[33] + a[6] * b[32];
  c[72] +=
      a[5] * b[29] + a[8] * b[28] + a[9] * b[27] + a[4] * b[33] + a[7] * b[32];
  c[73] += a[5] * b[33] + a[8] * b[32] + a[9] * b[31];
  c[74] += a[9] * b[9] + a[6] * b[19] + a[0] * b[34];
  c[75] += a[9] * b[15] + a[7] * b[19] + a[1] * b[34] + a[6] * b[29];
  c[76] += a[9] * b[25] + a[2] * b[34] + a[7] * b[29];
  c[77] += a[8] * b[19] + a[9] * b[18] + a[3] * b[34] + a[6] * b[33];
  c[78] += a[8] * b[29] + a[9] * b[28] + a[4] * b[34] + a[7] * b[33];
  c[79] += a[5] * b[34] + a[8] * b[33] + a[9] * b[32];
  c[80] += a[9] * b[19] + a[6] * b[34];
  c[81] += a[9] * b[29] + a[7] * b[34];
  c[82] += a[8] * b[34] + a[9] * b[33];
  c[83] += a[9] * b[34];
}

// Computes the matrix of coefficients for the 15 equations (in 84 monomials)
void setup_coeff_matrix(const std::vector<Eigen::Vector3d>& pp1,
                        const std::vector<Eigen::Vector3d>& xx1,
                        const std::vector<Eigen::Vector3d>& pp2,
                        const std::vector<Eigen::Vector3d>& xx2,
                        Eigen::Matrix<double, 84, 15>* M) {
  Eigen::Matrix<double, 10, 3> F1, F2, F3;

  double* f1 = F1.data();
  double* f2 = F2.data();
  double* f3 = F3.data();

  std::vector<Eigen::Vector3d> qq1(6);
  std::vector<Eigen::Vector3d> qq2(6);
  for (size_t k = 0; k < 6; ++k) {
    qq1[k] = xx1[k].cross(pp1[k]);
    qq2[k] = xx2[k].cross(pp2[k]);
  }
  M->setZero();
  for (size_t eq_k = 0; eq_k < 15; ++eq_k) {
    int i0 = pt_index[4 * eq_k];

    Eigen::Vector3d x1 = xx1[i0];
    Eigen::Vector3d p1 = pp1[i0];
    Eigen::Vector3d x2 = xx2[i0];
    Eigen::Vector3d p2 = pp2[i0];

    // Compute 3x3 matrix where each element is quadratic in cayley parameters
    // F1 is the first column of the matrix, etc..
    // This is for eliminating the translation.
    for (size_t i = 0; i < 3; ++i) {
      int i1 = pt_index[4 * eq_k + i + 1];

      Eigen::Vector3d xp1 = xx1[i1];
      Eigen::Vector3d qp1 = qq1[i1];
      Eigen::Vector3d xp2 = xx2[i1];
      Eigen::Vector3d qp2 = qq2[i1];

      F1(0, i) =
          qp1(0) * xp2(0) + qp2(0) * xp1(0) - qp1(1) * xp2(1) -
          qp2(1) * xp1(1) - qp1(2) * xp2(2) - qp2(2) * xp1(2) +
          xp1(0) * (xp2(2) * (p1(1) + p2(1)) - xp2(1) * (p1(2) + p2(2))) +
          xp1(2) * (xp2(0) * (p1(1) + p2(1)) + xp2(1) * (p1(0) - p2(0))) -
          xp1(1) * (xp2(0) * (p1(2) + p2(2)) + xp2(2) * (p1(0) - p2(0)));
      F1(1, i) =
          2 * qp1(0) * xp2(1) + 2 * qp1(1) * xp2(0) + 2 * qp2(0) * xp1(1) +
          2 * qp2(1) * xp1(0) -
          xp1(0) * (2 * p2(0) * xp2(2) - xp2(0) * (2 * p1(2) + 2 * p2(2))) +
          xp1(1) * (2 * p2(1) * xp2(2) - xp2(1) * (2 * p1(2) + 2 * p2(2))) -
          xp1(2) * (2 * p1(0) * xp2(0) - 2 * p1(1) * xp2(1));
      F1(2, i) =
          qp1(1) * xp2(1) - qp2(0) * xp1(0) - qp1(0) * xp2(0) +
          qp2(1) * xp1(1) - qp1(2) * xp2(2) - qp2(2) * xp1(2) -
          xp1(1) * (xp2(2) * (p1(0) + p2(0)) - xp2(0) * (p1(2) + p2(2))) -
          xp1(2) * (xp2(1) * (p1(0) + p2(0)) + xp2(0) * (p1(1) - p2(1))) +
          xp1(0) * (xp2(1) * (p1(2) + p2(2)) + xp2(2) * (p1(1) - p2(1)));
      F1(3, i) =
          2 * qp1(0) * xp2(2) + 2 * qp1(2) * xp2(0) + 2 * qp2(0) * xp1(2) +
          2 * qp2(2) * xp1(0) +
          xp1(0) * (2 * p2(0) * xp2(1) - xp2(0) * (2 * p1(1) + 2 * p2(1))) -
          xp1(2) * (2 * p2(2) * xp2(1) - xp2(2) * (2 * p1(1) + 2 * p2(1))) +
          xp1(1) * (2 * p1(0) * xp2(0) - 2 * p1(2) * xp2(2));
      F1(4, i) =
          2 * qp1(1) * xp2(2) + 2 * qp1(2) * xp2(1) + 2 * qp2(1) * xp1(2) +
          2 * qp2(2) * xp1(1) -
          xp1(1) * (2 * p2(1) * xp2(0) - xp2(1) * (2 * p1(0) + 2 * p2(0))) +
          xp1(2) * (2 * p2(2) * xp2(0) - xp2(2) * (2 * p1(0) + 2 * p2(0))) -
          xp1(0) * (2 * p1(1) * xp2(1) - 2 * p1(2) * xp2(2));
      F1(5, i) =
          qp1(2) * xp2(2) - qp2(0) * xp1(0) - qp1(1) * xp2(1) -
          qp2(1) * xp1(1) - qp1(0) * xp2(0) + qp2(2) * xp1(2) +
          xp1(2) * (xp2(1) * (p1(0) + p2(0)) - xp2(0) * (p1(1) + p2(1))) +
          xp1(1) * (xp2(2) * (p1(0) + p2(0)) + xp2(0) * (p1(2) - p2(2))) -
          xp1(0) * (xp2(2) * (p1(1) + p2(1)) + xp2(1) * (p1(2) - p2(2)));
      F1(6, i) =
          2 * qp1(1) * xp2(2) - 2 * qp1(2) * xp2(1) - 2 * qp2(1) * xp1(2) +
          2 * qp2(2) * xp1(1) -
          xp1(1) * (2 * p2(1) * xp2(0) + xp2(1) * (2 * p1(0) - 2 * p2(0))) -
          xp1(2) * (2 * p2(2) * xp2(0) + xp2(2) * (2 * p1(0) - 2 * p2(0))) +
          xp1(0) * (2 * p1(1) * xp2(1) + 2 * p1(2) * xp2(2));
      F1(7, i) =
          2 * qp1(2) * xp2(0) - 2 * qp1(0) * xp2(2) + 2 * qp2(0) * xp1(2) -
          2 * qp2(2) * xp1(0) -
          xp1(0) * (2 * p2(0) * xp2(1) + xp2(0) * (2 * p1(1) - 2 * p2(1))) -
          xp1(2) * (2 * p2(2) * xp2(1) + xp2(2) * (2 * p1(1) - 2 * p2(1))) +
          xp1(1) * (2 * p1(0) * xp2(0) + 2 * p1(2) * xp2(2));
      F1(8, i) =
          2 * qp1(0) * xp2(1) - 2 * qp1(1) * xp2(0) - 2 * qp2(0) * xp1(1) +
          2 * qp2(1) * xp1(0) -
          xp1(0) * (2 * p2(0) * xp2(2) + xp2(0) * (2 * p1(2) - 2 * p2(2))) -
          xp1(1) * (2 * p2(1) * xp2(2) + xp2(1) * (2 * p1(2) - 2 * p2(2))) +
          xp1(2) * (2 * p1(0) * xp2(0) + 2 * p1(1) * xp2(1));
      F1(9, i) =
          xp1(1) * (xp2(2) * (p1(0) - p2(0)) - xp2(0) * (p1(2) - p2(2))) -
          xp1(2) * (xp2(1) * (p1(0) - p2(0)) - xp2(0) * (p1(1) - p2(1))) -
          xp1(0) * (xp2(2) * (p1(1) - p2(1)) - xp2(1) * (p1(2) - p2(2))) +
          qp1(0) * xp2(0) + qp2(0) * xp1(0) + qp1(1) * xp2(1) +
          qp2(1) * xp1(1) + qp1(2) * xp2(2) + qp2(2) * xp1(2);
      F2(0, i) = xp1(2) * (x1(0) * xp2(1) + x1(1) * xp2(0)) -
                 xp1(1) * (x1(0) * xp2(2) + x1(2) * xp2(0)) +
                 xp1(0) * (x1(1) * xp2(2) - x1(2) * xp2(1));
      F2(1, i) = 2 * x1(2) * xp1(0) * xp2(0) -
                 xp1(2) * (2 * x1(0) * xp2(0) - 2 * x1(1) * xp2(1)) -
                 2 * x1(2) * xp1(1) * xp2(1);
      F2(2, i) = xp1(0) * (x1(1) * xp2(2) + x1(2) * xp2(1)) -
                 xp1(2) * (x1(0) * xp2(1) + x1(1) * xp2(0)) -
                 xp1(1) * (x1(0) * xp2(2) - x1(2) * xp2(0));
      F2(3, i) = xp1(1) * (2 * x1(0) * xp2(0) - 2 * x1(2) * xp2(2)) -
                 2 * x1(1) * xp1(0) * xp2(0) + 2 * x1(1) * xp1(2) * xp2(2);
      F2(4, i) = 2 * x1(0) * xp1(1) * xp2(1) -
                 xp1(0) * (2 * x1(1) * xp2(1) - 2 * x1(2) * xp2(2)) -
                 2 * x1(0) * xp1(2) * xp2(2);
      F2(5, i) = xp1(1) * (x1(0) * xp2(2) + x1(2) * xp2(0)) +
                 xp1(2) * (x1(0) * xp2(1) - x1(1) * xp2(0)) -
                 xp1(0) * (x1(1) * xp2(2) + x1(2) * xp2(1));
      F2(6, i) = xp1(0) * (2 * x1(1) * xp2(1) + 2 * x1(2) * xp2(2)) -
                 2 * x1(0) * xp1(1) * xp2(1) - 2 * x1(0) * xp1(2) * xp2(2);
      F2(7, i) = xp1(1) * (2 * x1(0) * xp2(0) + 2 * x1(2) * xp2(2)) -
                 2 * x1(1) * xp1(0) * xp2(0) - 2 * x1(1) * xp1(2) * xp2(2);
      F2(8, i) = xp1(2) * (2 * x1(0) * xp2(0) + 2 * x1(1) * xp2(1)) -
                 2 * x1(2) * xp1(0) * xp2(0) - 2 * x1(2) * xp1(1) * xp2(1);
      F2(9, i) = xp1(1) * (x1(0) * xp2(2) - x1(2) * xp2(0)) -
                 xp1(2) * (x1(0) * xp2(1) - x1(1) * xp2(0)) -
                 xp1(0) * (x1(1) * xp2(2) - x1(2) * xp2(1));
      F3(0, i) = xp1(1) * (x2(0) * xp2(2) - x2(2) * xp2(0)) -
                 xp1(2) * (x2(0) * xp2(1) - x2(1) * xp2(0)) +
                 xp1(0) * (x2(1) * xp2(2) - x2(2) * xp2(1));
      F3(1, i) = xp1(1) * (2 * x2(1) * xp2(2) - 2 * x2(2) * xp2(1)) -
                 xp1(0) * (2 * x2(0) * xp2(2) - 2 * x2(2) * xp2(0));
      F3(2, i) = -xp1(2) * (x2(0) * xp2(1) - x2(1) * xp2(0)) -
                 xp1(1) * (x2(0) * xp2(2) - x2(2) * xp2(0)) -
                 xp1(0) * (x2(1) * xp2(2) - x2(2) * xp2(1));
      F3(3, i) = xp1(0) * (2 * x2(0) * xp2(1) - 2 * x2(1) * xp2(0)) +
                 xp1(2) * (2 * x2(1) * xp2(2) - 2 * x2(2) * xp2(1));
      F3(4, i) = xp1(1) * (2 * x2(0) * xp2(1) - 2 * x2(1) * xp2(0)) -
                 xp1(2) * (2 * x2(0) * xp2(2) - 2 * x2(2) * xp2(0));
      F3(5, i) = xp1(2) * (x2(0) * xp2(1) - x2(1) * xp2(0)) +
                 xp1(1) * (x2(0) * xp2(2) - x2(2) * xp2(0)) -
                 xp1(0) * (x2(1) * xp2(2) - x2(2) * xp2(1));
      F3(6, i) = xp1(1) * (2 * x2(0) * xp2(1) - 2 * x2(1) * xp2(0)) +
                 xp1(2) * (2 * x2(0) * xp2(2) - 2 * x2(2) * xp2(0));
      F3(7, i) = xp1(2) * (2 * x2(1) * xp2(2) - 2 * x2(2) * xp2(1)) -
                 xp1(0) * (2 * x2(0) * xp2(1) - 2 * x2(1) * xp2(0));
      F3(8, i) = -xp1(0) * (2 * x2(0) * xp2(2) - 2 * x2(2) * xp2(0)) -
                 xp1(1) * (2 * x2(1) * xp2(2) - 2 * x2(2) * xp2(1));
      F3(9, i) = xp1(2) * (x2(0) * xp2(1) - x2(1) * xp2(0)) -
                 xp1(1) * (x2(0) * xp2(2) - x2(2) * xp2(0)) +
                 xp1(0) * (x2(1) * xp2(2) - x2(2) * xp2(1));
    }

    double p4[35];
    double* c = M->data() + 84 * eq_k;

    // Compute the determinant by expansion along first column
    mul2_2(f2 + 10, f3 + 20, p4);
    mul2_2m(f2 + 20, f3 + 10, p4);
    mul2_4p(f1, p4, c);

    mul2_2(f2 + 20, f3, p4);
    mul2_2m(f2, f3 + 20, p4);
    mul2_4p(f1 + 10, p4, c);

    mul2_2(f2, f3 + 10, p4);
    mul2_2m(f2 + 10, f3, p4);
    mul2_4p(f1 + 20, p4, c);
  }
}

#ifdef USE_FAST_EIGENVECTOR_SOLVER
// Solves for the eigenvector by using structured backsubstitution
// (i.e. substituting the eigenvalue into the eigenvector using the known
// structure to get a reduced linear system)
void fast_eigenvector_solver(double* eigv,
                             int neig,
                             const Eigen::Matrix<double, 64, 64>& AM,
                             Eigen::Matrix<double, 3, 64>& sols) {
  static const int ind[] = {5,  6,  7,  9,  10, 12, 15, 16, 18, 22, 26,
                            27, 29, 33, 38, 43, 44, 47, 51, 56, 63};
  // Truncated action matrix containing non-trivial rows
  Eigen::Matrix<double, 21, 64> AMs;
  double zi[8];

  for (int i = 0; i < 21; i++) {
    AMs.row(i) = AM.row(ind[i]);
  }
  for (int i = 0; i < neig; i++) {
    zi[0] = eigv[i];
    for (int j = 1; j < 8; j++) {
      zi[j] = zi[j - 1] * eigv[i];
    }
    Eigen::Matrix<double, 21, 21> AA;
    AA.col(0) = AMs.col(5);
    AA.col(1) = AMs.col(6);
    AA.col(2) = AMs.col(4) + zi[0] * AMs.col(7);
    AA.col(3) = AMs.col(9);
    AA.col(4) = AMs.col(8) + zi[0] * AMs.col(10);
    AA.col(5) = AMs.col(3) + zi[0] * AMs.col(11) + zi[1] * AMs.col(12);
    AA.col(6) = AMs.col(15);
    AA.col(7) = AMs.col(14) + zi[0] * AMs.col(16);
    AA.col(8) = AMs.col(13) + zi[0] * AMs.col(17) + zi[1] * AMs.col(18);
    AA.col(9) = AMs.col(2) + zi[0] * AMs.col(19) + zi[1] * AMs.col(20) +
                zi[2] * AMs.col(21) + zi[3] * AMs.col(22);
    AA.col(10) = AMs.col(26);
    AA.col(11) = AMs.col(25) + zi[0] * AMs.col(27);
    AA.col(12) = AMs.col(24) + zi[0] * AMs.col(28) + zi[1] * AMs.col(29);
    AA.col(13) = AMs.col(23) + zi[0] * AMs.col(30) + zi[1] * AMs.col(31) +
                 zi[2] * AMs.col(32) + zi[3] * AMs.col(33);
    AA.col(14) = AMs.col(1) + zi[0] * AMs.col(34) + zi[1] * AMs.col(35) +
                 zi[2] * AMs.col(36) + zi[3] * AMs.col(37) +
                 zi[4] * AMs.col(38);
    AA.col(15) = AMs.col(43);
    AA.col(16) = AMs.col(42) + zi[0] * AMs.col(44);
    AA.col(17) = AMs.col(41) + zi[0] * AMs.col(45) + zi[1] * AMs.col(46) +
                 zi[2] * AMs.col(47);
    AA.col(18) = AMs.col(40) + zi[0] * AMs.col(48) + zi[1] * AMs.col(49) +
                 zi[2] * AMs.col(50) + zi[3] * AMs.col(51);
    AA.col(19) = AMs.col(39) + zi[0] * AMs.col(52) + zi[1] * AMs.col(53) +
                 zi[2] * AMs.col(54) + zi[3] * AMs.col(55) +
                 zi[4] * AMs.col(56);
    AA.col(20) = AMs.col(0) + zi[0] * AMs.col(57) + zi[1] * AMs.col(58) +
                 zi[2] * AMs.col(59) + zi[3] * AMs.col(60) +
                 zi[4] * AMs.col(61) + zi[5] * AMs.col(62) +
                 zi[6] * AMs.col(63);
    AA(0, 0) = AA(0, 0) - zi[0];
    AA(1, 1) = AA(1, 1) - zi[0];
    AA(2, 2) = AA(2, 2) - zi[1];
    AA(3, 3) = AA(3, 3) - zi[0];
    AA(4, 4) = AA(4, 4) - zi[1];
    AA(5, 5) = AA(5, 5) - zi[2];
    AA(6, 6) = AA(6, 6) - zi[0];
    AA(7, 7) = AA(7, 7) - zi[1];
    AA(8, 8) = AA(8, 8) - zi[2];
    AA(9, 9) = AA(9, 9) - zi[4];
    AA(10, 10) = AA(10, 10) - zi[0];
    AA(11, 11) = AA(11, 11) - zi[1];
    AA(12, 12) = AA(12, 12) - zi[2];
    AA(13, 13) = AA(13, 13) - zi[4];
    AA(14, 14) = AA(14, 14) - zi[5];
    AA(15, 15) = AA(15, 15) - zi[0];
    AA(16, 16) = AA(16, 16) - zi[1];
    AA(17, 17) = AA(17, 17) - zi[3];
    AA(18, 18) = AA(18, 18) - zi[4];
    AA(19, 19) = AA(19, 19) - zi[5];
    AA(20, 20) = AA(20, 20) - zi[7];

    Eigen::Matrix<double, 20, 1> s =
        AA.leftCols(20).householderQr().solve(-AA.col(20));
    sols(0, i) = s(14);
    sols(1, i) = s(19);
    sols(2, i) = zi[0];
  }
}
#endif

// Performs Newton iterations on the epipolar constraints
void root_refinement(const std::vector<Eigen::Vector3d>& p1,
                     const std::vector<Eigen::Vector3d>& x1,
                     const std::vector<Eigen::Vector3d>& p2,
                     const std::vector<Eigen::Vector3d>& x2,
                     std::vector<CameraPose>* output) {
  Eigen::Matrix<double, 6, 6> J;
  Eigen::Matrix<double, 6, 1> res;
  Eigen::Matrix<double, 6, 1> dp;
  Eigen::Matrix<double, 3, 3> sw;
  sw.setZero();

  std::vector<Eigen::Vector3d> qq1(6), qq2(6);
  for (size_t pt_k = 0; pt_k < 6; ++pt_k) {
    qq1[pt_k] = x1[pt_k].cross(p1[pt_k]);
    qq2[pt_k] = x2[pt_k].cross(p2[pt_k]);
  }

  for (size_t pose_k = 0; pose_k < output->size(); ++pose_k) {
    CameraPose& pose = (*output)[pose_k];

    for (size_t iter = 0; iter < 5; ++iter) {
      // compute residual and jacobian
      for (size_t pt_k = 0; pt_k < 6; ++pt_k) {
        Eigen::Vector3d x2t = x2[pt_k].cross(pose.t);
        Eigen::Vector3d Rx1 = pose.rotate(x1[pt_k]);
        Eigen::Vector3d Rqq1 = pose.rotate(qq1[pt_k]);

        res(pt_k) = (x2t - qq2[pt_k]).dot(Rx1) - x2[pt_k].dot(Rqq1);
        J.block<1, 3>(pt_k, 0) =
            -x2t.cross(Rx1) + qq2[pt_k].cross(Rx1) + x2[pt_k].cross(Rqq1);
        J.block<1, 3>(pt_k, 3) = -x2[pt_k].cross(Rx1);
      }

      if (res.norm() < 1e-12) {
        break;
      }

      dp = J.partialPivLu().solve(res);

      Eigen::Vector3d w = -dp.block<3, 1>(0, 0);
      pose.q = quat_step_pre(pose.q, w);
      pose.t = pose.t - dp.block<3, 1>(3, 0);
    }
  }
}

int gen_relpose_6pt(const std::vector<Eigen::Vector3d>& p1,
                    const std::vector<Eigen::Vector3d>& x1,
                    const std::vector<Eigen::Vector3d>& p2,
                    const std::vector<Eigen::Vector3d>& x2,
                    std::vector<CameraPose>* output) {
  output->clear();

  Eigen::Matrix<double, 84, 15> M;
  setup_coeff_matrix(p1, x1, p2, x2, &M);

  double* coeffs = M.data();
  Eigen::MatrixXd C0 = Eigen::MatrixXd::Zero(99, 99);
  Eigen::MatrixXd C1 = Eigen::MatrixXd::Zero(99, 64);
  for (int i = 0; i < 4655; i++) {
    C0(C0_ind[i]) = coeffs[coeffs0_ind[i]];
  }
  for (int i = 0; i < 3661; i++) {
    C1(C1_ind[i]) = coeffs[coeffs1_ind[i]];
  }
  Eigen::MatrixXd C12 = C0.partialPivLu().solve(C1);

  // Setup action matrix
  Eigen::Matrix<double, 64, 64> AM;
  AM.setZero();
  AM(0, 57) = 1.0;
  AM(1, 34) = 1.0;
  AM(2, 19) = 1.0;
  AM(3, 11) = 1.0;
  AM(4, 7) = 1.0;
  AM.row(5) = -C12.row(78);
  AM.row(6) = -C12.row(79);
  AM.row(7) = -C12.row(80);
  AM(8, 10) = 1.0;
  AM.row(9) = -C12.row(81);
  AM.row(10) = -C12.row(82);
  AM(11, 12) = 1.0;
  AM.row(12) = -C12.row(83);
  AM(13, 17) = 1.0;
  AM(14, 16) = 1.0;
  AM.row(15) = -C12.row(84);
  AM.row(16) = -C12.row(85);
  AM(17, 18) = 1.0;
  AM.row(18) = -C12.row(86);
  AM(19, 20) = 1.0;
  AM(20, 21) = 1.0;
  AM(21, 22) = 1.0;
  AM.row(22) = -C12.row(87);
  AM(23, 30) = 1.0;
  AM(24, 28) = 1.0;
  AM(25, 27) = 1.0;
  AM.row(26) = -C12.row(88);
  AM.row(27) = -C12.row(89);
  AM(28, 29) = 1.0;
  AM.row(29) = -C12.row(90);
  AM(30, 31) = 1.0;
  AM(31, 32) = 1.0;
  AM(32, 33) = 1.0;
  AM.row(33) = -C12.row(91);
  AM(34, 35) = 1.0;
  AM(35, 36) = 1.0;
  AM(36, 37) = 1.0;
  AM(37, 38) = 1.0;
  AM.row(38) = -C12.row(92);
  AM(39, 52) = 1.0;
  AM(40, 48) = 1.0;
  AM(41, 45) = 1.0;
  AM(42, 44) = 1.0;
  AM.row(43) = -C12.row(93);
  AM.row(44) = -C12.row(94);
  AM(45, 46) = 1.0;
  AM(46, 47) = 1.0;
  AM.row(47) = -C12.row(95);
  AM(48, 49) = 1.0;
  AM(49, 50) = 1.0;
  AM(50, 51) = 1.0;
  AM.row(51) = -C12.row(96);
  AM(52, 53) = 1.0;
  AM(53, 54) = 1.0;
  AM(54, 55) = 1.0;
  AM(55, 56) = 1.0;
  AM.row(56) = -C12.row(97);
  AM(57, 58) = 1.0;
  AM(58, 59) = 1.0;
  AM(59, 60) = 1.0;
  AM(60, 61) = 1.0;
  AM(61, 62) = 1.0;
  AM(62, 63) = 1.0;
  AM.row(63) = -C12.row(98);

  Eigen::Matrix<double, 3, 64> sols;
  sols.setZero();
  int n_roots = 0;

#ifdef USE_FAST_EIGENVECTOR_SOLVER
  // Here we only compute eigenvalues and we use the structured backsubsitution
  // to solve for the eigenvectors
  Eigen::EigenSolver<Eigen::Matrix<double, 64, 64>> es(AM, false);
  Eigen::Matrix<std::complex<double>, 64, 1> D = es.eigenvalues();
  double eigv[64];
  for (int i = 0; i < 64; i++) {
    if (std::abs(D(i).imag()) < 1e-6) eigv[n_roots++] = D(i).real();
  }

  fast_eigenvector_solver(eigv, n_roots, AM, sols);
#else
  // Solve eigenvalue problem
  Eigen::EigenSolver<Eigen::Matrix<double, 64, 64>> es(AM);
  Eigen::ArrayXcd D = es.eigenvalues();
  Eigen::ArrayXXcd V = es.eigenvectors();

  // Extract solutions from eigenvectors
  for (size_t k = 0; k < 64; ++k) {
    if (std::abs(D(k).imag()) < 1e-6) {
      sols(0, n_roots) = V(1, k).real() / V(0, k).real();
      sols(1, n_roots) = V(39, k).real() / V(0, k).real();
      sols(2, n_roots) = D(k).real();
      n_roots++;
    }
  }
#endif

  output->clear();
  output->reserve(n_roots);
  for (int sol_k = 0; sol_k < n_roots; ++sol_k) {
    CameraPose pose;
    // From each solution we compute the rotation and solve for the translation

    Eigen::Vector3d w = sols.col(sol_k);
    pose.q << 1.0, w(0), w(1), w(2);
    pose.q.normalize();

    Eigen::Matrix3d R = quat_to_rotmat(pose.q);

    // Solve for the translation
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    A.setZero();
    b.setZero();
    for (size_t i = 0; i < 6; ++i) {
      Eigen::Vector3d u = (R * x1[i]).cross(x2[i]);
      Eigen::Vector3d v = p2[i] - R * p1[i];
      A += u * u.transpose();
      b += u * (u.dot(v));
    }
    pose.t = A.llt().solve(b);

    // Filter solution using cheirality
    bool cheiral_ok = true;
    for (size_t pt_k = 0; pt_k < 6; ++pt_k) {
      if (!check_cheirality(pose, p1[pt_k], x1[pt_k], p2[pt_k], x2[pt_k])) {
        cheiral_ok = false;
        break;
      }
    }
    if (!cheiral_ok) {
      continue;
    }

    output->push_back(pose);
  }
  root_refinement(p1, x1, p2, x2, output);

  return output->size();
}

}  // namespace
}  // namespace poselib

void GR6PEstimator::Estimate(const std::vector<X_t>& points1,
                             const std::vector<Y_t>& points2,
                             std::vector<M_t>* rigs2_from_rigs1) {
  THROW_CHECK_EQ(points1.size(), 6);
  THROW_CHECK_EQ(points2.size(), 6);
  THROW_CHECK(rigs2_from_rigs1 != nullptr);

  rigs2_from_rigs1->clear();

  thread_local std::vector<Eigen::Vector3d> proj_centers1(6);
  thread_local std::vector<Eigen::Vector3d> proj_centers2(6);
  thread_local std::vector<Eigen::Vector3d> rays1(6);
  thread_local std::vector<Eigen::Vector3d> rays2(6);
  for (int i = 0; i < 6; ++i) {
    proj_centers1[i] = points1[i].cam_from_rig.rotation.inverse() *
                       -points1[i].cam_from_rig.translation;
    proj_centers2[i] = points2[i].cam_from_rig.rotation.inverse() *
                       -points2[i].cam_from_rig.translation;
    rays1[i] =
        points1[i].cam_from_rig.rotation.inverse() * points1[i].ray_in_cam;
    rays2[i] =
        points2[i].cam_from_rig.rotation.inverse() * points2[i].ray_in_cam;
  }

  thread_local std::vector<poselib::CameraPose> poses;
  poselib::gen_relpose_6pt(proj_centers1, rays1, proj_centers2, rays2, &poses);

  rigs2_from_rigs1->reserve(poses.size());
  for (const auto& pose : poses) {
    rigs2_from_rigs1->emplace_back(
        Eigen::Quaterniond(pose.q(0), pose.q(1), pose.q(2), pose.q(3)), pose.t);
  }
}

void GR6PEstimator::Residuals(const std::vector<X_t>& points1,
                              const std::vector<Y_t>& points2,
                              const M_t& rig2_from_rig1,
                              std::vector<double>* residuals) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  residuals->resize(points1.size(), 0);
  for (size_t i = 0; i < points1.size(); ++i) {
    const Rigid3d cam2_from_cam1 = points2[i].cam_from_rig * rig2_from_rig1 *
                                   Inverse(points1[i].cam_from_rig);
    const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);
    const Eigen::Vector3d Ex1 =
        E * points1[i].ray_in_cam.hnormalized().homogeneous();
    const Eigen::Vector3d x2 =
        points2[i].ray_in_cam.hnormalized().homogeneous();
    const Eigen::Vector3d Etx2 = E.transpose() * x2;
    const double x2tEx1 = x2.transpose() * Ex1;
    (*residuals)[i] = x2tEx1 * x2tEx1 /
                      (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) +
                       Etx2(1) * Etx2(1));
  }
}

}  // namespace colmap
