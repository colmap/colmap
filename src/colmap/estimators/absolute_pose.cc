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

#include "colmap/estimators/absolute_pose.h"

#include "colmap/estimators/utils.h"
#include "colmap/math/polynomial.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Geometry>
#include <PoseLib/solvers/p3p.h>
#include <PoseLib/solvers/p4pf.h>

namespace colmap {

P3PEstimator::P3PEstimator(ImgFromCamFunc img_from_cam_func)
    : img_from_cam_func_(std::move(img_from_cam_func)) {}

void P3PEstimator::Estimate(const std::vector<X_t>& points2D,
                            const std::vector<Y_t>& points3D,
                            std::vector<M_t>* cams_from_world) const {
  THROW_CHECK_EQ(points2D.size(), 3);
  THROW_CHECK_EQ(points3D.size(), 3);
  THROW_CHECK_NOTNULL(cams_from_world);

  std::vector<Eigen::Vector3d> rays(3);
  for (int i = 0; i < 3; ++i) {
    rays[i] = points2D[i].camera_ray;
  }

  std::vector<poselib::CameraPose> poses;
  const int num_poses = poselib::p3p(rays, points3D, &poses);

  cams_from_world->resize(num_poses);
  for (int i = 0; i < num_poses; ++i) {
    (*cams_from_world)[i] = poses[i].Rt();
  }
}

void P3PEstimator::Residuals(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             const M_t& cam_from_world,
                             std::vector<double>* residuals) const {
  ComputeSquaredReprojectionError(
      points2D, points3D, cam_from_world, img_from_cam_func_, residuals);
}

void P4PFEstimator::Estimate(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             std::vector<M_t>* models) {
  THROW_CHECK_EQ(points2D.size(), 4);
  THROW_CHECK_EQ(points3D.size(), 4);
  THROW_CHECK_NOTNULL(models);

  std::vector<poselib::CameraPose> poses;
  std::vector<double> focals;
  const int num_poses = poselib::p4pf(
      points2D, points3D, &poses, &focals, /*filter_solutions=*/true);

  models->resize(num_poses);
  for (int i = 0; i < num_poses; ++i) {
    (*models)[i].cam_from_world = poses[i].Rt();
    (*models)[i].focal_length = focals[i];
  }
}

void P4PFEstimator::Residuals(const std::vector<X_t>& points2D,
                              const std::vector<Y_t>& points3D,
                              const M_t& model,
                              std::vector<double>* residuals) {
  const size_t num_points2D = points2D.size();
  CHECK_EQ(num_points2D, points3D.size());
  residuals->resize(num_points2D);
  for (size_t i = 0; i < num_points2D; ++i) {
    const Eigen::Vector3d point3D_in_cam =
        model.cam_from_world * points3D[i].homogeneous();
    // Check if 3D point is in front of camera.
    if (point3D_in_cam.z() > std::numeric_limits<double>::epsilon()) {
      (*residuals)[i] =
          (model.focal_length * point3D_in_cam.hnormalized() - points2D[i])
              .squaredNorm();
    } else {
      (*residuals)[i] = std::numeric_limits<double>::max();
    }
  }
}

EPNPEstimator::EPNPEstimator(ImgFromCamFunc img_from_cam_func)
    : img_from_cam_func_(std::move(img_from_cam_func)) {}

void EPNPEstimator::Estimate(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             std::vector<M_t>* cams_from_world) {
  THROW_CHECK_GE(points2D.size(), 4);
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  THROW_CHECK_NOTNULL(cams_from_world);

  cams_from_world->clear();

  M_t cam_from_world;
  if (!ComputePose(points2D, points3D, &cam_from_world)) {
    return;
  }

  cams_from_world->resize(1);
  (*cams_from_world)[0] = cam_from_world;
}

void EPNPEstimator::Residuals(const std::vector<X_t>& points2D,
                              const std::vector<Y_t>& points3D,
                              const M_t& cam_from_world,
                              std::vector<double>* residuals) const {
  ComputeSquaredReprojectionError(
      points2D, points3D, cam_from_world, img_from_cam_func_, residuals);
}

bool EPNPEstimator::ComputePose(const std::vector<X_t>& points2D,
                                const std::vector<Y_t>& points3D,
                                Eigen::Matrix3x4d* cam_from_world) {
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

  cam_from_world->leftCols<3>() = Rs[best_idx];
  cam_from_world->rightCols<1>() = ts[best_idx];

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
  const Eigen::Vector3d& D = svd.singularValues();
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
  Eigen::Matrix<double, Eigen::Dynamic, 12> M(3 * points2D_->size(), 12);
  for (size_t i = 0; i < points3D_->size(); ++i) {
    const Eigen::Vector3d& ray = (*points2D_)[i].camera_ray;
    for (size_t j = 0; j < 4; ++j) {
      M(3 * i, 3 * j) = 0.0;
      M(3 * i, 3 * j + 1) = -alphas_[i][j] * ray.z();
      M(3 * i, 3 * j + 2) = alphas_[i][j] * ray.y();

      M(3 * i + 1, 3 * j) = alphas_[i][j] * ray.z();
      M(3 * i + 1, 3 * j + 1) = 0.0;
      M(3 * i + 1, 3 * j + 2) = -alphas_[i][j] * ray.x();

      M(3 * i + 2, 3 * j) = -alphas_[i][j] * ray.y();
      M(3 * i + 2, 3 * j + 1) = alphas_[i][j] * ray.x();
      M(3 * i + 2, 3 * j + 2) = 0;
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
  const Eigen::Matrix<double, 4, 1> b4 = svd.solve(rho);

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
  const Eigen::Matrix<double, 3, 1> b3 = svd.solve(rho);

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
  const Eigen::Matrix<double, 5, 1> b5 = svd.solve(rho);

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
                                Eigen::Matrix3d* R,
                                Eigen::Vector3d* t) {
  ComputeCcs(betas, Ut);
  ComputePcs();

  SolveForSign();

  EstimateRT(R, t);

  return ComputeTotalError(*R, *t);
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
  const Eigen::Matrix3d& abt_U = svd.matrixU();
  const Eigen::Matrix3d& abt_V = svd.matrixV();

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

double EPNPEstimator::ComputeTotalError(const Eigen::Matrix3d& R,
                                        const Eigen::Vector3d& t) {
  Eigen::Matrix3x4d cam_from_world;
  cam_from_world.leftCols<3>() = R;
  cam_from_world.rightCols<1>() = t;

  std::vector<double> residuals;
  ComputeSquaredReprojectionError(
      *points2D_, *points3D_, cam_from_world, img_from_cam_func_, &residuals);

  double error = 0.0;
  for (const double residual : residuals) {
    error += std::sqrt(residual);
  }

  return error;
}

void ComputeSquaredReprojectionError(
    const std::vector<Point2DWithRay>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const ImgFromCamFunc& img_from_cam_func,
    std::vector<double>* residuals) {
  const size_t num_points = points2D.size();
  THROW_CHECK_EQ(num_points, points3D.size());
  residuals->resize(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    const Eigen::Vector3d point3D_in_cam =
        cam_from_world * points3D[i].homogeneous();
    const std::optional<Eigen::Vector2d> proj_image_point =
        img_from_cam_func(point3D_in_cam);
    if (proj_image_point) {
      (*residuals)[i] =
          (*proj_image_point - points2D[i].image_point).squaredNorm();
    } else {
      (*residuals)[i] = std::numeric_limits<double>::max();
    }
  }
}

}  // namespace colmap
