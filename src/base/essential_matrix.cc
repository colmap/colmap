// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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

#include "base/essential_matrix.h"

#include <array>

#include "base/pose.h"
#include "estimators/pose.h"

namespace colmap {

void DecomposeEssentialMatrix(const Eigen::Matrix3d& E, Eigen::Matrix3d* R1,
                              Eigen::Matrix3d* R2, Eigen::Vector3d* t) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV().transpose();

  if (U.determinant() < 0) {
    U *= -1;
  }
  if (V.determinant() < 0) {
    V *= -1;
  }

  Eigen::Matrix3d W;
  W << 0, 1, 0, -1, 0, 0, 0, 0, 1;

  *R1 = U * W * V;
  *R2 = U * W.transpose() * V;
  *t = U.col(2).normalized();
}

void PoseFromEssentialMatrix(const Eigen::Matrix3d& E,
                             const std::vector<Eigen::Vector2d>& points1,
                             const std::vector<Eigen::Vector2d>& points2,
                             Eigen::Matrix3d* R, Eigen::Vector3d* t,
                             std::vector<Eigen::Vector3d>* points3D) {
  CHECK_EQ(points1.size(), points2.size());

  Eigen::Matrix3d R1;
  Eigen::Matrix3d R2;
  DecomposeEssentialMatrix(E, &R1, &R2, t);

  // Generate all possible projection matrix combinations.
  const std::array<Eigen::Matrix3d, 4> R_cmbs{{R1, R2, R1, R2}};
  const std::array<Eigen::Vector3d, 4> t_cmbs{{*t, *t, -*t, -*t}};

  points3D->clear();
  for (size_t i = 0; i < R_cmbs.size(); ++i) {
    std::vector<Eigen::Vector3d> points3D_cmb;
    CheckCheirality(R_cmbs[i], t_cmbs[i], points1, points2, &points3D_cmb);
    if (points3D_cmb.size() >= points3D->size()) {
      *R = R_cmbs[i];
      *t = t_cmbs[i];
      *points3D = points3D_cmb;
    }
  }
}

Eigen::Matrix3d EssentialMatrixFromPose(const Eigen::Matrix3d& R,
                                        const Eigen::Vector3d& t) {
  return CrossProductMatrix(t.normalized()) * R;
}

Eigen::Matrix3d EssentialMatrixFromAbsolutePoses(
    const Eigen::Matrix3x4d& proj_matrix1,
    const Eigen::Matrix3x4d& proj_matrix2) {
  const Eigen::Matrix3d R1 = proj_matrix1.leftCols<3>();
  const Eigen::Matrix3d R2 = proj_matrix2.leftCols<3>();
  const Eigen::Vector3d t1 = proj_matrix1.rightCols<1>();
  const Eigen::Vector3d t2 = proj_matrix2.rightCols<1>();

  // Relative transformation between to cameras.
  const Eigen::Matrix3d R = R2 * R1.transpose();
  const Eigen::Vector3d t = t2 - R * t1;

  return EssentialMatrixFromPose(R, t);
}

void FindOptimalImageObservations(const Eigen::Matrix3d& E,
                                  const Eigen::Vector2d& point1,
                                  const Eigen::Vector2d& point2,
                                  Eigen::Vector2d* optimal_point1,
                                  Eigen::Vector2d* optimal_point2) {
  const Eigen::Vector3d& point1h = point1.homogeneous();
  const Eigen::Vector3d& point2h = point2.homogeneous();

  Eigen::Matrix<double, 2, 3> S;
  S << 1, 0, 0, 0, 1, 0;

  // Epipolar lines.
  Eigen::Vector2d n1 = S * E * point2h;
  Eigen::Vector2d n2 = S * E.transpose() * point1h;

  const Eigen::Matrix2d E_tilde = E.block<2, 2>(0, 0);

  const double a = n1.transpose() * E_tilde * n2;
  const double b = (n1.squaredNorm() + n2.squaredNorm()) / 2.0;
  const double c = point1h.transpose() * E * point2h;
  const double d = sqrt(b * b - a * c);
  double lambda = c / (b + d);

  n1 -= E_tilde * lambda * n1;
  n2 -= E_tilde.transpose() * lambda * n2;

  lambda *= (2.0 * d) / (n1.squaredNorm() + n2.squaredNorm());

  *optimal_point1 = (point1h - S.transpose() * lambda * n1).hnormalized();
  *optimal_point2 = (point2h - S.transpose() * lambda * n2).hnormalized();
}

Eigen::Vector3d EpipoleFromEssentialMatrix(const Eigen::Matrix3d& E,
                                           const bool left_image) {
  Eigen::Vector3d e;
  if (left_image) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullV);
    e = svd.matrixV().block<3, 1>(0, 2);
  } else {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E.transpose(), Eigen::ComputeFullV);
    e = svd.matrixV().block<3, 1>(0, 2);
  }
  return e;
}

Eigen::Matrix3d InvertEssentialMatrix(const Eigen::Matrix3d& E) {
  return E.transpose();
}

bool RefineEssentialMatrix(const ceres::Solver::Options& options,
                           const std::vector<Eigen::Vector2d>& points1,
                           const std::vector<Eigen::Vector2d>& points2,
                           const std::vector<char>& inlier_mask,
                           Eigen::Matrix3d* E) {
  CHECK_EQ(points1.size(), points2.size());
  CHECK_EQ(points1.size(), inlier_mask.size());

  // Extract inlier points for decomposing the essential matrix into
  // rotation and translation components.

  size_t num_inliers = 0;
  for (const auto inlier : inlier_mask) {
    if (inlier) {
      num_inliers += 1;
    }
  }

  std::vector<Eigen::Vector2d> inlier_points1(num_inliers);
  std::vector<Eigen::Vector2d> inlier_points2(num_inliers);
  size_t j = 0;
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      inlier_points1[j] = points1[i];
      inlier_points2[j] = points2[i];
      j += 1;
    }
  }

  // Extract relative pose from essential matrix.

  Eigen::Matrix3d R;
  Eigen::Vector3d tvec;
  std::vector<Eigen::Vector3d> points3D;
  PoseFromEssentialMatrix(*E, inlier_points1, inlier_points2, &R, &tvec,
                          &points3D);

  Eigen::Vector4d qvec = RotationMatrixToQuaternion(R);

  if (points3D.size() == 0) {
    return false;
  }

  // Refine essential matrix, use all points so that refinement is able to
  // consider points as inliers that were originally outliers.

  const bool refinement_success =
      RefineRelativePose(options, inlier_points1, inlier_points2, &qvec, &tvec);

  if (!refinement_success) {
    return false;
  }

  // Compose refined essential matrix.
  const Eigen::Matrix3d rot_mat = QuaternionToRotationMatrix(qvec);
  *E = EssentialMatrixFromPose(rot_mat, tvec);

  return true;
}

}  // namespace colmap
