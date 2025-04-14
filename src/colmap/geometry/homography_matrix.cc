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

#include "colmap/geometry/homography_matrix.h"

#include "colmap/geometry/pose.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/math/math.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <array>
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace colmap {
namespace {

double ComputeOppositeOfMinor(const Eigen::Matrix3d& matrix,
                              const int row,
                              const int col) {
  const int col1 = col == 0 ? 1 : 0;
  const int col2 = col == 2 ? 1 : 2;
  const int row1 = row == 0 ? 1 : 0;
  const int row2 = row == 2 ? 1 : 2;
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
                               std::vector<Rigid3d>* cams2_from_cams1,
                               std::vector<Eigen::Vector3d>* normals) {
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
  constexpr double kMinInfinityNorm = 1e-3;
  if (S.lpNorm<Eigen::Infinity>() < kMinInfinityNorm) {
    *cams2_from_cams1 = {
        Rigid3d(Eigen::Quaterniond(H_normalized), Eigen::Vector3d::Zero())};
    *normals = {Eigen::Vector3d::Zero()};
    return;
  }

  const double M00 = ComputeOppositeOfMinor(S, 0, 0);
  const double M11 = ComputeOppositeOfMinor(S, 1, 1);
  const double M22 = ComputeOppositeOfMinor(S, 2, 2);

  const double rtM00 = std::sqrt(std::max(M00, 0.));
  const double rtM11 = std::sqrt(std::max(M11, 0.));
  const double rtM22 = std::sqrt(std::max(M22, 0.));

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
  const double v =
      2.0 * std::sqrt(std::max(1.0 + traceS - M00 - M11 - M22, 0.));

  const double ESii = SignOfNumber(S(idx, idx));
  const double r_2 = 2 + traceS + v;
  const double nt_2 = 2 + traceS - v;

  const double r = std::sqrt(std::max(r_2, 0.));
  const double n_t = std::sqrt(std::max(nt_2, 0.));

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

  *cams2_from_cams1 = {Rigid3d(Eigen::Quaterniond(R1), t1),
                       Rigid3d(Eigen::Quaterniond(R1), -t1),
                       Rigid3d(Eigen::Quaterniond(R2), t2),
                       Rigid3d(Eigen::Quaterniond(R2), -t2)};
  *normals = {-n1, n1, -n2, n2};
}

namespace {

double CheckCheiralityAndReprojErrorSum(
    const Rigid3d& cam2_from_cam1,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    std::vector<Eigen::Vector3d>* points3D) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  const Eigen::Matrix3x4d cam1_from_world = Eigen::Matrix3x4d::Identity();
  const Eigen::Matrix3x4d cam2_from_world = cam2_from_cam1.ToMatrix();
  constexpr double kMinDepth = std::numeric_limits<double>::epsilon();
  const double max_depth = 1000.0 * cam2_from_cam1.translation.norm();
  double reproj_residual_sum = 0;
  points3D->clear();
  for (size_t i = 0; i < points1.size(); ++i) {
    Eigen::Vector3d point3D;
    if (!TriangulatePoint(cam1_from_world,
                          cam2_from_world,
                          points1[i],
                          points2[i],
                          &point3D)) {
      continue;
    }
    const Eigen::Vector3d point3D_in_cam1 =
        cam1_from_world * point3D.homogeneous();
    if (point3D_in_cam1.z() < kMinDepth || point3D_in_cam1.z() > max_depth) {
      continue;
    }
    const Eigen::Vector3d point3D_in_cam2 =
        cam2_from_world * point3D.homogeneous();
    if (point3D_in_cam2.z() < kMinDepth || point3D_in_cam2.z() > max_depth) {
      continue;
    }
    const double error1 =
        (points1[i] - point3D_in_cam1.hnormalized()).squaredNorm();
    const double error2 =
        (points2[i] - point3D_in_cam2.hnormalized()).squaredNorm();
    reproj_residual_sum += error1 + error2;
    points3D->push_back(point3D);
  }
  return reproj_residual_sum;
}

}  // namespace

void PoseFromHomographyMatrix(const Eigen::Matrix3d& H,
                              const Eigen::Matrix3d& K1,
                              const Eigen::Matrix3d& K2,
                              const std::vector<Eigen::Vector2d>& points1,
                              const std::vector<Eigen::Vector2d>& points2,
                              Rigid3d* cam2_from_cam1,
                              Eigen::Vector3d* normal,
                              std::vector<Eigen::Vector3d>* points3D) {
  THROW_CHECK_EQ(points1.size(), points2.size());

  std::vector<Rigid3d> cams2_from_cams1;
  std::vector<Eigen::Vector3d> normals;
  DecomposeHomographyMatrix(H, K1, K2, &cams2_from_cams1, &normals);
  THROW_CHECK_EQ(cams2_from_cams1.size(), normals.size());

  points3D->clear();
  std::vector<Eigen::Vector3d> tentative_points3D;
  double best_reproj_residual_sum = std::numeric_limits<double>::max();
  for (size_t i = 0; i < cams2_from_cams1.size(); ++i) {
    // Note that we can typically eliminate 2 of the 4 solutions using the
    // cheirality check. We can then typically narrow it down to 1 solution by
    // picking the solution with minimal overall squared reprojection error.
    // There is no principled reasoning for why choosing the sum of squared or
    // non-squared reprojection errors other than avoid sqrt for efficiency and
    // consistency with the RANSAC cost function.
    const double reproj_residual_sum = CheckCheiralityAndReprojErrorSum(
        cams2_from_cams1[i], points1, points2, &tentative_points3D);
    if (tentative_points3D.size() > points3D->size() ||
        (tentative_points3D.size() == points3D->size() &&
         reproj_residual_sum < best_reproj_residual_sum)) {
      best_reproj_residual_sum = reproj_residual_sum;
      *cam2_from_cam1 = cams2_from_cams1[i];
      *normal = normals[i];
      std::swap(*points3D, tentative_points3D);
    }
  }
}

Eigen::Matrix3d HomographyMatrixFromPose(const Eigen::Matrix3d& K1,
                                         const Eigen::Matrix3d& K2,
                                         const Eigen::Matrix3d& R,
                                         const Eigen::Vector3d& t,
                                         const Eigen::Vector3d& n,
                                         const double d) {
  THROW_CHECK_GT(d, 0);
  return K2 * (R - t * n.normalized().transpose() / d) * K1.inverse();
}

}  // namespace colmap
