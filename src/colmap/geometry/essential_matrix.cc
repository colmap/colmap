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

#include "colmap/geometry/essential_matrix.h"

#include "colmap/geometry/pose.h"

#include <array>

#include <glog/logging.h>

namespace colmap {

void DecomposeEssentialMatrix(const Eigen::Matrix3d& E,
                              Eigen::Matrix3d* R1,
                              Eigen::Matrix3d* R2,
                              Eigen::Vector3d* t) {
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
                             Eigen::Matrix3d* R,
                             Eigen::Vector3d* t,
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

Eigen::Matrix3d EssentialMatrixFromPose(const Rigid3d& cam2_from_cam1) {
  return CrossProductMatrix(cam2_from_cam1.translation.normalized()) *
         cam2_from_cam1.rotation.toRotationMatrix();
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
  const double d = std::sqrt(b * b - a * c);
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

}  // namespace colmap
