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
// Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

#include "base/projection.h"

#include "base/pose.h"
#include "util/matrix.h"

namespace colmap {

Eigen::Matrix3x4d ComposeProjectionMatrix(const Eigen::Vector4d& qvec,
                                          const Eigen::Vector3d& tvec) {
  Eigen::Matrix3x4d proj_matrix;
  proj_matrix.leftCols<3>() = QuaternionToRotationMatrix(qvec);
  proj_matrix.rightCols<1>() = tvec;
  return proj_matrix;
}

Eigen::Matrix3x4d ComposeProjectionMatrix(const Eigen::Matrix3d& R,
                                          const Eigen::Vector3d& T) {
  Eigen::Matrix3x4d proj_matrix;
  proj_matrix.leftCols<3>() = R;
  proj_matrix.rightCols<1>() = T;
  return proj_matrix;
}

Eigen::Matrix3x4d InvertProjectionMatrix(const Eigen::Matrix3x4d& proj_matrix) {
  Eigen::Matrix3x4d inv_proj_matrix;
  inv_proj_matrix.leftCols<3>() = proj_matrix.leftCols<3>().transpose();
  inv_proj_matrix.rightCols<1>() = ProjectionCenterFromProjMatrix(proj_matrix);
  return inv_proj_matrix;
}

Eigen::Matrix3d ComputeClosestRotationMatrix(const Eigen::Matrix3d& matrix) {
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R = svd.matrixU() * (svd.matrixV().transpose());
  if (R.determinant() < 0.0) {
    R *= -1.0;
  }
  return R;
}

bool DecomposeProjectionMatrix(const Eigen::Matrix3x4d& P, Eigen::Matrix3d* K,
                               Eigen::Matrix3d* R, Eigen::Vector3d* T) {
  Eigen::Matrix3d RR;
  Eigen::Matrix3d QQ;
  DecomposeMatrixRQ(P.leftCols<3>().eval(), &RR, &QQ);

  *R = ComputeClosestRotationMatrix(QQ);

  const double det_K = RR.determinant();
  if (det_K == 0) {
    return false;
  } else if (det_K > 0) {
    *K = RR;
  } else {
    *K = -RR;
  }

  for (int i = 0; i < 3; ++i) {
    if ((*K)(i, i) < 0.0) {
      K->col(i) = -K->col(i);
      R->row(i) = -R->row(i);
    }
  }

  *T = K->triangularView<Eigen::Upper>().solve(P.col(3));
  if (det_K < 0) {
    *T = -(*T);
  }

  return true;
}

Eigen::Vector2d ProjectPointToImage(const Eigen::Vector3d& point3D,
                                    const Eigen::Matrix3x4d& proj_matrix,
                                    const Camera& camera) {
  const Eigen::Vector3d world_point = proj_matrix * point3D.homogeneous();
  return camera.WorldToImage(world_point.hnormalized());
}

double CalculateReprojectionError(const Eigen::Vector2d& point2D,
                                  const Eigen::Vector3d& point3D,
                                  const Eigen::Matrix3x4d& proj_matrix,
                                  const Camera& camera) {
  const auto image_point = ProjectPointToImage(point3D, proj_matrix, camera);
  return (image_point - point2D).norm();
}

double CalculateAngularError(const Eigen::Vector2d& point2D,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Matrix3x4d& proj_matrix,
                             const Camera& camera) {
  return CalculateAngularError(camera.ImageToWorld(point2D), point3D,
                               proj_matrix);
}

double CalculateAngularError(const Eigen::Vector2d& point2D,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Matrix3x4d& proj_matrix) {
  const Eigen::Vector3d ray1 = point2D.homogeneous();
  const Eigen::Vector3d ray2 = proj_matrix * point3D.homogeneous();
  return std::acos(ray1.normalized().transpose() * ray2.normalized());
}

double CalculateDepth(const Eigen::Matrix3x4d& proj_matrix,
                      const Eigen::Vector3d& point3D) {
  const double d = (proj_matrix.row(2) * point3D.homogeneous()).sum();
  return d * proj_matrix.col(2).norm();
}

bool HasPointPositiveDepth(const Eigen::Matrix3x4d& proj_matrix,
                           const Eigen::Vector3d& point3D) {
  return (proj_matrix(2, 0) * point3D(0) + proj_matrix(2, 1) * point3D(1) +
          proj_matrix(2, 2) * point3D(2) + proj_matrix(2, 3)) >
         std::numeric_limits<double>::epsilon();
}

}  // namespace colmap
