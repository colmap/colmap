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

#include "colmap/geometry/pose.h"

#include "colmap/geometry/projection.h"
#include "colmap/geometry/triangulation.h"

#include <Eigen/Eigenvalues>

namespace colmap {

Eigen::Matrix3d CrossProductMatrix(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d matrix;
  matrix << 0, -vector(2), vector(1), vector(2), 0, -vector(0), -vector(1),
      vector(0), 0;
  return matrix;
}

void RotationMatrixToEulerAngles(const Eigen::Matrix3d& R,
                                 double* rx,
                                 double* ry,
                                 double* rz) {
  *rx = std::atan2(R(2, 1), R(2, 2));
  *ry = std::asin(-R(2, 0));
  *rz = std::atan2(R(1, 0), R(0, 0));

  *rx = std::isnan(*rx) ? 0 : *rx;
  *ry = std::isnan(*ry) ? 0 : *ry;
  *rz = std::isnan(*rz) ? 0 : *rz;
}

Eigen::Matrix3d EulerAnglesToRotationMatrix(const double rx,
                                            const double ry,
                                            const double rz) {
  const Eigen::Matrix3d Rx =
      Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()).toRotationMatrix();
  const Eigen::Matrix3d Ry =
      Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Matrix3d Rz =
      Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  return Rz * Ry * Rx;
}

Eigen::Quaterniond AverageQuaternions(
    const std::vector<Eigen::Quaterniond>& quats,
    const std::vector<double>& weights) {
  CHECK_EQ(quats.size(), weights.size());
  CHECK_GT(quats.size(), 0);

  if (quats.size() == 1) {
    return quats[0];
  }

  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  double weight_sum = 0;

  for (size_t i = 0; i < quats.size(); ++i) {
    CHECK_GT(weights[i], 0);
    const Eigen::Vector4d qvec = quats[i].normalized().coeffs();
    A += weights[i] * qvec * qvec.transpose();
    weight_sum += weights[i];
  }

  A.array() /= weight_sum;

  const Eigen::Matrix4d eigenvectors =
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>(A).eigenvectors();

  const Eigen::Vector4d average_qvec = eigenvectors.col(3);

  return Eigen::Quaterniond(
      average_qvec(3), average_qvec(0), average_qvec(1), average_qvec(2));
}

Rigid3d InterpolateCameraPoses(const Rigid3d& cam_from_world1,
                               const Rigid3d& cam_from_world2,
                               double t) {
  const Eigen::Vector3d translation12 =
      cam_from_world2.translation - cam_from_world1.translation;
  return Rigid3d(cam_from_world1.rotation.slerp(t, cam_from_world2.rotation),
                 cam_from_world1.translation + translation12 * t);
}

bool CheckCheirality(const Eigen::Matrix3d& R,
                     const Eigen::Vector3d& t,
                     const std::vector<Eigen::Vector2d>& points1,
                     const std::vector<Eigen::Vector2d>& points2,
                     std::vector<Eigen::Vector3d>* points3D) {
  CHECK_EQ(points1.size(), points2.size());
  const Eigen::Matrix3x4d proj_matrix1 = Eigen::Matrix3x4d::Identity();
  Eigen::Matrix3x4d proj_matrix2;
  proj_matrix2.leftCols<3>() = R;
  proj_matrix2.col(3) = t;
  const double kMinDepth = std::numeric_limits<double>::epsilon();
  const double max_depth = 1000.0f * (R.transpose() * t).norm();
  points3D->clear();
  for (size_t i = 0; i < points1.size(); ++i) {
    const Eigen::Vector3d point3D =
        TriangulatePoint(proj_matrix1, proj_matrix2, points1[i], points2[i]);
    const double depth1 = CalculateDepth(proj_matrix1, point3D);
    if (depth1 > kMinDepth && depth1 < max_depth) {
      const double depth2 = CalculateDepth(proj_matrix2, point3D);
      if (depth2 > kMinDepth && depth2 < max_depth) {
        points3D->push_back(point3D);
      }
    }
  }
  return !points3D->empty();
}

Rigid3d TransformCameraWorld(const Sim3d& new_from_old_world,
                             const Rigid3d& cam_from_world) {
  const Sim3d cam_from_new_world =
      Sim3d(1, cam_from_world.rotation, cam_from_world.translation) *
      Inverse(new_from_old_world);
  return Rigid3d(cam_from_new_world.rotation,
                 cam_from_new_world.translation * new_from_old_world.scale);
}

}  // namespace colmap
