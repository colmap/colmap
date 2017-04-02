// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "base/pose.h"

#include <Eigen/Eigenvalues>

#include "base/projection.h"
#include "base/triangulation.h"

namespace colmap {

Eigen::Matrix3d CrossProductMatrix(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d matrix;
  matrix << 0, -vector(2), vector(1), vector(2), 0, -vector(0), -vector(1),
      vector(0), 0;
  return matrix;
}

void RotationMatrixToEulerAngles(const Eigen::Matrix3d& R, double* rx,
                                 double* ry, double* rz) {
  *rx = std::atan2(-R(1, 2), R(2, 2));
  *ry = std::asin(R(0, 2));
  *rz = std::atan2(-R(0, 1), R(0, 0));

  *rx = IsNaN(*rx) ? 0 : *rx;
  *ry = IsNaN(*ry) ? 0 : *ry;
  *rz = IsNaN(*rz) ? 0 : *rz;
}

Eigen::Matrix3d EulerAnglesToRotationMatrix(const double rx, const double ry,
                                            const double rz) {
  const Eigen::Matrix3d Rx =
      Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()).toRotationMatrix();
  const Eigen::Matrix3d Ry =
      Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Matrix3d Rz =
      Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  return Rz * Ry * Rx;
}

Eigen::Vector4d RotationMatrixToQuaternion(const Eigen::Matrix3d& rot_mat) {
  const Eigen::Quaterniond quat(rot_mat);
  return Eigen::Vector4d(quat.w(), quat.x(), quat.y(), quat.z());
}

Eigen::Matrix3d QuaternionToRotationMatrix(const Eigen::Vector4d& qvec) {
  const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(qvec);
  const Eigen::Quaterniond quat(normalized_qvec(0), normalized_qvec(1),
                                normalized_qvec(2), normalized_qvec(3));
  return quat.toRotationMatrix();
}

Eigen::Vector4d NormalizeQuaternion(const Eigen::Vector4d& qvec) {
  const double norm = qvec.norm();
  if (norm == 0) {
    // We do not just use (1, 0, 0, 0) because that is a constant and when used
    // for automatic differentiation that would lead to a zero derivative.
    return Eigen::Vector4d(1.0, qvec(1), qvec(2), qvec(3));
  } else {
    const double inv_norm = 1.0 / norm;
    return inv_norm * qvec;
  }
}

Eigen::Vector4d InvertQuaternion(const Eigen::Vector4d& qvec) {
  const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(qvec);
  return Eigen::Vector4d(normalized_qvec(0), -normalized_qvec(1),
                         -normalized_qvec(2), -normalized_qvec(3));
}

Eigen::Vector4d ConcatenateQuaternions(const Eigen::Vector4d& qvec1,
                                       const Eigen::Vector4d& qvec2) {
  const Eigen::Vector4d normalized_qvec1 = NormalizeQuaternion(qvec1);
  const Eigen::Vector4d normalized_qvec2 = NormalizeQuaternion(qvec2);
  const Eigen::Quaterniond quat1(normalized_qvec1(0), normalized_qvec1(1),
                                 normalized_qvec1(2), normalized_qvec1(3));
  const Eigen::Quaterniond quat2(normalized_qvec2(0), normalized_qvec2(1),
                                 normalized_qvec2(2), normalized_qvec2(3));
  const Eigen::Quaterniond cat_quat = quat2 * quat1;
  return NormalizeQuaternion(
      Eigen::Vector4d(cat_quat.w(), cat_quat.x(), cat_quat.y(), cat_quat.z()));
}

Eigen::Vector3d QuaternionRotatePoint(const Eigen::Vector4d& qvec,
                                      const Eigen::Vector3d& point) {
  const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(qvec);
  const Eigen::Quaterniond quat(normalized_qvec(0), normalized_qvec(1),
                                normalized_qvec(2), normalized_qvec(3));
  return quat * point;
}

Eigen::Vector4d AverageQuaternions(const std::vector<Eigen::Vector4d>& qvecs,
                                   const std::vector<double>& weights) {
  CHECK_EQ(qvecs.size(), weights.size());
  CHECK_GT(qvecs.size(), 0);

  if (qvecs.size() == 1) {
    return qvecs[0];
  }

  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  double weight_sum = 0;

  for (size_t i = 0; i < qvecs.size(); ++i) {
    CHECK_GT(weights[i], 0);
    const Eigen::Vector4d qvec = NormalizeQuaternion(qvecs[i]);
    A += weights[i] * qvec * qvec.transpose();
    weight_sum += weights[i];
  }

  A.array() /= weight_sum;

  const Eigen::Matrix4d eigenvectors =
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>(A).eigenvectors();

  return eigenvectors.col(3);
}

Eigen::Matrix3d RotationFromUnitVectors(const Eigen::Vector3d& vector1,
                                        const Eigen::Vector3d& vector2) {
  const Eigen::Vector3d v1 = vector1.normalized();
  const Eigen::Vector3d v2 = vector2.normalized();
  const Eigen::Vector3d v = v1.cross(v2);
  const Eigen::Matrix3d v_x = CrossProductMatrix(v);
  const double c = v1.dot(v2);
  if (c == -1) {
    return Eigen::Matrix3d::Identity();
  } else {
    return Eigen::Matrix3d::Identity() + v_x + 1 / (1 + c) * (v_x * v_x);
  }
}

Eigen::Vector3d ProjectionCenterFromMatrix(
    const Eigen::Matrix3x4d& proj_matrix) {
  return -proj_matrix.leftCols<3>().transpose() * proj_matrix.rightCols<1>();
}

Eigen::Vector3d ProjectionCenterFromParameters(const Eigen::Vector4d& qvec,
                                               const Eigen::Vector3d& tvec) {
  // Inverse rotation as conjugate quaternion.
  const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(qvec);
  const Eigen::Quaterniond quat(normalized_qvec(0), -normalized_qvec(1),
                                -normalized_qvec(2), -normalized_qvec(3));
  return quat * -tvec;
}

void ComputeRelativePose(const Eigen::Vector4d& qvec1,
                         const Eigen::Vector3d& tvec1,
                         const Eigen::Vector4d& qvec2,
                         const Eigen::Vector3d& tvec2, Eigen::Vector4d* qvec12,
                         Eigen::Vector3d* tvec12) {
  const Eigen::Vector4d inv_qvec1 = InvertQuaternion(qvec1);
  *qvec12 = ConcatenateQuaternions(inv_qvec1, qvec2);
  *tvec12 = tvec2 - QuaternionRotatePoint(*qvec12, tvec1);
}

void ConcatenatePoses(const Eigen::Vector4d& qvec1,
                      const Eigen::Vector3d& tvec1,
                      const Eigen::Vector4d& qvec2,
                      const Eigen::Vector3d& tvec2, Eigen::Vector4d* qvec12,
                      Eigen::Vector3d* tvec12) {
  *qvec12 = ConcatenateQuaternions(qvec1, qvec2);
  *tvec12 = tvec2 + QuaternionRotatePoint(qvec2, tvec1);
}

void InvertPose(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                Eigen::Vector4d* inv_qvec, Eigen::Vector3d* inv_tvec) {
  *inv_qvec = InvertQuaternion(qvec);
  *inv_tvec = -QuaternionRotatePoint(*inv_qvec, tvec);
}

void InterpolatePose(const Eigen::Vector4d& qvec1, const Eigen::Vector3d& tvec1,
                     const Eigen::Vector4d& qvec2, const Eigen::Vector3d& tvec2,
                     const double t, Eigen::Vector4d* qveci,
                     Eigen::Vector3d* tveci) {
  const Eigen::Vector4d normalized_qvec1 = NormalizeQuaternion(qvec1);
  const Eigen::Vector4d normalized_qvec2 = NormalizeQuaternion(qvec2);
  const Eigen::Quaterniond quat1(normalized_qvec1(0), normalized_qvec1(1),
                                 normalized_qvec1(2), normalized_qvec1(3));
  const Eigen::Quaterniond quat2(normalized_qvec2(0), normalized_qvec2(1),
                                 normalized_qvec2(2), normalized_qvec2(3));
  const Eigen::Vector3d tvec12 = tvec2 - tvec1;

  const Eigen::Quaterniond quati = quat1.slerp(t, quat2);

  *qveci = Eigen::Vector4d(quati.w(), quati.x(), quati.y(), quati.z());
  *tveci = tvec1 + tvec12 * t;
}

Eigen::Vector3d CalculateBaseline(const Eigen::Vector4d& qvec1,
                                  const Eigen::Vector3d& tvec1,
                                  const Eigen::Vector4d& qvec2,
                                  const Eigen::Vector3d& tvec2) {
  const Eigen::Vector3d center1 = ProjectionCenterFromParameters(qvec1, tvec1);
  const Eigen::Vector3d center2 = ProjectionCenterFromParameters(qvec2, tvec2);
  return center2 - center1;
}

bool CheckCheirality(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                     const std::vector<Eigen::Vector2d>& points1,
                     const std::vector<Eigen::Vector2d>& points2,
                     std::vector<Eigen::Vector3d>* points3D) {
  CHECK_EQ(points1.size(), points2.size());
  const Eigen::Matrix3x4d proj_matrix1 = Eigen::Matrix3x4d::Identity();
  const Eigen::Matrix3x4d proj_matrix2 = ComposeProjectionMatrix(R, t);
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

}  // namespace colmap
