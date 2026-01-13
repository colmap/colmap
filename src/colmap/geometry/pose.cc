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

#include "colmap/geometry/pose.h"

#include "colmap/geometry/triangulation.h"
#include "colmap/math/matrix.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

namespace colmap {

Eigen::VectorXd AverageUnitVectors(const Eigen::MatrixXd& vectors,
                                   const Eigen::VectorXd& weights) {
  THROW_CHECK_GT(vectors.cols(), 0) << "Cannot average empty set of vectors";
  THROW_CHECK(weights.size() == 0 || weights.size() == vectors.cols())
      << "Weights size must match vectors size";

  if (vectors.cols() == 1) {
    return vectors.col(0).normalized();
  }

  // Determine weights: use provided weights or uniform weights.
  const Eigen::VectorXd w =
      weights.size() > 0 ? weights : Eigen::VectorXd::Ones(vectors.cols());
  THROW_CHECK((w.array() > 0).all()) << "Weights must be positive";

  // Normalize all columns and build weighted outer product sum matrix:
  // A = N * diag(w) * N^T / sum(w)
  const Eigen::MatrixXd normalized = vectors.colwise().normalized();
  const Eigen::MatrixXd A =
      normalized * w.asDiagonal() * normalized.transpose() / w.sum();

  // The first singular vector corresponds to the principal direction.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU);
  Eigen::VectorXd average = svd.matrixU().col(0);

  // Ensure consistent sign by aligning with majority of input vectors.
  // Compute dot products of all vectors with the average.
  const Eigen::VectorXd dots = vectors.transpose() * average;
  const Eigen::ArrayXd negative_mask = (dots.array() < 0).cast<double>();
  const double negative_weight = (negative_mask * w.array()).sum();
  if (negative_weight > w.sum() - negative_weight) {
    average = -average;
  }

  return average;
}

Eigen::Vector3d AverageDirections(
    const std::vector<Eigen::Vector3d>& directions,
    const std::vector<double>& weights) {
  Eigen::Matrix3Xd mat(3, directions.size());
  for (size_t i = 0; i < directions.size(); ++i) {
    mat.col(i) = directions[i];
  }
  return AverageUnitVectors(
      mat, Eigen::Map<const Eigen::VectorXd>(weights.data(), weights.size()));
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

bool DecomposeProjectionMatrix(const Eigen::Matrix3x4d& P,
                               Eigen::Matrix3d* K,
                               Eigen::Matrix3d* R,
                               Eigen::Vector3d* T) {
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

Eigen::Vector3d RotationMatrixToAngleAxis(const Eigen::Matrix3d& R) {
  const Eigen::AngleAxisd aa(R);
  return aa.angle() * aa.axis();
}

Eigen::Matrix3d AngleAxisToRotationMatrix(const Eigen::Vector3d& w) {
  const double angle = w.norm();
  if (angle > 1e-12) {
    return Eigen::AngleAxis<double>(angle, w / angle).toRotationMatrix();
  } else {
    // Small angle approximation: I + [w]_x.
    Eigen::Matrix3d R;
    R(0, 0) = 1;
    R(1, 0) = w[2];
    R(2, 0) = -w[1];
    R(0, 1) = -w[2];
    R(1, 1) = 1;
    R(2, 1) = w[0];
    R(0, 2) = w[1];
    R(1, 2) = -w[0];
    R(2, 2) = 1;
    return R;
  }
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
  THROW_CHECK_EQ(quats.size(), weights.size());

  // Convert quaternions to coefficient matrix (each column is a quaternion).
  Eigen::Matrix4Xd qmat(4, quats.size());
  for (size_t i = 0; i < quats.size(); ++i) {
    qmat.col(i) = quats[i].normalized().coeffs();
  }

  // Average using the unified unit vector averaging.
  const Eigen::VectorXd avg = AverageUnitVectors(
      qmat, Eigen::Map<const Eigen::VectorXd>(weights.data(), weights.size()));

  // Convert back to quaternion (Eigen order: x, y, z, w in coeffs).
  return Eigen::Quaterniond(avg(3), avg(0), avg(1), avg(2));
}

Rigid3d InterpolateCameraPoses(const Rigid3d& cam1_from_world,
                               const Rigid3d& cam2_from_world,
                               double t) {
  const Eigen::Vector3d translation12 =
      cam2_from_world.translation - cam1_from_world.translation;
  return Rigid3d(cam1_from_world.rotation.slerp(t, cam2_from_world.rotation),
                 cam1_from_world.translation + translation12 * t);
}

bool CheckCheirality(const Rigid3d& cam2_from_cam1,
                     const std::vector<Eigen::Vector3d>& cam_rays1,
                     const std::vector<Eigen::Vector3d>& cam_rays2,
                     std::vector<Eigen::Vector3d>* points3D) {
  THROW_CHECK_EQ(cam_rays1.size(), cam_rays2.size());
  points3D->clear();
  for (size_t i = 0; i < cam_rays1.size(); ++i) {
    Eigen::Vector3d point3D_in_cam1;
    if (!TriangulateMidPoint(
            cam2_from_cam1, cam_rays1[i], cam_rays2[i], &point3D_in_cam1)) {
      continue;
    }
    points3D->push_back(point3D_in_cam1);
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

Eigen::Matrix3d GravityAlignedRotation(const Eigen::Vector3d& gravity) {
  THROW_CHECK_LT(std::abs(gravity.norm() - 1.0), 1e-6)
      << "Gravity vector must be normalized";

  Eigen::Matrix3d R;
  R.col(1) = gravity;

  // Use Householder QR to find orthonormal basis vectors for the null space.
  Eigen::Matrix3d Q = gravity.householderQr().householderQ();
  Eigen::Matrix<double, 3, 2> N = Q.rightCols(2);
  R.col(0) = N.col(0);
  R.col(2) = N.col(1);

  // Ensure right-handed coordinate system.
  if (R.determinant() < 0) {
    R.col(2) = -R.col(2);
  }

  return R;
}

double YAxisAngleFromRotation(const Eigen::Matrix3d& rotation) {
  return RotationMatrixToAngleAxis(rotation)[1];
}

Eigen::Matrix3d RotationFromYAxisAngle(double angle) {
  return AngleAxisToRotationMatrix(Eigen::Vector3d(0, angle, 0));
}

}  // namespace colmap
