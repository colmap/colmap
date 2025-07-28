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

#include "colmap/geometry/pose_prior.h"

namespace colmap {
namespace {

inline Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d& v) {
  Eigen::Matrix3d skew;
  // clang-format off
  skew <<     0, -v.z(),  v.y(), 
          v.z(),      0, -v.x(), 
         -v.y(),  v.x(),      0;
  return skew;
  // clang-format on
}
}  // namespace

const Eigen::Vector3d PosePrior::kInvalidTranslation =
    Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
const Eigen::Quaterniond PosePrior::kInvalidRotation =
    Eigen::Quaterniond(std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN());
const Eigen::Matrix3d PosePrior::kInvalidCovariance3x3 =
    Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
const Eigen::Matrix6d PosePrior::kInvalidCovariance6x6 =
    Eigen::Matrix6d::Constant(std::numeric_limits<double>::quiet_NaN());

Eigen::Matrix3d PosePrior::TranslationCovariance() const {
  const Eigen::Matrix3d R = cam_from_world.rotation.toRotationMatrix();

  const Eigen::Matrix3d J_position = -R;
  const Eigen::Matrix3d J_rotation = SkewSymmetric(-R * Position());

  Eigen::Matrix<double, 3, 6> J;
  J.leftCols<3>() = J_position;
  J.rightCols<3>() = J_rotation;

  Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();
  covariance.topLeftCorner<3, 3>() = position_covariance;
  covariance.bottomRightCorner<3, 3>() = rotation_covariance;

  return J * covariance * J.transpose();
}

void PosePrior::SetTranslationCovariance(
    const Eigen::Matrix3d& cov_translation) {
  const Eigen::Matrix3d R = cam_from_world.rotation.toRotationMatrix();

  const Eigen::Matrix3d J_rotation = SkewSymmetric(-R * Position());
  const Eigen::Matrix3d rotation_term =
      J_rotation * rotation_covariance * J_rotation.transpose();

  Eigen::Matrix3d corrected_cov =
      R.transpose() * (cov_translation - rotation_term) * R;

  // Ensure symmetry
  position_covariance = 0.5 * (corrected_cov + corrected_cov.transpose());
}

Eigen::Matrix<double, 6, 6> PosePrior::PoseCovariance() const {
  const Eigen::Matrix3d R = cam_from_world.rotation.toRotationMatrix();

  const Eigen::Matrix3d J_rotation = SkewSymmetric(-R * Position());

  Eigen::Matrix<double, 6, 6> pose_covariance =
      Eigen::Matrix<double, 6, 6>::Zero();

  pose_covariance.block<3, 3>(0, 0) = rotation_covariance;

  pose_covariance.block<3, 3>(3, 3) =
      J_rotation * rotation_covariance * J_rotation.transpose() +
      R * position_covariance * R.transpose();

  pose_covariance.block<3, 3>(3, 0) = J_rotation * rotation_covariance;
  pose_covariance.block<3, 3>(0, 3) =
      pose_covariance.block<3, 3>(3, 0).transpose();

  return pose_covariance;
}

void PosePrior::SetPoseCovariance(
    const Eigen::Matrix<double, 6, 6>& pose_covariance) {
  const Eigen::Matrix3d R = cam_from_world.rotation.toRotationMatrix();

  const Eigen::Matrix3d rotation_cov = pose_covariance.block<3, 3>(0, 0);
  const Eigen::Matrix3d translation_cov = pose_covariance.block<3, 3>(3, 3);

  const Eigen::Matrix3d J_rotation = SkewSymmetric(-R * Position());

  rotation_covariance = rotation_cov;
  position_covariance =
      R.transpose() *
      (translation_cov - J_rotation * rotation_cov * J_rotation.transpose()) *
      R;
}

std::ostream& operator<<(std::ostream& stream, const PosePrior& prior) {
  const static Eigen::IOFormat kVecFmt(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ");

  stream << "PosePrior(\n"
         << "  position=[" << prior.Position().format(kVecFmt) << "],\n"
         << "  position_covariance=["
         << prior.PositionCovariance().format(kVecFmt) << "],\n"
         << "  rotation=["
         << prior.Rotation().coeffs().transpose().format(kVecFmt)
         << "],  // [x, y, z, w]\n"
         << "  rotation_covariance=["
         << prior.RotationCovariance().format(kVecFmt) << "],\n"
         << "  coordinate_system="
         << PosePrior::CoordinateSystemToString(prior.coordinate_system) << "\n"
         << ")";
  return stream;
}

}  // namespace colmap
