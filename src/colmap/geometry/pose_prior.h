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

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/enum_utils.h"
#include "colmap/util/types.h"

#include <ostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

// Represents a pose prior defined by a world coordinate system and its relative
// transformation to the camera.
//
// Commonly used in absolute pose estimation and global bundle adjustment,
// leveraging prior pose information from sources such as GPS, IMU,
// previously reconstructed camera poses with covariances, or other systems
// providing pose estimates with uncertainty.
//
// Internally, the pose is stored as a rigid transformation (rotation +
// translation). When SetRotation() is called, the translation is adjusted to
// preserve the camera center. Likewise, SetPosition() updates the translation
// without altering the rotation.
//
// Special cases:
// - If the rotation is invalid, the translation is set to -position.
// - For WGS84 systems, the position is ordered as longitude, latitude, height,
//   and rotation is meaningless.
//
// The caller must check pose validity using HasValid*() before use.
struct PosePrior {
 public:
  static const Eigen::Vector3d kInvalidTranslation;
  static const Eigen::Quaterniond kInvalidRotation;
  static const Eigen::Matrix3d kInvalidCovariance3x3;
  static const Eigen::Matrix6d kInvalidCovariance6x6;

  MAKE_ENUM_CLASS(CoordinateSystem,
                  -1,
                  UNDEFINED,  // = -1
                  WGS84,      // = 0
                  CARTESIAN   // = 1
  );

  CoordinateSystem coordinate_system = CoordinateSystem::UNDEFINED;

  Rigid3d cam_from_world = Rigid3d(kInvalidRotation, kInvalidTranslation);

  PosePrior() = default;

  explicit PosePrior(const Eigen::Vector3d& position)
      : cam_from_world(kInvalidRotation, -position) {}

  PosePrior(CoordinateSystem system, const Eigen::Vector3d& position)
      : coordinate_system(system),
        cam_from_world(kInvalidRotation, -position) {}

  PosePrior(const Eigen::Vector3d& position,
            const Eigen::Matrix3d& position_covariance)
      : cam_from_world(kInvalidRotation, -position),
        position_covariance(position_covariance) {}

  PosePrior(CoordinateSystem system,
            const Eigen::Vector3d& position,
            const Eigen::Matrix3d& position_covariance)
      : coordinate_system(system),
        cam_from_world(kInvalidRotation, -position),
        position_covariance(position_covariance) {}

  PosePrior(const Eigen::Vector3d& position, const Eigen::Quaterniond& rotation)
      : coordinate_system(CoordinateSystem::CARTESIAN),
        cam_from_world(rotation, -(rotation * position)) {}

  PosePrior(const Eigen::Vector3d& position,
            const Eigen::Quaterniond& rotation,
            const Eigen::Matrix3d& position_covariance,
            const Eigen::Matrix3d& rotation_covariance)
      : coordinate_system(CoordinateSystem::CARTESIAN),
        cam_from_world(rotation, -(rotation * position)),
        position_covariance(position_covariance),
        rotation_covariance(rotation_covariance) {}

  explicit PosePrior(const Rigid3d& cam_from_world)
      : coordinate_system(CoordinateSystem::CARTESIAN),
        cam_from_world(cam_from_world) {}
  PosePrior(const Rigid3d& cam_from_world,
            const Eigen::Matrix3d& position_covariance,
            const Eigen::Matrix3d& rotation_covariance)
      : coordinate_system(CoordinateSystem::CARTESIAN),
        cam_from_world(cam_from_world),
        position_covariance(position_covariance),
        rotation_covariance(rotation_covariance) {}

  inline Eigen::Vector3d Translation() const {
    return cam_from_world.translation;
  }
  inline void SetTranslation(const Eigen::Vector3d& translation) {
    cam_from_world.translation = translation;
  }

  inline Eigen::Quaterniond Rotation() const { return cam_from_world.rotation; }
  inline void SetRotation(const Eigen::Quaterniond& rotation) {
    Eigen::Vector3d position = Position();
    cam_from_world.rotation = rotation;
    if (HasValidRotation()) {
      cam_from_world.translation = -(cam_from_world.rotation * position);
    }
  }
  // Use Eigen's [x, y, z, w] quaternion order (not [w, x, y, z]).
  inline void SetRotationFromCoeffs(const Eigen::Vector4d& coeffs) {
    Eigen::Vector3d position = Position();
    cam_from_world.rotation.coeffs() = coeffs;
    cam_from_world.rotation.normalize();
    if (HasValidRotation()) {
      cam_from_world.translation = -(cam_from_world.rotation * position);
    }
  }

  inline Eigen::Vector3d Position() const {
    if (HasValidRotation()) {
      return -(cam_from_world.rotation.inverse() * cam_from_world.translation);
    }
    // If rotation is invalid, assume position is -t
    return -cam_from_world.translation;
  }
  inline void SetPosition(const Eigen::Vector3d& position) {
    cam_from_world.translation =
        HasValidRotation() ? -(cam_from_world.rotation * position) : -position;
  }

  // Get the translation covariance
  //
  // Given t = -R * C, apply first-order approximation:
  //   \Sigma_t = J * Sigma_{se3} * J^T
  // where J is the Jacobian of translation t:
  //   J = [ -R | -R * skew(-R * C) ]
  Eigen::Matrix3d TranslationCovariance() const;

  // Set the internal position covariance from translation covariance
  //
  // Given t = -R * C, apply first-order approximation:
  //   \Sigma_t = J * Sigma_{se3} * J^T
  // Solve for \Sigma_t by rearranging:
  //   \Sigma_C = R^T * (\Sigma_t - rot_term) * R
  // where:
  //   rot_term = skew(-R * C) * \Sigma_R * skew(-R * C)^T
  void SetTranslationCovariance(const Eigen::Matrix3d& cov_translation);

  inline Eigen::Matrix3d RotationCovariance() const {
    return rotation_covariance;
  }
  inline void SetRotationCovariance(const Eigen::Matrix3d& covariance) {
    rotation_covariance = covariance;
  }

  inline Eigen::Matrix3d PositionCovariance() const {
    return position_covariance;
  }

  inline void SetPositionCovariance(const Eigen::Matrix3d& covariance) {
    position_covariance = covariance;
  }

  // Get full 6x6 pose covariance (rotation + translation).
  //
  // Full covariance matrix structure:
  //   [     \Sigma_R,                           \Sigma_R * J^T ]
  //   [ J * \Sigma_R,  J * \Sigma_R * J^T + R * \Sigma_C * R^T ]
  // where J = skew(-R * C).
  Eigen::Matrix<double, 6, 6> PoseCovariance() const;

  // Set rotation_covariance and position_covariance from full 6x6 pose
  // covariance matrix.
  //
  // Input covariance matrix structure:
  //   [     \Sigma_R, \Sigma_{R,t} ]
  //   [ \Sigma_{t,R},     \Sigma_t ]
  // The position covariance is extracted by removing rotation-induced terms:
  //   \Sigma_C = R^T * (\Sigma_t - J * \Sigma_R * J^T) * R,
  // where J = skew(-R * C).
  void SetPoseCovariance(const Eigen::Matrix<double, 6, 6>& pose_covariance);

  inline bool HasValidTranslation() const { return Translation().allFinite(); }
  inline bool HasValidTranslationCovariance() const {
    return TranslationCovariance().allFinite();
  }

  inline bool HasValidRotation() const {
    return cam_from_world.rotation.coeffs().allFinite();
  }
  inline bool HasValidRotationCovariance() const {
    return RotationCovariance().allFinite();
  }

  inline bool HasValidPosition() const { return Position().allFinite(); }
  inline bool HasValidPositionCovariance() const {
    return PositionCovariance().allFinite();
  }

  inline bool HasValidPose() const {
    return HasValidTranslation() && HasValidRotation();
  }
  inline bool HasValidPoseCovariance() const {
    return PoseCovariance().allFinite();
  }

  inline bool operator==(const PosePrior& other) const;
  inline bool operator!=(const PosePrior& other) const;

 private:
  // Position uncertainty covariance, assumed independent of rotation.
  Eigen::Matrix3d position_covariance = kInvalidCovariance3x3;

  // Rotation uncertainty covariance (axis-angle), assumed independent of
  // position.
  Eigen::Matrix3d rotation_covariance = kInvalidCovariance3x3;
};

std::ostream& operator<<(std::ostream& stream, const PosePrior& prior);

bool PosePrior::operator==(const PosePrior& other) const {
  return coordinate_system == other.coordinate_system &&
         cam_from_world == other.cam_from_world &&
         position_covariance == other.position_covariance &&
         rotation_covariance == other.rotation_covariance;
}

bool PosePrior::operator!=(const PosePrior& other) const {
  return !(*this == other);
}

}  // namespace colmap
