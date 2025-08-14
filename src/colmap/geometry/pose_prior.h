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

// Represent a pose prior defined by a world coordinate system with
// world-from-camera transformation and its associated uncertainty.
//
// For WGS84 system, the position is ordered as longitude, latitude, altitude,
// and rotation is meaningless. Users must check validity via HasValid*() before
// use, the cross-covariance between position and rotation is assumed zero
// internally.
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

  // Prior pose represented as camera position and orientation.
  Rigid3d world_from_cam = Rigid3d(kInvalidRotation, kInvalidTranslation);

  // Position and rotation covariance (3×3).
  Eigen::Matrix3d position_covariance = kInvalidCovariance3x3;
  Eigen::Matrix3d rotation_covariance = kInvalidCovariance3x3;

  PosePrior() = default;

  explicit PosePrior(const Eigen::Vector3d& position);
  PosePrior(CoordinateSystem system, const Eigen::Vector3d& position);
  PosePrior(const Eigen::Vector3d& position,
            const Eigen::Matrix3d& position_covariance);
  PosePrior(CoordinateSystem system,
            const Eigen::Vector3d& position,
            const Eigen::Matrix3d& position_covar);

  explicit PosePrior(const Eigen::Quaterniond& rotation);
  PosePrior(const Eigen::Quaterniond& rotation,
            const Eigen::Matrix3d& rotation_covar);

  PosePrior(const Eigen::Vector3d& position,
            const Eigen::Quaterniond& rotation);
  PosePrior(const Eigen::Vector3d& position,
            const Eigen::Quaterniond& rotation,
            const Eigen::Matrix3d& position_covar,
            const Eigen::Matrix3d& rotation_covar);

  // Get the full 6×6 world_from_cam_covar, cross-covariance is assumed zero.
  inline Eigen::Matrix6d WorldFromCamCovariance() const;
  // Set rotation and position covariance from full 6×6 world_from_cam_covar,
  // cross-covariance is assumed zero.
  inline void SetWorldFromCamCovariance(
      const Eigen::Matrix6d& world_from_cam_covar);

  inline bool HasValidRotation() const;
  inline bool HasValidRotationCovariance() const;

  inline bool HasValidPosition() const;
  inline bool HasValidPositionCovariance() const;

  inline bool HasValidWorldFromCam() const;
  inline bool HasValidWorldFromCamCovariance() const;

  inline bool operator==(const PosePrior& other) const;
  inline bool operator!=(const PosePrior& other) const;
};

std::ostream& operator<<(std::ostream& stream, const PosePrior& prior);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

Eigen::Matrix6d PosePrior::WorldFromCamCovariance() const {
  Eigen::Matrix6d world_from_cam_covar = Eigen::Matrix6d::Zero();
  world_from_cam_covar.block<3, 3>(0, 0) = rotation_covariance;
  world_from_cam_covar.block<3, 3>(3, 3) = position_covariance;
  return world_from_cam_covar;
}

void PosePrior::SetWorldFromCamCovariance(
    const Eigen::Matrix6d& world_from_cam_covar) {
  rotation_covariance = world_from_cam_covar.block<3, 3>(0, 0);
  position_covariance = world_from_cam_covar.block<3, 3>(3, 3);
}

bool PosePrior::HasValidRotation() const {
  return world_from_cam.rotation.coeffs().allFinite();
}

bool PosePrior::HasValidRotationCovariance() const {
  return rotation_covariance.allFinite();
}

bool PosePrior::HasValidPosition() const {
  return world_from_cam.translation.allFinite();
}

bool PosePrior::HasValidPositionCovariance() const {
  return position_covariance.allFinite();
}

inline bool PosePrior::HasValidWorldFromCam() const {
  return HasValidPosition() && HasValidRotation();
}

bool PosePrior::HasValidWorldFromCamCovariance() const {
  return WorldFromCamCovariance().allFinite();
}

bool PosePrior::operator==(const PosePrior& other) const {
  return coordinate_system == other.coordinate_system &&
         world_from_cam == other.world_from_cam &&
         position_covariance == other.position_covariance &&
         rotation_covariance == other.rotation_covariance;
}

bool PosePrior::operator!=(const PosePrior& other) const {
  return !(*this == other);
}
}  // namespace colmap
