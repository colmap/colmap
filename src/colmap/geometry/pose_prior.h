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

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/enum_utils.h"
#include "colmap/util/types.h"

#include <optional>
#include <ostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

struct PosePrior {
 public:
  MAKE_ENUM_CLASS(CoordinateSystem,
                  -1,
                  UNDEFINED,  // = -1
                  WGS84,      // = 0
                  CARTESIAN   // = 1
  );

  constexpr static double kNaN = std::numeric_limits<double>::quiet_NaN();

  // The unique identifier of the pose prior.
  pose_prior_t pose_prior_id = kInvalidPosePriorId;

  // The identifier of the associated sensor for which this prior defines the
  // pose. For example, this can refer to a camera or an IMU sensor.
  data_t corr_data_id = kInvalidDataId;

  // The position of the associated sensor in the world coordinate system.
  Eigen::Vector3d position = Eigen::Vector3d::Constant(kNaN);
  // The position covariance in the Cartesian world coordinate system.
  Eigen::Matrix3d position_covariance = Eigen::Matrix3d::Constant(kNaN);
  // The coordinate system of the position in the world.
  CoordinateSystem coordinate_system = CoordinateSystem::UNDEFINED;

  // The gravity (down) in the sensor coordinate system.
  Eigen::Vector3d gravity = Eigen::Vector3d::Constant(kNaN);

  // The absolute orientation of the sensor as sensor_from_world. For
  // CoordinateSystem::CARTESIAN this is sensor_from_cartesian_world; for
  // CoordinateSystem::WGS84 this is sensor_from_local_enu_at_position.
  Eigen::Quaterniond rotation = Eigen::Quaterniond(kNaN, kNaN, kNaN, kNaN);
  // The rotation covariance in rad^2, right-multiplicative in the same world
  // basis as `rotation`.
  Eigen::Matrix3d rotation_covariance = Eigen::Matrix3d::Constant(kNaN);

  // Epsilon below which a nominally-finite gravity/direction vector is
  // treated as degenerate (zero or near-zero) rather than a usable
  // measurement -- a finite zero vector can reach here from a corrupt
  // database row or a non-archive producer; without this check it would
  // silently normalize to an arbitrary direction wherever it is consumed.
  constexpr static double kMinDirectionNorm = 1e-6;

  inline bool HasPosition() const { return position.allFinite(); }
  inline bool HasPositionCov() const { return position_covariance.allFinite(); }
  inline bool HasGravity() const {
    return gravity.allFinite() && gravity.norm() > kMinDirectionNorm;
  }
  inline bool HasRotation() const {
    return rotation.coeffs().allFinite() && rotation.norm() > 0;
  }
  inline bool HasRotationCov() const { return rotation_covariance.allFinite(); }

  bool operator==(const PosePrior& other) const;
  bool operator!=(const PosePrior& other) const;
};

std::ostream& operator<<(std::ostream& stream, const PosePrior& prior);

// Returns true if `cov` is finite, symmetric, and strictly positive definite
// -- the conditions that cov.inverse().llt() whitening silently assumes. A
// per-row covariance read from the database or an external archive can be
// finite (passing PosePrior::HasPositionCov()) while still being zero,
// singular, or only approximately symmetric, any of which corrupts the
// whitening rather than raising an error. Callers must treat a false return as
// "this row's declared covariance is not usable" and fall back to a declared
// sensor-class/fallback stddev instead of inverting it anyway.
bool IsValidPositionCovariance(const Eigen::Matrix3d& cov);

// Extract gravity vector from EXIF orientation. Returns std::nullopt if not an
// upright orientation (e.g. mirrored).
std::optional<Eigen::Vector3d> GravityFromExifOrientation(int orientation);

// Returns the number of 90 deg counter-clockwise rotations needed to make the
// sensor upright.
int ComputeRot90FromGravity(const Eigen::Vector3d& gravity);

}  // namespace colmap
