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

PosePrior::PosePrior(const Eigen::Vector3d& position)
    : world_from_cam(kInvalidRotation, position) {}

PosePrior::PosePrior(CoordinateSystem system, const Eigen::Vector3d& position)
    : coordinate_system(system), world_from_cam(kInvalidRotation, position) {}

PosePrior::PosePrior(const Eigen::Vector3d& position,
                     const Eigen::Matrix3d& position_covar)
    : world_from_cam(kInvalidRotation, position),
      position_covariance(position_covar) {}

PosePrior::PosePrior(CoordinateSystem system,
                     const Eigen::Vector3d& position,
                     const Eigen::Matrix3d& position_covar)
    : coordinate_system(system),
      world_from_cam(kInvalidRotation, position),
      position_covariance(position_covar) {}

PosePrior::PosePrior(const Eigen::Quaterniond& rotation)
    : coordinate_system(CoordinateSystem::CARTESIAN),
      world_from_cam(rotation, kInvalidTranslation) {}
PosePrior::PosePrior(const Eigen::Quaterniond& rotation,
                     const Eigen::Matrix3d& rotation_covar)
    : coordinate_system(CoordinateSystem::CARTESIAN),
      world_from_cam(rotation, kInvalidTranslation),
      rotation_covariance(rotation_covar) {}

PosePrior::PosePrior(const Eigen::Vector3d& position,
                     const Eigen::Quaterniond& rotation)
    : coordinate_system(CoordinateSystem::CARTESIAN),
      world_from_cam(rotation, position) {}

PosePrior::PosePrior(const Eigen::Vector3d& position,
                     const Eigen::Quaterniond& rotation,
                     const Eigen::Matrix3d& position_covar,
                     const Eigen::Matrix3d& rotation_covar)
    : coordinate_system(CoordinateSystem::CARTESIAN),
      world_from_cam(rotation, position),
      position_covariance(position_covar),
      rotation_covariance(rotation_covar) {}

std::ostream& operator<<(std::ostream& stream, const PosePrior& prior) {
  const static Eigen::IOFormat kVecFmt(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ");

  stream << "PosePrior(\n"
         << "  world_from_cam=[" << prior.world_from_cam << "],\n"
         << "  position_covariance=[" << prior.position_covariance << "],\n"
         << "  rotation_covariance=[" << prior.rotation_covariance << "],\n"
         << "  coordinate_system="
         << PosePrior::CoordinateSystemToString(prior.coordinate_system) << ")";
  return stream;
}

}  // namespace colmap
