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

#include <ostream>
#include <cmath>

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

  Eigen::Vector3d position =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();

  Eigen::Matrix3d position_covariance =
      Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Matrix3d rotation_covariance =
      Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());

  CoordinateSystem coordinate_system = CoordinateSystem::UNDEFINED;

PosePrior() = default;
explicit PosePrior(const Eigen::Vector3d& position_)
    : position(position_) {}

PosePrior(const Eigen::Vector3d& position_, const CoordinateSystem system)
    : position(position_), coordinate_system(system) {}

PosePrior(const Eigen::Vector3d& position_, const Eigen::Matrix3d& pos_cov)
    : position(position_), position_covariance(pos_cov) {}

PosePrior(const Eigen::Vector3d& position_, const Eigen::Matrix3d& pos_cov,
          const CoordinateSystem system)
    : position(position_), position_covariance(pos_cov),
      coordinate_system(system) {}

PosePrior(const Eigen::Vector3d& position_, const Eigen::Quaterniond& rotation_)
    : position(position_), rotation(rotation_) {}

PosePrior(const Eigen::Vector3d& position_, const Eigen::Quaterniond& rotation_,
          const CoordinateSystem system)
    : position(position_), rotation(rotation_), coordinate_system(system) {}

PosePrior(const Eigen::Vector3d& position_, const Eigen::Quaterniond& rotation_,
          const Eigen::Matrix3d& pos_cov)
    : position(position_), rotation(rotation_), position_covariance(pos_cov) {}

PosePrior(const Eigen::Vector3d& position_, const Eigen::Quaterniond& rotation_,
          const Eigen::Matrix3d& pos_cov, const CoordinateSystem system)
    : position(position_), rotation(rotation_), position_covariance(pos_cov),
      coordinate_system(system) {}

PosePrior(const Eigen::Vector3d& position_, const Eigen::Matrix3d& pos_cov,
          const Eigen::Matrix3d& rot_cov)
    : position(position_), position_covariance(pos_cov),
      rotation_covariance(rot_cov) {}

PosePrior(const Eigen::Vector3d& position_, const Eigen::Matrix3d& pos_cov,
          const Eigen::Matrix3d& rot_cov, const CoordinateSystem system)
    : position(position_), position_covariance(pos_cov),
      rotation_covariance(rot_cov), coordinate_system(system) {}

PosePrior(const Eigen::Vector3d& position_, const Eigen::Quaterniond& rotation_,
          const Eigen::Matrix3d& pos_cov, const Eigen::Matrix3d& rot_cov)
    : position(position_), rotation(rotation_),
      position_covariance(pos_cov), rotation_covariance(rot_cov) {}

PosePrior(const Eigen::Vector3d& position_, const Eigen::Quaterniond& rotation_,
          const Eigen::Matrix3d& pos_cov, const Eigen::Matrix3d& rot_cov,
          const CoordinateSystem system)
    : position(position_), rotation(rotation_),
      position_covariance(pos_cov), rotation_covariance(rot_cov),
      coordinate_system(system) {}

  inline bool IsValid() const { return position.allFinite(); }
  inline bool IsRotationValid() const {
    return rotation.coeffs().allFinite();
  }

  inline bool IsCovarianceValid() const {
    return position_covariance.allFinite();
  }
  inline bool IsRotationCovarianceValid() const { 
    return rotation_covariance.allFinite();
  }

  inline bool HasRotation() const {
    return rotation.coeffs().allFinite() && rotation.norm() > 0.0;
  }
  inline bool operator==(const PosePrior& other) const;
  inline bool operator!=(const PosePrior& other) const;
};

std::ostream& operator<<(std::ostream& stream, const PosePrior& prior);

bool PosePrior::operator==(const PosePrior& other) const {
  const double tol = 1e-9;

  auto eq_opt_vec3 = [tol](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    const bool a_set = a.allFinite(), b_set = b.allFinite();
    if (!a_set && !b_set) return true;
    if (a_set && b_set)   return a.isApprox(b, tol);
    return false;
  };

  auto eq_opt_mat3 = [tol](const Eigen::Matrix3d& A, const Eigen::Matrix3d& B) {
    const bool A_set = A.allFinite(), B_set = B.allFinite();
    if (!A_set && !B_set) return true;
    if (A_set && B_set)   return A.isApprox(B, tol);
    return false;
  };

  auto eq_quat = [tol](const Eigen::Quaterniond& a, const Eigen::Quaterniond& b) {
    const bool a_set = a.coeffs().allFinite() && a.norm() > 0.0;
    const bool b_set = b.coeffs().allFinite() && b.norm() > 0.0;
    if (!a_set && !b_set) return true;
    if (!(a_set && b_set)) return false;

    Eigen::Quaterniond qa = a, qb = b;
    qa.normalize(); qb.normalize();

    if (qa.coeffs().isApprox(qb.coeffs(), tol))  return true;
    if (qa.coeffs().isApprox(-qb.coeffs(), tol)) return true;

    double dot = std::abs(qa.dot(qb));
    dot = std::min(1.0, std::max(-1.0, dot));
    return 2.0 * std::acos(dot) <= tol;
  };

  return coordinate_system == other.coordinate_system &&
         eq_opt_vec3(position, other.position) &&
         eq_quat(rotation, other.rotation) &&
         eq_opt_mat3(position_covariance, other.position_covariance) &&
         eq_opt_mat3(rotation_covariance, other.rotation_covariance);
}

bool PosePrior::operator!=(const PosePrior& other) const {
  return !(*this == other);
}

}  // namespace colmap
