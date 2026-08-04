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

#include "colmap/geometry/gps.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/util/eigen_alignment.h"

#include <optional>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// The single shared WGS84 -> local-ENU tangent frame for a set of pose priors.
//
// This exists because the mapper and the report each used to derive an origin
// independently. They agreed on the datum but not on the rule: the mapper falls
// back to horizontal-only rows and a median altitude when no row carries a full
// position, while the report required full positions. On any capture where some
// rows lack altitude the two picked *different* origins, and every transform
// published against one of them is silently offset from the other. A downstream
// consumer composing the report's `geometry_from_ecef` onto geometry the mapper
// solved would land in the wrong place with nothing to indicate why.
//
// One rule, computed once, used by both.
class PosePriorEnuFrame {
 public:
  // Derives the frame from every prior that carries a usable WGS84 position.
  //
  // Origin: the geometric median of the ECEF positions, converted back to
  // WGS84, with the median altitude of the rows that declare one. The median is
  // used rather than a mean or the first row so a single gross GPS outlier --
  // which is a valid archive row, rejected later by robust fitting rather than
  // at import -- cannot move the frame every transform is expressed against.
  //
  // Returns nullopt when no prior carries a usable WGS84 position, in which
  // case there is no Earth frame and no caller may invent one.
  static std::optional<PosePriorEnuFrame> Derive(
      const std::vector<PosePrior>& pose_priors);

  // Whether a prior contributes to the frame and can be transformed by it:
  // it declares WGS84 and carries a finite latitude and longitude. Altitude
  // may be absent. Callers must use this rather than their own test, so that
  // the set of rows defining the origin is the same set the origin applies to.
  static bool IsUsable(const PosePrior& pose_prior);

  // Origin as (latitude_deg, longitude_deg, ellipsoidal_altitude_m).
  const Eigen::Vector3d& OriginWgs84() const { return origin_wgs84_; }
  const Eigen::Vector3d& OriginEcef() const { return origin_ecef_; }

  // True when at least one prior carried a finite altitude, i.e. the origin's
  // altitude is a real measurement rather than the 0.0 placeholder. Callers
  // that publish an ellipsoidal height must not present a placeholder as one.
  bool HasRealAltitude() const { return has_real_altitude_; }

  const Eigen::Matrix3d& EnuFromEcef() const { return enu_from_ecef_; }
  const Eigen::Matrix3d& EcefFromEnu() const { return ecef_from_enu_; }

  // Position of a WGS84 prior in this shared ENU frame.
  Eigen::Vector3d PositionInEnu(const PosePrior& pose_prior) const;

  // Rotation from the local ENU frame at `wgs84_position`'s latitude and
  // longitude into this shared frame.
  //
  // Every direction a prior declares -- its uncertainty axes, its measured
  // down, its azimuth from true north -- is defined against the tangent plane
  // at that row's own position, not at the origin. East, north and up all turn
  // as you move across the Earth, so a single constant direction reused for
  // every row is only correct at the origin and drifts from there outward.
  // Callers that consume a direction from a prior must route it through this.
  Eigen::Matrix3d SharedFromLocalEnu(
      const Eigen::Vector3d& wgs84_position) const;

  // Rotates a covariance from the prior's own local ENU frame into this shared
  // frame: C_shared = R * C_local * R^T, R = SharedFromLocalEnu(position).
  // Over a single reconstruction R is near-identity, but it is not identity,
  // and skipping it quietly rotates anisotropic uncertainty off-axis.
  Eigen::Matrix3d CovarianceInEnu(const PosePrior& pose_prior) const;

 private:
  Eigen::Vector3d origin_wgs84_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d origin_ecef_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d enu_from_ecef_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d ecef_from_enu_ = Eigen::Matrix3d::Identity();
  bool has_real_altitude_ = false;
};

}  // namespace colmap
