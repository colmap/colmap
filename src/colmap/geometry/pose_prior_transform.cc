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

#include "colmap/geometry/pose_prior_transform.h"

#include "colmap/math/geometric_median.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>

namespace colmap {

bool PosePriorEnuFrame::IsUsable(const PosePrior& pose_prior) {
  return pose_prior.coordinate_system == PosePrior::CoordinateSystem::WGS84 &&
         std::isfinite(pose_prior.position.x()) &&
         std::isfinite(pose_prior.position.y());
}

std::optional<PosePriorEnuFrame> PosePriorEnuFrame::Derive(
    const std::vector<PosePrior>& pose_priors) {
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);

  // Collect in a deterministic order. GeometricMedian accumulates in input
  // order, so two processes handed the same priors in different orders could
  // otherwise differ in the last bits of the origin -- and every transform is
  // expressed against that origin.
  std::vector<std::pair<data_t, const PosePrior*>> sorted;
  sorted.reserve(pose_priors.size());
  for (const PosePrior& pose_prior : pose_priors) {
    if (IsUsable(pose_prior)) {
      sorted.emplace_back(pose_prior.corr_data_id, &pose_prior);
    }
  }
  if (sorted.empty()) {
    return std::nullopt;
  }
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    if (a.first.sensor_id.type != b.first.sensor_id.type) {
      return a.first.sensor_id.type < b.first.sensor_id.type;
    }
    if (a.first.sensor_id.id != b.first.sensor_id.id) {
      return a.first.sensor_id.id < b.first.sensor_id.id;
    }
    return a.first.id < b.first.id;
  });

  // One rule for the origin, whether or not altitudes are present. Rows without
  // a finite altitude still anchor latitude/longitude -- discarding them would
  // make the origin depend on which rows happened to carry a height.
  std::vector<Eigen::Vector3d> lla;
  std::vector<double> altitudes;
  lla.reserve(sorted.size());
  for (const auto& [data_id, pose_prior] : sorted) {
    const double alt = std::isfinite(pose_prior->position.z())
                           ? pose_prior->position.z()
                           : 0.0;
    if (std::isfinite(pose_prior->position.z())) {
      altitudes.push_back(pose_prior->position.z());
    }
    lla.emplace_back(pose_prior->position.x(), pose_prior->position.y(), alt);
  }

  const std::vector<Eigen::Vector3d> ecef = gps_transform.EllipsoidToECEF(lla);
  const Eigen::Vector3d median_ecef = GeometricMedian(ecef);
  const Eigen::Vector3d median_lla =
      gps_transform.ECEFToEllipsoid({median_ecef})[0];

  PosePriorEnuFrame frame;
  frame.has_real_altitude_ = !altitudes.empty();
  double origin_alt = 0.0;
  if (frame.has_real_altitude_) {
    std::sort(altitudes.begin(), altitudes.end());
    origin_alt = altitudes[altitudes.size() / 2];
  }
  frame.origin_wgs84_ =
      Eigen::Vector3d(median_lla.x(), median_lla.y(), origin_alt);
  frame.origin_ecef_ = gps_transform.EllipsoidToECEF({frame.origin_wgs84_})[0];
  frame.enu_from_ecef_ = GPSTransform::ENUFromECEF(frame.origin_wgs84_.x(),
                                                   frame.origin_wgs84_.y());
  frame.ecef_from_enu_ = frame.enu_from_ecef_.transpose();

  THROW_CHECK(frame.origin_wgs84_.allFinite())
      << "derived a non-finite ENU origin";
  THROW_CHECK(frame.origin_ecef_.allFinite())
      << "derived a non-finite ECEF origin";
  return frame;
}

Eigen::Vector3d PosePriorEnuFrame::PositionInEnu(
    const PosePrior& pose_prior) const {
  THROW_CHECK(IsUsable(pose_prior))
      << "PositionInEnu requires a WGS84 prior with finite lat/lon";
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);
  const double alt =
      std::isfinite(pose_prior.position.z()) ? pose_prior.position.z() : 0.0;
  const Eigen::Vector3d ecef = gps_transform.EllipsoidToECEF({Eigen::Vector3d(
      pose_prior.position.x(), pose_prior.position.y(), alt)})[0];
  Eigen::Vector3d enu = enu_from_ecef_ * (ecef - origin_ecef_);
  if (!std::isfinite(pose_prior.position.z())) {
    // The row declared no height. Propagating the placeholder as a real
    // vertical coordinate would let a downstream consumer treat 0.0 as a
    // measurement, so keep it absent.
    enu.z() = std::numeric_limits<double>::quiet_NaN();
  }
  return enu;
}

Eigen::Matrix3d PosePriorEnuFrame::SharedFromLocalEnu(
    const Eigen::Vector3d& wgs84_position) const {
  THROW_CHECK(std::isfinite(wgs84_position.x()) &&
              std::isfinite(wgs84_position.y()))
      << "SharedFromLocalEnu requires a finite latitude and longitude";
  // shared_from_local = shared_from_ecef * ecef_from_local.
  return enu_from_ecef_ *
         GPSTransform::ECEFFromENU(wgs84_position.x(), wgs84_position.y());
}

Eigen::Matrix3d PosePriorEnuFrame::CovarianceInEnu(
    const PosePrior& pose_prior) const {
  THROW_CHECK(IsUsable(pose_prior))
      << "CovarianceInEnu requires a WGS84 prior with finite lat/lon";
  const Eigen::Matrix3d shared_from_local =
      SharedFromLocalEnu(pose_prior.position);
  return shared_from_local * pose_prior.position_covariance *
         shared_from_local.transpose();
}

}  // namespace colmap
