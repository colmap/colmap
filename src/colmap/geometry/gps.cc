// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/geometry/gps.h"

#include "colmap/math/math.h"

namespace colmap {

GPSTransform::GPSTransform(const int ellipsoid) {
  switch (ellipsoid) {
    case GRS80:
      a_ = 6378137.0;
      f_ = 1.0 / 298.257222100882711243162837;  // More accurate GRS80 ellipsoid
      b_ = (1.0 - f_) * a_;
      break;
    case WGS84:
      a_ = 6378137.0;
      f_ = 1.0 / 298.257223563;  // The WGS84 ellipsoid
      b_ = (1.0 - f_) * a_;
      break;
    default:
      a_ = std::numeric_limits<double>::quiet_NaN();
      b_ = std::numeric_limits<double>::quiet_NaN();
      f_ = std::numeric_limits<double>::quiet_NaN();
      throw std::invalid_argument("Ellipsoid not defined");
  }
  e2_ = f_ * (2.0 - f_);
}

std::vector<Eigen::Vector3d> GPSTransform::EllToXYZ(
    const std::vector<Eigen::Vector3d>& ell) const {
  std::vector<Eigen::Vector3d> xyz(ell.size());

  for (size_t i = 0; i < ell.size(); ++i) {
    const double lat = DegToRad(ell[i](0));
    const double lon = DegToRad(ell[i](1));
    const double alt = ell[i](2);

    const double sin_lat = std::sin(lat);
    const double sin_lon = std::sin(lon);
    const double cos_lat = std::cos(lat);
    const double cos_lon = std::cos(lon);

    // Normalized radius
    const double N = a_ / std::sqrt(1 - e2_ * sin_lat * sin_lat);

    xyz[i](0) = (N + alt) * cos_lat * cos_lon;
    xyz[i](1) = (N + alt) * cos_lat * sin_lon;
    xyz[i](2) = (N * (1 - e2_) + alt) * sin_lat;
  }

  return xyz;
}

std::vector<Eigen::Vector3d> GPSTransform::XYZToEll(
    const std::vector<Eigen::Vector3d>& xyz) const {
  std::vector<Eigen::Vector3d> ell(xyz.size());

  for (size_t i = 0; i < ell.size(); ++i) {
    const double x = xyz[i](0);
    const double y = xyz[i](1);
    const double z = xyz[i](2);

    const double radius_xy = std::sqrt(x * x + y * y);
    const double kEps = 1e-12;

    // Latitude
    double lat = std::atan2(z, radius_xy);
    double alt = 0.0;

    for (size_t j = 0; j < 100; ++j) {
      const double sin_lat0 = std::sin(lat);
      const double N = a_ / std::sqrt(1 - e2_ * sin_lat0 * sin_lat0);
      const double prev_alt = alt;
      alt = radius_xy / std::cos(lat) - N;
      const double prev_lat = lat;
      lat = std::atan((z / radius_xy) * 1 / (1 - e2_ * N / (N + alt)));

      if (std::abs(prev_lat - lat) < kEps && std::abs(prev_alt - alt) < kEps) {
        break;
      }
    }

    ell[i](0) = RadToDeg(lat);

    // Longitude
    ell[i](1) = RadToDeg(std::atan2(y, x));
    // Alt
    ell[i](2) = alt;
  }

  return ell;
}

std::vector<Eigen::Vector3d> GPSTransform::EllToENU(
    const std::vector<Eigen::Vector3d>& ell,
    const double lat0,
    const double lon0) const {
  // Convert GPS (lat / lon / alt) to ECEF
  std::vector<Eigen::Vector3d> xyz = EllToXYZ(ell);

  return XYZToENU(xyz, lat0, lon0);
}

std::vector<Eigen::Vector3d> GPSTransform::XYZToENU(
    const std::vector<Eigen::Vector3d>& xyz,
    const double lat0,
    const double lon0) const {
  std::vector<Eigen::Vector3d> enu(xyz.size());

  // https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU

  // ECEF to ENU Rot :
  const double cos_lat0 = std::cos(DegToRad(lat0));
  const double sin_lat0 = std::sin(DegToRad(lat0));

  const double cos_lon0 = std::cos(DegToRad(lon0));
  const double sin_lon0 = std::sin(DegToRad(lon0));

  Eigen::Matrix3d R;
  R << -sin_lon0, cos_lon0, 0., -sin_lat0 * cos_lon0, -sin_lat0 * sin_lon0,
      cos_lat0, cos_lat0 * cos_lon0, cos_lat0 * sin_lon0, sin_lat0;

  // Convert ECEF to ENU coords. (w.r.t. ECEF ref == xyz[0])
  for (size_t i = 0; i < xyz.size(); ++i) {
    enu[i] = R * (xyz[i] - xyz[0]);
  }

  return enu;
}

std::vector<Eigen::Vector3d> GPSTransform::ENUToEll(
    const std::vector<Eigen::Vector3d>& enu,
    const double lat0,
    const double lon0,
    const double alt0) const {
  return XYZToEll(ENUToXYZ(enu, lat0, lon0, alt0));
}

std::vector<Eigen::Vector3d> GPSTransform::ENUToXYZ(
    const std::vector<Eigen::Vector3d>& enu,
    const double lat0,
    const double lon0,
    const double alt0) const {
  std::vector<Eigen::Vector3d> xyz(enu.size());

  // ECEF ref (origin)
  const Eigen::Vector3d xyz_ref =
      EllToXYZ({Eigen::Vector3d(lat0, lon0, alt0)})[0];

  // ENU to ECEF Rot :
  const double cos_lat0 = std::cos(DegToRad(lat0));
  const double sin_lat0 = std::sin(DegToRad(lat0));

  const double cos_lon0 = std::cos(DegToRad(lon0));
  const double sin_lon0 = std::sin(DegToRad(lon0));

  Eigen::Matrix3d R;
  R << -sin_lon0, cos_lon0, 0., -sin_lat0 * cos_lon0, -sin_lat0 * sin_lon0,
      cos_lat0, cos_lat0 * cos_lon0, cos_lat0 * sin_lon0, sin_lat0;

  // R is ECEF to ENU so Transpose to get inverse
  R.transposeInPlace();

  // Convert ENU to ECEF coords.
  for (size_t i = 0; i < enu.size(); ++i) {
    xyz[i] = (R * enu[i]) + xyz_ref;
  }

  return xyz;
}

std::ostream& operator<<(std::ostream& stream, const PosePrior& prior) {
  const static Eigen::IOFormat kVecFmt(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ");
  stream << "PosePrior(position=[" << prior.position.format(kVecFmt)
         << "], position_covariance=["
         << prior.position_covariance.format(kVecFmt) << "], coordinate_system="
         << PosePrior::CoordinateSystemToString(prior.coordinate_system) << ")";
  return stream;
}

}  // namespace colmap
