// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "base/gps.h"

#include "util/math.h"

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

    const double sin_lat = sin(lat);
    const double sin_lon = sin(lon);
    const double cos_lat = cos(lat);
    const double cos_lon = cos(lon);

    // Normalized radius
    const double N = a_ / sqrt(1 - e2_ * sin_lat * sin_lat);

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

    const double radius_xy = sqrt(x * x + y * y);
    const double kEps = 1e-12;

    // Latitude
    double lat = atan2(z, radius_xy);
    double alt;

    for (size_t j = 0; j < 100; ++j) {
      const double sin_lat0 = sin(lat);
      const double N = a_ / sqrt(1 - e2_ * sin_lat0 * sin_lat0);
      alt = radius_xy / cos(lat) - N;
      const double prev_lat = lat;
      lat = atan((z / radius_xy) * 1 / (1 - e2_ * N / (N + alt)));

      if (std::abs(prev_lat - lat) < kEps) {
        break;
      }
    }

    ell[i](0) = RadToDeg(lat);

    // Longitude
    ell[i](1) = RadToDeg(atan2(y, x));
    // Alt
    ell[i](2) = alt;
  }

  return ell;
}

}  // namespace colmap
