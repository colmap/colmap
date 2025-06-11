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

#include "colmap/geometry/gps.h"

#include "colmap/math/math.h"

namespace colmap {
namespace {

struct UTMParams {
  // The order of the series expansion, determining the precision.
  static constexpr int kOrder = 4;

  // The following notations come from:
  // https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system

  // Powers of mathematical sign $n$, n[i] means $n^i$
  std::array<double, kOrder + 1> n;
  // Alpha coefficients for the series expansion
  std::array<double, kOrder> alpha;
  // Beta coefficients for the series expansion
  std::array<double, kOrder> beta;
  // Delta coefficients for the series expansion
  std::array<double, kOrder> delta;

  // Multiplicative factor
  double A;
  // UTM scale factor at the central meridian, typically 0.9996
  static constexpr double k0 = 0.9996;
  // Easting of the origin, typically 500 km
  static constexpr double E0 = 500;
  // Northing of the origin
  static double N0(double lat_or_hemi) { return lat_or_hemi > 0 ? 0 : 1e4; }

  UTMParams(double a, double f) {
    n[0] = 1;
    n[1] = f / (2.0 - f);
    for (size_t i = 2; i < kOrder + 1; ++i) {
      n[i] = n[1] * n[i - 1];
    }

    // clang-format off
    alpha = {1.0/2.0          * n[1] - 2.0/3.0     * n[2] + 5.0/16.0     * n[3] + 41.0/180.0 * n[4],
             13.0/48.0        * n[2] - 3.0/5.0     * n[3] + 557.0/1440.0 * n[4],
             61.0/240.0       * n[3] - 103.0/140.0 * n[4],
             49561.0/161280.0 * n[4]};

    beta = {1.0/2.0         * n[1] - 2.0/3.0    * n[2] + 37.0/96.0    * n[3] - 1.0/360.0 * n[4],
            1.0/48.0        * n[2] + 1.0/15.0   * n[3] - 437.0/1440.0 * n[4],
            17.0/480.0      * n[3] - 37.0/840.0 * n[4],
            4397.0/161280.0 * n[4]};

    delta = {2.0          * n[1] - 2.0/3.0    * n[2] - 2.0        * n[3] - 116.0/45.0 * n[4],
             7.0/3.0      * n[2] - 8.0/5.0    * n[3] - 227.0/45.0 * n[4],
             56.0/15.0    * n[3] - 136.0/35.0 * n[4],
             4279.0/630.0 * n[4]};
    // clang-format on

    A = a / (1.0 + n[1]) * (1.0 + n[2] / 4.0 + n[4] / 64.0);
  };

  static double ZoneToCentralMeridian(int zone) { return 6 * zone - 183; }

  static int MeridianToZone(double meridian) {
    return static_cast<int>(std::floor((meridian + 180) / 6)) + 1;
  }
};
}  // namespace

GPSTransform::GPSTransform(const Ellipsoid ellipsoid) {
  switch (ellipsoid) {
    case Ellipsoid::GRS80:
      a_ = 6378137.0;
      f_ = 1.0 / 298.257222100882711243162837;  // More accurate GRS80 ellipsoid
      b_ = (1.0 - f_) * a_;
      break;
    case Ellipsoid::WGS84:
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

std::vector<Eigen::Vector3d> GPSTransform::EllipsoidToECEF(
    const std::vector<Eigen::Vector3d>& lat_lon_alt) const {
  std::vector<Eigen::Vector3d> xyz_in_ecef(lat_lon_alt.size());

  for (size_t i = 0; i < lat_lon_alt.size(); ++i) {
    const double lat = DegToRad(lat_lon_alt[i](0));
    const double lon = DegToRad(lat_lon_alt[i](1));
    const double alt = lat_lon_alt[i](2);

    const double sin_lat = std::sin(lat);
    const double sin_lon = std::sin(lon);
    const double cos_lat = std::cos(lat);
    const double cos_lon = std::cos(lon);

    // Normalized radius
    const double N = a_ / std::sqrt(1 - e2_ * sin_lat * sin_lat);

    xyz_in_ecef[i](0) = (N + alt) * cos_lat * cos_lon;
    xyz_in_ecef[i](1) = (N + alt) * cos_lat * sin_lon;
    xyz_in_ecef[i](2) = (N * (1 - e2_) + alt) * sin_lat;
  }

  return xyz_in_ecef;
}

std::vector<Eigen::Vector3d> GPSTransform::ECEFToEllipsoid(
    const std::vector<Eigen::Vector3d>& xyz_in_ecef) const {
  std::vector<Eigen::Vector3d> lat_lon_alt(xyz_in_ecef.size());

  for (size_t i = 0; i < lat_lon_alt.size(); ++i) {
    const double x = xyz_in_ecef[i](0);
    const double y = xyz_in_ecef[i](1);
    const double z = xyz_in_ecef[i](2);

    const double radius_xy = std::sqrt(x * x + y * y);
    const double kEps = 1e-12;

    // Latitude
    double lat = std::atan2(z, radius_xy);
    double alt = 0.0;

    for (size_t j = 0; j < 100; ++j) {
      const double sin_lat = std::sin(lat);
      const double N = a_ / std::sqrt(1 - e2_ * sin_lat * sin_lat);
      const double prev_alt = alt;
      alt = radius_xy / std::cos(lat) - N;
      const double prev_lat = lat;
      lat = std::atan((z / radius_xy) * 1 / (1 - e2_ * N / (N + alt)));

      if (std::abs(prev_lat - lat) < kEps && std::abs(prev_alt - alt) < kEps) {
        break;
      }
    }

    lat_lon_alt[i](0) = RadToDeg(lat);

    // Longitude
    lat_lon_alt[i](1) = RadToDeg(std::atan2(y, x));
    // Alt
    lat_lon_alt[i](2) = alt;
  }

  return lat_lon_alt;
}

std::vector<Eigen::Vector3d> GPSTransform::EllipsoidToENU(
    const std::vector<Eigen::Vector3d>& lat_lon_alt,
    const double ref_lat,
    const double ref_lon) const {
  // Convert GPS (lat / lon / alt) to ECEF
  std::vector<Eigen::Vector3d> xyz_in_ecef = EllipsoidToECEF(lat_lon_alt);

  return ECEFToENU(xyz_in_ecef, ref_lat, ref_lon);
}

std::vector<Eigen::Vector3d> GPSTransform::ECEFToENU(
    const std::vector<Eigen::Vector3d>& xyz_in_ecef,
    const double ref_lat,
    const double ref_lon) const {
  std::vector<Eigen::Vector3d> xyz_in_enu(xyz_in_ecef.size());

  // https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU

  // ECEF to ENU Rot :
  const double cos_lat = std::cos(DegToRad(ref_lat));
  const double sin_lat = std::sin(DegToRad(ref_lat));

  const double cos_lon = std::cos(DegToRad(ref_lon));
  const double sin_lon = std::sin(DegToRad(ref_lon));

  Eigen::Matrix3d R;
  R << -sin_lon, cos_lon, 0., -sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat,
      cos_lat * cos_lon, cos_lat * sin_lon, sin_lat;

  // Convert ECEF to ENU coords. (w.r.t. ECEF ref == xyz_in_ecef[0])
  for (size_t i = 0; i < xyz_in_ecef.size(); ++i) {
    xyz_in_enu[i] = R * (xyz_in_ecef[i] - xyz_in_ecef[0]);
  }

  return xyz_in_enu;
}

std::vector<Eigen::Vector3d> GPSTransform::ENUToEllipsoid(
    const std::vector<Eigen::Vector3d>& xyz_in_enu,
    const double ref_lat,
    const double ref_lon,
    const double ref_alt) const {
  return ECEFToEllipsoid(ENUToECEF(xyz_in_enu, ref_lat, ref_lon, ref_alt));
}

std::vector<Eigen::Vector3d> GPSTransform::ENUToECEF(
    const std::vector<Eigen::Vector3d>& xyz_in_enu,
    const double ref_lat,
    const double ref_lon,
    const double ref_alt) const {
  std::vector<Eigen::Vector3d> xyz_in_ecef(xyz_in_enu.size());

  // ECEF ref (origin)
  const Eigen::Vector3d ref_xyz_in_ecef =
      EllipsoidToECEF({Eigen::Vector3d(ref_lat, ref_lon, ref_alt)})[0];

  // ENU to ECEF Rot :
  const double cos_lat = std::cos(DegToRad(ref_lat));
  const double sin_lat = std::sin(DegToRad(ref_lat));

  const double cos_lon = std::cos(DegToRad(ref_lon));
  const double sin_lon = std::sin(DegToRad(ref_lon));

  Eigen::Matrix3d R;
  R << -sin_lon, cos_lon, 0., -sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat,
      cos_lat * cos_lon, cos_lat * sin_lon, sin_lat;

  // R is ECEF to ENU so Transpose to get inverse
  R.transposeInPlace();

  // Convert ENU to ECEF coords.
  for (size_t i = 0; i < xyz_in_enu.size(); ++i) {
    xyz_in_ecef[i] = (R * xyz_in_enu[i]) + ref_xyz_in_ecef;
  }

  return xyz_in_ecef;
}

std::pair<std::vector<Eigen::Vector3d>, int> GPSTransform::EllipsoidToUTM(
    const std::vector<Eigen::Vector3d>& lat_lon_alt) const {
  // The following implementation is based on the formulas from:
  // https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system

  const UTMParams params(a_ / 1000.0, f_);  // converts to kilometers

  // For cases where points span different zones, we select the predominant zone
  // as the reference frame.
  std::array<int, 60> zone_counts{};
  for (const Eigen::Vector3d& lla : lat_lon_alt) {
    THROW_CHECK_GE(lla[0], -90);
    THROW_CHECK_LE(lla[0], 90);
    THROW_CHECK_GE(lla[1], -180);
    THROW_CHECK_LE(lla[1], 180);

    const size_t z_index =
        static_cast<std::size_t>(UTMParams::MeridianToZone(lla[1])) - 1;
    ++zone_counts[z_index];
  }
  const int zone =
      static_cast<int>(std::distance(
          zone_counts.begin(),
          std::max_element(zone_counts.begin(), zone_counts.end()))) +
      1;
  const double lambda0 = DegToRad(UTMParams::ZoneToCentralMeridian(zone));

  // Converts lla to utm
  std::vector<Eigen::Vector3d> xyz_in_utm;
  xyz_in_utm.reserve(lat_lon_alt.size());

  for (const Eigen::Vector3d& lla : lat_lon_alt) {
    const double phi = DegToRad(lla[0]);
    const double lambda = DegToRad(lla[1]);

    const double t = std::sinh(std::atanh(sin(phi)) -
                               2 * std::sqrt(params.n[1]) / (1 + params.n[1]) *
                                   std::atanh(2 * std::sqrt(params.n[1]) /
                                              (1 + params.n[1]) * sin(phi)));
    const double xi = std::atan(t / std::cos(lambda - lambda0));
    const double eta =
        std::atanh(std::sin(lambda - lambda0) / std::sqrt(1 + t * t));

    double E = eta, N = xi;
    for (size_t i = 0; i < params.kOrder; ++i) {
      double doubled_index = 2.0 * static_cast<double>(i + 1);
      E += params.alpha[i] * std::cos(doubled_index * xi) *
           std::sinh(doubled_index * eta);
      N += params.alpha[i] * std::sin(doubled_index * xi) *
           std::cosh(doubled_index * eta);
    }

    E = params.E0 + params.k0 * params.A * E;
    N = params.N0(lla[0]) + params.k0 * params.A * N;

    xyz_in_utm.emplace_back(
        E * 1000, N * 1000, lla[2]);  // converts back to meters
  }

  return std::make_pair(std::move(xyz_in_utm), zone);
}

std::vector<Eigen::Vector3d> GPSTransform::UTMToEllipsoid(
    const std::vector<Eigen::Vector3d>& xyz_in_utm,
    int zone,
    bool is_north) const {
  // The following implementation is based on the formulas from:
  // https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system

  THROW_CHECK_GE(zone, 1);
  THROW_CHECK_LE(zone, 60);

  // Setup params
  const UTMParams params(a_ / 1000.0, f_);  // converts to kilometers

  // Converts utm to ell
  std::vector<Eigen::Vector3d> lat_lon_alt;
  lat_lon_alt.reserve(xyz_in_utm.size());

  for (const Eigen::Vector3d& ena : xyz_in_utm) {
    const double xi =
        (ena[1] / 1000.0 - params.N0(is_north)) / (params.k0 * params.A);
    const double eta = (ena[0] / 1000.0 - params.E0) / (params.k0 * params.A);

    double xi_prime = 0.0, eta_prime = 0.0;
    for (size_t i = 0; i < params.kOrder; ++i) {
      double doubled_index = 2.0 * static_cast<double>(i + 1);
      xi_prime += params.beta[i] * std::sin(doubled_index * xi) *
                  std::cosh(doubled_index * eta);
      eta_prime += params.beta[i] * std::cos(doubled_index * xi) *
                   std::sinh(doubled_index * eta);
    }
    xi_prime = xi - xi_prime;
    eta_prime = eta - eta_prime;
    const double chi = std::asin(std::sin(xi_prime) / std::cosh(eta_prime));

    double phi = chi;
    for (size_t i = 0; i < params.kOrder; ++i) {
      double doubled_index = 2.0 * static_cast<double>(i + 1);
      phi += params.delta[i] * std::sin(doubled_index * chi);
    }

    const double lat = RadToDeg(phi);
    const double lon =
        UTMParams::ZoneToCentralMeridian(zone) +
        RadToDeg(std::atan(std::sinh(eta_prime) / std::cos(xi_prime)));

    lat_lon_alt.emplace_back(lat, lon, ena[2]);
  }

  return lat_lon_alt;
}

}  // namespace colmap
