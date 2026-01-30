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

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Transform ellipsoidal GPS coordinates to Cartesian coordinate systems
// and vice versa.
class GPSTransform {
 public:
  MAKE_ENUM_CLASS(Ellipsoid, 0, GRS80, WGS84);

  explicit GPSTransform(Ellipsoid ellipsoid = Ellipsoid::GRS80);

  // Convert ellipsoidal (lat/lon/alt) to ECEF coordinates.
  std::vector<Eigen::Vector3d> EllipsoidToECEF(
      const std::vector<Eigen::Vector3d>& lat_lon_alt) const;

  // Convert ECEF to ellipsoidal (lat/lon/alt) coordinates.
  std::vector<Eigen::Vector3d> ECEFToEllipsoid(
      const std::vector<Eigen::Vector3d>& xyz_in_ecef) const;

  // Convert ellipsoidal (lat/lon/alt) to ENU coordinates.
  // The reference point (ref_lat, ref_lon, ref_alt) defines the ENU origin.
  std::vector<Eigen::Vector3d> EllipsoidToENU(
      const std::vector<Eigen::Vector3d>& lat_lon_alt,
      double ref_lat,
      double ref_lon,
      double ref_alt) const;

  // Convert ECEF to ENU coordinates.
  // The reference point (ref_ecef) defines the ENU origin.
  std::vector<Eigen::Vector3d> ECEFToENU(
      const std::vector<Eigen::Vector3d>& xyz_in_ecef,
      const Eigen::Vector3d& ref_ecef) const;

  // Convert ENU to ellipsoidal (lat/lon/alt) coordinates.
  // The reference point (ref_lat, ref_lon, ref_alt) defines the ENU origin.
  std::vector<Eigen::Vector3d> ENUToEllipsoid(
      const std::vector<Eigen::Vector3d>& xyz_in_enu,
      double ref_lat,
      double ref_lon,
      double ref_alt) const;

  // Convert ENU to ECEF coordinates.
  // The reference point (ref_lat, ref_lon, ref_alt) defines the ENU origin.
  std::vector<Eigen::Vector3d> ENUToECEF(
      const std::vector<Eigen::Vector3d>& xyz_in_enu,
      double ref_lat,
      double ref_lon,
      double ref_alt) const;

  // Convert ellipsoidal (lat/lon/alt) to UTM coordinates.
  // Returns a pair of the converted coordinates and the zone number.
  // If the points span multiple zones, the zone with the most points is chosen.
  // The conversion uses a 4th-order expansion formula. The easting offset is
  // 500 km, and the northing offset is 10,000 km for the Southern Hemisphere.
  std::pair<std::vector<Eigen::Vector3d>, int> EllipsoidToUTM(
      const std::vector<Eigen::Vector3d>& lat_lon_alt) const;

  // Convert UTM to ellipsoidal (lat/lon/alt) coordinates.
  // Requires the zone number and hemisphere (true for north, false for south).
  std::vector<Eigen::Vector3d> UTMToEllipsoid(
      const std::vector<Eigen::Vector3d>& xyz_in_utm,
      int zone,
      bool is_north) const;

 private:
  double a_;   // Semimajor axis.
  double b_;   // Semiminor axis.
  double f_;   // Flattening.
  double e2_;  // Numerical eccentricity squared.
};

}  // namespace colmap
