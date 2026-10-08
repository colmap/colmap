// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/geometry/gps.h"

#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(GPS, EllipsoidToECEFGRS80) {
  std::vector<Eigen::Vector3d> ell;
  ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                   11 + 34. / 60 + 10.51777 / 3600,
                   561.1851);
  ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                   11 + 34. / 60 + 11.77179 / 3600,
                   561.1509);
  std::vector<Eigen::Vector3d> ref_xyz;
  ref_xyz.emplace_back(
      4.1772397090808507e6, 0.85515377993121441e6, 4.7282674046563692e6);
  ref_xyz.emplace_back(
      4.1772186604902023e6, 0.8551759313518483e6, 4.7282818502697079e6);

  GPSTransform gps_tform(GPSTransform::Ellipsoid::GRS80);

  const auto xyz = gps_tform.EllipsoidToECEF(ell);

  for (size_t i = 0; i < ell.size(); ++i) {
    EXPECT_THAT(xyz[i], EigenMatrixNear(ref_xyz[i], 1e-8));
  }
}

TEST(GPS, EllipsoidToECEFWGS84) {
  std::vector<Eigen::Vector3d> ell;
  ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                   11 + 34. / 60 + 10.51777 / 3600,
                   561.1851);
  ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                   11 + 34. / 60 + 11.77179 / 3600,
                   561.1509);
  std::vector<Eigen::Vector3d> ref_xyz;
  ref_xyz.emplace_back(
      4.177239709042750e6, 0.855153779923415e6, 4.728267404769168e6);
  ref_xyz.emplace_back(
      4.177218660452103e6, 0.855175931344048e6, 4.728281850382507e6);

  GPSTransform gps_tform(GPSTransform::Ellipsoid::WGS84);

  const auto xyz = gps_tform.EllipsoidToECEF(ell);

  for (size_t i = 0; i < ell.size(); ++i) {
    EXPECT_THAT(xyz[i], EigenMatrixNear(ref_xyz[i], 1e-8));
  }
}

TEST(GPS, ECEFToEllipsoid_GRS80) {
  std::vector<Eigen::Vector3d> xyz;
  xyz.emplace_back(
      4.1772397090808507e6, 0.85515377993121441e6, 4.7282674046563692e6);
  xyz.emplace_back(
      4.1772186604902023e6, 0.8551759313518483e6, 4.7282818502697079e6);
  std::vector<Eigen::Vector3d> ref_ell;
  ref_ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                       11 + 34. / 60 + 10.51777 / 3600,
                       561.1851);
  ref_ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                       11 + 34. / 60 + 11.77179 / 3600,
                       561.1509);

  GPSTransform gps_tform(GPSTransform::Ellipsoid::GRS80);

  const auto ell = gps_tform.ECEFToEllipsoid(xyz);

  for (size_t i = 0; i < xyz.size(); ++i) {
    EXPECT_THAT(ell[i], EigenMatrixNear(ref_ell[i], 1e-5));
  }
}

TEST(GPS, ECEFToEllipsoid_WGS84) {
  std::vector<Eigen::Vector3d> xyz;
  xyz.emplace_back(
      4.177239709042750e6, 0.855153779923415e6, 4.728267404769168e6);
  xyz.emplace_back(
      4.177218660452103e6, 0.855175931344048e6, 4.728281850382507e6);
  std::vector<Eigen::Vector3d> ref_ell;
  ref_ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                       11 + 34. / 60 + 10.51777 / 3600,
                       561.1851);
  ref_ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                       11 + 34. / 60 + 11.77179 / 3600,
                       561.1509);

  GPSTransform gps_tform(GPSTransform::Ellipsoid::WGS84);

  const auto ell = gps_tform.ECEFToEllipsoid(xyz);

  for (size_t i = 0; i < xyz.size(); ++i) {
    EXPECT_THAT(ell[i], EigenMatrixNear(ref_ell[i], 1e-5));
  }
}

TEST(GPS, ECEFToEllipsoidPoles) {
  for (const auto ellipsoid :
       {GPSTransform::Ellipsoid::GRS80, GPSTransform::Ellipsoid::WGS84}) {
    const double inverse_flattening =
        ellipsoid == GPSTransform::Ellipsoid::GRS80
            ? 298.257222100882711243162837
            : 298.257223563;
    const double semiminor_axis = 6378137.0 * (1.0 - 1.0 / inverse_flattening);
    const GPSTransform gps_tform(ellipsoid);
    for (const double sign : {-1.0, 1.0}) {
      for (const double altitude : {-100.0, 0.0, 500.0}) {
        const auto ell = gps_tform.ECEFToEllipsoid(
            {Eigen::Vector3d(0, 0, sign * (semiminor_axis + altitude))});
        EXPECT_THAT(
            ell[0],
            EigenMatrixNear(Eigen::Vector3d(sign * 90, 0, altitude), 1e-8));
      }
    }
  }
}

TEST(GPS, ECEFToEllipsoidNearPoles) {
  const GPSTransform gps_tform(GPSTransform::Ellipsoid::WGS84);
  // PROJ 9.8.1: EPSG:4978 -> EPSG:4979 for (0.001, 0, 6356852.314245179).
  const double semiminor_axis = 6356752.314245179;
  for (const double sign : {-1.0, 1.0}) {
    const auto ell = gps_tform.ECEFToEllipsoid(
        {Eigen::Vector3d(0.001, 0, sign * (semiminor_axis + 100)),
         Eigen::Vector3d(0, 0.001, sign * (semiminor_axis + 100))});
    EXPECT_THAT(ell[0],
                EigenMatrixNear(
                    Eigen::Vector3d(sign * 89.9999999910471, 0, 100), 1e-8));
    EXPECT_THAT(ell[1],
                EigenMatrixNear(
                    Eigen::Vector3d(sign * 89.9999999910471, 90, 100), 1e-8));
  }
}

TEST(GPS, ECEFToENUAtPoles) {
  const GPSTransform gps_tform(GPSTransform::Ellipsoid::WGS84);
  for (const double sign : {-1.0, 1.0}) {
    const Eigen::Vector3d origin(0, 0, sign * 6356752.314245179);
    const Eigen::Vector3d offset(1, 2, 3);
    const auto enu = gps_tform.ECEFToENU({origin, origin + offset}, origin);
    EXPECT_THAT(enu[0], EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-8));
    EXPECT_THAT(enu[1],
                EigenMatrixNear(Eigen::Vector3d(2, -sign, 3 * sign), 1e-8));
  }
}

TEST(GPS, ECEFToEllipsoidRoundTrip) {
  std::vector<Eigen::Vector3d> ell;
  for (const double latitude :
       {-85.0, -75.0, -45.0, -15.0, 0.0, 15.0, 45.0, 75.0, 85.0}) {
    for (const double longitude : {-179.0, -90.0, 0.0, 90.0, 179.0}) {
      for (const double altitude : {-1000.0, 0.0, 1000.0, 1000000.0}) {
        ell.emplace_back(latitude, longitude, altitude);
      }
    }
  }
  for (const auto ellipsoid :
       {GPSTransform::Ellipsoid::GRS80, GPSTransform::Ellipsoid::WGS84}) {
    const GPSTransform gps_tform(ellipsoid);
    const auto xyz = gps_tform.EllipsoidToECEF(ell);
    const auto actual = gps_tform.ECEFToEllipsoid(xyz);
    const auto repeated = GPSTransform(ellipsoid).ECEFToEllipsoid(xyz);
    const auto enu = gps_tform.EllipsoidToENU(ell, 35, 10, 50);
    const auto from_enu = gps_tform.ENUToEllipsoid(enu, 35, 10, 50);
    ASSERT_EQ(actual.size(), ell.size());
    ASSERT_EQ(repeated.size(), ell.size());
    ASSERT_EQ(from_enu.size(), ell.size());
    for (size_t i = 0; i < ell.size(); ++i) {
      EXPECT_THAT(actual[i], EigenMatrixNear(ell[i], 1e-5));
      EXPECT_THAT(repeated[i], EigenMatrixNear(actual[i], 1e-12));
      EXPECT_THAT(from_enu[i], EigenMatrixNear(ell[i], 1e-5));
    }
    EXPECT_TRUE(gps_tform.ECEFToEllipsoid({}).empty());
  }
}

TEST(GPS, ECEFToEllipsoidipsoidToECEF_GRS80) {
  std::vector<Eigen::Vector3d> xyz;
  xyz.emplace_back(
      4.177239709080851e6, 0.855153779931214e6, 4.728267404656370e6);
  xyz.emplace_back(
      4.177218660490202e6, 0.855175931351848e6, 4.728281850269709e6);

  GPSTransform gps_tform(GPSTransform::Ellipsoid::GRS80);

  const auto ell = gps_tform.ECEFToEllipsoid(xyz);
  const auto xyz2 = gps_tform.EllipsoidToECEF(ell);

  for (size_t i = 0; i < xyz.size(); ++i) {
    EXPECT_THAT(xyz[i], EigenMatrixNear(xyz2[i], 1e-5));
  }
}

TEST(GPS, ECEFToEllipsoidipsoidToECEF_WGS84) {
  std::vector<Eigen::Vector3d> xyz;
  xyz.emplace_back(
      4.177239709080851e6, 0.855153779931214e6, 4.728267404656370e6);
  xyz.emplace_back(
      4.177218660490202e6, 0.855175931351848e6, 4.728281850269709e6);

  GPSTransform gps_tform(GPSTransform::Ellipsoid::WGS84);

  const auto ell = gps_tform.ECEFToEllipsoid(xyz);
  const auto xyz2 = gps_tform.EllipsoidToECEF(ell);

  for (size_t i = 0; i < xyz.size(); ++i) {
    EXPECT_THAT(xyz[i], EigenMatrixNear(xyz2[i], 1e-5));
  }
}

TEST(GPS, EllipsoidToENUWGS84) {
  std::vector<Eigen::Vector3d> ell;
  ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                   11 + 34. / 60 + 10.51777 / 3600,
                   561.1851);
  ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                   11 + 34. / 60 + 11.77179 / 3600,
                   561.1509);
  std::vector<Eigen::Vector3d> ref_xyz;
  ref_xyz.emplace_back(
      4.177239709042750e6, 0.855153779923415e6, 4.728267404769168e6);
  ref_xyz.emplace_back(
      4.177218660452103e6, 0.855175931344048e6, 4.728281850382507e6);

  GPSTransform gps_tform(GPSTransform::Ellipsoid::WGS84);

  // Get lat0, lon0 origin from ref
  const auto ori_ell = gps_tform.ECEFToEllipsoid({ref_xyz[0]})[0];

  // Get ENU ref from ECEF ref
  const auto ref_enu = gps_tform.ECEFToENU(ref_xyz, ref_xyz[0]);

  // Get ENU from Ell
  const auto enu =
      gps_tform.EllipsoidToENU(ell, ori_ell(0), ori_ell(1), ori_ell(2));

  for (size_t i = 0; i < ell.size(); ++i) {
    EXPECT_THAT(enu[i], EigenMatrixNear(ref_enu[i], 1e-8));
  }
}

TEST(GPS, ECEFToENU) {
  std::vector<Eigen::Vector3d> ell;
  ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                   11 + 34. / 60 + 10.51777 / 3600,
                   561.1851);
  ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                   11 + 34. / 60 + 11.77179 / 3600,
                   561.1509);
  std::vector<Eigen::Vector3d> ref_xyz;
  ref_xyz.emplace_back(
      4.177239709042750e6, 0.855153779923415e6, 4.728267404769168e6);
  ref_xyz.emplace_back(
      4.177218660452103e6, 0.855175931344048e6, 4.728281850382507e6);

  GPSTransform gps_tform(GPSTransform::Ellipsoid::WGS84);

  const auto xyz = gps_tform.EllipsoidToECEF(ell);

  // Get ENU from ECEF ref
  const auto ref_enu = gps_tform.ECEFToENU(ref_xyz, ref_xyz[0]);

  // Get ENU from ECEF
  const auto enu = gps_tform.ECEFToENU(xyz, xyz[0]);

  for (size_t i = 0; i < ell.size(); ++i) {
    EXPECT_THAT(enu[i], EigenMatrixNear(ref_enu[i], 1e-8));
  }
}

TEST(GPS, ENUToEllipsoidWGS84) {
  std::vector<Eigen::Vector3d> ref_ell;
  ref_ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                       11 + 34. / 60 + 10.51777 / 3600,
                       561.1851);
  ref_ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                       11 + 34. / 60 + 11.77179 / 3600,
                       561.1509);

  std::vector<Eigen::Vector3d> xyz;
  xyz.emplace_back(
      4.177239709042750e6, 0.855153779923415e6, 4.728267404769168e6);
  xyz.emplace_back(
      4.177218660452103e6, 0.855175931344048e6, 4.728281850382507e6);

  GPSTransform gps_tform(GPSTransform::Ellipsoid::WGS84);

  // Get lat0, lon0 origin from ref
  const auto ori_ell = gps_tform.ECEFToEllipsoid(xyz);
  const double lat0 = ori_ell[0](0);
  const double lon0 = ori_ell[0](1);
  const double alt0 = ori_ell[0](2);

  // Get ENU from ECEF
  const auto enu = gps_tform.ECEFToENU(xyz, xyz[0]);

  const auto xyz_enu = gps_tform.ENUToECEF(enu, lat0, lon0, alt0);

  // Get Ell from ENU
  const auto ell = gps_tform.ENUToEllipsoid(enu, lat0, lon0, alt0);

  for (size_t i = 0; i < ell.size(); ++i) {
    EXPECT_THAT(ell[i], EigenMatrixNear(ref_ell[i], 1e-5));
  }
}

TEST(GPS, ENUToECEF) {
  std::vector<Eigen::Vector3d> ell;
  ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                   11 + 34. / 60 + 10.51777 / 3600,
                   561.1851);
  ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                   11 + 34. / 60 + 11.77179 / 3600,
                   561.1509);
  std::vector<Eigen::Vector3d> ref_xyz;
  ref_xyz.emplace_back(
      4.177239709042750e6, 0.855153779923415e6, 4.728267404769168e6);
  ref_xyz.emplace_back(
      4.177218660452103e6, 0.855175931344048e6, 4.728281850382507e6);

  GPSTransform gps_tform(GPSTransform::Ellipsoid::WGS84);

  // Get lat0, lon0 origin from Ell
  const double lat0 = ell[0](0);
  const double lon0 = ell[0](1);
  const double alt0 = ell[0](2);

  // Get ENU from Ell
  const auto enu = gps_tform.EllipsoidToENU(ell, lat0, lon0, alt0);

  // Get XYZ from ENU
  const auto xyz = gps_tform.ENUToECEF(enu, lat0, lon0, alt0);

  for (size_t i = 0; i < ell.size(); ++i) {
    EXPECT_THAT(xyz[i], EigenMatrixNear(ref_xyz[i], 1e-8));
  }
}

TEST(GPS, EllipsoidToUTMWGS84) {
  std::vector<Eigen::Vector3d> ell;
  ell.reserve(3);
  //(48.1476954472, 11.5695882694, 561.1851) zone32
  ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                   11 + 34. / 60 + 10.51777 / 3600,
                   561.1851);
  //(48.1478904861, 11.5699366083, 561.1509) zone32
  ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                   11 + 34. / 60 + 11.77179 / 3600,
                   561.1509);
  //(48.1478904861, 12.5699366083, 561.1509) zone33
  ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                   12 + 34. / 60 + 11.77179 / 3600,
                   561.1509);

  // Calculated from GeographicLib
  // echo 48.1476954472 11.5695882694 | TransverseMercatorProj -l 9 -p 9
  // echo 48.1478904861 11.5699366083 | TransverseMercatorProj -l 9 -p 9
  // echo 48.1478904861 12.5699366083 | TransverseMercatorProj -l 9 -p 9
  const double east_offset = 5.0e5;
  std::vector<Eigen::Vector3d> ref_utm;
  ref_utm.reserve(3);
  ref_utm.emplace_back(
      1.91125018424899e5 + east_offset, 5.335909515367108e6, 561.1851);
  ref_utm.emplace_back(
      1.91150201163177e5 + east_offset, 5.335932057413140e6, 561.1509);
  ref_utm.emplace_back(
      2.65520501819149e5 + east_offset, 5.338903134602814e6, 561.1509);

  GPSTransform gps_tform(GPSTransform::Ellipsoid::WGS84);
  const auto [utm, zone] = gps_tform.EllipsoidToUTM(ell);

  double tolerance = 1e-8;  // 10nm
  for (size_t i = 0; i < ell.size(); ++i) {
    EXPECT_THAT(utm[i], EigenMatrixNear(ref_utm[i], tolerance));
  }
  EXPECT_EQ(zone, 32);
}

TEST(GPS, EllipsoidToUTMGRS80) {
  std::vector<Eigen::Vector3d> ell;
  ell.reserve(3);
  //(48.1476954472, 11.5695882694, 561.1851) zone32
  ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                   11 + 34. / 60 + 10.51777 / 3600,
                   561.1851);
  //(48.1478904861, 11.5699366083, 561.1509) zone32
  ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                   11 + 34. / 60 + 11.77179 / 3600,
                   561.1509);
  //(48.1478904861, 12.5699366083, 561.1509) zone33
  ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                   12 + 34. / 60 + 11.77179 / 3600,
                   561.1509);

  // Calculated from GeographicLib
  // echo 48.1476954472 11.5695882694 | TransverseMercatorProj -l 9 -p 9
  // -e 6378137.0 1.0/298.257222100882711243162837
  // echo 48.1478904861 11.5699366083 | TransverseMercatorProj -l 9 -p 9
  // -e 6378137.0 1.0/298.257222100882711243162837
  // echo 48.1478904861 12.5699366083 | TransverseMercatorProj -l 9 -p 9
  // -e 6378137.0 1.0/298.257222100882711243162837
  const double east_offset = 5.0e5;
  std::vector<Eigen::Vector3d> ref_utm;
  ref_utm.reserve(3);
  ref_utm.emplace_back(
      1.91125018426643e5 + east_offset, 5.335909515244992e6, 561.1851);
  ref_utm.emplace_back(
      1.91150201164921e5 + east_offset, 5.335932057291023e6, 561.1509);
  ref_utm.emplace_back(
      2.65520501821572e5 + east_offset, 5.338903134480723e6, 561.1509);

  GPSTransform gps_tform(GPSTransform::Ellipsoid::GRS80);
  const auto [utm, zone] = gps_tform.EllipsoidToUTM(ell);

  double tolerance = 1e-8;  // 10nm
  for (size_t i = 0; i < ell.size(); ++i) {
    EXPECT_THAT(utm[i], EigenMatrixNear(ref_utm[i], tolerance));
  }
  EXPECT_EQ(zone, 32);
}

TEST(GPS, UTMToEllipsoidWGS84) {
  const double east_offset = 5.0e5;
  std::vector<Eigen::Vector3d> utm;
  utm.reserve(3);
  utm.emplace_back(
      1.91125018424899e5 + east_offset, 5.335909515367108e6, 561.1851);
  utm.emplace_back(
      1.91150201163177e5 + east_offset, 5.335932057413140e6, 561.1509);
  utm.emplace_back(
      2.65520501819149e5 + east_offset, 5.338903134602814e6, 561.1509);

  // Calculated from GeographicLib
  // echo 191125.018424899 5335909.515367108|TransverseMercatorProj -l 9 -p 9 -r
  // echo 191150.201163177 5335932.057413140|TransverseMercatorProj -l 9 -p 9 -r
  // echo 265520.501819149 5338903.134602814|TransverseMercatorProj -l 9 -p 9 -r
  std::vector<Eigen::Vector3d> ref_ell;
  ref_ell.reserve(3);
  //(48.1476954472, 11.5695882694, 561.1851) zone32
  ref_ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                       11 + 34. / 60 + 10.51777 / 3600,
                       561.1851);
  //(48.1478904861, 11.5699366083, 561.1509) zone32
  ref_ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                       11 + 34. / 60 + 11.77179 / 3600,
                       561.1509);
  //(48.1478904861, 12.5699366083, 561.1509) zone33
  ref_ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                       12 + 34. / 60 + 11.77179 / 3600,
                       561.1509);
  GPSTransform gps_tform(GPSTransform::Ellipsoid::WGS84);
  const auto ell = gps_tform.UTMToEllipsoid(utm, 32, true);

  double tolerance = 1e-8;
  for (size_t i = 0; i < ell.size(); ++i) {
    EXPECT_THAT(ell[i], EigenMatrixNear(ref_ell[i], tolerance));
  }
}

TEST(GPS, UTMToEllipsoidGRS80) {
  const double east_offset = 5.0e5;
  std::vector<Eigen::Vector3d> utm;
  utm.reserve(3);
  utm.emplace_back(
      1.91125018426643e5 + east_offset, 5.335909515244992e6, 561.1851);
  utm.emplace_back(
      1.91150201164921e5 + east_offset, 5.335932057291023e6, 561.1509);
  utm.emplace_back(
      2.65520501821572e5 + east_offset, 5.338903134480723e6, 561.1509);

  // Calculated from GeographicLib
  // echo 191125.018424899 5335909.515367108|TransverseMercatorProj -l 9 -p 9 -r
  // -e 6378137.0 1.0/298.257222100882711243162837
  // echo 191150.201163177 5335932.057413140|TransverseMercatorProj -l 9 -p 9 -r
  // -e 6378137.0 1.0/298.257222100882711243162837
  // echo 265520.501819149 5338903.134602814|TransverseMercatorProj -l 9 -p 9 -r
  // -e 6378137.0 1.0/298.257222100882711243162837
  std::vector<Eigen::Vector3d> ref_ell;
  ref_ell.reserve(3);
  //(48.1476954472, 11.5695882694, 561.1851) zone32
  ref_ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                       11 + 34. / 60 + 10.51777 / 3600,
                       561.1851);
  //(48.1478904861, 11.5699366083, 561.1509) zone32
  ref_ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                       11 + 34. / 60 + 11.77179 / 3600,
                       561.1509);
  //(48.1478904861, 12.5699366083, 561.1509) zone33
  ref_ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                       12 + 34. / 60 + 11.77179 / 3600,
                       561.1509);
  GPSTransform gps_tform(GPSTransform::Ellipsoid::WGS84);
  const auto ell = gps_tform.UTMToEllipsoid(utm, 32, true);

  double tolerance = 1e-8;
  for (size_t i = 0; i < ell.size(); ++i) {
    EXPECT_THAT(ell[i], EigenMatrixNear(ref_ell[i], tolerance));
  }
}

}  // namespace
}  // namespace colmap
