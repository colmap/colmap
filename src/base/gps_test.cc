// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#define TEST_NAME "base/gps"
#include "util/testing.h"

#include "base/gps.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEllToXYZGRS80) {
  std::vector<Eigen::Vector3d> ell;
  ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                   11 + 34. / 60 + 10.51777 / 3600, 561.1851);
  ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                   11 + 34. / 60 + 11.77179 / 3600, 561.1509);
  std::vector<Eigen::Vector3d> ref_xyz;
  ref_xyz.emplace_back(4.177239709080851e6, 0.855153779931214e6,
                       4.728267404656370e6);
  ref_xyz.emplace_back(4.177218660490202e6, 0.855175931351848e6,
                       4.728281850269709e6);

  GPSTransform gps_tform(GPSTransform::GRS80);

  const auto xyz = gps_tform.EllToXYZ(ell);

  for (size_t i = 0; i < ell.size(); ++i) {
    BOOST_CHECK(std::abs(xyz[i](0) - ref_xyz[i](0)) < 1e-8);
    BOOST_CHECK(std::abs(xyz[i](1) - ref_xyz[i](1)) < 1e-8);
    BOOST_CHECK(std::abs(xyz[i](2) - ref_xyz[i](2)) < 1e-8);
  }
}

BOOST_AUTO_TEST_CASE(TestEllToXYZWGS84) {
  std::vector<Eigen::Vector3d> ell;
  ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                   11 + 34. / 60 + 10.51777 / 3600, 561.1851);
  ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                   11 + 34. / 60 + 11.77179 / 3600, 561.1509);
  std::vector<Eigen::Vector3d> ref_xyz;
  ref_xyz.emplace_back(4.177239709042750e6, 0.855153779923415e6,
                       4.728267404769168e6);
  ref_xyz.emplace_back(4.177218660452103e6, 0.855175931344048e6,
                       4.728281850382507e6);

  GPSTransform gps_tform(GPSTransform::WGS84);

  const auto xyz = gps_tform.EllToXYZ(ell);

  for (size_t i = 0; i < ell.size(); ++i) {
    BOOST_CHECK(std::abs(xyz[i](0) - ref_xyz[i](0)) < 1e-8);
    BOOST_CHECK(std::abs(xyz[i](1) - ref_xyz[i](1)) < 1e-8);
    BOOST_CHECK(std::abs(xyz[i](2) - ref_xyz[i](2)) < 1e-8);
  }
}

BOOST_AUTO_TEST_CASE(TestXYZToEll_GRS80) {
  std::vector<Eigen::Vector3d> xyz;
  xyz.emplace_back(4.177239709080851e6, 0.855153779931214e6,
                   4.728267404656370e6);
  xyz.emplace_back(4.177218660490202e6, 0.855175931351848e6,
                   4.728281850269709e6);
  std::vector<Eigen::Vector3d> ref_ell;
  ref_ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                       11 + 34. / 60 + 10.51777 / 3600, 561.1851);
  ref_ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                       11 + 34. / 60 + 11.77179 / 3600, 561.1509);

  GPSTransform gps_tform(GPSTransform::GRS80);

  const auto ell = gps_tform.XYZToEll(xyz);

  for (size_t i = 0; i < xyz.size(); ++i) {
    BOOST_CHECK(std::abs(ell[i](0) - ref_ell[i](0)) < 1e-5);
    BOOST_CHECK(std::abs(ell[i](1) - ref_ell[i](1)) < 1e-5);
    BOOST_CHECK(std::abs(ell[i](2) - ref_ell[i](2)) < 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(TestXYZToEll_WGS84) {
  std::vector<Eigen::Vector3d> xyz;
  xyz.emplace_back(4.177239709042750e6, 0.855153779923415e6,
                   4.728267404769168e6);
  xyz.emplace_back(4.177218660452103e6, 0.855175931344048e6,
                   4.728281850382507e6);
  std::vector<Eigen::Vector3d> ref_ell;
  ref_ell.emplace_back(48 + 8. / 60 + 51.70361 / 3600,
                       11 + 34. / 60 + 10.51777 / 3600, 561.1851);
  ref_ell.emplace_back(48 + 8. / 60 + 52.40575 / 3600,
                       11 + 34. / 60 + 11.77179 / 3600, 561.1509);

  GPSTransform gps_tform(GPSTransform::WGS84);

  const auto ell = gps_tform.XYZToEll(xyz);

  for (size_t i = 0; i < xyz.size(); ++i) {
    BOOST_CHECK(std::abs(ell[i](0) - ref_ell[i](0)) < 1e-5);
    BOOST_CHECK(std::abs(ell[i](1) - ref_ell[i](1)) < 1e-5);
    BOOST_CHECK(std::abs(ell[i](2) - ref_ell[i](2)) < 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(TestXYZToEllToXYZ_GRS80) {
  std::vector<Eigen::Vector3d> xyz;
  xyz.emplace_back(4.177239709080851e6, 0.855153779931214e6,
                   4.728267404656370e6);
  xyz.emplace_back(4.177218660490202e6, 0.855175931351848e6,
                   4.728281850269709e6);

  GPSTransform gps_tform(GPSTransform::GRS80);

  const auto ell = gps_tform.XYZToEll(xyz);
  const auto xyz2 = gps_tform.EllToXYZ(ell);

  for (size_t i = 0; i < xyz.size(); ++i) {
    BOOST_CHECK(std::abs(xyz[i](0) - xyz2[i](0)) < 1e-5);
    BOOST_CHECK(std::abs(xyz[i](1) - xyz2[i](1)) < 1e-5);
    BOOST_CHECK(std::abs(xyz[i](2) - xyz2[i](2)) < 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(TestXYZToEllToXYZ_WGS84) {
  std::vector<Eigen::Vector3d> xyz;
  xyz.emplace_back(4.177239709080851e6, 0.855153779931214e6,
                   4.728267404656370e6);
  xyz.emplace_back(4.177218660490202e6, 0.855175931351848e6,
                   4.728281850269709e6);

  GPSTransform gps_tform(GPSTransform::WGS84);

  const auto ell = gps_tform.XYZToEll(xyz);
  const auto xyz2 = gps_tform.EllToXYZ(ell);

  for (size_t i = 0; i < xyz.size(); ++i) {
    BOOST_CHECK(std::abs(xyz[i](0) - xyz2[i](0)) < 1e-5);
    BOOST_CHECK(std::abs(xyz[i](1) - xyz2[i](1)) < 1e-5);
    BOOST_CHECK(std::abs(xyz[i](2) - xyz2[i](2)) < 1e-5);
  }
}
