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
  ref_xyz.emplace_back(4.1772397090808507e6, 0.85515377993121441e6,
                       4.7282674046563692e6);
  ref_xyz.emplace_back(4.1772186604902023e6, 0.8551759313518483e6,
                       4.7282818502697079e6);

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
  xyz.emplace_back(4.1772397090808507e6, 0.85515377993121441e6,
                   4.7282674046563692e6);
  xyz.emplace_back(4.1772186604902023e6, 0.8551759313518483e6,
                   4.7282818502697079e6);
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
