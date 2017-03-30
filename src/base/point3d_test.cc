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

#define TEST_NAME "base/point3d"
#include "util/testing.h"

#include "base/point3d.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestDefault) {
  Point3D point3D;
  BOOST_CHECK_EQUAL(point3D.X(), 0);
  BOOST_CHECK_EQUAL(point3D.Y(), 0);
  BOOST_CHECK_EQUAL(point3D.Z(), 0);
  BOOST_CHECK_EQUAL(point3D.XYZ()[0], point3D.X());
  BOOST_CHECK_EQUAL(point3D.XYZ()[1], point3D.Y());
  BOOST_CHECK_EQUAL(point3D.XYZ()[2], point3D.Z());
  BOOST_CHECK_EQUAL(point3D.Color()[0], 0);
  BOOST_CHECK_EQUAL(point3D.Color()[1], 0);
  BOOST_CHECK_EQUAL(point3D.Color()[2], 0);
  BOOST_CHECK_EQUAL(point3D.Error(), -1.0);
  BOOST_CHECK_EQUAL(point3D.HasError(), false);
  BOOST_CHECK_EQUAL(point3D.Track().Length(), 0);
}

BOOST_AUTO_TEST_CASE(TestXYZ) {
  Point3D point3D;
  BOOST_CHECK_EQUAL(point3D.X(), 0);
  BOOST_CHECK_EQUAL(point3D.Y(), 0);
  BOOST_CHECK_EQUAL(point3D.Z(), 0);
  BOOST_CHECK_EQUAL(point3D.XYZ()[0], point3D.X());
  BOOST_CHECK_EQUAL(point3D.XYZ()[1], point3D.Y());
  BOOST_CHECK_EQUAL(point3D.XYZ()[2], point3D.Z());
  point3D.SetXYZ(Eigen::Vector3d(0.1, 0.2, 0.3));
  BOOST_CHECK_EQUAL(point3D.X(), 0.1);
  BOOST_CHECK_EQUAL(point3D.Y(), 0.2);
  BOOST_CHECK_EQUAL(point3D.Z(), 0.3);
  BOOST_CHECK_EQUAL(point3D.XYZ()[0], point3D.X());
  BOOST_CHECK_EQUAL(point3D.XYZ()[1], point3D.Y());
  BOOST_CHECK_EQUAL(point3D.XYZ()[2], point3D.Z());
  point3D.XYZ() = Eigen::Vector3d(0.2, 0.3, 0.4);
  BOOST_CHECK_EQUAL(point3D.X(), 0.2);
  BOOST_CHECK_EQUAL(point3D.Y(), 0.3);
  BOOST_CHECK_EQUAL(point3D.Z(), 0.4);
  BOOST_CHECK_EQUAL(point3D.XYZ()[0], point3D.X());
  BOOST_CHECK_EQUAL(point3D.XYZ()[1], point3D.Y());
  BOOST_CHECK_EQUAL(point3D.XYZ()[2], point3D.Z());
}

BOOST_AUTO_TEST_CASE(TestRGB) {
  Point3D point3D;
  BOOST_CHECK_EQUAL(point3D.Color()[0], 0);
  BOOST_CHECK_EQUAL(point3D.Color()[1], 0);
  BOOST_CHECK_EQUAL(point3D.Color()[2], 0);
  point3D.SetColor(Eigen::Vector3ub(1, 2, 3));
  BOOST_CHECK_EQUAL(point3D.Color()[0], 1);
  BOOST_CHECK_EQUAL(point3D.Color()[1], 2);
  BOOST_CHECK_EQUAL(point3D.Color()[2], 3);
}

BOOST_AUTO_TEST_CASE(TestError) {
  Point3D point3D;
  BOOST_CHECK_EQUAL(point3D.Error(), -1.0);
  BOOST_CHECK_EQUAL(point3D.HasError(), false);
  point3D.SetError(1.0);
  BOOST_CHECK_EQUAL(point3D.Error(), 1.0);
  BOOST_CHECK_EQUAL(point3D.HasError(), true);
  point3D.SetError(-1.0);
  BOOST_CHECK_EQUAL(point3D.Error(), -1.0);
  BOOST_CHECK_EQUAL(point3D.HasError(), false);
}

BOOST_AUTO_TEST_CASE(TestTrack) {
  Point3D point3D;
  BOOST_CHECK_EQUAL(point3D.Track().Length(), 0);
  point3D.SetTrack(Track());
  BOOST_CHECK_EQUAL(point3D.Track().Length(), 0);
  Track track;
  track.AddElement(0, 1);
  track.AddElement(0, 2);
  point3D.SetTrack(track);
  BOOST_CHECK_EQUAL(point3D.Track().Length(), 2);
  track.AddElement(0, 3);
  BOOST_CHECK_EQUAL(point3D.Track().Length(), 2);
}
