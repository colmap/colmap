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

#define TEST_NAME "base/point2d"
#include "util/testing.h"

#include "base/point2d.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestDefault) {
  Point2D point2D;
  BOOST_CHECK_EQUAL(point2D.X(), 0);
  BOOST_CHECK_EQUAL(point2D.Y(), 0);
  BOOST_CHECK_EQUAL(point2D.XY()[0], point2D.X());
  BOOST_CHECK_EQUAL(point2D.XY()[1], point2D.Y());
  BOOST_CHECK_EQUAL(point2D.Point3DId(), kInvalidPoint3DId);
  BOOST_CHECK_EQUAL(point2D.HasPoint3D(), false);
}

BOOST_AUTO_TEST_CASE(TestXY) {
  Point2D point2D;
  BOOST_CHECK_EQUAL(point2D.X(), 0);
  BOOST_CHECK_EQUAL(point2D.Y(), 0);
  BOOST_CHECK_EQUAL(point2D.XY()[0], point2D.X());
  BOOST_CHECK_EQUAL(point2D.XY()[1], point2D.Y());
  point2D.SetXY(Eigen::Vector2d(0.1, 0.2));
  BOOST_CHECK_EQUAL(point2D.X(), 0.1);
  BOOST_CHECK_EQUAL(point2D.Y(), 0.2);
  BOOST_CHECK_EQUAL(point2D.XY()[0], point2D.X());
  BOOST_CHECK_EQUAL(point2D.XY()[1], point2D.Y());
}

BOOST_AUTO_TEST_CASE(TestPoint3DId) {
  Point2D point2D;
  BOOST_CHECK_EQUAL(point2D.Point3DId(), kInvalidPoint3DId);
  BOOST_CHECK_EQUAL(point2D.HasPoint3D(), false);
  point2D.SetPoint3DId(1);
  BOOST_CHECK_EQUAL(point2D.Point3DId(), 1);
  BOOST_CHECK_EQUAL(point2D.HasPoint3D(), true);
  point2D.SetPoint3DId(kInvalidPoint3DId);
  BOOST_CHECK_EQUAL(point2D.Point3DId(), kInvalidPoint3DId);
  BOOST_CHECK_EQUAL(point2D.HasPoint3D(), false);
}
