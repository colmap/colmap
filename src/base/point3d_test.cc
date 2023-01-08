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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

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
