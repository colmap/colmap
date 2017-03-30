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

#define TEST_NAME "base/camera_database"
#include "util/testing.h"

#include "base/camera_database.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestInitialization) {
  CameraDatabase database;
  camera_specs_t specs = InitializeCameraSpecs();
  BOOST_CHECK_EQUAL(database.NumEntries(), specs.size());
}

BOOST_AUTO_TEST_CASE(TestExactMatch) {
  CameraDatabase database;
  double sensor_width;
  BOOST_CHECK(
      database.QuerySensorWidth("canon", "digitalixus100is", &sensor_width));
  BOOST_CHECK_EQUAL(sensor_width, 6.1600f);
}

BOOST_AUTO_TEST_CASE(TestAmbiguousMatch) {
  CameraDatabase database;
  double sensor_width;
  BOOST_CHECK(
      !database.QuerySensorWidth("canon", "digitalixus", &sensor_width));
  BOOST_CHECK_EQUAL(sensor_width, 6.1600f);
}
