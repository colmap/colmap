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

#define TEST_NAME "base/camera_database"
#include "colmap/base/camera_database.h"

#include "colmap/util/testing.h"

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
