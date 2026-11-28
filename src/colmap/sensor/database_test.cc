// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sensor/database.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CameraDatabase, Initialization) {
  CameraDatabase database;
  camera_specs_t specs = InitializeCameraSpecs();
  EXPECT_EQ(database.NumEntries(), specs.size());
}

TEST(CameraDatabase, ExactMatch) {
  CameraDatabase database;
  double sensor_width;
  EXPECT_TRUE(
      database.QuerySensorWidth("canon", "digitalixus100is", &sensor_width));
  EXPECT_EQ(sensor_width, 6.1600f);
}

TEST(CameraDatabase, AmbiguousMatch) {
  CameraDatabase database;
  double sensor_width;
  // "digitalixus" itself is an exact database entry. This prefix is not.
  EXPECT_TRUE(
      !database.QuerySensorWidth("canon", "digitalixus1", &sensor_width));
  EXPECT_EQ(sensor_width, 6.1600f);
}

TEST(CameraDatabase, ExactMatchAfterAmbiguousPartialMatches) {
  CameraDatabase database;
  double sensor_width = 0;
  EXPECT_TRUE(database.QuerySensorWidth("canon", "digitalixus", &sensor_width));
  EXPECT_EQ(sensor_width, 5.3300f);
  EXPECT_TRUE(database.QuerySensorWidth("nikon", "z6iii", &sensor_width));
  EXPECT_EQ(sensor_width, 36.0000f);
  // Model prefix "NIKON" is shorter than the EXIF make "NIKON CORPORATION".
  EXPECT_TRUE(database.QuerySensorWidth(
      "NIKON CORPORATION", "NIKON D800", &sensor_width));
  EXPECT_EQ(sensor_width, 35.9000f);
}

}  // namespace
}  // namespace colmap
