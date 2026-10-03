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
  EXPECT_TRUE(
      !database.QuerySensorWidth("canon", "digitalixus", &sensor_width));
  EXPECT_EQ(sensor_width, 6.1600f);
}

}  // namespace
}  // namespace colmap
