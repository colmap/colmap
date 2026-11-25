// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sensor/database.h"

#include <tuple>

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

TEST(CameraDatabase, ExactMatchAfterPartialMatches) {
  CameraDatabase database;
  const std::vector<std::tuple<std::string, std::string, float>> cases = {
      {"canon", "digitalixus", 5.3300f},
      {"Canon", "Canon EOS-5D", 35.8000f},
      {"Nikon", "D800", 35.9000f},
      {"Nikon", "Z6III", 36.0000f}};
  for (const auto& [make, model, expected_width] : cases) {
    SCOPED_TRACE(testing::Message() << make << " " << model);
    double sensor_width = 0;
    EXPECT_TRUE(database.QuerySensorWidth(make, model, &sensor_width));
    EXPECT_EQ(sensor_width, expected_width);
  }
}

TEST(CameraDatabase, ExactMatchWithCanonicalMakePrefix) {
  CameraDatabase database;
  double sensor_width = 0;
  EXPECT_TRUE(database.QuerySensorWidth(
      "NIKON CORPORATION", "NIKON D800", &sensor_width));
  EXPECT_EQ(sensor_width, 35.9000f);
  EXPECT_TRUE(database.QuerySensorWidth(
      "NIKON CORPORATION", "NIKON Z6III", &sensor_width));
  EXPECT_EQ(sensor_width, 36.0000f);
}

TEST(CameraDatabase, UniquePartialMatch) {
  CameraDatabase database;
  double sensor_width = 0;
  EXPECT_TRUE(database.QuerySensorWidth(
      "canon", "powershotsx710hs-extra", &sensor_width));
  EXPECT_EQ(sensor_width, 6.1600f);
}

}  // namespace
}  // namespace colmap
