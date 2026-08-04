// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/geometry/pose_prior_io.h"

#include "colmap/util/eigen_matchers.h"
#include "colmap/util/testing.h"

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// M_PI is not standard C++ and needs _USE_MATH_DEFINES before <cmath> on
// MSVC, which is a compile break waiting to happen in a header-order change.
constexpr double kPi = 3.14159265358979323846;

std::filesystem::path WriteTestJSON(const std::string& content) {
  const auto path = CreateTestDir() / "test.json";
  std::ofstream file(path);
  file << content;
  file.close();
  return path;
}

PosePriorArchive ReadJSON(const std::string& content) {
  return ReadPosePriorArchive(WriteTestJSON(content));
}

// A well-formed archive with position and standard deviations only. Tests that
// probe one rule build on this so the rule under test is the only thing that
// differs from a known-good file.
constexpr const char* kMinimalArchive = R"({
  "schema_version": 1,
  "coordinate_system": "WGS84",
  "sensor_type": "CAMERA",
  "ellipsoid": "WGS84",
  "height_datum": "ELLIPSOIDAL",
  "position_covariance_frame": "LOCAL_ENU",
  "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
  "data": [
    ["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0],
    ["b.jpg", 45.001, -73.001, 41.0, 2.0, 2.0, 4.0]
  ]
})";

// Adds the gravity group in both schema and metadata.
constexpr const char* kGravityArchive = R"({
  "schema_version": 1,
  "coordinate_system": "WGS84",
  "sensor_type": "CAMERA",
  "ellipsoid": "WGS84",
  "height_datum": "ELLIPSOIDAL",
  "position_covariance_frame": "LOCAL_ENU",
  "gravity_frame": "CAMERA",
  "gravity_direction": "DOWN",
  "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
             "GX", "GY", "GZ"],
  "data": [
    ["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, 0.0, 1.0, 0.0],
    ["b.jpg", 45.001, -73.001, 41.0, 2.0, 2.0, 4.0, null, null, null]
  ]
})";

constexpr const char* kHeadingArchive = R"({
  "schema_version": 1,
  "coordinate_system": "WGS84",
  "sensor_type": "CAMERA",
  "ellipsoid": "WGS84",
  "height_datum": "ELLIPSOIDAL",
  "position_covariance_frame": "LOCAL_ENU",
  "gravity_frame": "CAMERA",
  "gravity_direction": "DOWN",
  "heading_reference": "TRUE_NORTH",
  "heading_axis": "CAMERA_FORWARD_PROJECTED_HORIZONTAL",
  "heading_rotation": "CLOCKWISE_FROM_NORTH",
  "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
             "GX", "GY", "GZ", "HEADING_DEG", "HEADING_STD_DEG"],
  "data": [
    ["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, 0.0, 1.0, 0.0, 90.0, 5.0],
    ["b.jpg", 45.001, -73.001, 41.0, 2.0, 2.0, 4.0, 0.0, 1.0, 0.0, null, null]
  ]
})";

////////////////////////////////////////////////////////////////////////////
// Accepting a valid archive
////////////////////////////////////////////////////////////////////////////

TEST(PosePriorArchive, ReadsPositionAndStandardDeviations) {
  const PosePriorArchive archive = ReadJSON(kMinimalArchive);
  ASSERT_EQ(archive.rows.size(), 2u);
  EXPECT_FALSE(archive.schema_has_gravity);
  EXPECT_FALSE(archive.schema_has_heading);

  EXPECT_EQ(archive.rows[0].name, "a.jpg");
  EXPECT_THAT(archive.rows[0].position_wgs84,
              EigenMatrixNear(Eigen::Vector3d(45.0, -73.0, 40.0), 1e-12));
  // Standard deviations become a diagonal covariance of their squares.
  EXPECT_THAT(
      archive.rows[0].position_covariance,
      EigenMatrixNear(
          Eigen::Matrix3d(Eigen::Vector3d(4.0, 4.0, 16.0).asDiagonal()),
          1e-12));
  EXPECT_FALSE(archive.rows[0].gravity.has_value());
  EXPECT_FALSE(archive.rows[0].heading_rad.has_value());
}

TEST(PosePriorArchive, ReadsFullCovariance) {
  const PosePriorArchive archive = ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT",
               "COV_TXX", "COV_TXY", "COV_TXZ",
               "COV_TYY", "COV_TYZ", "COV_TZZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 4.0, 1.0, 0.5, 9.0, 2.0, 16.0]]
  })");
  ASSERT_EQ(archive.rows.size(), 1u);
  Eigen::Matrix3d expected;
  expected << 4.0, 1.0, 0.5, 1.0, 9.0, 2.0, 0.5, 2.0, 16.0;
  EXPECT_THAT(archive.rows[0].position_covariance,
              EigenMatrixNear(expected, 1e-12));
}

TEST(PosePriorArchive, ColumnOrderIsFree) {
  const PosePriorArchive archive = ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["STD_TZ", "ALT", "NAME", "LON", "STD_TX", "LAT", "STD_TY"],
    "data": [[4.0, 40.0, "a.jpg", -73.0, 2.0, 45.0, 2.0]]
  })");
  ASSERT_EQ(archive.rows.size(), 1u);
  EXPECT_EQ(archive.rows[0].name, "a.jpg");
  EXPECT_THAT(archive.rows[0].position_wgs84,
              EigenMatrixNear(Eigen::Vector3d(45.0, -73.0, 40.0), 1e-12));
  EXPECT_THAT(
      archive.rows[0].position_covariance,
      EigenMatrixNear(
          Eigen::Matrix3d(Eigen::Vector3d(4.0, 4.0, 16.0).asDiagonal()),
          1e-12));
}

TEST(PosePriorArchive, ReadsAndNormalizesGravity) {
  const PosePriorArchive archive = ReadJSON(kGravityArchive);
  ASSERT_EQ(archive.rows.size(), 2u);
  EXPECT_TRUE(archive.schema_has_gravity);
  ASSERT_TRUE(archive.rows[0].gravity.has_value());
  EXPECT_THAT(*archive.rows[0].gravity,
              EigenMatrixNear(Eigen::Vector3d(0.0, 1.0, 0.0), 1e-12));
  // A whole-group null is how a row says it has no reading.
  EXPECT_FALSE(archive.rows[1].gravity.has_value());
}

TEST(PosePriorArchive, NormalizesGravityWithinTolerance) {
  const PosePriorArchive archive = ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "gravity_frame": "CAMERA",
    "gravity_direction": "DOWN",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
               "GX", "GY", "GZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, 0.0, 1.005, 0.0]]
  })");
  ASSERT_TRUE(archive.rows[0].gravity.has_value());
  EXPECT_NEAR(archive.rows[0].gravity->norm(), 1.0, 1e-12);
}

TEST(PosePriorArchive, ReadsHeadingInRadians) {
  const PosePriorArchive archive = ReadJSON(kHeadingArchive);
  ASSERT_EQ(archive.rows.size(), 2u);
  EXPECT_TRUE(archive.schema_has_heading);
  ASSERT_TRUE(archive.rows[0].heading_rad.has_value());
  EXPECT_NEAR(*archive.rows[0].heading_rad, 0.5 * kPi, 1e-12);
  EXPECT_NEAR(*archive.rows[0].heading_stddev_rad, 5.0 * kPi / 180.0, 1e-12);
  EXPECT_FALSE(archive.rows[1].heading_rad.has_value());
  // The gravity that heading depends on is still read for that row.
  EXPECT_TRUE(archive.rows[1].gravity.has_value());
}

TEST(PosePriorArchive, KeepsAGeographicallyDistantRow) {
  // A row on another continent is well-formed. Whether it belongs to this
  // capture is a question about the capture, which robust fitting answers --
  // dropping it here would hide a real GPS fault behind a clean import.
  const PosePriorArchive archive = ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [
      ["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0],
      ["b.jpg", -33.9, 151.2, 40.0, 2.0, 2.0, 4.0]
    ]
  })");
  EXPECT_EQ(archive.rows.size(), 2u);
}

////////////////////////////////////////////////////////////////////////////
// Metadata is fail-closed
////////////////////////////////////////////////////////////////////////////

TEST(PosePriorArchive, RejectsUnsupportedSchemaVersion) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 2,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsMissingSchemaVersion) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsMissingHeightDatum) {
  // An ellipsoidal height and an orthometric one differ by tens of metres.
  // Assuming one would put the whole scene at the wrong altitude, silently.
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsWrongHeightDatumValue) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ORTHOMETRIC",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsNonWGS84CoordinateSystem) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "CARTESIAN",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsUnknownTopLevelKey) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "translation_convention": "WORLD_FROM_CAM",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsGravityMetadataWithoutGravityColumns) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "gravity_frame": "CAMERA",
    "gravity_direction": "DOWN",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsGravityColumnsWithoutGravityMetadata) {
  // Without gravity_frame/gravity_direction there is nothing saying the
  // vector is camera-frame and points down, and the sign convention is
  // exactly the sort of thing two producers disagree about.
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
               "GX", "GY", "GZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, 0.0, 1.0, 0.0]]
  })"));
}

////////////////////////////////////////////////////////////////////////////
// Schema is fail-closed
////////////////////////////////////////////////////////////////////////////

TEST(PosePriorArchive, RejectsUnknownColumn) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
               "TIMESTAMP"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, 12.0]]
  })"));
}

TEST(PosePriorArchive, RejectsDuplicateColumn) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LAT", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 45.0, 45.0, 40.0, 2.0, 2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsMissingPositionColumn) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 45.0, -73.0, 2.0, 2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsBothUncertaintyForms) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
               "COV_TXX", "COV_TXY", "COV_TXZ",
               "COV_TYY", "COV_TYZ", "COV_TZZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0,
              4.0, 0.0, 0.0, 4.0, 0.0, 16.0]]
  })"));
}

TEST(PosePriorArchive, RejectsNoUncertaintyForm) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT"],
    "data": [["a.jpg", 45.0, -73.0, 40.0]]
  })"));
}

TEST(PosePriorArchive, RejectsPartialGravityGroupInSchema) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "gravity_frame": "CAMERA",
    "gravity_direction": "DOWN",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
               "GX", "GY"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, 0.0, 1.0]]
  })"));
}

TEST(PosePriorArchive, RejectsHeadingWithoutGravityGroup) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "heading_reference": "TRUE_NORTH",
    "heading_axis": "CAMERA_FORWARD_PROJECTED_HORIZONTAL",
    "heading_rotation": "CLOCKWISE_FROM_NORTH",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
               "HEADING_DEG", "HEADING_STD_DEG"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, 90.0, 5.0]]
  })"));
}

////////////////////////////////////////////////////////////////////////////
// Rows are fail-closed
////////////////////////////////////////////////////////////////////////////

TEST(PosePriorArchive, RejectsEmptyData) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": []
  })"));
}

TEST(PosePriorArchive, RejectsWrongCellCount) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0]]
  })"));
}

TEST(PosePriorArchive, RejectsEmptyName) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsDuplicateName) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [
      ["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0],
      ["a.jpg", 45.5, -73.5, 41.0, 2.0, 2.0, 4.0]
    ]
  })"));
}

TEST(PosePriorArchive, RejectsNullPosition) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 45.0, -73.0, null, 2.0, 2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsOutOfRangeLatitude) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 91.0, -73.0, 40.0, 2.0, 2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsOutOfRangeLongitude) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 45.0, -181.0, 40.0, 2.0, 2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsNonNumericCell) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", "north", -73.0, 40.0, 2.0, 2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsZeroStandardDeviation) {
  // Zero uncertainty is an infinite weight: that row would pin the solve.
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 0.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsNegativeStandardDeviation) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, -2.0, 4.0]]
  })"));
}

TEST(PosePriorArchive, RejectsSingularCovariance) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT",
               "COV_TXX", "COV_TXY", "COV_TXZ",
               "COV_TYY", "COV_TYZ", "COV_TZZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 4.0, 0.0, 0.0, 0.0, 0.0, 16.0]]
  })"));
}

TEST(PosePriorArchive, RejectsIndefiniteCovariance) {
  // Correlation larger than the variances allow: not a covariance at all.
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT",
               "COV_TXX", "COV_TXY", "COV_TXZ",
               "COV_TYY", "COV_TYZ", "COV_TZZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 1.0, 5.0, 0.0, 1.0, 0.0, 1.0]]
  })"));
}

TEST(PosePriorArchive, RejectsPartialGravityRow) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "gravity_frame": "CAMERA",
    "gravity_direction": "DOWN",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
               "GX", "GY", "GZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, 0.0, 1.0, null]]
  })"));
}

TEST(PosePriorArchive, RejectsZeroGravity) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "gravity_frame": "CAMERA",
    "gravity_direction": "DOWN",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
               "GX", "GY", "GZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, 0.0, 0.0, 0.0]]
  })"));
}

TEST(PosePriorArchive, RejectsNonUnitGravity) {
  // 9.81 is what an archive written from raw accelerometer counts looks like.
  // This column is a normalized direction; accepting the magnitude would
  // rescale nothing and mean nothing.
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "gravity_frame": "CAMERA",
    "gravity_direction": "DOWN",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
               "GX", "GY", "GZ"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, 0.0, 9.80665, 0.0]]
  })"));
}

TEST(PosePriorArchive, RejectsPartialHeadingRow) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "gravity_frame": "CAMERA",
    "gravity_direction": "DOWN",
    "heading_reference": "TRUE_NORTH",
    "heading_axis": "CAMERA_FORWARD_PROJECTED_HORIZONTAL",
    "heading_rotation": "CLOCKWISE_FROM_NORTH",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
               "GX", "GY", "GZ", "HEADING_DEG", "HEADING_STD_DEG"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0,
              0.0, 1.0, 0.0, 90.0, null]]
  })"));
}

TEST(PosePriorArchive, RejectsHeadingOnARowWithoutGravity) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "gravity_frame": "CAMERA",
    "gravity_direction": "DOWN",
    "heading_reference": "TRUE_NORTH",
    "heading_axis": "CAMERA_FORWARD_PROJECTED_HORIZONTAL",
    "heading_rotation": "CLOCKWISE_FROM_NORTH",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
               "GX", "GY", "GZ", "HEADING_DEG", "HEADING_STD_DEG"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0,
              null, null, null, 90.0, 5.0]]
  })"));
}

TEST(PosePriorArchive, RejectsOutOfRangeHeading) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "gravity_frame": "CAMERA",
    "gravity_direction": "DOWN",
    "heading_reference": "TRUE_NORTH",
    "heading_axis": "CAMERA_FORWARD_PROJECTED_HORIZONTAL",
    "heading_rotation": "CLOCKWISE_FROM_NORTH",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
               "GX", "GY", "GZ", "HEADING_DEG", "HEADING_STD_DEG"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0,
              0.0, 1.0, 0.0, 360.0, 5.0]]
  })"));
}

TEST(PosePriorArchive, RejectsNonPositiveHeadingUncertainty) {
  EXPECT_ANY_THROW(ReadJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "gravity_frame": "CAMERA",
    "gravity_direction": "DOWN",
    "heading_reference": "TRUE_NORTH",
    "heading_axis": "CAMERA_FORWARD_PROJECTED_HORIZONTAL",
    "heading_rotation": "CLOCKWISE_FROM_NORTH",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ",
               "GX", "GY", "GZ", "HEADING_DEG", "HEADING_STD_DEG"],
    "data": [["a.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0,
              0.0, 1.0, 0.0, 90.0, 0.0]]
  })"));
}

TEST(PosePriorArchive, RejectsMalformedFile) {
  EXPECT_ANY_THROW(ReadJSON("{ not json"));
}

////////////////////////////////////////////////////////////////////////////
// ToPosePriors
////////////////////////////////////////////////////////////////////////////

TEST(PosePriorArchive, ToPosePriorsProducesOneCompletePriorPerRow) {
  const PosePriorArchive archive = ReadJSON(kHeadingArchive);
  const std::vector<data_t> data_ids = {
      data_t(sensor_t(SensorType::CAMERA, 1), 10),
      data_t(sensor_t(SensorType::CAMERA, 1), 11)};
  const std::vector<PosePrior> priors = archive.ToPosePriors(data_ids);
  ASSERT_EQ(priors.size(), 2u);

  EXPECT_EQ(priors[0].corr_data_id, data_ids[0]);
  EXPECT_EQ(priors[0].coordinate_system, PosePrior::CoordinateSystem::WGS84);
  EXPECT_THAT(priors[0].position,
              EigenMatrixNear(Eigen::Vector3d(45.0, -73.0, 40.0), 1e-12));
  EXPECT_TRUE(priors[0].HasPosition());
  EXPECT_TRUE(priors[0].HasPositionCov());
  EXPECT_TRUE(priors[0].HasGravity());
  EXPECT_TRUE(priors[0].HasHeading());

  // Row two declared no heading; the prior must not claim one.
  EXPECT_TRUE(priors[1].HasGravity());
  EXPECT_FALSE(priors[1].HasHeading());
}

TEST(PosePriorArchive, ToPosePriorsRequiresOneDataIdPerRow) {
  const PosePriorArchive archive = ReadJSON(kMinimalArchive);
  const std::vector<data_t> too_few = {
      data_t(sensor_t(SensorType::CAMERA, 1), 10)};
  EXPECT_ANY_THROW(archive.ToPosePriors(too_few));
}

}  // namespace
}  // namespace colmap
