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

#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

std::filesystem::path WriteTestJSON(const std::string& content) {
  const auto path = CreateTestDir() / "test.json";
  std::ofstream file(path);
  file << content;
  file.close();
  return path;
}

TEST(PosePriorArchive, EmptySchema) {
  PosePriorArchive archive;
  archive.metadata.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  EXPECT_FALSE(archive.IsValid());
}

TEST(PosePriorArchive, NumColumnsAndRows) {
  PosePriorArchive archive;
  archive.metadata.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  archive.schema.columns = {PosePriorArchive::ColumnId::NAME,
                            PosePriorArchive::ColumnId::LAT,
                            PosePriorArchive::ColumnId::LON,
                            PosePriorArchive::ColumnId::ALT};
  archive.data = {
      {std::string("img001.jpg"), 47.0, 8.0, 500.0},
      {std::string("img002.jpg"), 48.0, 9.0, 600.0},
  };
  EXPECT_EQ(archive.NumColumns(), 4);
  EXPECT_EQ(archive.NumRows(), 2);
}

TEST(PosePriorArchive, MetadataIsValid) {
  PosePriorArchive::Metadata metadata;
  EXPECT_FALSE(metadata.IsValid());

  metadata.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  EXPECT_FALSE(metadata.IsValid());  // Missing required ellipsoid.

  metadata.ellipsoid = GPSTransform::Ellipsoid::WGS84;
  EXPECT_TRUE(metadata.IsValid());

  metadata.cartesian_frame = PosePriorArchive::CartesianFrame::ENU;
  EXPECT_FALSE(metadata.IsValid());

  metadata = {};
  metadata.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  EXPECT_TRUE(metadata.IsValid());

  metadata.cartesian_frame = PosePriorArchive::CartesianFrame::ENU;
  EXPECT_FALSE(metadata.IsValid());

  metadata.enu_origin = Eigen::Vector3d::Zero();
  EXPECT_TRUE(metadata.IsValid());

  metadata.sensor_type = SensorType::INVALID;
  EXPECT_FALSE(metadata.IsValid());

  metadata.sensor_type = SensorType::CAMERA;
  metadata.height_datum = "WRONG_DATUM";
  EXPECT_FALSE(metadata.IsValid());
  metadata.height_datum = "ELLIPSOIDAL";
  EXPECT_TRUE(metadata.IsValid());

  // metadata.cartesian_frame is ENU here; rotation_world_frame must match it.
  metadata.rotation_world_frame = "LOCAL";
  EXPECT_FALSE(metadata.IsValid());
  metadata.rotation_world_frame = "ENU";
  EXPECT_TRUE(metadata.IsValid());
}

TEST(PosePriorArchive, SchemaIsValid) {
  PosePriorArchive::Metadata metadata;
  metadata.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  metadata.ellipsoid = GPSTransform::Ellipsoid::WGS84;
  metadata.height_datum = "ELLIPSOIDAL";

  EXPECT_FALSE(PosePriorArchive::Schema{}.IsValid(metadata));

  PosePriorArchive::Schema schema;

  schema.columns = {PosePriorArchive::ColumnId::NAME};
  EXPECT_TRUE(schema.IsValid(metadata));

  schema.columns = {PosePriorArchive::ColumnId::NAME,
                    PosePriorArchive::ColumnId::LAT,
                    PosePriorArchive::ColumnId::LON,
                    PosePriorArchive::ColumnId::ALT};
  EXPECT_TRUE(schema.IsValid(metadata));

  schema.columns = {PosePriorArchive::ColumnId::NAME,
                    PosePriorArchive::ColumnId::LAT,
                    PosePriorArchive::ColumnId::LON};
  EXPECT_FALSE(schema.IsValid(metadata));

  schema.columns = {PosePriorArchive::ColumnId::NAME,
                    PosePriorArchive::ColumnId::LAT,
                    PosePriorArchive::ColumnId::LON,
                    PosePriorArchive::ColumnId::ALT,
                    PosePriorArchive::ColumnId::TX};
  EXPECT_TRUE(schema.IsValid(metadata));

  metadata.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  metadata.position_covariance_frame = "CARTESIAN";

  schema.columns = {PosePriorArchive::ColumnId::NAME,
                    PosePriorArchive::ColumnId::TX,
                    PosePriorArchive::ColumnId::TY,
                    PosePriorArchive::ColumnId::TZ};
  EXPECT_TRUE(schema.IsValid(metadata));

  schema.columns = {PosePriorArchive::ColumnId::NAME,
                    PosePriorArchive::ColumnId::TX,
                    PosePriorArchive::ColumnId::TY};
  EXPECT_FALSE(schema.IsValid(metadata));

  schema.columns = {PosePriorArchive::ColumnId::NAME,
                    PosePriorArchive::ColumnId::TX,
                    PosePriorArchive::ColumnId::TY,
                    PosePriorArchive::ColumnId::TZ,
                    PosePriorArchive::ColumnId::LAT};
  EXPECT_TRUE(schema.IsValid(metadata));

  schema.columns = {PosePriorArchive::ColumnId::NAME,
                    PosePriorArchive::ColumnId::TX,
                    PosePriorArchive::ColumnId::TY,
                    PosePriorArchive::ColumnId::TZ,
                    PosePriorArchive::ColumnId::STD_TX,
                    PosePriorArchive::ColumnId::STD_TY};
  EXPECT_FALSE(schema.IsValid(metadata));

  schema.columns = {PosePriorArchive::ColumnId::NAME,
                    PosePriorArchive::ColumnId::TX,
                    PosePriorArchive::ColumnId::TY,
                    PosePriorArchive::ColumnId::TZ,
                    PosePriorArchive::ColumnId::COV_TXX,
                    PosePriorArchive::ColumnId::COV_TXY};
  EXPECT_FALSE(schema.IsValid(metadata));

  schema.columns = {PosePriorArchive::ColumnId::NAME,
                    PosePriorArchive::ColumnId::TX,
                    PosePriorArchive::ColumnId::TY,
                    PosePriorArchive::ColumnId::TZ,
                    PosePriorArchive::ColumnId::STD_TX,
                    PosePriorArchive::ColumnId::STD_TY,
                    PosePriorArchive::ColumnId::STD_TZ,
                    PosePriorArchive::ColumnId::COV_TXX,
                    PosePriorArchive::ColumnId::COV_TXY,
                    PosePriorArchive::ColumnId::COV_TXZ,
                    PosePriorArchive::ColumnId::COV_TYY,
                    PosePriorArchive::ColumnId::COV_TYZ,
                    PosePriorArchive::ColumnId::COV_TZZ};
  EXPECT_FALSE(schema.IsValid(metadata));

  schema.columns = {PosePriorArchive::ColumnId::NAME,
                    PosePriorArchive::ColumnId::TX,
                    PosePriorArchive::ColumnId::TY,
                    PosePriorArchive::ColumnId::TZ,
                    PosePriorArchive::ColumnId::STD_TX,
                    PosePriorArchive::ColumnId::STD_TY,
                    PosePriorArchive::ColumnId::STD_TZ};
  EXPECT_TRUE(schema.IsValid(metadata));

  schema.columns = {PosePriorArchive::ColumnId::NAME,
                    PosePriorArchive::ColumnId::TX,
                    PosePriorArchive::ColumnId::TY,
                    PosePriorArchive::ColumnId::TZ,
                    PosePriorArchive::ColumnId::COV_TXX,
                    PosePriorArchive::ColumnId::COV_TXY,
                    PosePriorArchive::ColumnId::COV_TXZ,
                    PosePriorArchive::ColumnId::COV_TYY,
                    PosePriorArchive::ColumnId::COV_TYZ,
                    PosePriorArchive::ColumnId::COV_TZZ};
  EXPECT_TRUE(schema.IsValid(metadata));

  metadata.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  schema.columns = {PosePriorArchive::ColumnId::NAME};
  EXPECT_TRUE(schema.IsValid(metadata));

  metadata.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  schema.columns = {PosePriorArchive::ColumnId::NAME,
                    PosePriorArchive::ColumnId::STD_TX,
                    PosePriorArchive::ColumnId::STD_TY,
                    PosePriorArchive::ColumnId::STD_TZ};
  EXPECT_TRUE(schema.IsValid(metadata));
}

TEST(PosePriorArchive, IsValid) {
  PosePriorArchive archive;
  EXPECT_FALSE(archive.IsValid());

  archive.metadata.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  EXPECT_FALSE(archive.IsValid());

  archive.metadata.ellipsoid = GPSTransform::Ellipsoid::WGS84;
  archive.metadata.height_datum = "ELLIPSOIDAL";
  archive.schema.columns = {PosePriorArchive::ColumnId::NAME,
                            PosePriorArchive::ColumnId::LAT,
                            PosePriorArchive::ColumnId::LON,
                            PosePriorArchive::ColumnId::ALT};
  EXPECT_TRUE(archive.IsValid());

  archive.metadata.sensor_type = SensorType::INVALID;
  EXPECT_FALSE(archive.IsValid());

  archive.metadata.sensor_type = SensorType::CAMERA;
  archive.schema.columns = {PosePriorArchive::ColumnId::NAME,
                            PosePriorArchive::ColumnId::NAME};
  EXPECT_FALSE(archive.IsValid());
}

TEST(PosePriorArchive, ReadPosePriorArchive_WGS84) {
  const auto path = WriteTestJSON(R"({
    "coordinate_system": "WGS84",
    "translation_convention": "WORLD_FROM_CAM",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "schema": ["NAME", "LAT", "LON", "ALT"],
    "data": [
      ["img001.jpg", 47.3769, 8.5417, 500.0],
      ["img002.jpg", 47.3770, 8.5418, 501.0]
    ]
  })");
  const auto archive = ReadPosePriorArchive(path);

  EXPECT_EQ(archive.metadata.coordinate_system,
            PosePrior::CoordinateSystem::WGS84);
  EXPECT_EQ(archive.metadata.translation_convention,
            PosePriorArchive::PoseConvention::WORLD_FROM_CAM);
  ASSERT_EQ(archive.schema.columns.size(), 4);
  ASSERT_EQ(archive.data.size(), 2);
  EXPECT_EQ(std::get<std::string>(archive.data[0][0]), "img001.jpg");
  EXPECT_DOUBLE_EQ(std::get<double>(archive.data[0][1]), 47.3769);
}

TEST(PosePriorArchive, ReadPosePriorArchive_WithENUMetadata) {
  const auto path = WriteTestJSON(R"({
    "coordinate_system": "CARTESIAN",
    "cartesian_frame": "ENU",
    "ellipsoid": "WGS84",
    "enu_origin": [47.0, 8.0, 500.0],
    "schema": ["NAME", "TX", "TY", "TZ"],
    "data": [
      ["img001.jpg", 1.0, 2.0, 3.0]
    ]
  })");
  const auto archive = ReadPosePriorArchive(path);

  EXPECT_EQ(archive.metadata.coordinate_system,
            PosePrior::CoordinateSystem::CARTESIAN);
  ASSERT_TRUE(archive.metadata.cartesian_frame.has_value());
  EXPECT_EQ(*archive.metadata.cartesian_frame,
            PosePriorArchive::CartesianFrame::ENU);
  ASSERT_TRUE(archive.metadata.ellipsoid.has_value());
  EXPECT_EQ(*archive.metadata.ellipsoid, GPSTransform::Ellipsoid::WGS84);
  EXPECT_DOUBLE_EQ(archive.metadata.enu_origin->x(), 47.0);
}

TEST(PosePriorArchive, ToPosePriors_WGS84) {
  const auto path = WriteTestJSON(R"({
    "coordinate_system": "WGS84",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "schema": ["NAME", "LAT", "LON", "ALT"],
    "data": [
      ["img001.jpg", 47.3769, 8.5417, 500.0],
      ["img002.jpg", 47.3770, 8.5418, 501.0]
    ]
  })");
  const auto archive = ReadPosePriorArchive(path);

  int next_id = 1;
  const auto resolve = [&](const std::string& name) -> std::optional<data_t> {
    if (name == "img001.jpg") {
      return data_t(sensor_t(SensorType::CAMERA, 1), next_id++);
    }
    if (name == "img002.jpg") {
      return data_t(sensor_t(SensorType::CAMERA, 1), next_id++);
    }
    return std::nullopt;
  };

  const auto priors = archive.ToPosePriors(resolve);
  ASSERT_EQ(priors.size(), 2);
  EXPECT_DOUBLE_EQ(priors[0].position.x(), 47.3769);
  EXPECT_DOUBLE_EQ(priors[0].position.y(), 8.5417);
  EXPECT_DOUBLE_EQ(priors[0].position.z(), 500.0);
  EXPECT_FALSE(priors[0].HasPositionCov());
}

TEST(PosePriorArchive, ToPosePriors_CartesianWithSTD) {
  const auto path = WriteTestJSON(R"({
    "coordinate_system": "CARTESIAN",
    "position_covariance_frame": "CARTESIAN",
    "schema": ["NAME", "TX", "TY", "TZ", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [
      ["img001.jpg", 1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    ]
  })");
  const auto archive = ReadPosePriorArchive(path);

  const auto resolve = [](const std::string& name) -> std::optional<data_t> {
    if (name == "img001.jpg") {
      return data_t(sensor_t(SensorType::CAMERA, 1), 1);
    }
    return std::nullopt;
  };

  const auto priors = archive.ToPosePriors(resolve);
  ASSERT_EQ(priors.size(), 1);
  EXPECT_TRUE(priors[0].HasPosition());
  EXPECT_TRUE(priors[0].HasPositionCov());
  EXPECT_DOUBLE_EQ(priors[0].position_covariance(0, 0), 0.01);
  EXPECT_DOUBLE_EQ(priors[0].position_covariance(1, 1), 0.04);
  EXPECT_DOUBLE_EQ(priors[0].position_covariance(2, 2), 0.09);
}

TEST(PosePriorArchive, ToPosePriors_UnresolvedName) {
  const auto path = WriteTestJSON(R"({
    "coordinate_system": "CARTESIAN",
    "schema": ["NAME", "TX", "TY", "TZ"],
    "data": [
      ["unknown.jpg", 1.0, 2.0, 3.0]
    ]
  })");
  const auto archive = ReadPosePriorArchive(path);
  const auto resolve = [](const std::string&) -> std::optional<data_t> {
    return std::nullopt;
  };
  const auto priors = archive.ToPosePriors(resolve);
  EXPECT_TRUE(priors.empty());
}

TEST(PosePriorArchive, ToPosePriors_CartesianWithSTDOnly) {
  const auto path = WriteTestJSON(R"({
    "coordinate_system": "CARTESIAN",
    "position_covariance_frame": "CARTESIAN",
    "schema": ["NAME", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [
      ["img001.jpg", 0.1, 0.2, 0.3]
    ]
  })");
  const auto archive = ReadPosePriorArchive(path);

  const auto resolve = [](const std::string& name) -> std::optional<data_t> {
    if (name == "img001.jpg") {
      return data_t(sensor_t(SensorType::CAMERA, 1), 1);
    }
    return std::nullopt;
  };

  const auto priors = archive.ToPosePriors(resolve);
  ASSERT_EQ(priors.size(), 1);
  EXPECT_FALSE(priors[0].HasPosition());
  EXPECT_TRUE(priors[0].HasPositionCov());
  EXPECT_DOUBLE_EQ(priors[0].position_covariance(0, 0), 0.01);
  EXPECT_DOUBLE_EQ(priors[0].position_covariance(1, 1), 0.04);
  EXPECT_DOUBLE_EQ(priors[0].position_covariance(2, 2), 0.09);
}

TEST(PosePriorArchive, UpdatePosePriors_Existing) {
  PosePriorArchive archive;
  archive.metadata.sensor_type = SensorType::CAMERA;
  archive.metadata.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  archive.metadata.translation_convention =
      PosePriorArchive::PoseConvention::WORLD_FROM_CAM;
  archive.schema.columns = {PosePriorArchive::ColumnId::NAME,
                            PosePriorArchive::ColumnId::TX,
                            PosePriorArchive::ColumnId::TY,
                            PosePriorArchive::ColumnId::TZ};
  archive.data = {
      {std::string("img001.jpg"), 10.0, 20.0, 30.0},
  };
  ASSERT_TRUE(archive.IsValid());

  const auto resolve = [](const std::string& name) -> std::optional<data_t> {
    if (name == "img001.jpg") {
      return data_t(sensor_t(SensorType::CAMERA, 1), 1);
    }
    return std::nullopt;
  };

  PosePrior old_prior;
  old_prior.pose_prior_id = 42;
  old_prior.corr_data_id = data_t(sensor_t(SensorType::CAMERA, 1), 1);
  old_prior.position = Eigen::Vector3d(1.0, 2.0, 3.0);
  old_prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;

  std::vector<PosePrior> priors = {old_prior};
  archive.UpdatePosePriors(resolve, /*allow_new_priors=*/false, priors);

  ASSERT_EQ(priors.size(), 1);
  EXPECT_EQ(priors[0].pose_prior_id, 42);
  EXPECT_DOUBLE_EQ(priors[0].position.x(), 10.0);
  EXPECT_DOUBLE_EQ(priors[0].position.y(), 20.0);
  EXPECT_DOUBLE_EQ(priors[0].position.z(), 30.0);
}

TEST(PosePriorArchive, UpdatePosePriors_PartialSTD) {
  PosePriorArchive archive;
  archive.metadata.sensor_type = SensorType::CAMERA;
  archive.metadata.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  archive.metadata.translation_convention =
      PosePriorArchive::PoseConvention::WORLD_FROM_CAM;
  archive.metadata.position_covariance_frame = "CARTESIAN";
  archive.schema.columns = {PosePriorArchive::ColumnId::NAME,
                            PosePriorArchive::ColumnId::STD_TX,
                            PosePriorArchive::ColumnId::STD_TY,
                            PosePriorArchive::ColumnId::STD_TZ};
  archive.data = {
      {std::string("img001.jpg"), 0.1, 0.2, 0.3},
  };
  ASSERT_TRUE(archive.IsValid());

  const auto resolve = [](const std::string& name) -> std::optional<data_t> {
    if (name == "img001.jpg") {
      return data_t(sensor_t(SensorType::CAMERA, 1), 1);
    }
    return std::nullopt;
  };

  PosePrior old_prior;
  old_prior.pose_prior_id = 42;
  old_prior.corr_data_id = data_t(sensor_t(SensorType::CAMERA, 1), 1);
  old_prior.position = Eigen::Vector3d(1.0, 2.0, 3.0);
  old_prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;

  std::vector<PosePrior> priors = {old_prior};
  archive.UpdatePosePriors(resolve, /*allow_new_priors=*/false, priors);

  ASSERT_EQ(priors.size(), 1);
  EXPECT_EQ(priors[0].pose_prior_id, 42);
  EXPECT_TRUE(priors[0].HasPosition());
  EXPECT_DOUBLE_EQ(priors[0].position.x(), 1.0);
  EXPECT_DOUBLE_EQ(priors[0].position.y(), 2.0);
  EXPECT_DOUBLE_EQ(priors[0].position.z(), 3.0);
  EXPECT_TRUE(priors[0].HasPositionCov());
  EXPECT_DOUBLE_EQ(priors[0].position_covariance(0, 0), 0.01);
  EXPECT_DOUBLE_EQ(priors[0].position_covariance(1, 1), 0.04);
  EXPECT_DOUBLE_EQ(priors[0].position_covariance(2, 2), 0.09);
}

TEST(PosePriorArchive, UpdatePosePriors_AllowNewPriors) {
  PosePriorArchive archive;
  archive.metadata.sensor_type = SensorType::CAMERA;
  archive.metadata.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  archive.metadata.translation_convention =
      PosePriorArchive::PoseConvention::WORLD_FROM_CAM;
  archive.schema.columns = {PosePriorArchive::ColumnId::NAME,
                            PosePriorArchive::ColumnId::TX,
                            PosePriorArchive::ColumnId::TY,
                            PosePriorArchive::ColumnId::TZ};
  archive.data = {
      {std::string("img001.jpg"), 10.0, 20.0, 30.0},
  };
  ASSERT_TRUE(archive.IsValid());

  const auto resolve = [](const std::string& name) -> std::optional<data_t> {
    if (name == "img001.jpg") {
      return data_t(sensor_t(SensorType::CAMERA, 1), 1);
    }
    return std::nullopt;
  };

  std::vector<PosePrior> priors;
  archive.UpdatePosePriors(resolve, /*allow_new_priors=*/true, priors);

  ASSERT_EQ(priors.size(), 1);
  EXPECT_TRUE(priors[0].HasPosition());
  EXPECT_DOUBLE_EQ(priors[0].position.x(), 10.0);
  EXPECT_DOUBLE_EQ(priors[0].position.y(), 20.0);
  EXPECT_DOUBLE_EQ(priors[0].position.z(), 30.0);
}

// Comprehensive archive import case. Covers: mixed full-position/
// horizontal-only/gravity-only/full-orientation rows; one non-identity
// archive-ENU-to-row-local conversion with a hand-computed rotation and a
// diagonal covariance whose axes swap; merge
// (UpdatePosePriors) vs replace (ToPosePriors) semantics; and datum,
// incomplete-group, duplicate-row, gross-norm, and non-PSD rejection.
TEST(PosePriorArchive, ComprehensiveWGS84ImportAndRejections) {
  using ColumnId = PosePriorArchive::ColumnId;
  using cell_t = PosePriorArchive::cell_t;
  const cell_t kAbsent = std::monostate{};

  // The archive's global ENU origin is (lat=0, lon=90). At this exact
  // latitude/longitude, GPSTransform::ENUFromECEF(0, 90) reduces to the
  // clean, hand-verifiable signed permutation matrix
  //   R90 = [[-1,0,0],[0,0,1],[0,1,0]]
  // (sin(90deg)=1, cos(90deg)=0, sin(0)=0, cos(0)=1 in the ENU-from-ECEF
  // formula), and R90 is its own transpose and its own inverse. Similarly
  // GPSTransform::ENUFromECEF(0, 0) reduces to R0 = [[0,1,0],[0,0,1],[1,0,0]].
  // For a row at (0, 0), archive_from_local = ENUFromECEF(0,90) *
  // ECEFFromENU(0,0) = R90 * R0^T, which is exactly the S matrix asserted
  // below — computed here by hand, not by calling GPSTransform.
  PosePriorArchive archive;
  archive.metadata.sensor_type = SensorType::CAMERA;
  archive.metadata.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  archive.metadata.translation_convention =
      PosePriorArchive::PoseConvention::WORLD_FROM_CAM;
  archive.metadata.ellipsoid = GPSTransform::Ellipsoid::WGS84;
  archive.metadata.height_datum = "ELLIPSOIDAL";
  archive.metadata.position_covariance_frame = "LOCAL_ENU";
  archive.metadata.rotation_convention = "SENSOR_FROM_WORLD";
  archive.metadata.rotation_world_frame = "ENU";
  archive.metadata.rotation_covariance_convention =
      "RIGHT_MULTIPLICATIVE_WORLD";
  archive.metadata.enu_origin = Eigen::Vector3d(0.0, 90.0, 500.0);

  archive.schema.columns = {
      ColumnId::NAME,       ColumnId::LAT,        ColumnId::LON,
      ColumnId::ALT,        ColumnId::COV_TXX,    ColumnId::COV_TXY,
      ColumnId::COV_TXZ,    ColumnId::COV_TYY,    ColumnId::COV_TYZ,
      ColumnId::COV_TZZ,    ColumnId::GX,         ColumnId::GY,
      ColumnId::GZ,         ColumnId::QW,         ColumnId::QX,
      ColumnId::QY,         ColumnId::QZ,         ColumnId::ROT_COV_XX,
      ColumnId::ROT_COV_XY, ColumnId::ROT_COV_XZ, ColumnId::ROT_COV_YY,
      ColumnId::ROT_COV_YZ, ColumnId::ROT_COV_ZZ};

  // Each row is built with exactly one cell per schema column (23 columns:
  // NAME, LAT, LON, ALT, COV_T{XX,XY,XZ,YY,YZ,ZZ}, G{X,Y,Z}, Q{W,X,Y,Z},
  // ROT_COV_{XX,XY,XZ,YY,YZ,ZZ}), one push_back per cell so the count can be
  // verified by inspection rather than by hand-counting an aggregate list.
  PosePriorArchive::row_t identity_row;
  identity_row.push_back(std::string("identity.jpg"));  // NAME
  identity_row.push_back(0.0);                          // LAT
  identity_row.push_back(90.0);                         // LON
  identity_row.push_back(500.0);                        // ALT
  identity_row.push_back(1.0);                          // COV_TXX
  identity_row.push_back(0.0);                          // COV_TXY
  identity_row.push_back(0.0);                          // COV_TXZ
  identity_row.push_back(1.0);                          // COV_TYY
  identity_row.push_back(0.0);                          // COV_TYZ
  identity_row.push_back(1.0);                          // COV_TZZ
  identity_row.push_back(kAbsent);                      // GX
  identity_row.push_back(kAbsent);                      // GY
  identity_row.push_back(kAbsent);                      // GZ
  identity_row.push_back(1.0);                          // QW
  identity_row.push_back(0.0);                          // QX
  identity_row.push_back(0.0);                          // QY
  identity_row.push_back(0.0);                          // QZ
  identity_row.push_back(1.0);                          // ROT_COV_XX
  identity_row.push_back(0.0);                          // ROT_COV_XY
  identity_row.push_back(0.0);                          // ROT_COV_XZ
  identity_row.push_back(1.0);                          // ROT_COV_YY
  identity_row.push_back(0.0);                          // ROT_COV_YZ
  identity_row.push_back(1.0);                          // ROT_COV_ZZ
  ASSERT_EQ(identity_row.size(), archive.schema.columns.size());

  // 90 degrees of longitude from the origin: archive_from_local = S =
  // [[0,0,-1],[0,1,0],[1,0,0]] (hand-derived above). sensor_from_archive is
  // identity, so the stored rotation must equal S exactly, and the diagonal
  // rotation covariance diag(1,4,9) must become diag(9,4,1) (axes 0 and 2
  // swap; axis 1 is fixed by S). No position covariance on this row.
  PosePriorArchive::row_t other_row;
  other_row.push_back(std::string("other.jpg"));  // NAME
  other_row.push_back(0.0);                       // LAT
  other_row.push_back(0.0);                       // LON
  other_row.push_back(500.0);                     // ALT
  other_row.push_back(kAbsent);                   // COV_TXX
  other_row.push_back(kAbsent);                   // COV_TXY
  other_row.push_back(kAbsent);                   // COV_TXZ
  other_row.push_back(kAbsent);                   // COV_TYY
  other_row.push_back(kAbsent);                   // COV_TYZ
  other_row.push_back(kAbsent);                   // COV_TZZ
  other_row.push_back(kAbsent);                   // GX
  other_row.push_back(kAbsent);                   // GY
  other_row.push_back(kAbsent);                   // GZ
  other_row.push_back(1.0);                       // QW
  other_row.push_back(0.0);                       // QX
  other_row.push_back(0.0);                       // QY
  other_row.push_back(0.0);                       // QZ
  other_row.push_back(1.0);                       // ROT_COV_XX
  other_row.push_back(0.0);                       // ROT_COV_XY
  other_row.push_back(0.0);                       // ROT_COV_XZ
  other_row.push_back(4.0);                       // ROT_COV_YY
  other_row.push_back(0.0);                       // ROT_COV_YZ
  other_row.push_back(9.0);                       // ROT_COV_ZZ
  ASSERT_EQ(other_row.size(), archive.schema.columns.size());

  // Horizontal-only: LAT/LON finite, ALT absent (the sole partial position
  // subgroup); no covariance, gravity, or rotation.
  PosePriorArchive::row_t horizontal_row;
  horizontal_row.push_back(std::string("horizontal.jpg"));  // NAME
  horizontal_row.push_back(10.0);                           // LAT
  horizontal_row.push_back(20.0);                           // LON
  horizontal_row.push_back(kAbsent);                        // ALT
  horizontal_row.push_back(kAbsent);                        // COV_TXX
  horizontal_row.push_back(kAbsent);                        // COV_TXY
  horizontal_row.push_back(kAbsent);                        // COV_TXZ
  horizontal_row.push_back(kAbsent);                        // COV_TYY
  horizontal_row.push_back(kAbsent);                        // COV_TYZ
  horizontal_row.push_back(kAbsent);                        // COV_TZZ
  horizontal_row.push_back(kAbsent);                        // GX
  horizontal_row.push_back(kAbsent);                        // GY
  horizontal_row.push_back(kAbsent);                        // GZ
  horizontal_row.push_back(kAbsent);                        // QW
  horizontal_row.push_back(kAbsent);                        // QX
  horizontal_row.push_back(kAbsent);                        // QY
  horizontal_row.push_back(kAbsent);                        // QZ
  horizontal_row.push_back(kAbsent);                        // ROT_COV_XX
  horizontal_row.push_back(kAbsent);                        // ROT_COV_XY
  horizontal_row.push_back(kAbsent);                        // ROT_COV_XZ
  horizontal_row.push_back(kAbsent);                        // ROT_COV_YY
  horizontal_row.push_back(kAbsent);                        // ROT_COV_YZ
  horizontal_row.push_back(kAbsent);                        // ROT_COV_ZZ
  ASSERT_EQ(horizontal_row.size(), archive.schema.columns.size());

  // Gravity-only: no position at all.
  PosePriorArchive::row_t gravity_row;
  gravity_row.push_back(std::string("gravity.jpg"));  // NAME
  gravity_row.push_back(kAbsent);                     // LAT
  gravity_row.push_back(kAbsent);                     // LON
  gravity_row.push_back(kAbsent);                     // ALT
  gravity_row.push_back(kAbsent);                     // COV_TXX
  gravity_row.push_back(kAbsent);                     // COV_TXY
  gravity_row.push_back(kAbsent);                     // COV_TXZ
  gravity_row.push_back(kAbsent);                     // COV_TYY
  gravity_row.push_back(kAbsent);                     // COV_TYZ
  gravity_row.push_back(kAbsent);                     // COV_TZZ
  gravity_row.push_back(0.0);                         // GX
  gravity_row.push_back(0.0);                         // GY
  gravity_row.push_back(1.0);                         // GZ
  gravity_row.push_back(kAbsent);                     // QW
  gravity_row.push_back(kAbsent);                     // QX
  gravity_row.push_back(kAbsent);                     // QY
  gravity_row.push_back(kAbsent);                     // QZ
  gravity_row.push_back(kAbsent);                     // ROT_COV_XX
  gravity_row.push_back(kAbsent);                     // ROT_COV_XY
  gravity_row.push_back(kAbsent);                     // ROT_COV_XZ
  gravity_row.push_back(kAbsent);                     // ROT_COV_YY
  gravity_row.push_back(kAbsent);                     // ROT_COV_YZ
  gravity_row.push_back(kAbsent);                     // ROT_COV_ZZ
  ASSERT_EQ(gravity_row.size(), archive.schema.columns.size());

  archive.data = {identity_row, other_row, horizontal_row, gravity_row};

  ASSERT_TRUE(archive.IsValid());

  const auto resolve = [](const std::string& name) -> std::optional<data_t> {
    if (name == "identity.jpg") {
      return data_t(sensor_t(SensorType::CAMERA, 1), 1);
    }
    if (name == "other.jpg") {
      return data_t(sensor_t(SensorType::CAMERA, 1), 2);
    }
    if (name == "horizontal.jpg") {
      return data_t(sensor_t(SensorType::CAMERA, 1), 3);
    }
    if (name == "gravity.jpg") {
      return data_t(sensor_t(SensorType::CAMERA, 1), 4);
    }
    return std::nullopt;
  };

  // --- replace semantics: ToPosePriors always builds a fresh PosePrior. ---
  const auto priors = archive.ToPosePriors(resolve);
  ASSERT_EQ(priors.size(), 4);

  const auto find_prior = [&](image_t id) -> const PosePrior& {
    for (const auto& p : priors) {
      if (p.corr_data_id.id == id) {
        return p;
      }
    }
    ADD_FAILURE() << "No prior for id " << id;
    static const PosePrior kDummy;
    return kDummy;
  };

  const Eigen::Matrix3d identity3 = Eigen::Matrix3d::Identity();
  const PosePrior& identity_prior = find_prior(1);
  EXPECT_TRUE(identity_prior.HasRotation());
  EXPECT_THAT(identity_prior.rotation.toRotationMatrix(),
              EigenMatrixNear(identity3, 1e-9));
  EXPECT_TRUE(identity_prior.HasRotationCov());
  EXPECT_THAT(identity_prior.rotation_covariance,
              EigenMatrixNear(identity3, 1e-9));

  const PosePrior& other_prior = find_prior(2);
  Eigen::Matrix3d expected_rotation;
  expected_rotation << 0, 0, -1, 0, 1, 0, 1, 0, 0;
  EXPECT_TRUE(other_prior.HasRotation());
  EXPECT_THAT(other_prior.rotation.toRotationMatrix(),
              EigenMatrixNear(expected_rotation, 1e-9));
  const Eigen::Matrix3d expected_rotation_cov =
      Eigen::Vector3d(9.0, 4.0, 1.0).asDiagonal();
  EXPECT_TRUE(other_prior.HasRotationCov());
  EXPECT_THAT(other_prior.rotation_covariance,
              EigenMatrixNear(expected_rotation_cov, 1e-9));

  const PosePrior& horizontal_prior = find_prior(3);
  EXPECT_TRUE(std::isfinite(horizontal_prior.position.x()));
  EXPECT_TRUE(std::isfinite(horizontal_prior.position.y()));
  EXPECT_FALSE(std::isfinite(horizontal_prior.position.z()));
  EXPECT_FALSE(horizontal_prior.HasPosition());

  const PosePrior& gravity_prior = find_prior(4);
  EXPECT_FALSE(std::isfinite(gravity_prior.position.x()));
  EXPECT_TRUE(gravity_prior.HasGravity());
  EXPECT_THAT(gravity_prior.gravity,
              EigenMatrixNear(Eigen::Vector3d(0.0, 0.0, 1.0), 1e-9));

  // --- merge semantics: only groups present in the row are updated; groups
  // absent from the row (here, position) are preserved from the existing
  // prior. ---
  {
    PosePrior existing;
    existing.pose_prior_id = 99;
    existing.corr_data_id = *resolve("gravity.jpg");
    existing.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    existing.coordinate_system = PosePrior::CoordinateSystem::WGS84;
    existing.gravity = Eigen::Vector3d(0.0, 0.0, -1.0);
    std::vector<PosePrior> merged = {existing};
    archive.UpdatePosePriors(resolve, /*allow_new_priors=*/false, merged);
    ASSERT_EQ(merged.size(), 1);
    EXPECT_THAT(merged[0].position,
                EigenMatrixNear(Eigen::Vector3d(1.0, 2.0, 3.0), 1e-12));
    EXPECT_THAT(merged[0].gravity,
                EigenMatrixNear(Eigen::Vector3d(0.0, 0.0, 1.0), 1e-9));
  }

  // --- error semantics primitive: duplicate resolved names. ---
  EXPECT_FALSE(archive.HasDuplicateResolvedNames(resolve));
  PosePriorArchive duplicate_archive = archive;
  duplicate_archive.data.push_back(duplicate_archive.data[0]);
  EXPECT_TRUE(duplicate_archive.HasDuplicateResolvedNames(resolve));

  // --- rejections ---

  // Missing height_datum for a WGS84 position schema.
  {
    PosePriorArchive bad = archive;
    bad.metadata.height_datum = std::nullopt;
    EXPECT_FALSE(bad.IsValid());
  }

  // Incomplete row-level gravity group: GX/GY finite, GZ absent.
  {
    PosePriorArchive bad = archive;
    bad.data[3][12] = kAbsent;
    EXPECT_ANY_THROW(bad.ToPosePriors(resolve));
  }

  // Gross non-unit-norm gravity (norm 5, outside the 1e-2 tolerance).
  {
    PosePriorArchive bad = archive;
    bad.data[3][10] = 5.0;
    bad.data[3][11] = 0.0;
    bad.data[3][12] = 0.0;
    EXPECT_ANY_THROW(bad.ToPosePriors(resolve));
  }

  // Non-PSD position covariance (off-diagonal dominates the diagonal).
  {
    PosePriorArchive bad = archive;
    bad.data[0][4] = 1.0;
    bad.data[0][5] = 10.0;
    bad.data[0][6] = 0.0;
    bad.data[0][7] = 1.0;
    bad.data[0][8] = 0.0;
    bad.data[0][9] = 1.0;
    EXPECT_ANY_THROW(bad.ToPosePriors(resolve));
  }
}

// Pure documentation-anchoring check for the axis conventions expected of
// external telemetry adapters (e.g. GoPro/Betaflight, see doc/pose_priors.rst):
// does not call any COLMAP adapter or archive/import code, and every matrix
// is written out from the axis definitions rather than derived from a
// COLMAP call.
//   NED (sensor telemetry): X=North, Y=East, Z=Down.
//   ENU (COLMAP world):     X=East,  Y=North, Z=Up.
//   FRD (sensor body):      X=Forward, Y=Right, Z=Down.
//   COLMAP camera:          X=Right, Y=Down, Z=Forward.
TEST(PosePriorArchive, HandComputedNEDFRDConventions) {
  Eigen::Matrix3d enu_from_ned;
  enu_from_ned << 0, 1, 0, 1, 0, 0, 0, 0, -1;
  Eigen::Matrix3d cam_from_frd;
  cam_from_frd << 0, 1, 0, 0, 0, 1, 1, 0, 0;

  // Both are proper orthonormal rotations (a reflection would silently
  // mirror the trajectory).
  const Eigen::Matrix3d identity3 = Eigen::Matrix3d::Identity();
  EXPECT_THAT(Eigen::Matrix3d(enu_from_ned * enu_from_ned.transpose()),
              EigenMatrixNear(identity3, 1e-12));
  EXPECT_NEAR(enu_from_ned.determinant(), 1.0, 1e-12);
  EXPECT_THAT(Eigen::Matrix3d(cam_from_frd * cam_from_frd.transpose()),
              EigenMatrixNear(identity3, 1e-12));
  EXPECT_NEAR(cam_from_frd.determinant(), 1.0, 1e-12);

  // North becomes ENU's Y (North) axis.
  EXPECT_THAT(enu_from_ned * Eigen::Vector3d(1.0, 0.0, 0.0),
              EigenMatrixNear(Eigen::Vector3d(0.0, 1.0, 0.0), 1e-12));
  // Down becomes ENU's -Z (Up flips sign).
  EXPECT_THAT(enu_from_ned * Eigen::Vector3d(0.0, 0.0, 1.0),
              EigenMatrixNear(Eigen::Vector3d(0.0, 0.0, -1.0), 1e-12));

  // Forward becomes the camera's principal (+Z) axis.
  EXPECT_THAT(cam_from_frd * Eigen::Vector3d(1.0, 0.0, 0.0),
              EigenMatrixNear(Eigen::Vector3d(0.0, 0.0, 1.0), 1e-12));
  // Right becomes the camera's +X axis.
  EXPECT_THAT(cam_from_frd * Eigen::Vector3d(0.0, 1.0, 0.0),
              EigenMatrixNear(Eigen::Vector3d(1.0, 0.0, 0.0), 1e-12));
}

}  // namespace
}  // namespace colmap
