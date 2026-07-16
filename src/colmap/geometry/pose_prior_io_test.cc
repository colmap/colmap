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
}

TEST(PosePriorArchive, SchemaIsValid) {
  PosePriorArchive::Metadata metadata;
  metadata.coordinate_system = PosePrior::CoordinateSystem::WGS84;

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

}  // namespace
}  // namespace colmap
