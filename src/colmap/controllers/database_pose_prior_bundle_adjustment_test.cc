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

#include "colmap/controllers/database_pose_prior_bundle_adjustment.h"

#include "colmap/geometry/pose_prior.h"
#include "colmap/sensor/models.h"
#include "colmap/util/testing.h"

#include <filesystem>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <sqlite3.h>

namespace colmap {
namespace {

template <typename MatrixType>
void BindMatrix(sqlite3_stmt* statement,
                const int column,
                const MatrixType& matrix) {
  ASSERT_EQ(sqlite3_bind_blob(statement,
                              column,
                              matrix.data(),
                              static_cast<int>(matrix.size() * sizeof(double)),
                              SQLITE_TRANSIENT),
            SQLITE_OK);
}

TEST(DatabasePosePriorBundleAdjustment, ReadPosePriors) {
  const std::filesystem::path test_path = CreateTestDir();
  const std::filesystem::path database_path = test_path / "database.db";

  sqlite3* database = nullptr;
  ASSERT_EQ(sqlite3_open(database_path.string().c_str(), &database), SQLITE_OK);
  ASSERT_NE(database, nullptr);

  const char* create_table_sql =
      "CREATE TABLE pose_priors("
      "pose_prior_id INTEGER PRIMARY KEY, "
      "corr_data_id INTEGER NOT NULL, "
      "corr_sensor_id INTEGER NOT NULL, "
      "corr_sensor_type INTEGER NOT NULL, "
      "position BLOB, "
      "position_covariance BLOB, "
      "gravity BLOB, "
      "coordinate_system INTEGER NOT NULL, "
      "rotation BLOB, "
      "rotation_covariance BLOB);";
  ASSERT_EQ(sqlite3_exec(database, create_table_sql, nullptr, nullptr, nullptr),
            SQLITE_OK);

  sqlite3_stmt* statement = nullptr;
  ASSERT_EQ(sqlite3_prepare_v2(
                database,
                "INSERT INTO pose_priors VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                -1,
                &statement,
                nullptr),
            SQLITE_OK);
  ASSERT_NE(statement, nullptr);

  const Eigen::Vector3d position(1.0, 2.0, 3.0);
  const Eigen::Matrix3d position_covariance =
      0.25 * Eigen::Matrix3d::Identity();
  const Eigen::Vector3d gravity(0.0, 1.0, 0.0);
  const Eigen::Vector4d rotation_wxyz(0.5, 0.5, 0.5, 0.5);
  const Eigen::Matrix3d rotation_covariance =
      0.01 * Eigen::Matrix3d::Identity();

  ASSERT_EQ(sqlite3_bind_int64(statement, 1, 1), SQLITE_OK);
  ASSERT_EQ(sqlite3_bind_int64(statement, 2, 7), SQLITE_OK);
  ASSERT_EQ(sqlite3_bind_int64(statement, 3, 9), SQLITE_OK);
  ASSERT_EQ(
      sqlite3_bind_int(statement, 4, static_cast<int>(SensorType::CAMERA)),
      SQLITE_OK);
  BindMatrix(statement, 5, position);
  BindMatrix(statement, 6, position_covariance);
  BindMatrix(statement, 7, gravity);
  ASSERT_EQ(sqlite3_bind_int(
                statement,
                8,
                static_cast<int>(PosePrior::CoordinateSystem::CARTESIAN)),
            SQLITE_OK);
  BindMatrix(statement, 9, rotation_wxyz);
  BindMatrix(statement, 10, rotation_covariance);
  ASSERT_EQ(sqlite3_step(statement), SQLITE_DONE);
  ASSERT_EQ(sqlite3_finalize(statement), SQLITE_OK);
  ASSERT_EQ(sqlite3_close(database), SQLITE_OK);

  const DatabasePosePriorBundleAdjustmentOptions options;
  const std::vector<DatabasePosePrior> priors =
      ReadDatabasePosePriors(database_path, options);
  ASSERT_EQ(priors.size(), 1);
  EXPECT_EQ(priors[0].image_id, 7);
  EXPECT_TRUE(priors[0].position.isApprox(position));
  EXPECT_TRUE(priors[0].position_covariance.isApprox(position_covariance));
  EXPECT_TRUE(priors[0].rotation.coeffs().isApprox(
      Eigen::Quaterniond(0.5, 0.5, 0.5, 0.5).coeffs()));
  EXPECT_TRUE(priors[0].rotation_covariance.isApprox(rotation_covariance));

  std::filesystem::remove_all(test_path);
}

}  // namespace
}  // namespace colmap
