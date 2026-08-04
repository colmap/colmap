// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "colmap/exe/database.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/math/math.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/database.h"
#include "colmap/scene/image.h"
#include "colmap/util/testing.h"
#include "colmap/util/types.h"

#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <sqlite3.h>

namespace colmap {
namespace {

std::filesystem::path WriteArchiveJSON(const std::string& content) {
  const auto path = CreateTestDir() / "priors.json";
  std::ofstream file(path);
  file << content;
  file.close();
  return path;
}

int RunImporter(const std::filesystem::path& database_path,
                const std::filesystem::path& pose_prior_path,
                const std::string& existing_policy) {
  std::vector<std::string> args{
      "pose_prior_importer",
      "--database_path",
      database_path.string(),
      "--pose_prior_path",
      pose_prior_path.string(),
      "--existing",
      existing_policy,
  };
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) {
    argv.push_back(arg.data());
  }
  return RunPosePriorImporter(static_cast<int>(argv.size()), argv.data());
}

// Creates two registered images ("keep.jpg", "update.jpg") sharing one
// camera, each with a full position+covariance+gravity+heading prior, and
// returns their assigned image ids in that order.
std::pair<image_t, image_t> CreateTwoImagesWithFullPriors(Database& database) {
  Camera camera = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 1.0, 100, 100);
  const camera_t camera_id = database.WriteCamera(camera);

  Image keep_image;
  keep_image.SetName("keep.jpg");
  keep_image.SetCameraId(camera_id);
  const image_t keep_image_id = database.WriteImage(keep_image);

  Image update_image;
  update_image.SetName("update.jpg");
  update_image.SetCameraId(camera_id);
  const image_t update_image_id = database.WriteImage(update_image);

  for (const image_t image_id : {keep_image_id, update_image_id}) {
    PosePrior prior;
    prior.corr_data_id =
        data_t(sensor_t(SensorType::CAMERA, camera_id), image_id);
    prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
    prior.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    prior.position_covariance = Eigen::Matrix3d::Identity() * 0.5;
    prior.gravity = Eigen::Vector3d(0.0, 0.0, -1.0);
    prior.heading_rad = 0.0;
    prior.heading_stddev_rad = DegToRad(5.0);
    database.WritePosePrior(prior);
  }
  return {keep_image_id, update_image_id};
}

TEST(PosePriorImporter, MergeUnrelatedPriorUnchangedNoUpdateIssued) {
  const auto database_path = CreateTestDir() / "database.db";
  PosePrior keep_prior_before;
  PosePrior update_prior_before;
  {
    auto database = Database::Open(database_path);
    const auto [keep_id, update_id] = CreateTwoImagesWithFullPriors(*database);
    for (const auto& prior : database->ReadAllPosePriors()) {
      if (prior.corr_data_id.id == keep_id) {
        keep_prior_before = prior;
      } else if (prior.corr_data_id.id == update_id) {
        update_prior_before = prior;
      }
    }
  }
  ASSERT_TRUE(keep_prior_before.HasPosition());
  ASSERT_TRUE(update_prior_before.HasPosition());

  // Install an audit trigger recording every UPDATE on pose_priors, so an
  // unrelated row's lack of any SQL UPDATE is provable, not merely
  // consistent with an unchanged value that could coincidentally result
  // from an UPDATE that wrote the same bytes back.
  {
    sqlite3* db = nullptr;
    ASSERT_EQ(sqlite3_open(database_path.string().c_str(), &db), SQLITE_OK);
    ASSERT_EQ(sqlite3_exec(db,
                           "CREATE TABLE update_audit(pose_prior_id INTEGER);"
                           "CREATE TRIGGER audit_pose_prior_update "
                           "AFTER UPDATE ON pose_priors BEGIN "
                           "INSERT INTO update_audit(pose_prior_id) "
                           "VALUES (OLD.pose_prior_id); END;",
                           nullptr,
                           nullptr,
                           nullptr),
              SQLITE_OK);
    sqlite3_close(db);
  }

  // Archive only names "update.jpg" with a new position; "keep.jpg" never
  // appears in the archive at all.
  const auto archive_path = WriteArchiveJSON(R"({
    "coordinate_system": "CARTESIAN",
    "schema": ["NAME", "TX", "TY", "TZ"],
    "data": [
      ["update.jpg", 10.0, 20.0, 30.0]
    ]
  })");
  ASSERT_EQ(RunImporter(database_path, archive_path, "merge"), EXIT_SUCCESS);

  {
    auto database = Database::Open(database_path);
    for (const auto& prior : database->ReadAllPosePriors()) {
      if (prior.corr_data_id.id == keep_prior_before.corr_data_id.id) {
        EXPECT_EQ(prior, keep_prior_before);
      } else if (prior.corr_data_id.id == update_prior_before.corr_data_id.id) {
        EXPECT_DOUBLE_EQ(prior.position.x(), 10.0);
        EXPECT_DOUBLE_EQ(prior.position.y(), 20.0);
        EXPECT_DOUBLE_EQ(prior.position.z(), 30.0);
        // Groups absent from the incoming row are preserved.
        EXPECT_TRUE(prior.HasGravity());
        EXPECT_TRUE(prior.HasHeading());
      }
    }
  }

  {
    sqlite3* db = nullptr;
    ASSERT_EQ(sqlite3_open(database_path.string().c_str(), &db), SQLITE_OK);
    sqlite3_stmt* stmt = nullptr;
    ASSERT_EQ(sqlite3_prepare_v2(db,
                                 "SELECT COUNT(*) FROM update_audit WHERE "
                                 "pose_prior_id = ?;",
                                 -1,
                                 &stmt,
                                 nullptr),
              SQLITE_OK);
    ASSERT_EQ(sqlite3_bind_int64(stmt, 1, keep_prior_before.pose_prior_id),
              SQLITE_OK);
    ASSERT_EQ(sqlite3_step(stmt), SQLITE_ROW);
    EXPECT_EQ(sqlite3_column_int64(stmt, 0), 0)
        << "Unrelated prior received an SQL UPDATE it should not have.";
    sqlite3_finalize(stmt);

    ASSERT_EQ(sqlite3_prepare_v2(db,
                                 "SELECT COUNT(*) FROM update_audit WHERE "
                                 "pose_prior_id = ?;",
                                 -1,
                                 &stmt,
                                 nullptr),
              SQLITE_OK);
    ASSERT_EQ(sqlite3_bind_int64(stmt, 1, update_prior_before.pose_prior_id),
              SQLITE_OK);
    ASSERT_EQ(sqlite3_step(stmt), SQLITE_ROW);
    EXPECT_GT(sqlite3_column_int64(stmt, 0), 0)
        << "The targeted prior should have received exactly one SQL UPDATE.";
    sqlite3_finalize(stmt);
    sqlite3_close(db);
  }
}

TEST(PosePriorImporter, MergePartialGravityPreservesPositionAndCovariance) {
  const auto database_path = CreateTestDir() / "database.db";
  {
    auto database = Database::Open(database_path);
    CreateTwoImagesWithFullPriors(*database);
  }

  const auto archive_path = WriteArchiveJSON(R"({
    "coordinate_system": "CARTESIAN",
    "schema": ["NAME", "GX", "GY", "GZ"],
    "data": [
      ["update.jpg", 0.0, 1.0, 0.0]
    ]
  })");
  ASSERT_EQ(RunImporter(database_path, archive_path, "merge"), EXIT_SUCCESS);

  auto database = Database::Open(database_path);
  bool found = false;
  for (const auto& prior : database->ReadAllPosePriors()) {
    const Image image =
        database->ReadImage(static_cast<image_t>(prior.corr_data_id.id));
    if (image.Name() != "update.jpg") {
      continue;
    }
    found = true;
    EXPECT_TRUE(prior.HasPosition());
    EXPECT_DOUBLE_EQ(prior.position.x(), 1.0);
    EXPECT_DOUBLE_EQ(prior.position.y(), 2.0);
    EXPECT_DOUBLE_EQ(prior.position.z(), 3.0);
    EXPECT_TRUE(prior.HasPositionCov());
    EXPECT_DOUBLE_EQ(prior.position_covariance(0, 0), 0.5);
    ASSERT_TRUE(prior.HasGravity());
    EXPECT_LT((prior.gravity - Eigen::Vector3d(0.0, 1.0, 0.0)).norm(), 1e-9);
  }
  EXPECT_TRUE(found);
}

TEST(PosePriorImporter, MergePartialPositionPreservesGravityAndRotation) {
  const auto database_path = CreateTestDir() / "database.db";
  {
    auto database = Database::Open(database_path);
    CreateTwoImagesWithFullPriors(*database);
  }

  const auto archive_path = WriteArchiveJSON(R"({
    "coordinate_system": "CARTESIAN",
    "schema": ["NAME", "TX", "TY", "TZ"],
    "data": [
      ["update.jpg", 7.0, 8.0, 9.0]
    ]
  })");
  ASSERT_EQ(RunImporter(database_path, archive_path, "merge"), EXIT_SUCCESS);

  auto database = Database::Open(database_path);
  bool found = false;
  for (const auto& prior : database->ReadAllPosePriors()) {
    const Image image =
        database->ReadImage(static_cast<image_t>(prior.corr_data_id.id));
    if (image.Name() != "update.jpg") {
      continue;
    }
    found = true;
    EXPECT_DOUBLE_EQ(prior.position.x(), 7.0);
    EXPECT_DOUBLE_EQ(prior.position.y(), 8.0);
    EXPECT_DOUBLE_EQ(prior.position.z(), 9.0);
    ASSERT_TRUE(prior.HasGravity());
    EXPECT_LT((prior.gravity - Eigen::Vector3d(0.0, 0.0, -1.0)).norm(), 1e-9);
    ASSERT_TRUE(prior.HasHeading());
    EXPECT_NEAR(prior.heading_rad, 0.0, 1e-9);
    EXPECT_NEAR(prior.heading_stddev_rad, DegToRad(5.0), 1e-9);
  }
  EXPECT_TRUE(found);
}

TEST(PosePriorImporter, MergeAddsNewRow) {
  const auto database_path = CreateTestDir() / "database.db";
  camera_t camera_id = kInvalidCameraId;
  {
    auto database = Database::Open(database_path);
    const auto [keep_id, update_id] = CreateTwoImagesWithFullPriors(*database);
    (void)keep_id;
    (void)update_id;
    Image new_image;
    new_image.SetName("new.jpg");
    Camera camera = Camera::CreateFromModelId(
        kInvalidCameraId, CameraModelId::kSimplePinhole, 1.0, 100, 100);
    camera_id = database->WriteCamera(camera);
    new_image.SetCameraId(camera_id);
    database->WriteImage(new_image);
    ASSERT_EQ(database->NumPosePriors(), 2);
  }

  const auto archive_path = WriteArchiveJSON(R"({
    "coordinate_system": "CARTESIAN",
    "schema": ["NAME", "TX", "TY", "TZ"],
    "data": [
      ["new.jpg", 4.0, 5.0, 6.0]
    ]
  })");
  ASSERT_EQ(RunImporter(database_path, archive_path, "merge"), EXIT_SUCCESS);

  auto database = Database::Open(database_path);
  EXPECT_EQ(database->NumPosePriors(), 3);
  bool found = false;
  for (const auto& prior : database->ReadAllPosePriors()) {
    const Image image =
        database->ReadImage(static_cast<image_t>(prior.corr_data_id.id));
    if (image.Name() == "new.jpg") {
      found = true;
      EXPECT_DOUBLE_EQ(prior.position.x(), 4.0);
    }
  }
  EXPECT_TRUE(found);
}

TEST(PosePriorImporter, MergeRejectsDuplicateResolvedNamesBeforeAnyWrite) {
  const auto database_path = CreateTestDir() / "database.db";
  {
    auto database = Database::Open(database_path);
    CreateTwoImagesWithFullPriors(*database);
  }

  // Two different archive rows both resolve to "update.jpg": ambiguous, and
  // must be rejected before any write.
  const auto archive_path = WriteArchiveJSON(R"({
    "coordinate_system": "CARTESIAN",
    "schema": ["NAME", "TX", "TY", "TZ"],
    "data": [
      ["update.jpg", 1.0, 1.0, 1.0],
      ["update.jpg", 2.0, 2.0, 2.0]
    ]
  })");
  EXPECT_ANY_THROW(RunImporter(database_path, archive_path, "merge"));

  auto database = Database::Open(database_path);
  for (const auto& prior : database->ReadAllPosePriors()) {
    const Image image =
        database->ReadImage(static_cast<image_t>(prior.corr_data_id.id));
    if (image.Name() == "update.jpg") {
      // Unchanged from CreateTwoImagesWithFullPriors' original position.
      EXPECT_DOUBLE_EQ(prior.position.x(), 1.0);
      EXPECT_DOUBLE_EQ(prior.position.y(), 2.0);
      EXPECT_DOUBLE_EQ(prior.position.z(), 3.0);
    }
  }
}

TEST(PosePriorImporter, MergeFailureLeavesDatabaseUnchanged) {
  // Mixing WGS84 and Cartesian coordinate-bearing groups for the *same*
  // existing prior across rows of one archive is rejected by
  // PosePriorArchive::UpdatePosePriors' internal consistency check; the
  // failure must leave the database completely unmodified.
  const auto database_path = CreateTestDir() / "database.db";
  PosePrior update_prior_before;
  {
    auto database = Database::Open(database_path);
    const auto [keep_id, update_id] = CreateTwoImagesWithFullPriors(*database);
    (void)keep_id;
    for (const auto& prior : database->ReadAllPosePriors()) {
      if (prior.corr_data_id.id == update_id) {
        update_prior_before = prior;
      }
    }
  }
  ASSERT_TRUE(update_prior_before.HasPosition());

  const auto archive_path = WriteArchiveJSON(R"({
    "coordinate_system": "WGS84",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "schema": ["NAME", "LAT", "LON", "ALT"],
    "data": [
      ["update.jpg", 47.0, 8.0, 500.0]
    ]
  })");
  EXPECT_ANY_THROW(RunImporter(database_path, archive_path, "merge"));

  auto database = Database::Open(database_path);
  for (const auto& prior : database->ReadAllPosePriors()) {
    if (prior.corr_data_id.id == update_prior_before.corr_data_id.id) {
      EXPECT_EQ(prior, update_prior_before);
    }
  }
}

}  // namespace
}  // namespace colmap
