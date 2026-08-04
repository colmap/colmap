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

// Registers two images ("keep.jpg", "update.jpg") sharing one camera, and
// returns their assigned image ids in that order. No priors are written.
std::pair<image_t, image_t> CreateTwoImages(Database& database) {
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

  return {keep_image_id, update_image_id};
}

// Gives both images a full position+covariance+gravity+heading prior.
void WriteFullPriors(Database& database,
                     camera_t camera_id,
                     image_t keep_image_id,
                     image_t update_image_id) {
  for (const image_t image_id : {keep_image_id, update_image_id}) {
    PosePrior prior;
    prior.corr_data_id =
        data_t(sensor_t(SensorType::CAMERA, camera_id), image_id);
    prior.coordinate_system = PosePrior::CoordinateSystem::WGS84;
    prior.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    prior.position_covariance = Eigen::Matrix3d::Identity() * 0.5;
    prior.gravity = Eigen::Vector3d(0.0, 0.0, -1.0);
    prior.heading_rad = 0.0;
    prior.heading_stddev_rad = DegToRad(5.0);
    database.WritePosePrior(prior);
  }
}

constexpr const char* kBothImagesArchive = R"({
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
    ["keep.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, 0.0, 1.0, 0.0],
    ["update.jpg", 45.001, -73.001, 41.0, 2.0, 2.0, 4.0, null, null, null]
  ]
})";

TEST(PosePriorImporter, ImportsEveryRowIntoAnEmptyDatabase) {
  const auto database_path = CreateTestDir() / "database.db";
  {
    auto database = Database::Open(database_path);
    CreateTwoImages(*database);
  }

  const auto archive_path = WriteArchiveJSON(kBothImagesArchive);
  ASSERT_EQ(RunImporter(database_path, archive_path, "error"), EXIT_SUCCESS);

  auto database = Database::Open(database_path);
  ASSERT_EQ(database->NumPosePriors(), 2);
  for (const auto& prior : database->ReadAllPosePriors()) {
    const Image image =
        database->ReadImage(static_cast<image_t>(prior.corr_data_id.id));
    EXPECT_EQ(prior.coordinate_system, PosePrior::CoordinateSystem::WGS84);
    EXPECT_TRUE(prior.HasPosition());
    EXPECT_TRUE(prior.HasPositionCov());
    if (image.Name() == "keep.jpg") {
      EXPECT_DOUBLE_EQ(prior.position.x(), 45.0);
      EXPECT_TRUE(prior.HasGravity());
    } else {
      EXPECT_DOUBLE_EQ(prior.position.x(), 45.001);
      // Its gravity group was null, so it has none -- not a zero vector.
      EXPECT_FALSE(prior.HasGravity());
    }
  }
}

TEST(PosePriorImporter, ExistingErrorRefusesAndWritesNothing) {
  const auto database_path = CreateTestDir() / "database.db";
  std::vector<PosePrior> before;
  {
    auto database = Database::Open(database_path);
    const auto [keep_id, update_id] = CreateTwoImages(*database);
    const camera_t camera_id = database->ReadImage(keep_id).CameraId();
    WriteFullPriors(*database, camera_id, keep_id, update_id);
    before = database->ReadAllPosePriors();
  }
  ASSERT_EQ(before.size(), 2u);

  const auto archive_path = WriteArchiveJSON(kBothImagesArchive);
  EXPECT_EQ(RunImporter(database_path, archive_path, "error"), EXIT_FAILURE);

  // Every stored prior is byte-for-byte what it was: the refusal happens
  // before the transaction opens, not partway through it.
  auto database = Database::Open(database_path);
  const std::vector<PosePrior> after = database->ReadAllPosePriors();
  ASSERT_EQ(after.size(), before.size());
  for (size_t i = 0; i < after.size(); ++i) {
    EXPECT_EQ(after[i], before[i]);
  }
}

TEST(PosePriorImporter, ExistingReplaceOverwritesTheWholeRecord) {
  const auto database_path = CreateTestDir() / "database.db";
  {
    auto database = Database::Open(database_path);
    const auto [keep_id, update_id] = CreateTwoImages(*database);
    const camera_t camera_id = database->ReadImage(keep_id).CameraId();
    WriteFullPriors(*database, camera_id, keep_id, update_id);
  }

  const auto archive_path = WriteArchiveJSON(kBothImagesArchive);
  ASSERT_EQ(RunImporter(database_path, archive_path, "replace"), EXIT_SUCCESS);

  auto database = Database::Open(database_path);
  EXPECT_EQ(database->NumPosePriors(), 2);
  for (const auto& prior : database->ReadAllPosePriors()) {
    const Image image =
        database->ReadImage(static_cast<image_t>(prior.corr_data_id.id));
    if (image.Name() != "update.jpg") {
      continue;
    }
    EXPECT_DOUBLE_EQ(prior.position.x(), 45.001);
    // The stored prior had gravity and heading; the archive row declares a
    // null gravity group and no heading columns at all. Replace means
    // replace: what the archive does not state, the record no longer has.
    // A merge here would leave a prior whose fields came from two different
    // captures with nothing recording which came from where.
    EXPECT_FALSE(prior.HasGravity());
    EXPECT_FALSE(prior.HasHeading());
  }
}

TEST(PosePriorImporter, UnresolvedNamesAreAllReportedAndNothingIsWritten) {
  const auto database_path = CreateTestDir() / "database.db";
  {
    auto database = Database::Open(database_path);
    CreateTwoImages(*database);
  }

  const auto archive_path = WriteArchiveJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [
      ["keep.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0],
      ["missing-one.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0],
      ["missing-two.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0]
    ]
  })");
  EXPECT_EQ(RunImporter(database_path, archive_path, "error"), EXIT_FAILURE);

  // The resolvable row must not have been imported either. Importing what
  // resolved and reporting the rest would leave the operator to discover the
  // naming mismatch one run at a time.
  auto database = Database::Open(database_path);
  EXPECT_EQ(database->NumPosePriors(), 0);
}

TEST(PosePriorImporter, DuplicateNamesAreRejectedBeforeAnyWrite) {
  const auto database_path = CreateTestDir() / "database.db";
  {
    auto database = Database::Open(database_path);
    CreateTwoImages(*database);
  }

  const auto archive_path = WriteArchiveJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [
      ["update.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0],
      ["update.jpg", 46.0, -74.0, 50.0, 2.0, 2.0, 4.0]
    ]
  })");
  EXPECT_ANY_THROW(RunImporter(database_path, archive_path, "error"));

  auto database = Database::Open(database_path);
  EXPECT_EQ(database->NumPosePriors(), 0);
}

TEST(PosePriorImporter, MergeIsNotAPolicy) {
  const auto database_path = CreateTestDir() / "database.db";
  {
    auto database = Database::Open(database_path);
    CreateTwoImages(*database);
  }
  const auto archive_path = WriteArchiveJSON(kBothImagesArchive);
  EXPECT_EQ(RunImporter(database_path, archive_path, "merge"), EXIT_FAILURE);

  auto database = Database::Open(database_path);
  EXPECT_EQ(database->NumPosePriors(), 0);
}

TEST(PosePriorImporter, AMalformedArchiveLeavesTheDatabaseUnchanged) {
  const auto database_path = CreateTestDir() / "database.db";
  std::vector<PosePrior> before;
  {
    auto database = Database::Open(database_path);
    const auto [keep_id, update_id] = CreateTwoImages(*database);
    const camera_t camera_id = database->ReadImage(keep_id).CameraId();
    WriteFullPriors(*database, camera_id, keep_id, update_id);
    before = database->ReadAllPosePriors();
  }

  // Valid JSON, valid schema, but one row's covariance is singular. The
  // archive is read and validated in full before the first write, so this
  // must not leave the earlier rows imported.
  const auto archive_path = WriteArchiveJSON(R"({
    "schema_version": 1,
    "coordinate_system": "WGS84",
    "sensor_type": "CAMERA",
    "ellipsoid": "WGS84",
    "height_datum": "ELLIPSOIDAL",
    "position_covariance_frame": "LOCAL_ENU",
    "schema": ["NAME", "LAT", "LON", "ALT", "STD_TX", "STD_TY", "STD_TZ"],
    "data": [
      ["keep.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0],
      ["update.jpg", 45.0, -73.0, 40.0, 2.0, 0.0, 4.0]
    ]
  })");
  EXPECT_ANY_THROW(RunImporter(database_path, archive_path, "replace"));

  auto database = Database::Open(database_path);
  const std::vector<PosePrior> after = database->ReadAllPosePriors();
  ASSERT_EQ(after.size(), before.size());
  for (size_t i = 0; i < after.size(); ++i) {
    EXPECT_EQ(after[i], before[i]);
  }
}

}  // namespace
}  // namespace colmap
