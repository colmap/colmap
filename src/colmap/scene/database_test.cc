// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/database.h"

#include "colmap/math/random_eigen.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <filesystem>
#include <thread>

#include <Eigen/Geometry>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sqlite3.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace colmap {
namespace {

class ParameterizedDatabaseTests
    : public ::testing::TestWithParam<std::function<std::shared_ptr<Database>(
          const std::filesystem::path&)>> {};

TEST_P(ParameterizedDatabaseTests, OpenInMemory) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
}

TEST_P(ParameterizedDatabaseTests, OpenCloseInMemory) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  database->Close();
  // Any database operation after closing the database should fail.
  EXPECT_ANY_THROW(database->ExistsCamera(42));
  database->Close();
}

TEST_P(ParameterizedDatabaseTests, OpenFile) {
  std::shared_ptr<Database> database =
      GetParam()(CreateTestDir() / "database.db");
}

TEST_P(ParameterizedDatabaseTests, OpenCloseFile) {
  std::shared_ptr<Database> database =
      GetParam()(CreateTestDir() / "database.db");
  database->Close();
  // Any database operation after closing the database should fail.
  EXPECT_ANY_THROW(database->ExistsCamera(42));
  database->Close();
}

TEST_P(ParameterizedDatabaseTests, OpenFileWithNonASCIIPath) {
  const auto database_path = CreateTestDir() / u8"äöü時临.db";
  std::shared_ptr<Database> database = GetParam()(database_path);
  EXPECT_TRUE(ExistsPath(database_path));
}

TEST_P(ParameterizedDatabaseTests, Transaction) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  DatabaseTransaction database_transaction(database.get());
}

TEST_P(ParameterizedDatabaseTests, TransactionMultiThreaded) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);

  constexpr int kNumThreads = 3;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&database]() {
      DatabaseTransaction database_transaction(database.get());
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_P(ParameterizedDatabaseTests, Empty) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  EXPECT_EQ(database->NumCameras(), 0);
  EXPECT_EQ(database->NumFrames(), 0);
  EXPECT_EQ(database->NumImages(), 0);
  EXPECT_EQ(database->NumKeypoints(), 0);
  EXPECT_EQ(database->MaxNumKeypoints(), 0);
  EXPECT_EQ(database->NumDescriptors(), 0);
  EXPECT_EQ(database->MaxNumDescriptors(), 0);
  EXPECT_EQ(database->NumMatches(), 0);
  EXPECT_EQ(database->NumMatchedImagePairs(), 0);
  EXPECT_EQ(database->NumVerifiedImagePairs(), 0);
}

TEST_P(ParameterizedDatabaseTests, Rig) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  EXPECT_EQ(database->NumRigs(), 0);
  Rig rig;
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, 1));
  rig.SetRigId(database->WriteRig(rig));
  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_TRUE(database->ExistsRig(rig.RigId()));
  EXPECT_EQ(database->ReadRig(rig.RigId()), rig);

  database->ClearRigs();
  EXPECT_EQ(database->NumRigs(), 0);

  rig.AddSensor(sensor_t(SensorType::CAMERA, 2),
                Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>()));
  rig.AddSensor(sensor_t(SensorType::IMU, 3));
  rig.AddSensor(sensor_t(SensorType::IMU, 4),
                Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>()));
  rig.SetRigId(database->WriteRig(rig));
  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_TRUE(database->ExistsRig(rig.RigId()));
  EXPECT_EQ(database->ReadRig(rig.RigId()), rig);
  EXPECT_EQ(database->ReadRigWithSensor(sensor_t(SensorType::CAMERA, 1)), rig);
  EXPECT_EQ(database->ReadRigWithSensor(sensor_t(SensorType::IMU, 4)), rig);
  EXPECT_EQ(database->ReadRigWithSensor(sensor_t(SensorType::IMU, 42)),
            std::nullopt);
  rig.SensorFromRig(sensor_t(SensorType::CAMERA, 2)) =
      Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>());
  database->UpdateRig(rig);
  EXPECT_EQ(database->ReadRig(rig.RigId()), rig);
  Rig rig2;
  rig2.AddRefSensor(sensor_t(SensorType::IMU, 10));
  rig2.SetRigId(rig.RigId() + 1);
  database->WriteRig(rig2, /*use_rig_id=*/true);
  EXPECT_EQ(database->NumRigs(), 2);
  EXPECT_TRUE(database->ExistsRig(rig.RigId()));
  EXPECT_TRUE(database->ExistsRig(rig2.RigId()));
  EXPECT_EQ(database->ReadAllRigs().size(), 2);
  EXPECT_EQ(database->ReadAllRigs()[0].RigId(), rig.RigId());
  EXPECT_EQ(database->ReadAllRigs()[1].RigId(), rig2.RigId());
  database->ClearRigs();
  EXPECT_EQ(database->NumRigs(), 0);
}

TEST_P(ParameterizedDatabaseTests, Camera) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  EXPECT_EQ(database->NumCameras(), 0);
  Camera camera = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 1.0, 1, 1);
  camera.camera_id = database->WriteCamera(camera);
  EXPECT_EQ(database->NumCameras(), 1);
  EXPECT_TRUE(database->ExistsCamera(camera.camera_id));
  EXPECT_EQ(database->ReadCamera(camera.camera_id).camera_id, camera.camera_id);
  EXPECT_EQ(database->ReadCamera(camera.camera_id).model_id, camera.model_id);
  EXPECT_EQ(database->ReadCamera(camera.camera_id), camera);
  camera.SetFocalLength(2 * camera.FocalLength());
  database->UpdateCamera(camera);
  EXPECT_EQ(database->ReadCamera(camera.camera_id), camera);
  Camera camera2 = camera;
  camera2.camera_id = camera.camera_id + 1;
  database->WriteCamera(camera2, true);
  EXPECT_EQ(database->NumCameras(), 2);
  EXPECT_TRUE(database->ExistsCamera(camera.camera_id));
  EXPECT_TRUE(database->ExistsCamera(camera2.camera_id));
  EXPECT_EQ(database->ReadAllCameras().size(), 2);
  EXPECT_THAT(database->ReadAllCameras(),
              testing::ElementsAre(camera, camera2));
  database->ClearCameras();
  EXPECT_EQ(database->NumCameras(), 0);
}

TEST_P(ParameterizedDatabaseTests, CameraCalibrations) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  EXPECT_EQ(database->NumCameras(), 0);

  // 1. Initial write with GUESS.
  Camera camera_guess = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 100.0, 1000, 1000);
  camera_guess.source = CameraSource::GUESS;
  const camera_t camera_id = database->WriteCamera(camera_guess);
  camera_guess.camera_id = camera_id;

  EXPECT_EQ(database->NumCameras(), 1);
  EXPECT_TRUE(database->ExistsCamera(camera_id));
  EXPECT_TRUE(database->ExistsCamera(camera_id, CameraSource::BEST));
  EXPECT_TRUE(database->ExistsCamera(camera_id, CameraSource::GUESS));
  EXPECT_FALSE(database->ExistsCamera(camera_id, CameraSource::EXIF));
  EXPECT_FALSE(database->ExistsCamera(camera_id, CameraSource::SINGLE_VIEW));
  EXPECT_FALSE(database->ExistsCamera(camera_id, CameraSource::USER));
  EXPECT_FALSE(database->ExistsCamera(camera_id, CameraSource::VIEW_GRAPH));

  EXPECT_EQ(database->ReadCamera(camera_id), camera_guess);
  EXPECT_EQ(database->ReadCamera(camera_id, CameraSource::BEST), camera_guess);
  EXPECT_EQ(database->ReadCamera(camera_id, CameraSource::GUESS), camera_guess);

  // 2. Add EXIF calibration (higher priority: EXIF > GUESS).
  Camera camera_exif = camera_guess;
  camera_exif.source = CameraSource::EXIF;
  camera_exif.SetFocalLength(120.0);
  database->UpdateCamera(camera_exif);

  EXPECT_EQ(database->NumCameras(), 1);
  EXPECT_TRUE(database->ExistsCamera(camera_id, CameraSource::EXIF));
  EXPECT_TRUE(database->ExistsCamera(camera_id, CameraSource::GUESS));
  EXPECT_EQ(database->ReadCamera(camera_id), camera_exif);
  EXPECT_EQ(database->ReadCamera(camera_id, CameraSource::BEST), camera_exif);
  EXPECT_EQ(database->ReadCamera(camera_id, CameraSource::EXIF), camera_exif);
  EXPECT_EQ(database->ReadCamera(camera_id, CameraSource::GUESS), camera_guess);

  // 3. Add SINGLE_VIEW calibration (SINGLE_VIEW > EXIF).
  Camera camera_sv = camera_guess;
  camera_sv.source = CameraSource::SINGLE_VIEW;
  camera_sv.SetFocalLength(130.0);
  database->UpdateCamera(camera_sv);

  EXPECT_EQ(database->ReadCamera(camera_id), camera_sv);
  EXPECT_EQ(database->ReadCamera(camera_id, CameraSource::BEST), camera_sv);
  EXPECT_EQ(database->ReadCamera(camera_id, CameraSource::SINGLE_VIEW),
            camera_sv);

  // 4. Add VIEW_GRAPH calibration (VIEW_GRAPH > USER > SINGLE_VIEW).
  Camera camera_vg = camera_guess;
  camera_vg.source = CameraSource::VIEW_GRAPH;
  camera_vg.SetFocalLength(140.0);
  database->UpdateCamera(camera_vg);

  EXPECT_EQ(database->ReadCamera(camera_id), camera_vg);
  EXPECT_EQ(database->ReadCamera(camera_id, CameraSource::BEST), camera_vg);
  EXPECT_EQ(database->ReadCamera(camera_id, CameraSource::VIEW_GRAPH),
            camera_vg);

  // 5. Add USER calibration (USER < VIEW_GRAPH).
  // Writing USER should store the calibration, but BEST remains VIEW_GRAPH.
  Camera camera_user = camera_guess;
  camera_user.source = CameraSource::USER;
  camera_user.SetFocalLength(135.0);
  database->UpdateCamera(camera_user);

  EXPECT_EQ(database->ReadCamera(camera_id), camera_vg);
  EXPECT_EQ(database->ReadCamera(camera_id, CameraSource::BEST), camera_vg);
  EXPECT_EQ(database->ReadCamera(camera_id, CameraSource::USER), camera_user);

  // 6. Test ReadAllCameraCalibrations(camera_id).
  const auto calibrations = database->ReadAllCameraCalibrations(camera_id);
  EXPECT_EQ(calibrations.size(), 5);
  EXPECT_EQ(calibrations.at(CameraSource::GUESS), camera_guess);
  EXPECT_EQ(calibrations.at(CameraSource::EXIF), camera_exif);
  EXPECT_EQ(calibrations.at(CameraSource::SINGLE_VIEW), camera_sv);
  EXPECT_EQ(calibrations.at(CameraSource::USER), camera_user);
  EXPECT_EQ(calibrations.at(CameraSource::VIEW_GRAPH), camera_vg);

  // 7. Test ReadAllCameraCalibrations() (all cameras).
  const auto all_calibrations = database->ReadAllCameraCalibrations();
  EXPECT_EQ(all_calibrations.size(), 1);
  EXPECT_EQ(all_calibrations.at(camera_id).size(), 5);

  // 8. Test ReadAllCameras(source).
  EXPECT_THAT(database->ReadAllCameras(CameraSource::BEST),
              testing::ElementsAre(camera_vg));
  EXPECT_THAT(database->ReadAllCameras(CameraSource::VIEW_GRAPH),
              testing::ElementsAre(camera_vg));
  EXPECT_THAT(database->ReadAllCameras(CameraSource::USER),
              testing::ElementsAre(camera_user));
  EXPECT_THAT(database->ReadAllCameras(CameraSource::SINGLE_VIEW),
              testing::ElementsAre(camera_sv));
  EXPECT_THAT(database->ReadAllCameras(CameraSource::EXIF),
              testing::ElementsAre(camera_exif));
  EXPECT_THAT(database->ReadAllCameras(CameraSource::GUESS),
              testing::ElementsAre(camera_guess));
  EXPECT_TRUE(database->ReadAllCameras(CameraSource::UNKNOWN).empty());

  // 9. Test ReadCameraExcludingSources and ReadAllCamerasExcludingSources.
  EXPECT_EQ(database->ReadCameraExcludingSources(camera_id,
                                                 {CameraSource::VIEW_GRAPH}),
            camera_user);
  EXPECT_EQ(database->ReadCameraExcludingSources(
                camera_id, {CameraSource::VIEW_GRAPH, CameraSource::USER}),
            camera_sv);
  EXPECT_EQ(database->ReadCameraExcludingSources(camera_id,
                                                 {CameraSource::VIEW_GRAPH,
                                                  CameraSource::USER,
                                                  CameraSource::SINGLE_VIEW}),
            camera_exif);
  EXPECT_EQ(database->ReadCameraExcludingSources(camera_id,
                                                 {CameraSource::VIEW_GRAPH,
                                                  CameraSource::USER,
                                                  CameraSource::SINGLE_VIEW,
                                                  CameraSource::EXIF}),
            camera_guess);
  // Fallback to highest available when all are excluded.
  EXPECT_EQ(database->ReadCameraExcludingSources(camera_id,
                                                 {CameraSource::VIEW_GRAPH,
                                                  CameraSource::USER,
                                                  CameraSource::SINGLE_VIEW,
                                                  CameraSource::EXIF,
                                                  CameraSource::GUESS}),
            camera_vg);

  const auto cams_ex_vg =
      database->ReadAllCamerasExcludingSources({CameraSource::VIEW_GRAPH});
  EXPECT_EQ(cams_ex_vg.size(), 1);
  EXPECT_EQ(cams_ex_vg.at(camera_id), camera_user);

  const auto cams_ex_sv_vg = database->ReadAllCamerasExcludingSources(
      {CameraSource::SINGLE_VIEW, CameraSource::VIEW_GRAPH});
  EXPECT_EQ(cams_ex_sv_vg.size(), 1);
  EXPECT_EQ(cams_ex_sv_vg.at(camera_id), camera_user);

  // 10. Test DeleteCameraCalibration for a specific source (VIEW_GRAPH).
  // Deleting VIEW_GRAPH should fallback to USER as BEST.
  database->DeleteCameraCalibration(camera_id, CameraSource::VIEW_GRAPH);
  EXPECT_FALSE(database->ExistsCamera(camera_id, CameraSource::VIEW_GRAPH));
  EXPECT_EQ(database->NumCameras(), 1);
  EXPECT_EQ(database->ReadCamera(camera_id), camera_user);
  EXPECT_EQ(database->ReadCamera(camera_id, CameraSource::BEST), camera_user);
  EXPECT_EQ(database->ReadAllCameraCalibrations(camera_id).size(), 4);

  // 10. Delete another source (USER). Fallback to SINGLE_VIEW.
  database->DeleteCameraCalibration(camera_id, CameraSource::USER);
  EXPECT_FALSE(database->ExistsCamera(camera_id, CameraSource::USER));
  EXPECT_EQ(database->ReadCamera(camera_id), camera_sv);

  // 11. Delete with BEST deletes the entire camera and all remaining
  // calibrations.
  database->DeleteCameraCalibration(camera_id, CameraSource::BEST);
  EXPECT_EQ(database->NumCameras(), 0);
  EXPECT_FALSE(database->ExistsCamera(camera_id));
  EXPECT_TRUE(database->ReadAllCameraCalibrations(camera_id).empty());
}

TEST_P(ParameterizedDatabaseTests, CameraMergeCalibrations) {
  std::shared_ptr<Database> database1 = GetParam()(kInMemorySqliteDatabasePath);
  std::shared_ptr<Database> database2 = GetParam()(kInMemorySqliteDatabasePath);

  Camera cam1_guess = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 100.0, 1000, 1000);
  cam1_guess.source = CameraSource::GUESS;
  const camera_t cam1_id = database1->WriteCamera(cam1_guess);
  cam1_guess.camera_id = cam1_id;

  Camera cam1_exif = cam1_guess;
  cam1_exif.source = CameraSource::EXIF;
  cam1_exif.SetFocalLength(120.0);
  database1->UpdateCamera(cam1_exif);

  Camera cam2_guess = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 200.0, 800, 800);
  cam2_guess.source = CameraSource::GUESS;
  const camera_t cam2_id = database2->WriteCamera(cam2_guess);
  cam2_guess.camera_id = cam2_id;

  Camera cam2_user = cam2_guess;
  cam2_user.source = CameraSource::USER;
  cam2_user.SetFocalLength(250.0);
  database2->UpdateCamera(cam2_user);

  std::shared_ptr<Database> merged_database =
      GetParam()(kInMemorySqliteDatabasePath);
  Database::Merge(*database1, *database2, merged_database.get());

  EXPECT_EQ(merged_database->NumCameras(), 2);

  // Check camera 1 in merged db.
  EXPECT_EQ(merged_database->ReadCamera(1), cam1_exif);
  EXPECT_EQ(merged_database->ReadCamera(1, CameraSource::GUESS).FocalLength(),
            100.0);
  EXPECT_EQ(merged_database->ReadCamera(1, CameraSource::EXIF).FocalLength(),
            120.0);
  const auto calibs1 = merged_database->ReadAllCameraCalibrations(1);
  EXPECT_EQ(calibs1.size(), 2);

  // Check camera 2 in merged db.
  cam2_user.camera_id = 2;
  EXPECT_EQ(merged_database->ReadCamera(2), cam2_user);
  EXPECT_EQ(merged_database->ReadCamera(2, CameraSource::GUESS).FocalLength(),
            200.0);
  EXPECT_EQ(merged_database->ReadCamera(2, CameraSource::USER).FocalLength(),
            250.0);
  const auto calibs2 = merged_database->ReadAllCameraCalibrations(2);
  EXPECT_EQ(calibs2.size(), 2);
}

TEST_P(ParameterizedDatabaseTests, CameraAutoIncrement) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  EXPECT_EQ(database->NumCameras(), 0);

  // 1. Write camera with use_camera_id=false. ID should be 1.
  Camera camera1 = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 100.0, 1000, 1000);
  camera1.source = CameraSource::GUESS;
  const camera_t id1 = database->WriteCamera(camera1, /*use_camera_id=*/false);
  EXPECT_EQ(id1, 1);
  EXPECT_EQ(database->NumCameras(), 1);

  // 2. Add multiple calibrations for camera 1.
  Camera camera1_exif = camera1;
  camera1_exif.camera_id = id1;
  camera1_exif.source = CameraSource::EXIF;
  camera1_exif.SetFocalLength(120.0);
  database->UpdateCamera(camera1_exif);

  Camera camera1_user = camera1;
  camera1_user.camera_id = id1;
  camera1_user.source = CameraSource::USER;
  camera1_user.SetFocalLength(130.0);
  database->UpdateCamera(camera1_user);

  EXPECT_EQ(database->NumCameras(), 1);
  EXPECT_EQ(database->ReadAllCameraCalibrations(id1).size(), 3);

  // 3. Write second camera with use_camera_id=false. ID should be 2.
  Camera camera2 = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 200.0, 800, 800);
  const camera_t id2 = database->WriteCamera(camera2, /*use_camera_id=*/false);
  EXPECT_EQ(id2, 2);
  EXPECT_EQ(database->NumCameras(), 2);

  // 4. Write third camera with use_camera_id=false. ID should be 3.
  Camera camera3 = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 300.0, 600, 600);
  const camera_t id3 = database->WriteCamera(camera3, /*use_camera_id=*/false);
  EXPECT_EQ(id3, 3);
  EXPECT_EQ(database->NumCameras(), 3);

  // 5. Delete camera 2 completely.
  database->DeleteCameraCalibration(id2, CameraSource::BEST);
  EXPECT_EQ(database->NumCameras(), 2);
  EXPECT_FALSE(database->ExistsCamera(id2));

  // 6. Write fourth camera with use_camera_id=false.
  // MAX(camera_id) is 3, so next ID must be 4.
  Camera camera4 = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 400.0, 400, 400);
  const camera_t id4 = database->WriteCamera(camera4, /*use_camera_id=*/false);
  EXPECT_EQ(id4, 4);
  EXPECT_EQ(database->NumCameras(), 3);

  EXPECT_TRUE(database->ExistsCamera(1));
  EXPECT_FALSE(database->ExistsCamera(2));
  EXPECT_TRUE(database->ExistsCamera(3));
  EXPECT_TRUE(database->ExistsCamera(4));
}

TEST_P(ParameterizedDatabaseTests, Frame) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  Rig rig;
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, 1));
  rig.SetRigId(database->WriteRig(rig));
  EXPECT_EQ(database->NumFrames(), 0);

  Frame frame;
  frame.SetRigId(rig.RigId());
  frame.SetFrameId(database->WriteFrame(frame));
  EXPECT_EQ(database->NumFrames(), 1);
  EXPECT_TRUE(database->ExistsFrame(frame.FrameId()));
  EXPECT_EQ(database->ReadFrame(frame.FrameId()), frame);

  database->ClearFrames();
  EXPECT_EQ(database->NumFrames(), 0);

  frame.AddDataId(data_t(sensor_t(SensorType::IMU, 1), 2));
  frame.AddDataId(data_t(sensor_t(SensorType::CAMERA, 1), 3));
  frame.SetFrameId(database->WriteFrame(frame));
  EXPECT_EQ(database->NumFrames(), 1);
  EXPECT_TRUE(database->ExistsFrame(frame.FrameId()));
  EXPECT_EQ(database->ReadFrame(frame.FrameId()), frame);

  frame.AddDataId(data_t(sensor_t(SensorType::CAMERA, 2), 4));
  database->UpdateFrame(frame);
  EXPECT_EQ(database->ReadFrame(frame.FrameId()), frame);
  Frame frame2;
  frame2.SetRigId(rig.RigId());
  frame2.AddDataId(data_t(sensor_t(SensorType::CAMERA, 2), 5));
  frame2.SetFrameId(frame.FrameId() + 1);
  database->WriteFrame(frame2, /*use_frame_id=*/true);
  EXPECT_EQ(database->NumFrames(), 2);
  EXPECT_TRUE(database->ExistsFrame(frame.FrameId()));
  EXPECT_TRUE(database->ExistsFrame(frame2.FrameId()));
  EXPECT_EQ(database->ReadAllFrames().size(), 2);
  EXPECT_EQ(database->ReadAllFrames()[0].FrameId(), frame.FrameId());
  EXPECT_EQ(database->ReadAllFrames()[1].FrameId(), frame2.FrameId());

  database->ClearFrames();
  EXPECT_EQ(database->NumFrames(), 0);
}

TEST_P(ParameterizedDatabaseTests, Image) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  Camera camera = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 1.0, 1, 1);
  camera.camera_id = database->WriteCamera(camera);
  Rig rig;
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, camera.camera_id));
  rig.SetRigId(database->WriteRig(rig));
  EXPECT_EQ(database->NumImages(), 0);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.camera_id);
  image.SetImageId(database->WriteImage(image));
  Frame frame;
  frame.SetRigId(rig.RigId());
  frame.AddDataId(image.DataId());
  frame.SetFrameId(database->WriteFrame(frame));
  image.SetFrameId(frame.FrameId());
  EXPECT_EQ(database->NumImages(), 1);
  EXPECT_TRUE(database->ExistsImage(image.ImageId()));
  EXPECT_EQ(database->ReadImage(image.ImageId()), image);
  EXPECT_EQ(database->ReadImageWithName(image.Name()), image);
  EXPECT_EQ(database->ReadImageWithName("foobar"), std::nullopt);
  image.SetName("test_changed");
  database->UpdateImage(image);
  EXPECT_EQ(database->ReadImage(image.ImageId()), image);
  Image image2 = image;
  image2.SetName("test2");
  image2.SetImageId(image.ImageId() + 1);
  frame.AddDataId(image2.DataId());
  database->UpdateFrame(frame);
  EXPECT_EQ(database->WriteImage(image2, /*use_image_id=*/true),
            image2.ImageId());
  EXPECT_EQ(database->NumImages(), 2);
  EXPECT_TRUE(database->ExistsImage(image.ImageId()));
  EXPECT_TRUE(database->ExistsImage(image2.ImageId()));
  EXPECT_THAT(database->ReadAllImages(), testing::ElementsAre(image, image2));
  database->ClearImages();
  EXPECT_EQ(database->NumImages(), 0);
}

TEST_P(ParameterizedDatabaseTests, PosePrior) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  Camera camera;
  camera.camera_id = database->WriteCamera(camera);
  Image image;
  image.SetCameraId(camera.camera_id);
  EXPECT_EQ(database->NumPosePriors(), 0);
  PosePrior pose_prior;
  pose_prior.corr_data_id = image.DataId();
  pose_prior.position = Eigen::Vector3d(0.1, 0.2, 0.3);
  pose_prior.position_covariance = RandomEigenMatrixd<3, 3>();
  pose_prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  pose_prior.gravity = RandomEigenVectord<3>();
  pose_prior.pose_prior_id = database->WritePosePrior(pose_prior);
  EXPECT_ANY_THROW(database->WritePosePrior(pose_prior));
  EXPECT_EQ(database->NumPosePriors(), 1);
  EXPECT_EQ(database->ReadPosePrior(pose_prior.pose_prior_id,
                                    /*is_deprecated_image_prior=*/false),
            pose_prior);
  pose_prior.position_covariance = Eigen::Matrix3d::Identity();
  database->UpdatePosePrior(pose_prior);
  EXPECT_EQ(database->ReadPosePrior(pose_prior.pose_prior_id,
                                    /*is_deprecated_image_prior=*/false),
            pose_prior);
  EXPECT_THAT(database->ReadAllPosePriors(), testing::ElementsAre(pose_prior));
  database->ClearPosePriors();
  EXPECT_EQ(database->NumPosePriors(), 0);
}

TEST_P(ParameterizedDatabaseTests, Keypoints) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  Camera camera;
  camera.camera_id = database->WriteCamera(camera);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.camera_id);
  image.SetImageId(database->WriteImage(image));
  EXPECT_EQ(database->NumKeypoints(), 0);
  EXPECT_EQ(database->NumKeypointsForImage(image.ImageId()), 0);
  const FeatureKeypoints keypoints = FeatureKeypoints(10);
  database->WriteKeypoints(image.ImageId(), keypoints);
  EXPECT_EQ(keypoints, database->ReadKeypoints(image.ImageId()));
  EXPECT_EQ(database->NumKeypoints(), 10);
  EXPECT_EQ(database->MaxNumKeypoints(), 10);
  EXPECT_EQ(database->NumKeypointsForImage(image.ImageId()), 10);
  FeatureKeypoints keypoints2 = FeatureKeypoints(20);
  image.SetName("test2");
  image.SetImageId(database->WriteImage(image));
  database->WriteKeypoints(image.ImageId(), keypoints2);
  EXPECT_EQ(keypoints2, database->ReadKeypoints(image.ImageId()));
  EXPECT_EQ(database->NumKeypoints(), 30);
  EXPECT_EQ(database->MaxNumKeypoints(), 20);
  EXPECT_EQ(database->NumKeypointsForImage(image.ImageId()), 20);
  keypoints2[0].x += 1;
  database->UpdateKeypoints(image.ImageId(), keypoints2);
  EXPECT_EQ(keypoints2, database->ReadKeypoints(image.ImageId()));
  database->ClearKeypoints();
  EXPECT_EQ(database->NumKeypoints(), 0);
  EXPECT_EQ(database->MaxNumKeypoints(), 0);
  EXPECT_EQ(database->NumKeypointsForImage(image.ImageId()), 0);
}

TEST_P(ParameterizedDatabaseTests, ReadKeypointsEmpty) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  Camera camera;
  camera.camera_id = database->WriteCamera(camera);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.camera_id);
  image.SetImageId(database->WriteImage(image));
  // Reading keypoints for an image with no keypoints should return empty.
  const FeatureKeypoints keypoints = database->ReadKeypoints(image.ImageId());
  EXPECT_TRUE(keypoints.empty());
}

TEST_P(ParameterizedDatabaseTests, Descriptors) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  Camera camera;
  camera.camera_id = database->WriteCamera(camera);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.camera_id);
  image.SetImageId(database->WriteImage(image));
  EXPECT_EQ(database->NumDescriptors(), 0);
  EXPECT_EQ(database->NumDescriptorsForImage(image.ImageId()), 0);
  const FeatureDescriptors descriptors(FeatureExtractorType::SIFT,
                                       FeatureDescriptorsData::Random(10, 128));
  database->WriteDescriptors(image.ImageId(), descriptors);
  const FeatureDescriptors descriptors_read =
      database->ReadDescriptors(image.ImageId());
  EXPECT_EQ(descriptors.data.rows(), descriptors_read.data.rows());
  EXPECT_EQ(descriptors.data.cols(), descriptors_read.data.cols());
  EXPECT_EQ(descriptors.type, descriptors_read.type);
  for (Eigen::Index r = 0; r < descriptors.data.rows(); ++r) {
    for (Eigen::Index c = 0; c < descriptors.data.cols(); ++c) {
      EXPECT_EQ(descriptors.data(r, c), descriptors_read.data(r, c));
    }
  }
  EXPECT_EQ(database->NumDescriptors(), 10);
  EXPECT_EQ(database->MaxNumDescriptors(), 10);
  EXPECT_EQ(database->NumDescriptorsForImage(image.ImageId()), 10);
  const FeatureDescriptors descriptors2(FeatureExtractorType::UNDEFINED,
                                        FeatureDescriptorsData(20, 128));
  image.SetName("test2");
  image.SetImageId(database->WriteImage(image));
  database->WriteDescriptors(image.ImageId(), descriptors2);
  const FeatureDescriptors descriptors2_read =
      database->ReadDescriptors(image.ImageId());
  EXPECT_EQ(descriptors2.type, descriptors2_read.type);
  EXPECT_EQ(database->NumDescriptors(), 30);
  EXPECT_EQ(database->MaxNumDescriptors(), 20);
  EXPECT_EQ(database->NumDescriptorsForImage(image.ImageId()), 20);
  database->ClearDescriptors();
  EXPECT_EQ(database->NumDescriptors(), 0);
  EXPECT_EQ(database->MaxNumDescriptors(), 0);
  EXPECT_EQ(database->NumDescriptorsForImage(image.ImageId()), 0);
}

TEST_P(ParameterizedDatabaseTests, DescriptorFeatureTypeDefault) {
  // Test that descriptors written with UNDEFINED type are read back correctly,
  // and that the database default (for migration) is SIFT (0).
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  Camera camera;
  camera.camera_id = database->WriteCamera(camera);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.camera_id);
  image.SetImageId(database->WriteImage(image));

  // Write descriptors with UNDEFINED type (default).
  FeatureDescriptors descriptors;
  descriptors.data = FeatureDescriptorsData(5, 128);
  EXPECT_EQ(descriptors.type, FeatureExtractorType::UNDEFINED);
  database->WriteDescriptors(image.ImageId(), descriptors);

  // Read back and verify the type is preserved.
  const FeatureDescriptors descriptors_read =
      database->ReadDescriptors(image.ImageId());
  EXPECT_EQ(descriptors_read.type, FeatureExtractorType::UNDEFINED);

  // Write another image with SIFT type.
  image.SetName("test2");
  image.SetImageId(database->WriteImage(image));
  const FeatureDescriptors descriptors_sift(FeatureExtractorType::SIFT,
                                            FeatureDescriptorsData(5, 128));
  database->WriteDescriptors(image.ImageId(), descriptors_sift);

  const FeatureDescriptors descriptors_sift_read =
      database->ReadDescriptors(image.ImageId());
  EXPECT_EQ(descriptors_sift_read.type, FeatureExtractorType::SIFT);
}

TEST_P(ParameterizedDatabaseTests, Matches) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  const image_t image_id1 = 1;
  const image_t image_id2 = 2;
  constexpr int kNumMatches = 1000;
  FeatureMatches matches12(kNumMatches);
  FeatureMatches matches21(kNumMatches);
  for (size_t i = 0; i < matches12.size(); ++i) {
    matches12[i].point2D_idx1 = i;
    matches12[i].point2D_idx2 = 10000 + i;
    matches21[i].point2D_idx1 = 10000 + i;
    matches21[i].point2D_idx2 = i;
  }

  auto expectValidMatches = [&]() {
    EXPECT_EQ(database->NumMatchedImagePairs(), 1);
    const FeatureMatches matches_read12 =
        database->ReadMatches(image_id1, image_id2);
    EXPECT_EQ(matches12.size(), matches_read12.size());
    for (size_t i = 0; i < matches12.size(); ++i) {
      EXPECT_EQ(matches12[i].point2D_idx1, matches_read12[i].point2D_idx1);
      EXPECT_EQ(matches12[i].point2D_idx2, matches_read12[i].point2D_idx2);
    }
    const FeatureMatches matches_read21 =
        database->ReadMatches(image_id2, image_id1);
    EXPECT_EQ(matches12.size(), matches_read21.size());
    for (size_t i = 0; i < matches12.size(); ++i) {
      EXPECT_EQ(matches12[i].point2D_idx1, matches_read21[i].point2D_idx2);
      EXPECT_EQ(matches12[i].point2D_idx2, matches_read21[i].point2D_idx1);
    }
  };

  EXPECT_EQ(database->NumMatchedImagePairs(), 0);
  database->WriteMatches(image_id1, image_id2, matches12);
  expectValidMatches();
  database->DeleteMatches(image_id1, image_id2);
  EXPECT_EQ(database->NumMatchedImagePairs(), 0);
  database->WriteMatches(image_id2, image_id1, matches21);
  expectValidMatches();

  EXPECT_EQ(database->ReadAllMatchesBlob().size(), 1);
  EXPECT_EQ(database->ReadAllMatchesBlob()[0].first,
            ImagePairToPairId(image_id1, image_id2));
  const std::vector<std::pair<image_pair_t, FeatureMatches>> matches =
      database->ReadAllMatches();
  EXPECT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0].first, ImagePairToPairId(image_id1, image_id2));
  const std::vector<std::pair<image_pair_t, int>> pair_ids_and_num_matches =
      database->ReadNumMatches();
  EXPECT_EQ(pair_ids_and_num_matches.size(), 1);
  EXPECT_EQ(pair_ids_and_num_matches[0].first,
            ImagePairToPairId(image_id1, image_id2));
  EXPECT_EQ(pair_ids_and_num_matches[0].second, matches[0].second.size());
  EXPECT_EQ(database->NumMatches(), kNumMatches);
  database->DeleteMatches(image_id1, image_id2);
  EXPECT_EQ(database->NumMatches(), 0);
  database->WriteMatches(image_id1, image_id2, matches12);
  EXPECT_EQ(database->NumMatches(), kNumMatches);
  database->ClearMatches();
  EXPECT_EQ(database->NumMatches(), 0);
}

TEST_P(ParameterizedDatabaseTests, TwoViewGeometry) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  const image_t image_id1 = 1;
  const image_t image_id2 = 2;
  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches = FeatureMatches(1000);
  two_view_geometry.config =
      TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
  two_view_geometry.F = RandomEigenMatrixd<3, 3>();
  two_view_geometry.E = RandomEigenMatrixd<3, 3>();
  two_view_geometry.H = RandomEigenMatrixd<3, 3>();
  two_view_geometry.cam2_from_cam1 =
      Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>());
  // Distinct cameras so the swap performed on inverse reads is observable.
  two_view_geometry.camera1 = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 640, 480);
  two_view_geometry.camera2 =
      Camera::CreateFromModelId(2, CameraModelId::kPinhole, 200.0, 800, 600);
  database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  const TwoViewGeometry two_view_geometry_read =
      database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(),
            two_view_geometry_read.inlier_matches.size());
  for (size_t i = 0; i < two_view_geometry_read.inlier_matches.size(); ++i) {
    EXPECT_EQ(two_view_geometry.inlier_matches[i].point2D_idx1,
              two_view_geometry_read.inlier_matches[i].point2D_idx1);
    EXPECT_EQ(two_view_geometry.inlier_matches[i].point2D_idx2,
              two_view_geometry_read.inlier_matches[i].point2D_idx2);
  }

  EXPECT_EQ(two_view_geometry.config, two_view_geometry_read.config);
  EXPECT_EQ(two_view_geometry.F, two_view_geometry_read.F);
  EXPECT_EQ(two_view_geometry.E, two_view_geometry_read.E);
  EXPECT_EQ(two_view_geometry.H, two_view_geometry_read.H);
  EXPECT_TRUE(two_view_geometry.cam2_from_cam1.has_value());
  EXPECT_TRUE(two_view_geometry_read.cam2_from_cam1.has_value());
  EXPECT_EQ(two_view_geometry.cam2_from_cam1->rotation().coeffs(),
            two_view_geometry_read.cam2_from_cam1->rotation().coeffs());
  EXPECT_EQ(two_view_geometry.cam2_from_cam1->translation(),
            two_view_geometry_read.cam2_from_cam1->translation());
  EXPECT_EQ(two_view_geometry.camera1, two_view_geometry_read.camera1);
  EXPECT_EQ(two_view_geometry.camera2, two_view_geometry_read.camera2);

  const TwoViewGeometry two_view_geometry_read_inv =
      database->ReadTwoViewGeometry(image_id2, image_id1);
  EXPECT_EQ(two_view_geometry_read_inv.inlier_matches.size(),
            two_view_geometry_read.inlier_matches.size());
  for (size_t i = 0; i < two_view_geometry_read.inlier_matches.size(); ++i) {
    EXPECT_EQ(two_view_geometry_read_inv.inlier_matches[i].point2D_idx2,
              two_view_geometry_read.inlier_matches[i].point2D_idx1);
    EXPECT_EQ(two_view_geometry_read_inv.inlier_matches[i].point2D_idx1,
              two_view_geometry_read.inlier_matches[i].point2D_idx2);
  }

  EXPECT_EQ(two_view_geometry_read_inv.config, two_view_geometry_read.config);
  EXPECT_EQ(two_view_geometry_read_inv.F.value().transpose(),
            two_view_geometry_read.F.value());
  EXPECT_EQ(two_view_geometry_read_inv.E.value().transpose(),
            two_view_geometry_read.E.value());
  EXPECT_TRUE(two_view_geometry_read_inv.H.value().inverse().eval().isApprox(
      two_view_geometry_read.H.value()));
  EXPECT_TRUE(two_view_geometry_read_inv.cam2_from_cam1.has_value());
  EXPECT_TRUE(two_view_geometry_read_inv.cam2_from_cam1->rotation().isApprox(
      Inverse(*two_view_geometry_read.cam2_from_cam1).rotation()));
  EXPECT_TRUE(two_view_geometry_read_inv.cam2_from_cam1->translation().isApprox(
      Inverse(*two_view_geometry_read.cam2_from_cam1).translation()));
  // The inverse read swaps the two cameras.
  EXPECT_EQ(two_view_geometry_read_inv.camera1, two_view_geometry.camera2);
  EXPECT_EQ(two_view_geometry_read_inv.camera2, two_view_geometry.camera1);

  const std::vector<std::pair<image_pair_t, TwoViewGeometry>>
      two_view_geometries = database->ReadTwoViewGeometries();
  EXPECT_EQ(two_view_geometries.size(), 1);
  EXPECT_EQ(two_view_geometries[0].first,
            ImagePairToPairId(image_id1, image_id2));
  EXPECT_EQ(two_view_geometry.config, two_view_geometries[0].second.config);
  EXPECT_EQ(two_view_geometry.F, two_view_geometries[0].second.F);
  EXPECT_EQ(two_view_geometry.E, two_view_geometries[0].second.E);
  EXPECT_EQ(two_view_geometry.H, two_view_geometries[0].second.H);
  EXPECT_TRUE(two_view_geometries[0].second.cam2_from_cam1.has_value());
  EXPECT_EQ(two_view_geometry.cam2_from_cam1->rotation().coeffs(),
            two_view_geometries[0].second.cam2_from_cam1->rotation().coeffs());
  EXPECT_EQ(two_view_geometry.cam2_from_cam1->translation(),
            two_view_geometries[0].second.cam2_from_cam1->translation());
  EXPECT_EQ(two_view_geometry.camera1, two_view_geometries[0].second.camera1);
  EXPECT_EQ(two_view_geometry.camera2, two_view_geometries[0].second.camera2);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(),
            two_view_geometries[0].second.inlier_matches.size());
  const std::vector<std::pair<image_pair_t, int>> pair_ids_and_num_inliers =
      database->ReadTwoViewGeometryNumInliers();
  EXPECT_EQ(pair_ids_and_num_inliers.size(), 1);
  EXPECT_EQ(pair_ids_and_num_inliers[0].first,
            ImagePairToPairId(image_id1, image_id2));
  EXPECT_EQ(pair_ids_and_num_inliers[0].second,
            two_view_geometry.inlier_matches.size());
  EXPECT_EQ(database->NumInlierMatches(), 1000);
  database->DeleteInlierMatches(image_id1, image_id2);
  EXPECT_TRUE(database->ExistsTwoViewGeometry(image_id1, image_id2));
  EXPECT_EQ(database->NumInlierMatches(), 0);
  database->DeleteTwoViewGeometry(image_id1, image_id2);
  EXPECT_FALSE(database->ExistsTwoViewGeometry(image_id1, image_id2));
  EXPECT_EQ(database->NumInlierMatches(), 0);
  database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  EXPECT_ANY_THROW(
      database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry));
  EXPECT_EQ(database->NumInlierMatches(), 1000);
  database->ClearTwoViewGeometries();
  EXPECT_EQ(database->NumInlierMatches(), 0);
  two_view_geometry.inlier_matches.clear();
  database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  EXPECT_EQ(two_view_geometry.cam2_from_cam1,
            database->ReadTwoViewGeometry(image_id1, image_id2).cam2_from_cam1);

  // Test with E and F set, but H missing.
  database->ClearTwoViewGeometries();
  TwoViewGeometry two_view_geometry_no_h;
  two_view_geometry_no_h.inlier_matches = FeatureMatches(10);
  two_view_geometry_no_h.config =
      TwoViewGeometry::ConfigurationType::CALIBRATED;
  two_view_geometry_no_h.E = RandomEigenMatrixd<3, 3>();
  two_view_geometry_no_h.F = RandomEigenMatrixd<3, 3>();
  database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry_no_h);
  const TwoViewGeometry two_view_geometry_no_h_read =
      database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_TRUE(two_view_geometry_no_h_read.E.has_value());
  EXPECT_TRUE(two_view_geometry_no_h_read.F.has_value());
  EXPECT_EQ(two_view_geometry_no_h.E, two_view_geometry_no_h_read.E);
  EXPECT_EQ(two_view_geometry_no_h.F, two_view_geometry_no_h_read.F);
  EXPECT_FALSE(two_view_geometry_no_h_read.H.has_value());
}

TEST_P(ParameterizedDatabaseTests, TwoViewGeometryWithoutCameras) {
  std::shared_ptr<Database> database = GetParam()(kInMemorySqliteDatabasePath);
  const image_t image_id1 = 1;
  const image_t image_id2 = 2;

  // A geometry that leaves the estimated cameras unset (as for configurations
  // that consume fixed intrinsics) round-trips them as nullopt.
  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::ConfigurationType::CALIBRATED;
  two_view_geometry.E = RandomEigenMatrixd<3, 3>();
  ASSERT_FALSE(two_view_geometry.camera1.has_value());
  ASSERT_FALSE(two_view_geometry.camera2.has_value());
  database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);

  const TwoViewGeometry read =
      database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_FALSE(read.camera1.has_value());
  EXPECT_FALSE(read.camera2.has_value());

  const std::vector<std::pair<image_pair_t, TwoViewGeometry>> all =
      database->ReadTwoViewGeometries();
  ASSERT_EQ(all.size(), 1);
  EXPECT_FALSE(all[0].second.camera1.has_value());
  EXPECT_FALSE(all[0].second.camera2.has_value());
}

TEST_P(ParameterizedDatabaseTests, Merge) {
  std::shared_ptr<Database> database1 = GetParam()(kInMemorySqliteDatabasePath);
  std::shared_ptr<Database> database2 = GetParam()(kInMemorySqliteDatabasePath);

  // This test intentionally uses custom, large, partially overlapping IDs from
  // rigs/frames/images/cameras which then require remapping of the IDs. This is
  // to ensure that the database can handle this case.

  Camera camera1 = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 1.0, 1, 1);
  camera1.camera_id = 50;
  database1->WriteCamera(camera1, /*use_camera_id=*/true);
  Camera camera2 = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 1.0, 1, 1);
  camera2.camera_id = 60;
  database1->WriteCamera(camera2, /*use_camera_id=*/true);
  Camera camera3 = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 1.0, 1, 1);
  camera3.camera_id = 55;
  database2->WriteCamera(camera3, /*use_camera_id=*/true);
  Camera camera4 = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 1.0, 1, 1);
  camera4.camera_id = 60;
  database2->WriteCamera(camera4, /*use_camera_id=*/true);

  Rig rig1;
  rig1.SetRigId(100);
  rig1.AddRefSensor(camera1.SensorId());
  rig1.AddSensor(camera2.SensorId(), Rigid3d());
  database1->WriteRig(rig1, /*use_rig_id=*/true);

  Rig rig2;
  rig2.SetRigId(200);
  rig2.AddRefSensor(camera3.SensorId());
  rig2.AddSensor(camera4.SensorId(), Rigid3d());
  database2->WriteRig(rig2, /*use_rig_id=*/true);

  const image_t image_id1 = 300;
  const image_t image_id2 = 400;
  const image_t image_id3 = 350;
  const image_t image_id4 = 400;

  Image image;
  image.SetImageId(image_id1);
  image.SetCameraId(camera1.camera_id);
  image.SetName("test1");
  database1->WriteImage(image, /*use_image_id=*/true);
  image.SetImageId(image_id2);
  image.SetCameraId(camera2.camera_id);
  image.SetName("test2");
  database1->WriteImage(image, /*use_image_id=*/true);

  image.SetImageId(image_id3);
  image.SetCameraId(camera3.camera_id);
  image.SetName("test3");
  database2->WriteImage(image, /*use_image_id=*/true);
  image.SetImageId(image_id4);
  image.SetCameraId(camera4.camera_id);
  image.SetName("test4");
  database2->WriteImage(image, /*use_image_id=*/true);

  Frame frame1;
  frame1.SetRigId(rig1.RigId());
  frame1.AddDataId(data_t(camera1.SensorId(), image_id1));
  frame1.AddDataId(data_t(camera2.SensorId(), image_id2));
  frame1.SetFrameId(1000);
  database1->WriteFrame(frame1, /*use_frame_id=*/true);
  Frame frame2;
  frame2.SetRigId(rig2.RigId());
  frame2.AddDataId(data_t(camera3.SensorId(), image_id3));
  frame2.AddDataId(data_t(camera4.SensorId(), image_id4));
  frame2.SetFrameId(2000);
  database2->WriteFrame(frame2, /*use_frame_id=*/true);

  PosePrior pose_prior1;
  pose_prior1.corr_data_id = data_t(camera1.SensorId(), image_id1);
  pose_prior1.position = RandomEigenVectord<3>();
  pose_prior1.pose_prior_id = database1->WritePosePrior(pose_prior1);

  PosePrior pose_prior2;
  pose_prior2.corr_data_id = data_t(camera3.SensorId(), image_id3);
  pose_prior2.position = RandomEigenVectord<3>();
  pose_prior2.pose_prior_id = database2->WritePosePrior(pose_prior2);

  auto keypoints1 = FeatureKeypoints(10);
  keypoints1[0].x = 100;
  auto keypoints2 = FeatureKeypoints(20);
  keypoints2[0].x = 200;
  auto keypoints3 = FeatureKeypoints(30);
  keypoints3[0].x = 300;
  auto keypoints4 = FeatureKeypoints(40);
  keypoints4[0].x = 400;

  const FeatureDescriptors descriptors1(
      FeatureExtractorType::UNDEFINED, FeatureDescriptorsData::Random(10, 128));
  const FeatureDescriptors descriptors2(
      FeatureExtractorType::UNDEFINED, FeatureDescriptorsData::Random(20, 128));
  const FeatureDescriptors descriptors3(
      FeatureExtractorType::UNDEFINED, FeatureDescriptorsData::Random(30, 128));
  const FeatureDescriptors descriptors4(
      FeatureExtractorType::UNDEFINED, FeatureDescriptorsData::Random(40, 128));

  database1->WriteKeypoints(image_id1, keypoints1);
  database1->WriteKeypoints(image_id2, keypoints2);
  database2->WriteKeypoints(image_id3, keypoints3);
  database2->WriteKeypoints(image_id4, keypoints4);
  database1->WriteDescriptors(image_id1, descriptors1);
  database1->WriteDescriptors(image_id2, descriptors2);
  database2->WriteDescriptors(image_id3, descriptors3);
  database2->WriteDescriptors(image_id4, descriptors4);
  database1->WriteMatches(image_id1, image_id2, FeatureMatches(10));
  database2->WriteMatches(image_id3, image_id4, FeatureMatches(10));
  database1->WriteTwoViewGeometry(image_id1, image_id2, TwoViewGeometry());
  database2->WriteTwoViewGeometry(image_id3, image_id4, TwoViewGeometry());

  std::shared_ptr<Database> merged_database =
      GetParam()(kInMemorySqliteDatabasePath);
  Database::Merge(*database1, *database2, merged_database.get());
  EXPECT_EQ(merged_database->NumRigs(), 2);
  EXPECT_EQ(merged_database->NumCameras(), 4);
  EXPECT_EQ(merged_database->NumFrames(), 2);
  EXPECT_EQ(merged_database->NumImages(), 4);
  EXPECT_EQ(merged_database->NumPosePriors(), 2);
  EXPECT_EQ(merged_database->NumKeypoints(), 100);
  EXPECT_EQ(merged_database->NumDescriptors(), 100);
  EXPECT_EQ(merged_database->NumMatches(), 20);
  EXPECT_EQ(merged_database->NumInlierMatches(), 0);
  EXPECT_EQ(merged_database->ReadAllFrames()[0].NumDataIds(),
            frame1.NumDataIds());
  EXPECT_EQ(merged_database->ReadAllFrames()[1].NumDataIds(),
            frame2.NumDataIds());
  for (const auto& frame : merged_database->ReadAllFrames()) {
    for (const auto& data_id : frame.DataIds()) {
      switch (data_id.sensor_id.type) {
        case SensorType::CAMERA:
          EXPECT_TRUE(merged_database->ExistsCamera(data_id.sensor_id.id));
          EXPECT_TRUE(merged_database->ExistsImage(data_id.id));
          break;
        default:
          GTEST_FAIL() << "Unexpected sensor type: " << data_id.sensor_id.type;
          break;
      }
    }
  }
  for (const auto& pose_prior : merged_database->ReadAllPosePriors()) {
    switch (pose_prior.corr_data_id.sensor_id.type) {
      case SensorType::CAMERA:
        EXPECT_TRUE(merged_database->ExistsCamera(
            pose_prior.corr_data_id.sensor_id.id));
        EXPECT_TRUE(merged_database->ExistsImage(pose_prior.corr_data_id.id));
        break;
      default:
        GTEST_FAIL() << "Unexpected sensor type: "
                     << pose_prior.corr_data_id.sensor_id.type;
        break;
    }
  }

  EXPECT_EQ(merged_database->ReadAllImages()[0].CameraId(), 1);
  EXPECT_EQ(merged_database->ReadAllImages()[1].CameraId(), 2);
  EXPECT_EQ(merged_database->ReadAllImages()[2].CameraId(), 3);
  EXPECT_EQ(merged_database->ReadAllImages()[3].CameraId(), 4);
  EXPECT_EQ(merged_database->ReadKeypoints(1).size(), 10);
  EXPECT_EQ(merged_database->ReadKeypoints(2).size(), 20);
  EXPECT_EQ(merged_database->ReadKeypoints(3).size(), 30);
  EXPECT_EQ(merged_database->ReadKeypoints(4).size(), 40);
  EXPECT_EQ(merged_database->ReadKeypoints(1)[0].x, 100);
  EXPECT_EQ(merged_database->ReadKeypoints(2)[0].x, 200);
  EXPECT_EQ(merged_database->ReadKeypoints(3)[0].x, 300);
  EXPECT_EQ(merged_database->ReadKeypoints(4)[0].x, 400);
  EXPECT_EQ(merged_database->ReadDescriptors(1).type, descriptors1.type);
  EXPECT_EQ(merged_database->ReadDescriptors(1).data.size(),
            descriptors1.data.size());
  EXPECT_EQ(merged_database->ReadDescriptors(2).type, descriptors2.type);
  EXPECT_EQ(merged_database->ReadDescriptors(2).data.size(),
            descriptors2.data.size());
  EXPECT_EQ(merged_database->ReadDescriptors(3).type, descriptors3.type);
  EXPECT_EQ(merged_database->ReadDescriptors(3).data.size(),
            descriptors3.data.size());
  EXPECT_EQ(merged_database->ReadDescriptors(4).type, descriptors4.type);
  EXPECT_EQ(merged_database->ReadDescriptors(4).data.size(),
            descriptors4.data.size());
  EXPECT_TRUE(merged_database->ExistsMatches(1, 2));
  EXPECT_FALSE(merged_database->ExistsMatches(2, 3));
  EXPECT_FALSE(merged_database->ExistsMatches(2, 4));
  EXPECT_TRUE(merged_database->ExistsMatches(3, 4));

  merged_database->ClearAllTables();
  EXPECT_EQ(merged_database->NumRigs(), 0);
  EXPECT_EQ(merged_database->NumCameras(), 0);
  EXPECT_EQ(merged_database->NumFrames(), 0);
  EXPECT_EQ(merged_database->NumImages(), 0);
  EXPECT_EQ(merged_database->NumPosePriors(), 0);
  EXPECT_EQ(merged_database->NumKeypoints(), 0);
  EXPECT_EQ(merged_database->NumDescriptors(), 0);
  EXPECT_EQ(merged_database->NumMatches(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    DatabaseTests,
    ParameterizedDatabaseTests,
    ::testing::Values([](const std::filesystem::path& path) {
      return Database::Open(path);
    }));

// Helper to create a database file with images and descriptors.
std::shared_ptr<Database> CreateDatabaseWithRandomDescriptors(
    const std::vector<int>& num_descriptors_per_image) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  const int num_images = num_descriptors_per_image.size();

  Camera camera;
  camera.camera_id = database->WriteCamera(camera);

  for (int i = 0; i < num_images; ++i) {
    Image image;
    image.SetName("image" + std::to_string(i));
    image.SetCameraId(camera.camera_id);
    image.SetImageId(database->WriteImage(image));
    database->WriteDescriptors(
        image.ImageId(),
        FeatureDescriptors(
            FeatureExtractorType::SIFT,
            FeatureDescriptorsData::Random(num_descriptors_per_image[i], 128)));
  }

  return database;
}

TEST(LoadRandomDatabaseDescriptorsTest, LoadEmpty) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  const auto result = LoadRandomDatabaseDescriptors(*database, -1);
  EXPECT_EQ(result.data.rows(), 0);
  EXPECT_EQ(result.data.cols(), 0);
  EXPECT_EQ(result.type, FeatureExtractorType::UNDEFINED);
}

TEST(LoadRandomDatabaseDescriptorsTest, LoadAll) {
  const auto database = CreateDatabaseWithRandomDescriptors({10, 20, 30});
  const auto result = LoadRandomDatabaseDescriptors(*database, -1);
  EXPECT_EQ(result.data.rows(), 60);
  EXPECT_EQ(result.data.cols(), 128);
  EXPECT_EQ(result.type, FeatureExtractorType::SIFT);
}

TEST(LoadRandomDatabaseDescriptorsTest, LoadAllWithLargeMax) {
  const auto database = CreateDatabaseWithRandomDescriptors({15, 15});
  const auto result = LoadRandomDatabaseDescriptors(*database, 1000);
  EXPECT_EQ(result.data.rows(), 30);
  EXPECT_EQ(result.data.cols(), 128);
}

TEST(LoadRandomDatabaseDescriptorsTest, LoadSubset) {
  const auto database = CreateDatabaseWithRandomDescriptors({10, 20, 30});
  const auto result = LoadRandomDatabaseDescriptors(*database, 10);
  EXPECT_EQ(result.data.rows(), 10);
  EXPECT_EQ(result.data.cols(), 128);
  EXPECT_EQ(result.type, FeatureExtractorType::SIFT);
}

TEST(LoadRandomDatabaseDescriptorsTest, LoadSubsetWithSomeEmpty) {
  const auto database =
      CreateDatabaseWithRandomDescriptors({0, 10, 0, 15, 0, 20, 0});
  const auto result = LoadRandomDatabaseDescriptors(*database, 15);
  EXPECT_EQ(result.data.rows(), 15);
  EXPECT_EQ(result.data.cols(), 128);
  EXPECT_EQ(result.type, FeatureExtractorType::SIFT);
}

TEST(LoadRandomDatabaseDescriptorsTest, LoadExactTotal) {
  const auto database = CreateDatabaseWithRandomDescriptors({0, 10, 0, 10, 0});
  const auto result = LoadRandomDatabaseDescriptors(*database, 20);
  EXPECT_EQ(result.data.rows(), 20);
  EXPECT_EQ(result.data.cols(), 128);
}

TEST(DatabaseMigrationTest, LegacyCameraTableMigration) {
  const auto database_path = CreateTestDir() / "legacy_cameras.db";
  sqlite3* db = nullptr;
  ASSERT_EQ(sqlite3_open_v2(database_path.string().c_str(),
                            &db,
                            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                            nullptr),
            SQLITE_OK);

  // Set user_version to 4.2.0 (before 4.3.0 camera migration).
  ASSERT_EQ(
      sqlite3_exec(
          db, "PRAGMA user_version = 4020000;", nullptr, nullptr, nullptr),
      SQLITE_OK);

  // Create legacy schema with single-column PRIMARY KEY cameras and images with
  // FK to cameras.
  const char* kLegacySchema =
      "CREATE TABLE cameras ("
      "  camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
      "  model INTEGER NOT NULL,"
      "  width INTEGER NOT NULL,"
      "  height INTEGER NOT NULL,"
      "  params BLOB,"
      "  prior_focal_length INTEGER NOT NULL);"
      "CREATE TABLE images ("
      "  image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
      "  name TEXT NOT NULL UNIQUE,"
      "  camera_id INTEGER NOT NULL,"
      "  CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < "
      "2147483647),"
      "  FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));"
      "CREATE TABLE keypoints ("
      "  image_id INTEGER PRIMARY KEY NOT NULL,"
      "  rows INTEGER NOT NULL,"
      "  cols INTEGER NOT NULL,"
      "  data BLOB,"
      "  FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);";
  ASSERT_EQ(sqlite3_exec(db, kLegacySchema, nullptr, nullptr, nullptr),
            SQLITE_OK);

  // Insert legacy camera 1 with prior_focal_length = 1.
  Camera cam1 = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 1000, 1000);
  sqlite3_stmt* stmt;
  ASSERT_EQ(sqlite3_prepare_v2(
                db,
                "INSERT INTO cameras (camera_id, model, width, height, params, "
                "prior_focal_length) VALUES (?, ?, ?, ?, ?, ?);",
                -1,
                &stmt,
                nullptr),
            SQLITE_OK);
  sqlite3_bind_int64(stmt, 1, 1);
  sqlite3_bind_int64(stmt, 2, static_cast<int>(cam1.model_id));
  sqlite3_bind_int64(stmt, 3, cam1.width);
  sqlite3_bind_int64(stmt, 4, cam1.height);
  sqlite3_bind_blob(stmt,
                    5,
                    cam1.params.data(),
                    cam1.params.size() * sizeof(double),
                    SQLITE_STATIC);
  sqlite3_bind_int64(stmt, 6, 1);
  ASSERT_EQ(sqlite3_step(stmt), SQLITE_DONE);
  sqlite3_finalize(stmt);

  // Insert legacy camera 2 with prior_focal_length = 0.
  Camera cam2 = Camera::CreateFromModelId(
      2, CameraModelId::kSimplePinhole, 200.0, 800, 800);
  ASSERT_EQ(sqlite3_prepare_v2(
                db,
                "INSERT INTO cameras (camera_id, model, width, height, params, "
                "prior_focal_length) VALUES (?, ?, ?, ?, ?, ?);",
                -1,
                &stmt,
                nullptr),
            SQLITE_OK);
  sqlite3_bind_int64(stmt, 1, 2);
  sqlite3_bind_int64(stmt, 2, static_cast<int>(cam2.model_id));
  sqlite3_bind_int64(stmt, 3, cam2.width);
  sqlite3_bind_int64(stmt, 4, cam2.height);
  sqlite3_bind_blob(stmt,
                    5,
                    cam2.params.data(),
                    cam2.params.size() * sizeof(double),
                    SQLITE_STATIC);
  sqlite3_bind_int64(stmt, 6, 0);
  ASSERT_EQ(sqlite3_step(stmt), SQLITE_DONE);
  sqlite3_finalize(stmt);

  // Insert legacy image 1.
  ASSERT_EQ(sqlite3_exec(db,
                         "INSERT INTO images (image_id, name, camera_id) "
                         "VALUES (1, 'image1.jpg', 1);",
                         nullptr,
                         nullptr,
                         nullptr),
            SQLITE_OK);

  // Insert keypoints for image 1.
  const FeatureKeypoints keypoints(5);
  ASSERT_EQ(sqlite3_prepare_v2(db,
                               "INSERT INTO keypoints (image_id, rows, cols, "
                               "data) VALUES (1, ?, ?, ?);",
                               -1,
                               &stmt,
                               nullptr),
            SQLITE_OK);
  sqlite3_bind_int64(stmt, 1, keypoints.size());
  sqlite3_bind_int64(stmt, 2, sizeof(FeatureKeypoint) / sizeof(float));
  sqlite3_bind_blob(stmt,
                    3,
                    keypoints.data(),
                    keypoints.size() * sizeof(FeatureKeypoint),
                    SQLITE_STATIC);
  ASSERT_EQ(sqlite3_step(stmt), SQLITE_DONE);
  sqlite3_finalize(stmt);

  ASSERT_EQ(sqlite3_close(db), SQLITE_OK);

  // Open with Database::Open, triggering migration.
  auto database = Database::Open(database_path);

  // 1. Verify cameras were migrated with correct sources:
  // cam1 had prior_focal_length=1 -> EXIF
  // cam2 had prior_focal_length=0 -> GUESS
  EXPECT_EQ(database->NumCameras(), 2);
  EXPECT_TRUE(database->ExistsCamera(1));
  EXPECT_TRUE(database->ExistsCamera(2));

  const Camera cam1_migrated = database->ReadCamera(1);
  EXPECT_EQ(cam1_migrated.source, CameraSource::EXIF);
  EXPECT_TRUE(cam1_migrated.HasPriorFocalLength());
  EXPECT_EQ(cam1_migrated.params, cam1.params);

  const Camera cam2_migrated = database->ReadCamera(2);
  EXPECT_EQ(cam2_migrated.source, CameraSource::GUESS);
  EXPECT_FALSE(cam2_migrated.HasPriorFocalLength());
  EXPECT_EQ(cam2_migrated.params, cam2.params);

  // 2. Verify images and keypoints migrated intact.
  EXPECT_EQ(database->NumImages(), 1);
  EXPECT_TRUE(database->ExistsImage(1));
  EXPECT_EQ(database->ReadImage(1).Name(), "image1.jpg");
  EXPECT_EQ(database->NumKeypoints(), 5);
  EXPECT_EQ(database->ReadKeypoints(1).size(), 5);

  // 3. Verify writing additional calibrations to migrated camera works.
  Camera cam1_vgc = cam1_migrated;
  cam1_vgc.source = CameraSource::VIEW_GRAPH;
  cam1_vgc.SetFocalLength(150.0);
  database->UpdateCamera(cam1_vgc);
  EXPECT_EQ(database->ReadCamera(1).source, CameraSource::VIEW_GRAPH);
  EXPECT_EQ(database->ReadCamera(1, CameraSource::EXIF).source,
            CameraSource::EXIF);
  EXPECT_EQ(database->ReadAllCameraCalibrations(1).size(), 2);

  // 4. Verify auto-increment for new camera works (next ID is 3).
  Camera cam3 = Camera::CreateFromModelId(
      kInvalidCameraId, CameraModelId::kSimplePinhole, 300.0, 500, 500);
  const camera_t cam3_id = database->WriteCamera(cam3, /*use_camera_id=*/false);
  EXPECT_EQ(cam3_id, 3);
  EXPECT_EQ(database->NumCameras(), 3);

  // 5. Verify foreign key cascade still works for images -> keypoints.
  database->ClearImages();
  EXPECT_EQ(database->NumImages(), 0);
  EXPECT_EQ(database->NumKeypoints(), 0);

  // 6. Verify PRAGMA foreign_key_check returns 0 violations.
  database->Close();
  ASSERT_EQ(
      sqlite3_open_v2(
          database_path.string().c_str(), &db, SQLITE_OPEN_READONLY, nullptr),
      SQLITE_OK);
  sqlite3_stmt* fk_stmt;
  ASSERT_EQ(sqlite3_prepare_v2(
                db, "PRAGMA foreign_key_check;", -1, &fk_stmt, nullptr),
            SQLITE_OK);
  EXPECT_EQ(sqlite3_step(fk_stmt), SQLITE_DONE);
  sqlite3_finalize(fk_stmt);

  // 7. Verify prior_focal_length column was removed from cameras table.
  sqlite3_stmt* info_stmt;
  ASSERT_EQ(sqlite3_prepare_v2(
                db, "PRAGMA table_info(cameras);", -1, &info_stmt, nullptr),
            SQLITE_OK);
  bool has_prior_col = false;
  while (sqlite3_step(info_stmt) == SQLITE_ROW) {
    const std::string col =
        reinterpret_cast<const char*>(sqlite3_column_text(info_stmt, 1));
    if (col == "prior_focal_length") {
      has_prior_col = true;
    }
  }
  sqlite3_finalize(info_stmt);
  EXPECT_FALSE(has_prior_col);

  sqlite3_close(db);
}

TEST(DatabaseMigrationTest, IntermediateCameraCalibrationsTableMigration) {
  const auto database_path = CreateTestDir() / "intermediate_calibs.db";
  sqlite3* db = nullptr;
  ASSERT_EQ(sqlite3_open_v2(database_path.string().c_str(),
                            &db,
                            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                            nullptr),
            SQLITE_OK);

  // Set user_version to 4.2.0.
  ASSERT_EQ(
      sqlite3_exec(
          db, "PRAGMA user_version = 4020000;", nullptr, nullptr, nullptr),
      SQLITE_OK);

  const char* kSchema =
      "CREATE TABLE cameras ("
      "  camera_id INTEGER NOT NULL,"
      "  model INTEGER NOT NULL,"
      "  width INTEGER NOT NULL,"
      "  height INTEGER NOT NULL,"
      "  params BLOB,"
      "  prior_focal_length INTEGER NOT NULL,"
      "  source INTEGER NOT NULL,"
      "  PRIMARY KEY(camera_id, source));"
      "CREATE TABLE camera_calibrations ("
      "  camera_id INTEGER NOT NULL,"
      "  model INTEGER NOT NULL,"
      "  width INTEGER NOT NULL,"
      "  height INTEGER NOT NULL,"
      "  params BLOB,"
      "  prior_focal_length INTEGER NOT NULL,"
      "  source INTEGER NOT NULL,"
      "  PRIMARY KEY(camera_id, source));"
      "CREATE TABLE images ("
      "  image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
      "  name TEXT NOT NULL UNIQUE,"
      "  camera_id INTEGER NOT NULL,"
      "  CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < "
      "2147483647));";
  ASSERT_EQ(sqlite3_exec(db, kSchema, nullptr, nullptr, nullptr), SQLITE_OK);

  Camera cam1_guess = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 1000, 1000);
  sqlite3_stmt* stmt;
  ASSERT_EQ(sqlite3_prepare_v2(
                db,
                "INSERT INTO cameras (camera_id, model, width, height, params, "
                "prior_focal_length, source) VALUES (?, ?, ?, ?, ?, ?, ?);",
                -1,
                &stmt,
                nullptr),
            SQLITE_OK);
  sqlite3_bind_int64(stmt, 1, 1);
  sqlite3_bind_int64(stmt, 2, static_cast<int>(cam1_guess.model_id));
  sqlite3_bind_int64(stmt, 3, cam1_guess.width);
  sqlite3_bind_int64(stmt, 4, cam1_guess.height);
  sqlite3_bind_blob(stmt,
                    5,
                    cam1_guess.params.data(),
                    cam1_guess.params.size() * sizeof(double),
                    SQLITE_STATIC);
  sqlite3_bind_int64(stmt, 6, 0);
  sqlite3_bind_int64(stmt, 7, static_cast<int64_t>(CameraSource::GUESS));
  ASSERT_EQ(sqlite3_step(stmt), SQLITE_DONE);
  sqlite3_finalize(stmt);

  // Insert USER calibration into camera_calibrations table.
  Camera cam1_user = cam1_guess;
  cam1_user.SetFocalLength(120.0);
  ASSERT_EQ(sqlite3_prepare_v2(
                db,
                "INSERT INTO camera_calibrations (camera_id, model, width, "
                "height, params, prior_focal_length, source) VALUES (?, ?, ?, "
                "?, ?, ?, ?);",
                -1,
                &stmt,
                nullptr),
            SQLITE_OK);
  sqlite3_bind_int64(stmt, 1, 1);
  sqlite3_bind_int64(stmt, 2, static_cast<int>(cam1_user.model_id));
  sqlite3_bind_int64(stmt, 3, cam1_user.width);
  sqlite3_bind_int64(stmt, 4, cam1_user.height);
  sqlite3_bind_blob(stmt,
                    5,
                    cam1_user.params.data(),
                    cam1_user.params.size() * sizeof(double),
                    SQLITE_STATIC);
  sqlite3_bind_int64(stmt, 6, 1);
  sqlite3_bind_int64(stmt, 7, static_cast<int64_t>(CameraSource::USER));
  ASSERT_EQ(sqlite3_step(stmt), SQLITE_DONE);
  sqlite3_finalize(stmt);

  // Insert VIEW_GRAPH calibration into camera_calibrations table.
  Camera cam1_vg = cam1_guess;
  cam1_vg.SetFocalLength(130.0);
  ASSERT_EQ(sqlite3_prepare_v2(
                db,
                "INSERT INTO camera_calibrations (camera_id, model, width, "
                "height, params, prior_focal_length, source) VALUES (?, ?, ?, "
                "?, ?, ?, ?);",
                -1,
                &stmt,
                nullptr),
            SQLITE_OK);
  sqlite3_bind_int64(stmt, 1, 1);
  sqlite3_bind_int64(stmt, 2, static_cast<int>(cam1_vg.model_id));
  sqlite3_bind_int64(stmt, 3, cam1_vg.width);
  sqlite3_bind_int64(stmt, 4, cam1_vg.height);
  sqlite3_bind_blob(stmt,
                    5,
                    cam1_vg.params.data(),
                    cam1_vg.params.size() * sizeof(double),
                    SQLITE_STATIC);
  sqlite3_bind_int64(stmt, 6, 1);
  sqlite3_bind_int64(stmt, 7, static_cast<int64_t>(CameraSource::VIEW_GRAPH));
  ASSERT_EQ(sqlite3_step(stmt), SQLITE_DONE);
  sqlite3_finalize(stmt);

  ASSERT_EQ(sqlite3_close(db), SQLITE_OK);

  // Open with Database::Open, triggering migration.
  auto database = Database::Open(database_path);

  EXPECT_EQ(database->NumCameras(), 1);
  EXPECT_EQ(database->ReadAllCameraCalibrations(1).size(), 3);
  EXPECT_EQ(database->ReadCamera(1).source, CameraSource::VIEW_GRAPH);
  EXPECT_EQ(database->ReadCamera(1, CameraSource::USER).source,
            CameraSource::USER);
  EXPECT_EQ(database->ReadCamera(1, CameraSource::GUESS).source,
            CameraSource::GUESS);

  // Verify camera_calibrations table was dropped.
  database->Close();
  ASSERT_EQ(
      sqlite3_open_v2(
          database_path.string().c_str(), &db, SQLITE_OPEN_READONLY, nullptr),
      SQLITE_OK);
  sqlite3_stmt* check_stmt;
  ASSERT_EQ(sqlite3_prepare_v2(db,
                               "SELECT name FROM sqlite_master WHERE "
                               "type='table' AND name='camera_calibrations';",
                               -1,
                               &check_stmt,
                               nullptr),
            SQLITE_OK);
  EXPECT_EQ(sqlite3_step(check_stmt), SQLITE_DONE);
  sqlite3_finalize(check_stmt);
  sqlite3_close(db);
}

TEST(Database, CameraSourceBestRejected) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Camera camera = Camera::CreateFromModelId(
      1, SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  camera.source = CameraSource::BEST;
  EXPECT_THROW(database->WriteCamera(camera), std::invalid_argument);

  camera.source = CameraSource::USER;
  database->WriteCamera(camera);

  camera.source = CameraSource::BEST;
  EXPECT_THROW(database->UpdateCamera(camera), std::invalid_argument);
}

}  // namespace
}  // namespace colmap
