// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/rig.h"

#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random_eigen.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <cstdint>
#include <fstream>
#include <locale>
#include <map>
#include <memory>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sqlite3.h>

namespace colmap {
namespace {

std::filesystem::path WriteTestConfig(const std::string& config) {
  const auto file_path = CreateTestDir() / "config.json";
  std::ofstream file(file_path);
  file << config << '\n';
  return file_path;
}

// Custom numpunct facet that uses comma as decimal separator, for testing
// locale independence without requiring a specific system locale.
struct CommaDecimalFacet : std::numpunct<char> {
 protected:
  char do_decimal_point() const override { return ','; }
};

TEST(ReadRigConfig, Empty) {
  EXPECT_THAT(ReadRigConfig(WriteTestConfig("[]")), testing::IsEmpty());
}

TEST(ReadRigConfig, InvalidJson) {
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig("")));
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig("[{")));
}

TEST(ReadRigConfig, MissingImagePrefix) {
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig(R"(
          [
            {
              "cameras": [
                {
                    "ref_sensor": true,
                }
              ]
            }
          ]
          )")));
}

TEST(ReadRigConfig, InvalidRefSensor) {
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig(R"(
          [
            {
              "cameras": [
                {
                    "image_prefix": "rig1/camera1/",
                    "ref_sensor": true,
                },
                {
                    "image_prefix": "rig1/camera2/",
                    "ref_sensor": true,
                }
              ]
            }
          ]
          )")));
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig(R"(
              [
                {
                  "cameras": [
                    {
                        "image_prefix": "rig1/camera1/",
                    },
                    {
                        "image_prefix": "rig1/camera2/",
                    }
                  ]
                }
              ]
              )")));
}

TEST(ReadRigConfig, Nominal) {
  const std::vector<RigConfig> configs = ReadRigConfig(WriteTestConfig(R"(
[
  {
    "cameras": [
      {
          "image_prefix": "rig1/camera1/",
          "ref_sensor": true,
          "camera_model_name": "OPENCV",
          "camera_params": [640, 480, 320, 240, 0.1, 0.2, 0.3, 0.4]
      },
      {
          "image_prefix": "rig1/camera2/",
          "cam_from_rig_rotation": [0, 1, 0, 0],
          "cam_from_rig_translation": [1, 2, 3]
      }
    ]
  },
  {
    "cameras": [
      {
          "image_prefix": "rig2/camera1/",
          "ref_sensor": true
      },
      {
          "image_prefix": "rig2/camera2/"
      }
    ]
  }
]
)"));
  ASSERT_EQ(configs.size(), 2);
  ASSERT_EQ(configs[0].cameras.size(), 2);
  ASSERT_EQ(configs[1].cameras.size(), 2);

  EXPECT_EQ(configs[0].cameras[0].image_prefix, "rig1/camera1/");
  EXPECT_TRUE(configs[0].cameras[0].ref_sensor);
  ASSERT_FALSE(configs[0].cameras[0].cam_from_rig.has_value());
  ASSERT_TRUE(configs[0].cameras[0].camera.has_value());
  EXPECT_EQ(configs[0].cameras[0].camera->model_id, CameraModelId::kOpenCV);
  EXPECT_TRUE(configs[0].cameras[0].camera->HasPriorFocalLength());
  EXPECT_EQ(configs[0].cameras[0].camera->source, CameraSource::USER);
  EXPECT_THAT(configs[0].cameras[0].camera->params,
              testing::ElementsAre(640, 480, 320, 240, 0.1, 0.2, 0.3, 0.4));

  EXPECT_EQ(configs[0].cameras[1].image_prefix, "rig1/camera2/");
  EXPECT_FALSE(configs[0].cameras[1].ref_sensor);
  ASSERT_TRUE(configs[0].cameras[1].cam_from_rig.has_value());
  EXPECT_EQ(configs[0].cameras[1].cam_from_rig->rotation().coeffs(),
            Eigen::Vector4d(1, 0, 0, 0));
  EXPECT_EQ(configs[0].cameras[1].cam_from_rig->translation(),
            Eigen::Vector3d(1, 2, 3));
  ASSERT_FALSE(configs[0].cameras[1].camera.has_value());

  EXPECT_EQ(configs[1].cameras[0].image_prefix, "rig2/camera1/");
  EXPECT_TRUE(configs[1].cameras[0].ref_sensor);
  ASSERT_FALSE(configs[1].cameras[0].cam_from_rig.has_value());
  ASSERT_FALSE(configs[1].cameras[0].camera.has_value());

  EXPECT_EQ(configs[1].cameras[1].image_prefix, "rig2/camera2/");
  EXPECT_FALSE(configs[1].cameras[1].ref_sensor);
  ASSERT_FALSE(configs[1].cameras[1].cam_from_rig.has_value());
  ASSERT_FALSE(configs[1].cameras[1].camera.has_value());
}

TEST(ReadRigConfig, LocaleIndependence) {
  // Install a global locale that uses comma as decimal separator.
  const std::locale original_locale = std::locale::global(
      std::locale(std::locale::classic(), new CommaDecimalFacet));

  const std::vector<RigConfig> configs = ReadRigConfig(WriteTestConfig(R"(
[
  {
    "cameras": [
      {
          "image_prefix": "rig1/camera1/",
          "ref_sensor": true,
          "camera_model_name": "OPENCV",
          "camera_params": [640.5, 480.25, 320.125, 240.0625, 0.1, 0.2, 0.3, 0.4]
      },
      {
          "image_prefix": "rig1/camera2/",
          "cam_from_rig_rotation": [0.5, 0.5, 0.5, 0.5],
          "cam_from_rig_translation": [1.5, 2.25, 3.125]
      }
    ]
  }
]
)"));
  ASSERT_EQ(configs.size(), 1);
  ASSERT_EQ(configs[0].cameras.size(), 2);

  ASSERT_TRUE(configs[0].cameras[0].camera.has_value());
  EXPECT_THAT(configs[0].cameras[0].camera->params,
              testing::ElementsAre(
                  640.5, 480.25, 320.125, 240.0625, 0.1, 0.2, 0.3, 0.4));

  ASSERT_TRUE(configs[0].cameras[1].cam_from_rig.has_value());
  EXPECT_EQ(configs[0].cameras[1].cam_from_rig->rotation().coeffs(),
            Eigen::Vector4d(0.5, 0.5, 0.5, 0.5));
  EXPECT_EQ(configs[0].cameras[1].cam_from_rig->translation(),
            Eigen::Vector3d(1.5, 2.25, 3.125));

  // Restore original locale.
  std::locale::global(original_locale);
}

void CreateTestData(int num_frames,
                    int num_cameras_per_rig,
                    Database* database,
                    Reconstruction* reconstruction) {
  // Create a synthetic dataset with one trivial rig per camera, so that the
  // images of each camera in the rig config have their own camera.
  SyntheticDatasetOptions options;
  options.num_rigs = num_cameras_per_rig;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = num_frames;
  SynthesizeDataset(options, reconstruction, database);

  // Manually overwrite the image names to match the expected rig config.
  std::map<camera_t, int> camera_id_to_idx;
  std::map<camera_t, int> camera_id_to_num_images;
  for (auto& image : database->ReadAllImages()) {
    const int camera_idx =
        camera_id_to_idx
            .emplace(image.CameraId(),
                     static_cast<int>(camera_id_to_idx.size()))
            .first->second;
    const int frame_idx = ++camera_id_to_num_images[image.CameraId()];
    image.SetName("camera" + std::to_string(camera_idx) + "_frame" +
                  std::to_string(frame_idx));
    reconstruction->Image(image.ImageId()).SetName(image.Name());
    database->UpdateImage(image);
  }
}

void CreateRemappedPartialReconstruction(const Reconstruction& source,
                                         const camera_t id_offset,
                                         const std::string& omitted_image_name,
                                         Reconstruction* destination) {
  for (const auto& [_, camera] : source.Cameras()) {
    Camera remapped_camera = camera;
    remapped_camera.camera_id += id_offset;
    destination->AddCameraWithTrivialRig(std::move(remapped_camera));
  }
  for (const auto& [_, image] : source.Images()) {
    if (image.Name() == omitted_image_name) {
      continue;
    }
    const Rigid3d cam_from_world = image.CamFromWorld();
    Image remapped_image = image;
    remapped_image.ResetCameraPtr();
    remapped_image.ResetFramePtr();
    remapped_image.SetImageId(image.ImageId() + id_offset);
    remapped_image.SetCameraId(image.CameraId() + id_offset);
    remapped_image.SetFrameId(remapped_image.ImageId());
    destination->AddImageWithTrivialFrame(std::move(remapped_image),
                                          cam_from_world);
  }
}

void ExpectIntrinsicsEqual(const Camera& actual, const Camera& expected) {
  EXPECT_EQ(actual.model_id, expected.model_id);
  EXPECT_EQ(actual.params, expected.params);
  EXPECT_EQ(actual.source, expected.source);
}

std::uintptr_t Address(const void* value) {
  return reinterpret_cast<std::uintptr_t>(value);
}

struct RigConfigDatabaseState {
  std::vector<Rig> rigs;
  std::vector<Frame> frames;
  std::vector<Image> images;
  std::map<camera_t, std::map<CameraSource, Camera>> cameras;
};

RigConfigDatabaseState CaptureDatabaseState(const Database& database) {
  RigConfigDatabaseState state{database.ReadAllRigs(),
                               database.ReadAllFrames(),
                               database.ReadAllImages(),
                               {}};
  for (const auto& [id, sources] : database.ReadAllCameraSources()) {
    state.cameras.emplace(id, sources);
  }
  return state;
}

void ExpectDatabaseState(const Database& database,
                         const RigConfigDatabaseState& state) {
  EXPECT_EQ(database.ReadAllRigs(), state.rigs);
  EXPECT_EQ(database.ReadAllFrames(), state.frames);
  EXPECT_EQ(database.ReadAllImages(), state.images);
  std::map<camera_t, std::map<CameraSource, Camera>> cameras;
  for (const auto& [id, sources] : database.ReadAllCameraSources()) {
    cameras.emplace(id, sources);
  }
  EXPECT_EQ(cameras, state.cameras);
}

struct RigConfigLiveState {
  Reconstruction reconstruction;
  std::map<rig_t, std::uintptr_t> rigs;
  std::map<frame_t, std::tuple<std::uintptr_t, std::uintptr_t, bool>> frames;
  std::map<camera_t, std::uintptr_t> cameras;
  std::map<image_t, std::tuple<std::uintptr_t, std::uintptr_t, std::uintptr_t>>
      images;
};

RigConfigLiveState CaptureLiveState(const Reconstruction& reconstruction) {
  RigConfigLiveState state;
  state.reconstruction = reconstruction;
  for (const auto& [id, rig] : reconstruction.Rigs()) {
    state.rigs.emplace(id, Address(&rig));
  }
  for (const auto& [id, frame] : reconstruction.Frames()) {
    state.frames.emplace(
        id,
        std::make_tuple(Address(&frame),
                        frame.HasRigPtr() ? Address(frame.RigPtr()) : 0,
                        frame.HasFinalDataIds()));
  }
  for (const auto& [id, camera] : reconstruction.Cameras()) {
    state.cameras.emplace(id, Address(&camera));
  }
  for (const auto& [id, image] : reconstruction.Images()) {
    state.images.emplace(
        id,
        std::make_tuple(Address(&image),
                        image.HasCameraPtr() ? Address(image.CameraPtr()) : 0,
                        image.HasFramePtr() ? Address(image.FramePtr()) : 0));
  }
  return state;
}

void ExpectLiveState(const Reconstruction& reconstruction,
                     const RigConfigLiveState& before) {
  ASSERT_EQ(reconstruction.NumRigs(), before.rigs.size());
  ASSERT_EQ(reconstruction.NumFrames(), before.frames.size());
  ASSERT_EQ(reconstruction.NumCameras(), before.cameras.size());
  ASSERT_EQ(reconstruction.NumImages(), before.images.size());
  // Validate ownership before the equality matcher follows image/frame links.
  // The old implementation can leave those links dangling after rejection.
  for (const auto& [id, expected] : before.frames) {
    ASSERT_TRUE(reconstruction.ExistsFrame(id));
    const Frame& frame = reconstruction.Frame(id);
    ASSERT_TRUE(reconstruction.ExistsRig(frame.RigId()));
    ASSERT_TRUE(frame.HasRigPtr());
    ASSERT_EQ(frame.RigPtr(), &reconstruction.Rig(frame.RigId()));
  }
  for (const auto& [id, expected] : before.images) {
    ASSERT_TRUE(reconstruction.ExistsImage(id));
    const Image& image = reconstruction.Image(id);
    ASSERT_TRUE(reconstruction.ExistsCamera(image.CameraId()));
    ASSERT_TRUE(reconstruction.ExistsFrame(image.FrameId()));
    ASSERT_TRUE(image.HasCameraPtr());
    ASSERT_TRUE(image.HasFramePtr());
    ASSERT_EQ(image.CameraPtr(), &reconstruction.Camera(image.CameraId()));
    ASSERT_EQ(image.FramePtr(), &reconstruction.Frame(image.FrameId()));
  }
  const RigConfigLiveState after = CaptureLiveState(reconstruction);
  EXPECT_EQ(after.rigs, before.rigs);
  EXPECT_EQ(after.frames, before.frames);
  EXPECT_EQ(after.cameras, before.cameras);
  EXPECT_EQ(after.images, before.images);
  EXPECT_EQ(reconstruction.RegFrameIds(), before.reconstruction.RegFrameIds());
  EXPECT_EQ(reconstruction.NumRegImages(),
            before.reconstruction.NumRegImages());
  EXPECT_THAT(reconstruction, ReconstructionEq(before.reconstruction));
}

TEST(ApplyRigConfig, WithReconstruction) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  CreateTestData(
      /*num_frames=*/5,
      /*num_cameras_per_rig=*/2,
      database.get(),
      &reconstruction);

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera0_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera1_";

  ApplyRigConfig(configs, *database, &reconstruction);
  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_EQ(database->NumFrames(), 5);
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), 5);
  EXPECT_EQ(reconstruction.NumRegFrames(), 5);
}

TEST(ApplyRigConfig, WithDifferingDatabaseAndReconstructionIds) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  CreateTestData(
      /*num_frames=*/5,
      /*num_cameras_per_rig=*/2,
      database.get(),
      &reconstruction);

  // Create a reconstruction with differing rig/camera/frame/image ids.
  Reconstruction differing_reconstruction;
  for (const auto& [_, camera] : reconstruction.Cameras()) {
    auto differing_camera = camera;
    differing_camera.camera_id = camera.camera_id + 1;
    differing_reconstruction.AddCamera(differing_camera);
  }
  for (const auto& [_, rig] : reconstruction.Rigs()) {
    Rig differing_rig;
    differing_rig.SetRigId(rig.RigId() + 1);
    differing_rig.AddRefSensor(
        sensor_t(rig.RefSensorId().type, rig.RefSensorId().id + 1));
    for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
      differing_rig.AddSensor(sensor_t(sensor_id.type, sensor_id.id + 1),
                              sensor_from_rig);
    }
    differing_reconstruction.AddRig(differing_rig);
  }
  for (const auto& [_, frame] : reconstruction.Frames()) {
    auto differing_frame = frame;
    differing_frame.ResetRigPtr();
    differing_frame.SetFrameId(frame.FrameId() + 1);
    differing_frame.SetRigId(frame.RigId() + 1);
    differing_frame.ClearDataIds();
    for (auto& data_id : frame.DataIds()) {
      differing_frame.AddDataId(
          data_t(sensor_t(data_id.sensor_id.type, data_id.sensor_id.id + 1),
                 data_id.id + 1));
    }
    differing_reconstruction.AddFrame(differing_frame);
  }
  for (const auto& [_, image] : reconstruction.Images()) {
    auto differing_image = image;
    differing_image.ResetCameraPtr();
    differing_image.ResetFramePtr();
    differing_image.SetImageId(image.ImageId() + 1);
    differing_image.SetCameraId(image.CameraId() + 1);
    differing_image.SetFrameId(image.FrameId() + 1);
    differing_reconstruction.AddImage(differing_image);
  }

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera0_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera1_";

  ApplyRigConfig(configs, *database, &differing_reconstruction);
  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_EQ(database->NumFrames(), 5);
  EXPECT_EQ(differing_reconstruction.NumRigs(), 1);
  EXPECT_EQ(differing_reconstruction.NumFrames(), 5);
  EXPECT_EQ(differing_reconstruction.NumRegFrames(), 5);
}

TEST(ApplyRigConfig, ExplicitIntrinsicsWithMatchingCameraIds) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  CreateTestData(5, 2, database.get(), &reconstruction);
  const Image* first_image = reconstruction.FindImageWithName("camera0_frame1");
  const Image* other_image = reconstruction.FindImageWithName("camera1_frame1");
  ASSERT_NE(first_image, nullptr);
  ASSERT_NE(other_image, nullptr);
  const camera_t camera_id = first_image->CameraId();
  const Camera* camera_ptr = &reconstruction.Camera(camera_id);
  const Camera before_other = reconstruction.Camera(other_image->CameraId());
  const Camera specified = Camera::CreateFromModelId(
      camera_id, CameraModelId::kOpenCV, 1200.0, 1024, 768);

  RigConfig config;
  config.cameras.emplace_back().image_prefix = "camera0_";
  config.cameras[0].ref_sensor = true;
  config.cameras[0].camera = specified;
  config.cameras.emplace_back().image_prefix = "camera1_";

  ApplyRigConfig({config}, *database, &reconstruction);
  EXPECT_EQ(&reconstruction.Camera(camera_id), camera_ptr);
  ExpectIntrinsicsEqual(database->ReadCamera(camera_id), specified);
  ExpectIntrinsicsEqual(reconstruction.Camera(camera_id), specified);
  ExpectIntrinsicsEqual(reconstruction.Camera(before_other.camera_id),
                        before_other);
  // All five database-matched images of this sensor share one camera.
  for (int frame_idx = 1; frame_idx <= 5; ++frame_idx) {
    const Image* image = reconstruction.FindImageWithName(
        "camera0_frame" + std::to_string(frame_idx));
    ASSERT_NE(image, nullptr);
    EXPECT_EQ(image->CameraId(), camera_id);
  }
}

TEST(ApplyRigConfig, ExplicitIntrinsicsWithCollisionAndPartialModel) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction source;
  CreateTestData(5, 2, database.get(), &source);
  const Image* first_database_image =
      source.FindImageWithName("camera0_frame1");
  const Image* second_sensor_image = source.FindImageWithName("camera1_frame1");
  ASSERT_NE(first_database_image, nullptr);
  ASSERT_NE(second_sensor_image, nullptr);
  const camera_t database_camera_id = first_database_image->CameraId();
  const camera_t second_database_camera_id = second_sensor_image->CameraId();

  Reconstruction partial;
  constexpr camera_t kIdOffset = 10;
  CreateRemappedPartialReconstruction(
      source, kIdOffset, "camera0_frame1", &partial);
  ASSERT_EQ(partial.FindImageWithName("camera0_frame1"), nullptr);
  ASSERT_NE(partial.FindImageWithName("camera0_frame2"), nullptr);
  Camera unrelated = source.Camera(database_camera_id);
  partial.AddCamera(unrelated);  // Its ID collides with the database ID.
  const Camera unrelated_before = partial.Camera(database_camera_id);
  const camera_t matched_camera_id = database_camera_id + kIdOffset;
  const Camera* matched_camera_ptr = &partial.Camera(matched_camera_id);
  const Camera specified = Camera::CreateFromModelId(
      database_camera_id, CameraModelId::kOpenCV, 1200.0, 1024, 768);
  const Camera second_specified = Camera::CreateFromModelId(
      second_database_camera_id, CameraModelId::kOpenCV, 900.0, 1024, 768);

  RigConfig config;
  config.cameras.emplace_back().image_prefix = "camera0_";
  config.cameras[0].ref_sensor = true;
  config.cameras[0].camera = specified;
  config.cameras.emplace_back().image_prefix = "camera1_";
  config.cameras[1].camera = second_specified;

  ApplyRigConfig({config}, *database, &partial);
  EXPECT_EQ(&partial.Camera(matched_camera_id), matched_camera_ptr);
  ExpectIntrinsicsEqual(database->ReadCamera(database_camera_id), specified);
  ExpectIntrinsicsEqual(partial.Camera(matched_camera_id), specified);
  ExpectIntrinsicsEqual(database->ReadCamera(second_database_camera_id),
                        second_specified);
  ExpectIntrinsicsEqual(partial.Camera(second_database_camera_id + kIdOffset),
                        second_specified);
  ExpectIntrinsicsEqual(partial.Camera(database_camera_id), unrelated_before);
  EXPECT_EQ(partial.NumImages(), 9);
}

TEST(ApplyRigConfig, ConflictingReconstructionCameraRemainsUnsupported) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction source;
  CreateTestData(2, 2, database.get(), &source);
  const Image* first_image = source.FindImageWithName("camera0_frame1");
  const Image* second_image = source.FindImageWithName("camera1_frame1");
  ASSERT_NE(first_image, nullptr);
  ASSERT_NE(second_image, nullptr);

  // Distinct database sensors are mapped to one reconstruction camera.
  const camera_t shared_camera_id = first_image->CameraId() + 100;
  Reconstruction shared;
  Camera shared_camera = source.Camera(first_image->CameraId());
  shared_camera.camera_id = shared_camera_id;
  shared.AddCameraWithTrivialRig(std::move(shared_camera));
  for (const auto& [_, source_image] : source.Images()) {
    const Rigid3d cam_from_world = source_image.CamFromWorld();
    Image image = source_image;
    image.ResetCameraPtr();
    image.ResetFramePtr();
    image.SetImageId(source_image.ImageId() + 100);
    image.SetCameraId(shared_camera_id);
    image.SetFrameId(image.ImageId());
    shared.AddImageWithTrivialFrame(std::move(image), cam_from_world);
  }

  RigConfig config;
  config.cameras.emplace_back().image_prefix = "camera0_";
  config.cameras[0].ref_sensor = true;
  config.cameras[0].camera = Camera::CreateFromModelId(
      first_image->CameraId(), CameraModelId::kOpenCV, 1200.0, 1024, 768);
  config.cameras.emplace_back().image_prefix = "camera1_";
  config.cameras[1].camera = Camera::CreateFromModelId(
      second_image->CameraId(), CameraModelId::kOpenCV, 900.0, 1024, 768);

  // Rig rebuild already refuses a sensor shared by distinct config sensors.
  // This does not test or imply rollback of earlier database writes.
  try {
    ApplyRigConfig({config}, *database, &shared);
    ADD_FAILURE() << "Expected a shared-sensor rig rebuild failure";
  } catch (const std::exception& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr("inserted twice into the rig"));
  }
}

TEST(ApplyRigConfig, WithPartialReconstruction) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  CreateTestData(
      /*num_frames=*/5,
      /*num_cameras_per_rig=*/2,
      database.get(),
      &reconstruction);

  // Trivial frames 1-5 hold the camera0_ images and 6-10 the camera1_ images.
  reconstruction.DeRegisterFrame(1);
  reconstruction.DeRegisterFrame(6);
  // De-register only one image in the frame.
  // The other image allows us to register the frame.
  reconstruction.DeRegisterFrame(7);

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera0_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera1_";

  ApplyRigConfig(configs, *database, &reconstruction);
  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_EQ(database->NumFrames(), 5);
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), 5);
  EXPECT_EQ(reconstruction.NumRegFrames(), 4);
}

TEST(ApplyRigConfig, WithoutReconstruction) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  CreateTestData(
      /*num_frames=*/5,
      /*num_cameras_per_rig=*/2,
      database.get(),
      &reconstruction);

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera0_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera1_";
  camera2.camera =
      Camera::CreateFromModelId(2, CameraModelId::kOpenCV, 2.0, 1024, 768);
  camera2.cam_from_rig =
      Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>());

  ApplyRigConfig(configs, *database);
  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_EQ(database->NumFrames(), 5);
  const auto [sensor_id2, sensor2_from_rig] =
      *database->ReadAllRigs().at(0).NonRefSensors().begin();
  EXPECT_EQ(sensor2_from_rig.value().rotation().coeffs(),
            camera2.cam_from_rig.value().rotation().coeffs());
  EXPECT_EQ(sensor2_from_rig.value().translation(),
            camera2.cam_from_rig.value().translation());
  EXPECT_EQ(database->ReadCamera(sensor_id2.id), camera2.camera);
}

TEST(ApplyRigConfig, RejectsCameraSharedBetweenSensors) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 4;
  SynthesizeDataset(options, &reconstruction, database.get());

  // Split the images of the single camera between two rig sensors, as happens
  // when images of identical cameras were extracted into a shared camera.
  int image_idx = 0;
  for (auto& image : database->ReadAllImages()) {
    image.SetName("camera" + std::to_string(image_idx % 2) + "_frame" +
                  std::to_string(image_idx / 2));
    database->UpdateImage(image);
    ++image_idx;
  }

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera0_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera1_";

  try {
    ApplyRigConfig(configs, *database);
    ADD_FAILURE() << "Expected ApplyRigConfig to throw";
  } catch (const std::exception& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr("share camera"));
  }
}

TEST(ApplyRigConfig, WithUnconfiguredSingleAndConfiguredMultiCameraRigs) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 3;
  options.num_frames_per_rig = 5;
  SynthesizeDataset(options, &reconstruction, database.get());

  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 3;
  SynthesizeDataset(options, &reconstruction, database.get());

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera000001_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera000002_";

  ApplyRigConfig(configs, *database, &reconstruction);
  EXPECT_EQ(database->NumRigs(), 3);
  EXPECT_EQ(database->NumFrames(), 13);
  EXPECT_EQ(reconstruction.NumRigs(), 3);
  EXPECT_EQ(reconstruction.NumFrames(), 13);
  EXPECT_EQ(reconstruction.NumRegFrames(), 13);

  int num_non_trivial_rigs = 0;
  for (const auto& rig : database->ReadAllRigs()) {
    num_non_trivial_rigs += rig.NumSensors() > 1;
  }
  EXPECT_EQ(num_non_trivial_rigs, 1);

  int num_non_trivial_frames = 0;
  for (const auto& rig : database->ReadAllFrames()) {
    num_non_trivial_frames += rig.NumDataIds() > 1;
  }
  EXPECT_EQ(num_non_trivial_frames, 5);
}

TEST(ApplyRigConfig, PrefersRefSensorForFramePoses) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 2;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 5;
  SynthesizeDataset(options, &reconstruction, database.get());

  // Record reference-sensor image poses before conversion. The two images
  // grouped into each new frame have independent (inconsistent) poses, since
  // they come from different trivial rigs.
  NodeHashMap<std::string, Rigid3d> ref_name_to_cam_from_world;
  for (const auto& [_, image] : reconstruction.Images()) {
    if (image.CameraId() == 1) {
      ref_name_to_cam_from_world.emplace(image.Name(), image.CamFromWorld());
    }
  }
  EXPECT_EQ(ref_name_to_cam_from_world.size(), 5);

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera000001_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera000002_";

  ApplyRigConfig(configs, *database, &reconstruction);
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), 5);

  // Converted frame poses must match the reference-sensor images, not an
  // arbitrary (e.g. last-visited) image in the frame.
  for (const auto& [_, image] : reconstruction.Images()) {
    if (image.CameraId() != 1) {
      continue;
    }
    const Rigid3d& expected = ref_name_to_cam_from_world.at(image.Name());
    EXPECT_THAT(image.CamFromWorld(),
                Rigid3dNear(expected, /*rtol=*/1e-6, /*ttol=*/1e-6));
  }
}

TEST(ApplyRigConfig, LateInvalidConfigRestoresDatabaseAndLiveOwners) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  CreateTestData(2, 2, database.get(), &reconstruction);
  const auto* first = reconstruction.FindImageWithName("camera0_frame1");
  ASSERT_NE(first, nullptr);
  const auto database_before = CaptureDatabaseState(*database);
  const auto live_before = CaptureLiveState(reconstruction);

  RigConfig first_config;
  first_config.cameras.emplace_back().image_prefix = "camera0_";
  first_config.cameras[0].ref_sensor = true;
  first_config.cameras[0].camera = Camera::CreateFromModelId(
      first->CameraId(), CameraModelId::kOpenCV, 1200.0, 1024, 768);
  first_config.cameras.emplace_back().image_prefix = "camera1_";
  RigConfig invalid_later;
  invalid_later.cameras.emplace_back().image_prefix = "missing_";
  invalid_later.cameras[0].ref_sensor = true;

  try {
    ApplyRigConfig({first_config, invalid_later}, *database, &reconstruction);
    ADD_FAILURE() << "Expected late invalid sensor rejection";
  } catch (const std::exception& error) {
    EXPECT_THAT(error.what(), testing::HasSubstr("At least one image"));
  }
  ExpectDatabaseState(*database, database_before);
  ExpectLiveState(reconstruction, live_before);
}

TEST(ApplyRigConfig, WriteAndReleaseFailuresPreserveDatabaseAndLiveOwners) {
  for (const bool fail_on_release : {false, true}) {
    SCOPED_TRACE(fail_on_release);
    const auto database_path =
        CreateTestDir() /
        (fail_on_release ? "rig_release_failure.db" : "rig_insert_failure.db");
    auto database = Database::Open(database_path);
    Reconstruction reconstruction;
    CreateTestData(2, 2, database.get(), &reconstruction);
    const auto database_before = CaptureDatabaseState(*database);
    const auto live_before = CaptureLiveState(reconstruction);

    // Use a real SQL write failure or a deferred foreign-key failure. The
    // latter permits every frame write and all reconstruction preparation, but
    // rejects RELEASE of the top-level savepoint before live state is
    // published.
    sqlite3* raw_database = nullptr;
    const int open_result =
        sqlite3_open(database_path.string().c_str(), &raw_database);
    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> raw(raw_database,
                                                           &sqlite3_close);
    ASSERT_EQ(open_result, SQLITE_OK);
    const char* failure_sql =
        fail_on_release
            ? "CREATE TABLE rollback_parent (id INTEGER PRIMARY KEY); "
              "CREATE TABLE rollback_child (parent_id INTEGER REFERENCES "
              "rollback_parent(id) DEFERRABLE INITIALLY DEFERRED); "
              "CREATE TRIGGER fail_rig_frame AFTER INSERT ON frames "
              "BEGIN INSERT INTO rollback_child VALUES (1); END;"
            : "CREATE TRIGGER fail_rig_frame BEFORE INSERT ON frames "
              "WHEN (SELECT COUNT(*) FROM rigs) = 1 "
              "BEGIN SELECT RAISE(ABORT, 'forced frame insert failure'); END;";
    ASSERT_EQ(sqlite3_exec(raw.get(), failure_sql, nullptr, nullptr, nullptr),
              SQLITE_OK);
    raw.reset();

    RigConfig config;
    config.cameras.emplace_back().image_prefix = "camera0_";
    config.cameras[0].ref_sensor = true;
    config.cameras.emplace_back().image_prefix = "camera1_";
    try {
      ApplyRigConfig({config}, *database, &reconstruction);
      ADD_FAILURE() << "Expected SQLite write or release failure";
    } catch (const std::exception& error) {
      EXPECT_THAT(error.what(), testing::HasSubstr("SQLite error"));
      if (fail_on_release) {
        EXPECT_THAT(error.what(),
                    testing::HasSubstr("FOREIGN KEY constraint failed"));
      }
    }
    ExpectDatabaseState(*database, database_before);
    ExpectLiveState(reconstruction, live_before);

    // Remove the external failure and reuse the same owners. This also checks
    // that the failed statement and savepoint left the connection usable.
    raw_database = nullptr;
    const int reopen_result =
        sqlite3_open(database_path.string().c_str(), &raw_database);
    raw.reset(raw_database);
    ASSERT_EQ(reopen_result, SQLITE_OK);
    ASSERT_EQ(sqlite3_exec(raw.get(),
                           "DROP TRIGGER fail_rig_frame",
                           nullptr,
                           nullptr,
                           nullptr),
              SQLITE_OK);
    raw.reset();

    ASSERT_NO_THROW(ApplyRigConfig({config}, *database, &reconstruction));
    EXPECT_EQ(database->NumRigs(), 1);
    EXPECT_EQ(database->NumFrames(), 2);
    EXPECT_EQ(reconstruction.NumRigs(), 1);
    EXPECT_EQ(reconstruction.NumFrames(), 2);
    EXPECT_EQ(reconstruction.NumRegFrames(), 2);
    for (const auto& [_, frame] : reconstruction.Frames()) {
      ASSERT_TRUE(reconstruction.ExistsRig(frame.RigId()));
      ASSERT_TRUE(frame.HasRigPtr());
      EXPECT_EQ(frame.RigPtr(), &reconstruction.Rig(frame.RigId()));
    }
    for (const auto& [_, image] : reconstruction.Images()) {
      ASSERT_TRUE(reconstruction.ExistsFrame(image.FrameId()));
      ASSERT_TRUE(image.HasFramePtr());
      ASSERT_EQ(image.FramePtr(), &reconstruction.Frame(image.FrameId()));
      EXPECT_TRUE(image.FramePtr()->HasDataId(image.DataId()));
    }
  }
}

TEST(ApplyRigConfig, OuterRollbackAfterSuccessfulNestedCallIsCallerOwned) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  CreateTestData(2, 2, database.get(), &reconstruction);
  const auto database_before = CaptureDatabaseState(*database);
  RigConfig config;
  config.cameras.emplace_back().image_prefix = "camera0_";
  config.cameras[0].ref_sensor = true;
  config.cameras.emplace_back().image_prefix = "camera1_";

  auto outer = database->CreateSavepoint();
  ASSERT_NO_THROW(ApplyRigConfig({config}, *database, &reconstruction));
  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  outer->Rollback();
  ExpectDatabaseState(*database, database_before);
  // The call succeeded before the caller rolled back its outer scope. The
  // reconstruction intentionally remains published; callers own this later
  // database/reconstruction lifecycle boundary.
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), 2);
  EXPECT_EQ(reconstruction.NumRegFrames(), 2);
}

}  // namespace
}  // namespace colmap
