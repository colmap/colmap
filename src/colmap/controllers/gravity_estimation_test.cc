// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/gravity_estimation.h"

#include "colmap/scene/database.h"
#include "colmap/util/eigen_matchers.h"
#include "colmap/util/testing.h"

#include <filesystem>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

void WriteSolidImage(const std::filesystem::path& path, int width, int height) {
  Bitmap bitmap(width, height, /*as_rgb=*/true);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      bitmap.SetPixel(c, r, BitmapColor<uint8_t>(128, 128, 128));
    }
  }
  ASSERT_TRUE(bitmap.Write(path));
}

class FakeGeoCalib : public GeoCalib {
 public:
  FakeGeoCalib(const Camera& gt_camera,
               const std::vector<Eigen::Vector3d>& gt_gravities)
      : gt_camera_(gt_camera), gt_gravities_(gt_gravities) {}

  PerspectiveField PredictPerspectiveField(
      const Bitmap& /*bitmap*/) const override {
    const Eigen::Vector3d g =
        gt_gravities_[std::min(call_idx_++, gt_gravities_.size() - 1)];
    return ComputePerspectiveFieldFromCameraAndGravity(gt_camera_, g, 32, 24);
  }

  FittedPerspectiveFields Calibrate(const Bitmap& bitmap,
                                    Camera* camera,
                                    const bool refine_camera) const override {
    const PerspectiveField field = PredictPerspectiveField(bitmap);
    return FitPerspectiveField(
        PerspectiveFieldFittingOptions(), field, camera, refine_camera);
  }

 private:
  Camera gt_camera_;
  std::vector<Eigen::Vector3d> gt_gravities_;
  mutable size_t call_idx_ = 0;
};

TEST(GravityEstimationControllerTest,
     SingleCameraEstimatesGravityAndMedianIntrinsics) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  WriteSolidImage(test_dir / "image0.png", 64, 48);
  WriteSolidImage(test_dir / "image1.png", 64, 48);

  const Camera gt_camera =
      Camera::CreateFromModelName(1, "SIMPLE_PINHOLE", 72.0, 64, 48);
  const Eigen::Vector3d gt_g0 = Eigen::Vector3d(0.2, 0.88, -0.42).normalized();
  const Eigen::Vector3d gt_g1 = Eigen::Vector3d(-0.15, 0.90, 0.41).normalized();

  {
    auto database = Database::Open(database_path);
    Camera init_cam =
        Camera::CreateFromModelName(1, "SIMPLE_PINHOLE", 45.0, 64, 48);
    init_cam.has_prior_focal_length = false;
    const camera_t camera_id = database->WriteCamera(init_cam);

    Image img0;
    img0.SetName("image0.png");
    img0.SetCameraId(camera_id);
    database->WriteImage(img0);

    Image img1;
    img1.SetName("image1.png");
    img1.SetCameraId(camera_id);
    database->WriteImage(img1);
  }

  GravityEstimationOptions options;
  options.refine_intrinsics = true;
  options.geocalib.fitting.stride = 2;

  auto controller = CreateGravityEstimationController(
      database_path,
      test_dir,
      options,
      /*image_names=*/{},
      [&](const GeoCalibOptions&) {
        return std::make_unique<FakeGeoCalib>(
            gt_camera, std::vector<Eigen::Vector3d>{gt_g0, gt_g1});
      });
  controller->Start();
  controller->Wait();

  auto database = Database::Open(database_path);
  const std::vector<PosePrior> priors = database->ReadAllPosePriors();
  ASSERT_EQ(priors.size(), 2);
  EXPECT_THAT(priors[0].gravity, EigenMatrixNear(gt_g0, 1e-3));
  EXPECT_THAT(priors[1].gravity, EigenMatrixNear(gt_g1, 1e-3));

  const std::vector<Camera> cameras = database->ReadAllCameras();
  ASSERT_EQ(cameras.size(), 1);
  EXPECT_TRUE(cameras[0].has_prior_focal_length);
  EXPECT_NEAR(cameras[0].MeanFocalLength(), 72.0, 1e-2);
}

TEST(GravityEstimationControllerTest, MultiCameraRigEstimatesSingleRigGravity) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  WriteSolidImage(test_dir / "cam0.png", 64, 48);
  WriteSolidImage(test_dir / "cam1.png", 64, 48);

  const Camera gt_camera =
      Camera::CreateFromModelName(1, "SIMPLE_PINHOLE", 70.0, 64, 48);
  const Eigen::Quaterniond cam1_from_rig(
      Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitY()));
  const Eigen::Vector3d gt_g_rig =
      Eigen::Vector3d(0.1, 0.92, -0.38).normalized();
  const Eigen::Vector3d gt_g_cam0 = gt_g_rig;
  const Eigen::Vector3d gt_g_cam1 = cam1_from_rig * gt_g_rig;

  {
    auto database = Database::Open(database_path);
    Camera cam0 = gt_camera;
    cam0.has_prior_focal_length = true;
    const camera_t cam0_id = database->WriteCamera(cam0);

    Camera cam1 = gt_camera;
    cam1.has_prior_focal_length = true;
    const camera_t cam1_id = database->WriteCamera(cam1);

    Rig rig;
    rig.AddRefSensor(sensor_t(SensorType::CAMERA, cam0_id));
    rig.AddSensor(sensor_t(SensorType::CAMERA, cam1_id),
                  Rigid3d(cam1_from_rig, Eigen::Vector3d(0.2, 0.0, 0.0)));
    const rig_t rig_id = database->WriteRig(rig);

    Frame frame;
    frame.SetRigId(rig_id);
    frame.AddDataId(data_t(sensor_t(SensorType::CAMERA, cam0_id), 1));
    frame.AddDataId(data_t(sensor_t(SensorType::CAMERA, cam1_id), 2));
    const frame_t frame_id = database->WriteFrame(frame);

    Image img0;
    img0.SetImageId(1);
    img0.SetName("cam0.png");
    img0.SetCameraId(cam0_id);
    img0.SetFrameId(frame_id);
    database->WriteImage(img0, /*use_image_id=*/true);

    Image img1;
    img1.SetImageId(2);
    img1.SetName("cam1.png");
    img1.SetCameraId(cam1_id);
    img1.SetFrameId(frame_id);
    database->WriteImage(img1, /*use_image_id=*/true);
  }

  GravityEstimationOptions options;
  options.geocalib.fitting.stride = 2;

  auto controller = CreateGravityEstimationController(
      database_path,
      test_dir,
      options,
      /*image_names=*/{},
      [&](const GeoCalibOptions&) {
        return std::make_unique<FakeGeoCalib>(
            gt_camera, std::vector<Eigen::Vector3d>{gt_g_cam0, gt_g_cam1});
      });
  controller->Start();
  controller->Wait();

  auto database = Database::Open(database_path);
  const std::vector<PosePrior> priors = database->ReadAllPosePriors();
  ASSERT_EQ(priors.size(), 2);
  for (const auto& prior : priors) {
    if (prior.corr_data_id.id == 1) {
      EXPECT_THAT(prior.gravity, EigenMatrixNear(gt_g_cam0, 1e-4));
    } else if (prior.corr_data_id.id == 2) {
      EXPECT_THAT(prior.gravity, EigenMatrixNear(gt_g_cam1, 1e-4));
    } else {
      FAIL() << "Unexpected image id: " << prior.corr_data_id.id;
    }
  }
}

}  // namespace
}  // namespace colmap
