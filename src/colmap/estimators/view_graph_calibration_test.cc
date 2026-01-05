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

#include "colmap/estimators/view_graph_calibration.h"

#include "colmap/math/random.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CalibrateViewGraph, Nominal) {
  SetPRNGSeed(42);

  auto database = Database::Open(kInMemorySqliteDatabasePath);

  SyntheticDatasetOptions options;
  options.num_rigs = 10;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 1;
  options.num_points3D = 200;
  options.camera_model_id = SimplePinholeCameraModel::model_id;
  options.camera_params = {1280, 512, 384};
  options.camera_has_prior_focal_length = false;

  Reconstruction reconstruction;
  SynthesizeDataset(options, &reconstruction, database.get());

  // Store ground truth focal lengths.
  std::unordered_map<camera_t, double> gt_focals;
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    gt_focals[camera_id] = camera.MeanFocalLength();
  }

  // Change pairs to UNCALIBRATED so F matrices are used directly.
  for (const auto& [pair_id, tvg] : database->ReadTwoViewGeometries()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    TwoViewGeometry uncalib_tvg = tvg;
    uncalib_tvg.config = TwoViewGeometry::UNCALIBRATED;
    database->UpdateTwoViewGeometry(image_id1, image_id2, uncalib_tvg);
  }

  // Add noise to focal lengths of the first two cameras.
  // TODO: investigate view graph calibration cost functor and use more
  // challenging test setup.
  for (const auto& [camera_id, _] : reconstruction.Cameras()) {
    if (camera_id >= 2) continue;
    Camera camera = database->ReadCamera(camera_id);
    const double noise = RandomUniformReal(-50.0, 50.0);
    for (const size_t idx : camera.FocalLengthIdxs()) {
      camera.params[idx] += noise;
    }
    database->UpdateCamera(camera);
  }

  ViewGraphCalibrationOptions calib_options;
  calib_options.random_seed = 42;
  calib_options.reestimate_relative_pose = false;
  EXPECT_TRUE(CalibrateViewGraph(calib_options, database.get()));

  // Verify focal lengths are calibrated close to ground truth.
  for (const auto& [camera_id, gt_focal] : gt_focals) {
    const Camera camera = database->ReadCamera(camera_id);
    EXPECT_NEAR(camera.MeanFocalLength(), gt_focal, 0.1);
  }

  // Verify pairs are now CALIBRATED with valid E matrices.
  for (const auto& [pair_id, tvg] : database->ReadTwoViewGeometries()) {
    EXPECT_EQ(tvg.config, TwoViewGeometry::CALIBRATED);
    EXPECT_FALSE(tvg.E.isZero());
  }
}

TEST(CalibrateViewGraph, PriorFocalLength) {
  SetPRNGSeed(42);

  auto database = Database::Open(kInMemorySqliteDatabasePath);

  SyntheticDatasetOptions options;
  options.num_rigs = 10;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 1;
  options.num_points3D = 200;
  options.camera_model_id = SimplePinholeCameraModel::model_id;
  options.camera_params = {1280, 512, 384};
  options.camera_has_prior_focal_length = true;

  Reconstruction reconstruction;
  SynthesizeDataset(options, &reconstruction, database.get());

  // Store original focal lengths (which have priors).
  std::unordered_map<camera_t, double> original_focals;
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    original_focals[camera_id] = camera.MeanFocalLength();
    EXPECT_TRUE(camera.has_prior_focal_length);
  }

  ViewGraphCalibrationOptions calib_options;
  calib_options.random_seed = 42;
  calib_options.reestimate_relative_pose = false;
  EXPECT_TRUE(CalibrateViewGraph(calib_options, database.get()));

  // Verify cameras with priors are unchanged.
  for (const auto& [camera_id, original_focal] : original_focals) {
    const Camera camera = database->ReadCamera(camera_id);
    EXPECT_EQ(camera.MeanFocalLength(), original_focal);
  }
}

TEST(CalibrateViewGraph, ConfigTagging) {
  SetPRNGSeed(42);

  auto database = Database::Open(kInMemorySqliteDatabasePath);

  SyntheticDatasetOptions options;
  options.num_rigs = 10;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 1;
  options.num_points3D = 200;
  options.camera_model_id = SimplePinholeCameraModel::model_id;
  options.camera_params = {1280, 512, 384};
  options.camera_has_prior_focal_length = false;

  Reconstruction reconstruction;
  SynthesizeDataset(options, &reconstruction, database.get());

  // Set very strict calibration error threshold to trigger DEGENERATE_VGC.
  ViewGraphCalibrationOptions calib_options;
  calib_options.random_seed = 42;
  calib_options.reestimate_relative_pose = false;
  calib_options.max_calibration_error = 0.001;  // Very strict threshold
  EXPECT_TRUE(CalibrateViewGraph(calib_options, database.get()));

  // With strict threshold, some pairs should be tagged as DEGENERATE_VGC.
  size_t calibrated_count = 0;
  size_t degenerate_count = 0;
  for (const auto& [pid, geom] : database->ReadTwoViewGeometries()) {
    if (geom.config == TwoViewGeometry::CALIBRATED) {
      calibrated_count++;
    } else if (geom.config == TwoViewGeometry::DEGENERATE_VGC) {
      degenerate_count++;
    }
  }
  // At least some pairs should be processed.
  EXPECT_GT(calibrated_count + degenerate_count, 0);
}

TEST(CalibrateViewGraph, RelativePoseReestimation) {
  SetPRNGSeed(42);

  auto database = Database::Open(kInMemorySqliteDatabasePath);

  SyntheticDatasetOptions options;
  options.num_rigs = 10;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 1;
  options.num_points3D = 200;
  options.camera_model_id = SimplePinholeCameraModel::model_id;
  options.camera_params = {1280, 512, 384};
  options.camera_has_prior_focal_length = false;

  Reconstruction reconstruction;
  SynthesizeDataset(options, &reconstruction, database.get());

  // Store ground truth relative poses.
  std::unordered_map<image_pair_t, Rigid3d> gt_poses;
  for (const auto& [pair_id, tvg] : database->ReadTwoViewGeometries()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const Image& image1 = reconstruction.Image(image_id1);
    const Image& image2 = reconstruction.Image(image_id2);
    gt_poses[pair_id] = image2.CamFromWorld() * Inverse(image1.CamFromWorld());
  }

  ViewGraphCalibrationOptions calib_options;
  calib_options.random_seed = 42;
  calib_options.reestimate_relative_pose = true;
  EXPECT_TRUE(CalibrateViewGraph(calib_options, database.get()));

  // Verify relative poses are estimated correctly.
  for (const auto& [pair_id, tvg] : database->ReadTwoViewGeometries()) {
    if (tvg.config != TwoViewGeometry::CALIBRATED) continue;

    ASSERT_TRUE(tvg.cam2_from_cam1.has_value());
    const Rigid3d& gt_pose = gt_poses.at(pair_id);

    const double rotation_error =
        tvg.cam2_from_cam1->rotation.angularDistance(gt_pose.rotation);
    EXPECT_LT(rotation_error, 0.1);

    const Eigen::Vector3d gt_translation_normalized =
        gt_pose.translation.normalized();
    const Eigen::Vector3d est_translation_normalized =
        tvg.cam2_from_cam1->translation.normalized();
    const double translation_error =
        (gt_translation_normalized - est_translation_normalized).norm();
    EXPECT_LT(translation_error, 0.1);
  }
}

}  // namespace
}  // namespace colmap
