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

#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_matchers.h"

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
  calib_options.reestimate_relative_pose = false;
  EXPECT_TRUE(CalibrateViewGraph(calib_options, database.get()));

  // Verify focal lengths are calibrated close to ground truth.
  for (const auto& [camera_id, gt_focal] : gt_focals) {
    const Camera camera = database->ReadCamera(camera_id);
    EXPECT_NEAR(camera.MeanFocalLength(), gt_focal, 1.0);
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
  }

  ViewGraphCalibrationOptions calib_options;
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

  // Add large noise to F matrices for 3 pairs to ensure they become degenerate.
  std::unordered_set<image_pair_t> perturbed_pairs;
  for (const auto& [pair_id, tvg] : database->ReadTwoViewGeometries()) {
    if (perturbed_pairs.size() >= 3) break;
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    TwoViewGeometry perturbed_tvg = tvg;
    perturbed_tvg.F += Eigen::Matrix3d::Constant(1.0);
    database->UpdateTwoViewGeometry(image_id1, image_id2, perturbed_tvg);
    perturbed_pairs.insert(pair_id);
  }

  ViewGraphCalibrationOptions calib_options;
  calib_options.reestimate_relative_pose = false;
  calib_options.max_calibration_error = 0.01;
  EXPECT_TRUE(CalibrateViewGraph(calib_options, database.get()));

  // Verify perturbed pairs became DEGENERATE, others became CALIBRATED.
  for (const auto& [pair_id, tvg] : database->ReadTwoViewGeometries()) {
    if (perturbed_pairs.count(pair_id)) {
      EXPECT_EQ(tvg.config, TwoViewGeometry::DEGENERATE);
    } else {
      EXPECT_EQ(tvg.config, TwoViewGeometry::CALIBRATED);
    }
  }
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

  // Store ground truth relative poses and perturb them in the database.
  // The perturbation must exceed test thresholds (0.1 rad rotation, 0.1
  // normalized translation error) to ensure re-estimation actually runs.
  std::unordered_map<image_pair_t, Rigid3d> gt_poses;
  for (const auto& [pair_id, tvg] : database->ReadTwoViewGeometries()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const Image& image1 = reconstruction.Image(image_id1);
    const Image& image2 = reconstruction.Image(image_id2);
    gt_poses[pair_id] = image2.CamFromWorld() * Inverse(image1.CamFromWorld());

    // Perturb the relative pose stored in database.
    TwoViewGeometry perturbed_tvg = tvg;
    if (perturbed_tvg.cam2_from_cam1.has_value()) {
      const Rigid3d perturbation(
          Eigen::Quaterniond(
              Eigen::AngleAxisd(RandomUniformReal(0.3, 0.7),
                                Eigen::Vector3d(RandomUniformReal(-1.0, 1.0),
                                                RandomUniformReal(-1.0, 1.0),
                                                RandomUniformReal(-1.0, 1.0))
                                    .normalized())),
          Eigen::Vector3d(RandomUniformReal(-1.0, 1.0),
                          RandomUniformReal(-1.0, 1.0),
                          RandomUniformReal(-1.0, 1.0)));
      perturbed_tvg.cam2_from_cam1 =
          perturbation * *perturbed_tvg.cam2_from_cam1;
      perturbed_tvg.cam2_from_cam1->translation.normalize();
    }
    database->UpdateTwoViewGeometry(image_id1, image_id2, perturbed_tvg);
  }

  ViewGraphCalibrationOptions calib_options;
  calib_options.reestimate_relative_pose = true;
  EXPECT_TRUE(CalibrateViewGraph(calib_options, database.get()));

  // Verify relative poses are estimated correctly.
  for (const auto& [pair_id, tvg] : database->ReadTwoViewGeometries()) {
    if (tvg.config != TwoViewGeometry::CALIBRATED) continue;

    ASSERT_TRUE(tvg.cam2_from_cam1.has_value());
    EXPECT_NEAR(tvg.cam2_from_cam1->translation.norm(), 1.0, 1e-6);

    // Normalize ground truth translation since estimated pose has unit scale.
    const Rigid3d& gt_pose = gt_poses.at(pair_id);
    const Rigid3d gt_pose_normalized(gt_pose.rotation,
                                     gt_pose.translation.normalized());
    EXPECT_THAT(*tvg.cam2_from_cam1,
                Rigid3dNear(gt_pose_normalized, 0.01, 0.01));
  }
}

}  // namespace
}  // namespace colmap
