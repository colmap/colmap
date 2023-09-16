// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/estimators/generalized_pose.h"

#include "colmap/estimators/generalized_absolute_pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <numeric>

#include <gtest/gtest.h>

namespace colmap {
namespace {

void BuildGeneralizedCameraProblem(Rigid3d& gt_rig_from_world,
                                   std::vector<Eigen::Vector2d>& points2D,
                                   std::vector<Eigen::Vector3d>& points3D,
                                   std::vector<size_t>& camera_idxs,
                                   std::vector<Rigid3d>& cams_from_rig,
                                   std::vector<Camera>& cameras) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_cameras = 3;
  synthetic_dataset_options.num_images = 3;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.point2D_stddev = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  gt_rig_from_world =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    for (const auto& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        points2D.push_back(point2D.xy);
        points3D.push_back(reconstruction.Point3D(point2D.point3D_id).XYZ());
        camera_idxs.push_back(cameras.size());
      }
    }
    cameras.push_back(reconstruction.Camera(image.CameraId()));
    cams_from_rig.push_back(image.CamFromWorld() * Inverse(gt_rig_from_world));
  }
}

TEST(EstimateGeneralizedAbsolutePose, Nominal) {
  Rigid3d gt_rig_from_world;
  std::vector<Eigen::Vector2d> points2D;
  std::vector<Eigen::Vector3d> points3D;
  std::vector<size_t> camera_idxs;
  std::vector<Rigid3d> cams_from_rig;
  std::vector<Camera> cameras;
  BuildGeneralizedCameraProblem(gt_rig_from_world,
                                points2D,
                                points3D,
                                camera_idxs,
                                cams_from_rig,
                                cameras);

  const double gt_inlier_ratio = 0.8;
  const double outlier_distance = 50;
  const size_t gt_num_outliers =
      std::max(static_cast<size_t>((1.0 - gt_inlier_ratio) * points2D.size()),
               static_cast<size_t>(GP3PEstimator::kMinNumSamples));
  std::vector<char> gt_inlier_mask(points2D.size(), true);
  std::vector<size_t> outlier_indices(points2D.size());
  std::iota(outlier_indices.begin(), outlier_indices.end(), 0);
  std::shuffle(outlier_indices.begin(), outlier_indices.end(), *PRNG);
  for (size_t i = 0; i < gt_num_outliers; ++i) {
    points2D[outlier_indices[i]] +=
        Eigen::Vector2d::Random().normalized() * outlier_distance;
    gt_inlier_mask[outlier_indices[i]] = false;
  }

  RANSACOptions ransac_options;
  ransac_options.max_error = 8;
  ransac_options.min_inlier_ratio = gt_inlier_ratio - 0.1;

  Rigid3d rig_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  EXPECT_TRUE(EstimateGeneralizedAbsolutePose(ransac_options,
                                              points2D,
                                              points3D,
                                              camera_idxs,
                                              cams_from_rig,
                                              cameras,
                                              &rig_from_world,
                                              &num_inliers,
                                              &inlier_mask));
  EXPECT_EQ(num_inliers, points2D.size() - gt_num_outliers);
  EXPECT_EQ(inlier_mask, gt_inlier_mask);
  EXPECT_LT(gt_rig_from_world.rotation.angularDistance(rig_from_world.rotation),
            1e-6);
  EXPECT_LT((gt_rig_from_world.translation - rig_from_world.translation).norm(),
            1e-6);
}

TEST(RefineGeneralizedAbsolutePose, Nominal) {
  Rigid3d gt_rig_from_world;
  std::vector<Eigen::Vector2d> points2D;
  std::vector<Eigen::Vector3d> points3D;
  std::vector<size_t> camera_idxs;
  std::vector<Rigid3d> cams_from_rig;
  std::vector<Camera> cameras;
  BuildGeneralizedCameraProblem(gt_rig_from_world,
                                points2D,
                                points3D,
                                camera_idxs,
                                cams_from_rig,
                                cameras);
  const std::vector<char> gt_inlier_mask(points2D.size(), true);

  const double rotation_noise_degree = 1;
  const double translation_noise = 0.1;
  const Rigid3d rig_from_gt_rig(Eigen::Quaterniond(Eigen::AngleAxisd(
                                    DegToRad(rotation_noise_degree),
                                    Eigen::Vector3d::Random().normalized())),
                                Eigen::Vector3d::Random() * translation_noise);
  Rigid3d rig_from_world = rig_from_gt_rig * gt_rig_from_world;

  AbsolutePoseRefinementOptions options;
  options.refine_focal_length = false;
  options.refine_extra_params = false;
  EXPECT_TRUE(RefineGeneralizedAbsolutePose(options,
                                            gt_inlier_mask,
                                            points2D,
                                            points3D,
                                            camera_idxs,
                                            cams_from_rig,
                                            &rig_from_world,
                                            &cameras));
  EXPECT_LT(gt_rig_from_world.rotation.angularDistance(rig_from_world.rotation),
            1e-6);
  EXPECT_LT((gt_rig_from_world.translation - rig_from_world.translation).norm(),
            1e-6);
}

}  // namespace
}  // namespace colmap
