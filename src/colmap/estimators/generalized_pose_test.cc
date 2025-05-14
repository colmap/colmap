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

#include "colmap/estimators/generalized_pose.h"

#include "colmap/estimators/generalized_absolute_pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <numeric>

#include <gtest/gtest.h>

namespace colmap {
namespace {

struct GeneralizedAbsolutePoseProblem {
  Rigid3d gt_rig_from_world;
  std::vector<Eigen::Vector2d> points2D;
  std::vector<Eigen::Vector3d> points3D;
  std::vector<size_t> point3D_ids;
  std::vector<size_t> camera_idxs;
  std::vector<Rigid3d> cams_from_rig;
  std::vector<Camera> cameras;
};

GeneralizedAbsolutePoseProblem BuildGeneralizedAbsolutePoseProblem() {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.point2D_stddev = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  GeneralizedAbsolutePoseProblem problem;
  problem.gt_rig_from_world =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    for (const auto& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        problem.points2D.push_back(point2D.xy);
        problem.points3D.push_back(
            reconstruction.Point3D(point2D.point3D_id).xyz);
        problem.point3D_ids.push_back(point2D.point3D_id);
        problem.camera_idxs.push_back(problem.cameras.size());
      }
    }
    problem.cameras.push_back(*image.CameraPtr());
    problem.cams_from_rig.push_back(image.CamFromWorld() *
                                    Inverse(problem.gt_rig_from_world));
  }
  return problem;
}

TEST(EstimateGeneralizedAbsolutePose, Nominal) {
  SetPRNGSeed();

  GeneralizedAbsolutePoseProblem problem =
      BuildGeneralizedAbsolutePoseProblem();
  const size_t num_points = problem.points2D.size();

  const double gt_inlier_ratio = 0.8;
  const double outlier_distance = 50;
  const size_t gt_num_inliers =
      std::max(static_cast<size_t>(gt_inlier_ratio * num_points),
               static_cast<size_t>(GP3PEstimator::kMinNumSamples));
  std::vector<size_t> shuffled_idxs(num_points);
  std::iota(shuffled_idxs.begin(), shuffled_idxs.end(), 0);
  std::shuffle(shuffled_idxs.begin(), shuffled_idxs.end(), *PRNG);

  std::unordered_set<size_t> unique_inlier_ids;
  unique_inlier_ids.reserve(gt_num_inliers);
  for (size_t i = 0; i < gt_num_inliers; ++i) {
    unique_inlier_ids.insert(problem.point3D_ids[shuffled_idxs[i]]);
  }

  std::vector<char> gt_inlier_mask(num_points, true);
  for (size_t i = gt_num_inliers; i < num_points; ++i) {
    problem.points2D[shuffled_idxs[i]] +=
        Eigen::Vector2d::Random().normalized() * outlier_distance;
    gt_inlier_mask[shuffled_idxs[i]] = false;
  }

  RANSACOptions ransac_options;
  ransac_options.max_error = 2;
  ransac_options.min_inlier_ratio = gt_inlier_ratio / 2;
  ransac_options.confidence = 0.99999;

  Rigid3d rig_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  EXPECT_TRUE(EstimateGeneralizedAbsolutePose(ransac_options,
                                              problem.points2D,
                                              problem.points3D,
                                              problem.camera_idxs,
                                              problem.cams_from_rig,
                                              problem.cameras,
                                              &rig_from_world,
                                              &num_inliers,
                                              &inlier_mask));
  EXPECT_EQ(num_inliers, unique_inlier_ids.size());
  EXPECT_EQ(inlier_mask, gt_inlier_mask);
  EXPECT_THAT(
      rig_from_world,
      Rigid3dNear(problem.gt_rig_from_world, /*rtol=*/1e-6, /*ttol=*/1e-6));
}

TEST(RefineGeneralizedAbsolutePose, Nominal) {
  GeneralizedAbsolutePoseProblem problem =
      BuildGeneralizedAbsolutePoseProblem();
  const std::vector<char> gt_inlier_mask(problem.points2D.size(), true);

  const double rotation_noise_degree = 1;
  const double translation_noise = 0.1;
  const Rigid3d rig_from_gt_rig(Eigen::Quaterniond(Eigen::AngleAxisd(
                                    DegToRad(rotation_noise_degree),
                                    Eigen::Vector3d::Random().normalized())),
                                Eigen::Vector3d::Random() * translation_noise);
  Rigid3d rig_from_world = rig_from_gt_rig * problem.gt_rig_from_world;

  AbsolutePoseRefinementOptions options;
  options.refine_focal_length = false;
  options.refine_extra_params = false;
  EXPECT_TRUE(RefineGeneralizedAbsolutePose(options,
                                            gt_inlier_mask,
                                            problem.points2D,
                                            problem.points3D,
                                            problem.camera_idxs,
                                            problem.cams_from_rig,
                                            &rig_from_world,
                                            &problem.cameras));
  EXPECT_THAT(
      rig_from_world,
      Rigid3dNear(problem.gt_rig_from_world, /*rtol=*/1e-6, /*ttol=*/1e-6));
}

struct GeneralizedRelativePoseProblem {
  Rigid3d gt_rig2_from_rig1;
  std::vector<Eigen::Vector2d> points2D1;
  std::vector<Eigen::Vector2d> points2D2;
  std::vector<size_t> camera_idxs1;
  std::vector<size_t> camera_idxs2;
  std::vector<Rigid3d> cams_from_rig;
  std::vector<Camera> cameras;
};

GeneralizedRelativePoseProblem BuildGeneralizedRelativePoseProblem(
    int num_cameras_per_rig) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = num_cameras_per_rig;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.point2D_stddev = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const Frame& frame1 = reconstruction.Frame(1);
  const Frame& frame2 = reconstruction.Frame(2);
  CHECK_NE(frame1.RigId(), frame2.RigId());

  GeneralizedRelativePoseProblem problem;
  problem.gt_rig2_from_rig1 =
      frame2.RigFromWorld() * Inverse(frame1.RigFromWorld());

  std::unordered_map<point3D_t, std::pair<const Image*, point2D_t>>
      observations2;
  for (const data_t& data_id : frame2.ImageIds()) {
    const auto& image = reconstruction.Image(data_id.id);
    for (size_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      const auto& point2D = image.Point2D(point2D_idx);
      if (point2D.HasPoint3D()) {
        observations2[point2D.point3D_id] = std::make_pair(&image, point2D_idx);
      }
    }
  }

  std::unordered_map<camera_t, size_t> camera_id_to_idx;
  for (const data_t& data_id : frame1.ImageIds()) {
    const auto& image1 = reconstruction.Image(data_id.id);
    for (size_t point2D_idx1 = 0; point2D_idx1 < image1.NumPoints2D();
         ++point2D_idx1) {
      const auto& point2D1 = image1.Point2D(point2D_idx1);
      const auto observation_it = observations2.find(point2D1.point3D_id);
      if (observation_it == observations2.end()) {
        continue;
      }

      const auto& [image2_ptr, point2D_idx2] = observation_it->second;

      auto maybe_add_and_get_camera = [&problem,
                                       &camera_id_to_idx](const Image& image) {
        auto [it, inserted] =
            camera_id_to_idx.emplace(image.CameraId(), problem.cameras.size());
        if (inserted) {
          problem.cameras.push_back(*image.CameraPtr());
          const Rig& rig = *image.FramePtr()->RigPtr();
          if (rig.IsRefSensor(image.CameraPtr()->SensorId())) {
            problem.cams_from_rig.push_back(Rigid3d());
          } else {
            problem.cams_from_rig.push_back(
                image.FramePtr()->RigPtr()->SensorFromRig(
                    image.CameraPtr()->SensorId()));
          }
        }
        return it->second;
      };

      problem.points2D1.push_back(point2D1.xy);
      problem.points2D2.push_back(image2_ptr->Point2D(point2D_idx2).xy);
      problem.camera_idxs1.push_back(maybe_add_and_get_camera(image1));
      problem.camera_idxs2.push_back(maybe_add_and_get_camera(*image2_ptr));
    }
  }

  return problem;
}

TEST(EstimateGeneralizedRelativePose, Nominal) {
  SetPRNGSeed();

  GeneralizedRelativePoseProblem problem =
      BuildGeneralizedRelativePoseProblem(/*num_cameras_per_rig=*/3);

  RANSACOptions ransac_options;
  ransac_options.max_error = 1e-2;

  std::optional<Rigid3d> rig2_from_rig1;
  std::optional<Rigid3d> cam2_from_cam1;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  EXPECT_TRUE(EstimateGeneralizedRelativePose(ransac_options,
                                              problem.points2D1,
                                              problem.points2D2,
                                              problem.camera_idxs1,
                                              problem.camera_idxs2,
                                              problem.cams_from_rig,
                                              problem.cameras,
                                              &rig2_from_rig1,
                                              &cam2_from_cam1,
                                              &num_inliers,
                                              &inlier_mask));
  EXPECT_EQ(num_inliers, problem.points2D1.size());
  EXPECT_THAT(inlier_mask, testing::Each(testing::Eq(true)));
  ASSERT_TRUE(rig2_from_rig1.has_value());
  ASSERT_FALSE(cam2_from_cam1.has_value());
  EXPECT_THAT(
      *rig2_from_rig1,
      Rigid3dNear(problem.gt_rig2_from_rig1, /*rtol=*/1e-6, /*ttol=*/1e-6));
}

TEST(EstimateGeneralizedRelativePose, Panoramic) {
  SetPRNGSeed();

  GeneralizedRelativePoseProblem problem =
      BuildGeneralizedRelativePoseProblem(/*num_cameras_per_rig=*/1);

  RANSACOptions ransac_options;
  ransac_options.max_error = 1e-2;

  std::optional<Rigid3d> rig2_from_rig1;
  std::optional<Rigid3d> cam2_from_cam1;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  EXPECT_TRUE(EstimateGeneralizedRelativePose(ransac_options,
                                              problem.points2D1,
                                              problem.points2D2,
                                              problem.camera_idxs1,
                                              problem.camera_idxs2,
                                              problem.cams_from_rig,
                                              problem.cameras,
                                              &rig2_from_rig1,
                                              &cam2_from_cam1,
                                              &num_inliers,
                                              &inlier_mask));
  EXPECT_EQ(num_inliers, problem.points2D1.size());
  EXPECT_THAT(inlier_mask, testing::Each(testing::Eq(true)));
  ASSERT_FALSE(rig2_from_rig1.has_value());
  ASSERT_TRUE(cam2_from_cam1.has_value());
  EXPECT_THAT(
      *cam2_from_cam1,
      Rigid3dNear(Rigid3d(problem.gt_rig2_from_rig1.rotation,
                          problem.gt_rig2_from_rig1.translation.normalized()),
                  /*rtol=*/1e-6,
                  /*ttol=*/1e-6));
}

}  // namespace
}  // namespace colmap
