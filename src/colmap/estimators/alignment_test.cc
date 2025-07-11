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

#include "colmap/estimators/alignment.h"

#include "colmap/geometry/sim3.h"
#include "colmap/math/random.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

Sim3d TestSim3d() {
  return Sim3d(RandomUniformReal<double>(0.5, 2),
               Eigen::Quaterniond::UnitRandom(),
               Eigen::Vector3d::Random());
}

void ExpectEqualSim3d(const Sim3d& gt_tgt_from_src, const Sim3d& tgt_from_src) {
  EXPECT_NEAR(gt_tgt_from_src.scale, tgt_from_src.scale, 1e-6);
  EXPECT_LT(gt_tgt_from_src.rotation.angularDistance(tgt_from_src.rotation),
            1e-6);
  EXPECT_LT((gt_tgt_from_src.translation - tgt_from_src.translation).norm(),
            1e-6);
}

Reconstruction GenerateReconstructionForAlignment() {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.point2D_stddev = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  return reconstruction;
}

TEST(Alignment, AlignReconstructionToLocations) {
  Reconstruction src_reconstruction = GenerateReconstructionForAlignment();
  Reconstruction tgt_reconstruction = src_reconstruction;

  Sim3d gt_tgt_from_src = TestSim3d();
  tgt_reconstruction.Transform(gt_tgt_from_src);

  std::vector<std::string> tgt_image_names;
  std::vector<Eigen::Vector3d> tgt_image_locations;
  for (const auto& [_, image] : tgt_reconstruction.Images()) {
    tgt_image_names.push_back(image.Name());
    tgt_image_locations.push_back(image.ProjectionCenter());
  }

  RANSACOptions ransac_options;
  ransac_options.max_error = 1e-2;

  Sim3d tgt_from_src;
  ASSERT_FALSE(AlignReconstructionToLocations(
      src_reconstruction,
      tgt_image_names,
      tgt_image_locations,
      /*min_common_images=*/tgt_image_names.size() + 1,
      ransac_options,
      &tgt_from_src));
  ASSERT_TRUE(AlignReconstructionToLocations(src_reconstruction,
                                             tgt_image_names,
                                             tgt_image_locations,
                                             /*min_common_images=*/3,
                                             ransac_options,
                                             &tgt_from_src));
  ExpectEqualSim3d(gt_tgt_from_src, tgt_from_src);
}

TEST(Alignment, AlignReconstructionToPosePriors) {
  Reconstruction src_reconstruction = GenerateReconstructionForAlignment();
  Reconstruction tgt_reconstruction = src_reconstruction;

  Sim3d gt_tgt_from_src = TestSim3d();
  tgt_reconstruction.Transform(gt_tgt_from_src);

  std::unordered_map<image_t, PosePrior> tgt_pose_priors;
  for (const auto& [image_id, image] : tgt_reconstruction.Images()) {
    PosePrior& pose_prior = tgt_pose_priors[image_id];
    pose_prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
    pose_prior.position = image.ProjectionCenter();
    pose_prior.position_covariance = 1e-2 * Eigen::Matrix3d::Identity();
  }

  RANSACOptions ransac_options;
  ransac_options.max_error = 1e-2;

  Sim3d tgt_from_src;
  ASSERT_TRUE(AlignReconstructionToPosePriors(
      src_reconstruction, tgt_pose_priors, ransac_options, &tgt_from_src));
  ExpectEqualSim3d(gt_tgt_from_src, tgt_from_src);
}

TEST(Alignment, AlignReconstructionsViaReprojections) {
  Reconstruction src_reconstruction = GenerateReconstructionForAlignment();
  Reconstruction tgt_reconstruction = src_reconstruction;

  Sim3d gt_tgt_from_src = TestSim3d();
  tgt_reconstruction.Transform(gt_tgt_from_src);

  Sim3d tgt_from_src;
  ASSERT_TRUE(
      AlignReconstructionsViaReprojections(src_reconstruction,
                                           tgt_reconstruction,
                                           /*min_inlier_observations=*/0.9,
                                           /*max_reproj_error=*/2,
                                           &tgt_from_src));
  ExpectEqualSim3d(gt_tgt_from_src, tgt_from_src);
}

TEST(Alignment, AlignReconstructionsViaProjCenters) {
  Reconstruction src_reconstruction = GenerateReconstructionForAlignment();
  Reconstruction tgt_reconstruction = src_reconstruction;

  Sim3d gt_tgt_from_src = TestSim3d();
  tgt_reconstruction.Transform(gt_tgt_from_src);

  Sim3d tgt_from_src;
  ASSERT_TRUE(AlignReconstructionsViaProjCenters(src_reconstruction,
                                                 tgt_reconstruction,
                                                 /*max_proj_center_error=*/0.1,
                                                 &tgt_from_src));
  ExpectEqualSim3d(gt_tgt_from_src, tgt_from_src);
}

TEST(Alignment, AlignReconstructionsViaPoints) {
  Reconstruction src_reconstruction = GenerateReconstructionForAlignment();
  Reconstruction tgt_reconstruction = src_reconstruction;

  Sim3d gt_tgt_from_src = TestSim3d();
  tgt_reconstruction.Transform(gt_tgt_from_src);

  Sim3d tgt_from_src;
  ASSERT_TRUE(AlignReconstructionsViaPoints(src_reconstruction,
                                            tgt_reconstruction,
                                            /*min_common_observations=*/3,
                                            /*max_error=*/0.01,
                                            /*min_inlier_ratio=*/0.9,
                                            &tgt_from_src));
  ExpectEqualSim3d(gt_tgt_from_src, tgt_from_src);
}

TEST(Alignment, MergeReconstructions) {
  // Synthesize a reconstruction which has at least two cameras
  Reconstruction src_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 3;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.point2D_stddev = 0;
  SynthesizeDataset(synthetic_dataset_options, &src_reconstruction);
  Reconstruction orig_reconstruction = src_reconstruction;
  Reconstruction tgt_reconstruction = src_reconstruction;

  auto remove_rig_frames = [](Reconstruction& reconstruction, rig_t rig_id) {
    const std::vector<frame_t> frame_ids = reconstruction.RegFrameIds();
    for (const auto& frame_id : frame_ids) {
      if (reconstruction.Frame(frame_id).RigId() == rig_id) {
        reconstruction.DeRegisterFrame(frame_id);
      }
    }
  };

  remove_rig_frames(src_reconstruction, 1);
  remove_rig_frames(tgt_reconstruction, 2);

  // Remove all unregistered rigs/cameras/frames/images.
  src_reconstruction.TearDown();
  tgt_reconstruction.TearDown();
  EXPECT_EQ(src_reconstruction.NumRigs(), 2);
  EXPECT_EQ(src_reconstruction.NumCameras(), 2);
  EXPECT_EQ(src_reconstruction.NumFrames(), 20);
  EXPECT_EQ(src_reconstruction.NumRegFrames(), 20);
  EXPECT_EQ(src_reconstruction.NumImages(), 20);
  EXPECT_EQ(tgt_reconstruction.NumRigs(), 2);
  EXPECT_EQ(tgt_reconstruction.NumCameras(), 2);
  EXPECT_EQ(tgt_reconstruction.NumFrames(), 20);
  EXPECT_EQ(tgt_reconstruction.NumRegFrames(), 20);
  EXPECT_EQ(tgt_reconstruction.NumImages(), 20);

  // Merge reconstructions.
  ASSERT_TRUE(MergeReconstructions(
      /*max_reproj_error=*/1e-4, src_reconstruction, tgt_reconstruction));
  EXPECT_EQ(tgt_reconstruction.NumRigs(), 3);
  EXPECT_EQ(tgt_reconstruction.NumCameras(), 3);
  EXPECT_EQ(tgt_reconstruction.NumFrames(), 30);
  EXPECT_EQ(tgt_reconstruction.NumRegFrames(), 30);
  EXPECT_EQ(tgt_reconstruction.NumImages(), 30);
  EXPECT_EQ(tgt_reconstruction.NumPoints3D(), 50);
  EXPECT_EQ(tgt_reconstruction.ComputeNumObservations(),
            orig_reconstruction.ComputeNumObservations());
}

}  // namespace
}  // namespace colmap
