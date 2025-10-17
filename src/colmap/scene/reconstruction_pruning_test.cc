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

#include "colmap/scene/reconstruction_pruning.h"

#include "colmap/scene/synthetic.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Geometry>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(FindRedundantPoints3D, Empty) {
  Reconstruction reconstruction;
  EXPECT_THAT(FindRedundantPoints3D(/*min_coverage_gain=*/0, reconstruction),
              testing::IsEmpty());
}

TEST(FindRedundantPoints3D, VaryingCoverageGain) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  EXPECT_THAT(FindRedundantPoints3D(/*min_coverage_gain=*/0, reconstruction),
              testing::IsEmpty());
  size_t prev_num_redundant_points3D = 0;
  for (const double min_coverage_gain : {0.1, 0.4, 0.7, 10.0}) {
    const std::vector<point3D_t> redundant_point3D_ids =
        FindRedundantPoints3D(min_coverage_gain, reconstruction);
    EXPECT_GT(redundant_point3D_ids.size(), prev_num_redundant_points3D);
    prev_num_redundant_points3D = redundant_point3D_ids.size();
  }
  EXPECT_EQ(prev_num_redundant_points3D, reconstruction.NumPoints3D());
}

TEST(FindRedundantPoints3D, VaryingTrackLength) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 5;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  std::vector<point3D_t> expected_redundant_point3D_ids;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    const auto track_el = point3D.track.Element(0);
    reconstruction.Image(track_el.image_id)
        .ResetPoint3DForPoint2D(track_el.point2D_idx);
    reconstruction.Point3D(point3D_id).track.DeleteElement(0);
    expected_redundant_point3D_ids.push_back(point3D_id);
    EXPECT_THAT(
        FindRedundantPoints3D(/*min_coverage_gain=*/0.1, reconstruction),
        testing::UnorderedElementsAreArray(expected_redundant_point3D_ids));
  }
}

TEST(FindRedundantPoints3D, VaryingSpatialDistribution) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 4;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 4;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  // Ensure all cameras in the rig have identical poses.
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0.0;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 0.0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  // Generate a synthetic dataset where all points are identically distributed
  // in a circle around the camera center, i.e., each of them has the same
  // coverage.
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CHECK_EQ(point3D.track.Length(),
             synthetic_dataset_options.num_cameras_per_rig);
    for (int i = 0; i < synthetic_dataset_options.num_cameras_per_rig; ++i) {
      const auto& track_el = point3D.track.Element(i);
      auto& point2D =
          reconstruction.Image(track_el.image_id).Point2D(track_el.point2D_idx);
      const double angle = EIGEN_PI / 4 + i * EIGEN_PI / 2;
      const double radius =
          0.25 * std::sqrt(synthetic_dataset_options.camera_width *
                               synthetic_dataset_options.camera_width +
                           synthetic_dataset_options.camera_height *
                               synthetic_dataset_options.camera_height);
      point2D.xy =
          Eigen::Vector2d(std::sin(angle) * radius +
                              synthetic_dataset_options.camera_width / 2.,
                          std::cos(angle) * radius +
                              synthetic_dataset_options.camera_height / 2.);
    }
  }

  auto get_point2D = [&reconstruction](point3D_t point3D_id,
                                       int track_el_idx) -> Point2D& {
    const auto& track_el =
        reconstruction.Point3D(point3D_id).track.Element(track_el_idx);
    return reconstruction.Image(track_el.image_id)
        .Point2D(track_el.point2D_idx);
  };

  // Make sure point 3 has the largest coverage and selected first.
  reconstruction.Point3D(3).track.AddElement(1, 4);
  Point2D point2D;
  point2D.xy = Eigen::Vector2d::Zero();
  point2D.point3D_id = 3;
  reconstruction.Image(1).Points2D().push_back(point2D);

  // Make sure point 1 has larger coverage gain by moving it to another
  // tile, while point 2 and 4 have redundant and thus smaller coverage.
  get_point2D(1, 0).xy *= 2;
  EXPECT_THAT(FindRedundantPoints3D(/*min_coverage_gain=*/0.6, reconstruction),
              testing::UnorderedElementsAre(2, 4));

  // Now change point 2 to have redundant coverage with point 1 and point 4 to
  // have unique coverage.
  get_point2D(2, 0).xy *= 2;
  get_point2D(4, 2).xy *= 2;
  EXPECT_THAT(FindRedundantPoints3D(/*min_coverage_gain=*/0.6, reconstruction),
              testing::UnorderedElementsAre(1, 2));
}

}  // namespace
}  // namespace colmap
