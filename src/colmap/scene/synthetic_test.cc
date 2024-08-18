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

#include "colmap/scene/synthetic.h"

#include "colmap/geometry/triangulation.h"
#include "colmap/scene/projection.h"
#include "colmap/util/misc.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(SynthesizeDataset, Nominal) {
  Database database(Database::kInMemoryDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  SynthesizeDataset(options, &reconstruction, &database);

  const std::string test_dir = CreateTestDir();
  const std::string sparse_path = test_dir + "/sparse";
  CreateDirIfNotExists(sparse_path);
  reconstruction.Write(sparse_path);

  EXPECT_EQ(database.NumCameras(), options.num_cameras);
  EXPECT_EQ(reconstruction.NumCameras(), options.num_cameras);
  for (const auto& camera : reconstruction.Cameras()) {
    EXPECT_EQ(camera.second.ParamsToString(),
              database.ReadCamera(camera.first).ParamsToString());
  }

  EXPECT_EQ(database.NumImages(), options.num_images);
  EXPECT_EQ(reconstruction.NumImages(), options.num_images);
  EXPECT_EQ(reconstruction.NumRegImages(), options.num_images);
  for (const auto& image : reconstruction.Images()) {
    EXPECT_EQ(image.second.Name(), database.ReadImage(image.first).Name());
    EXPECT_EQ(image.second.NumPoints2D(),
              database.ReadKeypoints(image.first).size());
    EXPECT_EQ(image.second.NumPoints2D(),
              database.ReadDescriptors(image.first).rows());
    EXPECT_EQ(database.ReadDescriptors(image.first).cols(), 128);
    EXPECT_EQ(image.second.NumPoints2D(),
              options.num_points3D + options.num_points2D_without_point3D);
    EXPECT_EQ(image.second.NumPoints3D(), options.num_points3D);
  }

  const int num_image_pairs = options.num_images * (options.num_images - 1) / 2;
  EXPECT_EQ(database.NumVerifiedImagePairs(), num_image_pairs);
  EXPECT_EQ(database.NumInlierMatches(),
            num_image_pairs * options.num_points3D);

  EXPECT_NEAR(reconstruction.ComputeMeanReprojectionError(), 0, 1e-6);
  EXPECT_NEAR(reconstruction.ComputeCentroid(0, 1).norm(), 0, 0.2);
  EXPECT_NEAR(reconstruction.ComputeMeanTrackLength(), options.num_images, 0.1);
  EXPECT_EQ(reconstruction.ComputeNumObservations(),
            options.num_images * options.num_points3D);

  // All observations should be perfect and have sufficient triangulation angle.
  // No points or observations should be filtered.
  const double kMaxReprojError = 1e-3;
  const double kMinTriAngleDeg = 0.4;
  std::unordered_map<image_t, Eigen::Vector3d> proj_centers;
  for (const auto& point3D_id : reconstruction.Point3DIds()) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);

    // Make sure all descriptors of the same 3D point have identical features.
    const FeatureDescriptor descriptors =
        database.ReadDescriptors(point3D.track.Element(0).image_id)
            .row(point3D.track.Element(0).point2D_idx);

    for (size_t i1 = 0; i1 < point3D.track.Length(); ++i1) {
      const auto& track_el = point3D.track.Element(i1);
      const image_t image_id1 = track_el.image_id;
      const Image& image1 = reconstruction.Image(image_id1);
      const Camera& camera1 = reconstruction.Camera(image1.CameraId());
      const Point2D& point2D = image1.Point2D(track_el.point2D_idx);
      const double squared_reproj_error = CalculateSquaredReprojectionError(
          point2D.xy, point3D.xyz, image1.CamFromWorld(), camera1);
      EXPECT_LE(squared_reproj_error, kMaxReprojError * kMaxReprojError);
      EXPECT_EQ(descriptors,
                database.ReadDescriptors(point3D.track.Element(i1).image_id)
                    .row(point3D.track.Element(i1).point2D_idx));

      Eigen::Vector3d proj_center1;
      if (proj_centers.count(image_id1) == 0) {
        proj_center1 = image1.ProjectionCenter();
        proj_centers.emplace(image_id1, proj_center1);
      } else {
        proj_center1 = proj_centers.at(image_id1);
      }

      for (size_t i2 = 0; i2 < i1; ++i2) {
        const image_t image_id2 = point3D.track.Element(i2).image_id;
        const Eigen::Vector3d proj_center2 = proj_centers.at(image_id2);

        const double tri_angle = CalculateTriangulationAngle(
            proj_center1, proj_center2, point3D.xyz);
        EXPECT_GE(tri_angle, DegToRad(kMinTriAngleDeg));
      }
    }
  }
}

TEST(SynthesizeDataset, WithNoise) {
  Database database(Database::kInMemoryDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.point2D_stddev = 2.0;
  SynthesizeDataset(options, &reconstruction, &database);

  EXPECT_NEAR(reconstruction.ComputeMeanReprojectionError(),
              options.point2D_stddev,
              0.5 * options.point2D_stddev);
  EXPECT_NEAR(reconstruction.ComputeMeanTrackLength(), options.num_images, 0.1);
}

TEST(SynthesizeDataset, MultiReconstruction) {
  Database database(Database::kInMemoryDatabasePath);
  Reconstruction reconstruction1;
  Reconstruction reconstruction2;
  SyntheticDatasetOptions options;
  SynthesizeDataset(options, &reconstruction1, &database);
  SynthesizeDataset(options, &reconstruction2, &database);

  EXPECT_EQ(database.NumCameras(), 2 * options.num_cameras);
  EXPECT_EQ(reconstruction1.NumCameras(), options.num_cameras);
  EXPECT_EQ(reconstruction1.NumCameras(), options.num_cameras);
  EXPECT_EQ(database.NumImages(), 2 * options.num_images);
  EXPECT_EQ(reconstruction1.NumImages(), options.num_images);
  EXPECT_EQ(reconstruction2.NumImages(), options.num_images);
  EXPECT_EQ(reconstruction1.NumRegImages(), options.num_images);
  EXPECT_EQ(reconstruction2.NumRegImages(), options.num_images);
  const int num_image_pairs = options.num_images * (options.num_images - 1) / 2;
  EXPECT_EQ(database.NumVerifiedImagePairs(), 2 * num_image_pairs);
  EXPECT_EQ(database.NumInlierMatches(),
            2 * num_image_pairs * options.num_points3D);
}

TEST(SynthesizeDataset, ExhaustiveMatches) {
  Database database(Database::kInMemoryDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.match_config = SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE;
  SynthesizeDataset(options, &reconstruction, &database);

  const int num_image_pairs = options.num_images * (options.num_images - 1) / 2;
  EXPECT_EQ(database.NumVerifiedImagePairs(), num_image_pairs);
  EXPECT_EQ(database.NumInlierMatches(),
            num_image_pairs * options.num_points3D);
}

TEST(SynthesizeDataset, ChainedMatches) {
  Database database(Database::kInMemoryDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.match_config = SyntheticDatasetOptions::MatchConfig::CHAINED;
  SynthesizeDataset(options, &reconstruction, &database);

  const int num_image_pairs = options.num_images * (options.num_images - 1) / 2;
  EXPECT_EQ(database.NumVerifiedImagePairs(), num_image_pairs);
  EXPECT_EQ(database.NumInlierMatches(),
            (options.num_images - 1) * options.num_points3D);
}

TEST(SynthesizeDataset, NoDatabase) {
  Database database(Database::kInMemoryDatabasePath);
  SyntheticDatasetOptions options;
  Reconstruction reconstruction;
  SynthesizeDataset(options, &reconstruction);
}

}  // namespace
}  // namespace colmap
