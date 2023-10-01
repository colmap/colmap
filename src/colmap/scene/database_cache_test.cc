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

#include "colmap/scene/database_cache.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(DatabaseCache, Empty) {
  DatabaseCache cache;
  EXPECT_EQ(cache.NumCameras(), 0);
  EXPECT_EQ(cache.NumImages(), 0);
}

TEST(DatabaseCache, Nominal) {
  Database database(Database::kInMemoryDatabasePath);
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1, 1, 1);
  const camera_t camera_id = database.WriteCamera(camera);
  Image image1;
  image1.SetName("image1");
  image1.SetCameraId(camera_id);
  Image image2;
  image2.SetName("image2");
  image2.SetCameraId(camera_id);
  const image_t image_id1 = database.WriteImage(image1);
  const image_t image_id2 = database.WriteImage(image2);
  database.WriteKeypoints(image_id1, FeatureKeypoints(10));
  database.WriteKeypoints(image_id2, FeatureKeypoints(5));
  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches = {{0, 1}};
  two_view_geometry.config =
      TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
  two_view_geometry.F = Eigen::Matrix3d::Random();
  two_view_geometry.E = Eigen::Matrix3d::Random();
  two_view_geometry.H = Eigen::Matrix3d::Random();
  two_view_geometry.cam2_from_cam1 =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  database.WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  auto cache = DatabaseCache::Create(database,
                                     /*min_num_matches=*/0,
                                     /*ignore_watermarks=*/false,
                                     /*image_names=*/{});
  EXPECT_EQ(cache->NumCameras(), 1);
  EXPECT_EQ(cache->NumImages(), 2);
  EXPECT_TRUE(cache->ExistsCamera(camera_id));
  EXPECT_EQ(cache->Camera(camera_id).ModelId(), camera.ModelId());
  EXPECT_TRUE(cache->ExistsImage(image_id1));
  EXPECT_TRUE(cache->ExistsImage(image_id2));
  EXPECT_EQ(cache->Image(image_id1).NumPoints2D(), 10);
  EXPECT_EQ(cache->Image(image_id2).NumPoints2D(), 5);
  const auto correspondence_graph = cache->CorrespondenceGraph();
  EXPECT_TRUE(cache->CorrespondenceGraph()->ExistsImage(image_id1));
  EXPECT_EQ(cache->CorrespondenceGraph()->NumCorrespondencesForImage(image_id1),
            1);
  EXPECT_EQ(cache->CorrespondenceGraph()->NumObservationsForImage(image_id1),
            1);
  EXPECT_TRUE(cache->CorrespondenceGraph()->ExistsImage(image_id2));
  EXPECT_EQ(cache->CorrespondenceGraph()->NumCorrespondencesForImage(image_id2),
            1);
  EXPECT_EQ(cache->CorrespondenceGraph()->NumObservationsForImage(image_id2),
            1);
}

}  // namespace
}  // namespace colmap
