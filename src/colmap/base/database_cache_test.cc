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

#include "colmap/base/database_cache.h"

#include <gtest/gtest.h>

namespace colmap {

TEST(DatabaseCache, Empty) {
  DatabaseCache cache;
  EXPECT_EQ(cache.NumCameras(), 0);
  EXPECT_EQ(cache.NumImages(), 0);
}

TEST(DatabaseCache, AddCamera) {
  DatabaseCache cache;
  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1, 1, 1);
  cache.AddCamera(camera);
  EXPECT_EQ(cache.NumCameras(), 1);
  EXPECT_EQ(cache.NumImages(), 0);
  EXPECT_TRUE(cache.ExistsCamera(camera.CameraId()));
  EXPECT_EQ(cache.Camera(camera.CameraId()).ModelId(), camera.ModelId());
}

TEST(DatabaseCache, DegenerateCamera) {
  DatabaseCache cache;
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1, 1, 1);
  cache.AddCamera(camera);
  EXPECT_EQ(cache.NumCameras(), 1);
  EXPECT_EQ(cache.NumImages(), 0);
  EXPECT_TRUE(cache.ExistsCamera(camera.CameraId()));
  EXPECT_EQ(cache.Camera(camera.CameraId()).MeanFocalLength(), 1);
}

TEST(DatabaseCache, AddImage) {
  DatabaseCache cache;
  Image image;
  image.SetImageId(1);
  image.SetPoints2D(std::vector<Eigen::Vector2d>(10));
  cache.AddImage(image);
  EXPECT_EQ(cache.NumCameras(), 0);
  EXPECT_EQ(cache.NumImages(), 1);
  EXPECT_TRUE(cache.ExistsImage(image.ImageId()));
  EXPECT_EQ(cache.Image(image.ImageId()).NumPoints2D(), image.NumPoints2D());
  EXPECT_TRUE(cache.CorrespondenceGraph().ExistsImage(image.ImageId()));
  EXPECT_EQ(
      cache.CorrespondenceGraph().NumCorrespondencesForImage(image.ImageId()),
      0);
  EXPECT_EQ(
      cache.CorrespondenceGraph().NumObservationsForImage(image.ImageId()), 0);
}

}  // namespace colmap
