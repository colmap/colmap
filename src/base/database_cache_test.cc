// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#define TEST_NAME "base/database_cache"
#include "util/testing.h"

#include "base/database_cache.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  DatabaseCache cache;
  BOOST_CHECK_EQUAL(cache.NumCameras(), 0);
  BOOST_CHECK_EQUAL(cache.NumImages(), 0);
}

BOOST_AUTO_TEST_CASE(TestAddCamera) {
  DatabaseCache cache;
  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1, 1, 1);
  cache.AddCamera(camera);
  BOOST_CHECK_EQUAL(cache.NumCameras(), 1);
  BOOST_CHECK_EQUAL(cache.NumImages(), 0);
  BOOST_CHECK(cache.ExistsCamera(camera.CameraId()));
  BOOST_CHECK_EQUAL(cache.Camera(camera.CameraId()).ModelId(),
                    camera.ModelId());
}

BOOST_AUTO_TEST_CASE(TestDegenerateCamera) {
  DatabaseCache cache;
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1, 1, 1);
  cache.AddCamera(camera);
  BOOST_CHECK_EQUAL(cache.NumCameras(), 1);
  BOOST_CHECK_EQUAL(cache.NumImages(), 0);
  BOOST_CHECK(cache.ExistsCamera(camera.CameraId()));
  BOOST_CHECK_EQUAL(cache.Camera(camera.CameraId()).MeanFocalLength(), 1);
}

BOOST_AUTO_TEST_CASE(TestAddImage) {
  DatabaseCache cache;
  Image image;
  image.SetImageId(1);
  image.SetPoints2D(std::vector<Eigen::Vector2d>(10));
  cache.AddImage(image);
  BOOST_CHECK_EQUAL(cache.NumCameras(), 0);
  BOOST_CHECK_EQUAL(cache.NumImages(), 1);
  BOOST_CHECK(cache.ExistsImage(image.ImageId()));
  BOOST_CHECK_EQUAL(cache.Image(image.ImageId()).NumPoints2D(),
                    image.NumPoints2D());
  BOOST_CHECK(cache.CorrespondenceGraph().ExistsImage(image.ImageId()));
  BOOST_CHECK_EQUAL(
      cache.CorrespondenceGraph().NumCorrespondencesForImage(image.ImageId()),
      0);
  BOOST_CHECK_EQUAL(
      cache.CorrespondenceGraph().NumObservationsForImage(image.ImageId()), 0);
}
