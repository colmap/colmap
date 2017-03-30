// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
  BOOST_CHECK(cache.SceneGraph().ExistsImage(image.ImageId()));
  BOOST_CHECK_EQUAL(
      cache.SceneGraph().NumCorrespondencesForImage(image.ImageId()), 0);
  BOOST_CHECK_EQUAL(cache.SceneGraph().NumObservationsForImage(image.ImageId()),
                    0);
}
