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

#define TEST_NAME "base/database"
#include "util/testing.h"

#include <thread>

#include "base/database.h"

using namespace colmap;

const static std::string kMemoryDatabasePath = ":memory:";

BOOST_AUTO_TEST_CASE(TestOpenCloseConstructorDestructor) {
  Database database(kMemoryDatabasePath);
}

BOOST_AUTO_TEST_CASE(TestOpenClose) {
  Database database(kMemoryDatabasePath);
  database.Close();
}

BOOST_AUTO_TEST_CASE(TestTransaction) {
  Database database(kMemoryDatabasePath);
  DatabaseTransaction database_transaction(&database);
}

BOOST_AUTO_TEST_CASE(TestTransactionMultiThreaded) {
  Database database(kMemoryDatabasePath);

  std::thread thread1([&database]() {
    DatabaseTransaction database_transaction(&database);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  });

  std::thread thread2([&database]() {
    DatabaseTransaction database_transaction(&database);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  });

  thread1.join();
  thread2.join();
}

BOOST_AUTO_TEST_CASE(TestEmpty) {
  Database database(kMemoryDatabasePath);
  BOOST_CHECK_EQUAL(database.NumCameras(), 0);
  BOOST_CHECK_EQUAL(database.NumImages(), 0);
  BOOST_CHECK_EQUAL(database.NumKeypoints(), 0);
  BOOST_CHECK_EQUAL(database.MaxNumKeypoints(), 0);
  BOOST_CHECK_EQUAL(database.NumDescriptors(), 0);
  BOOST_CHECK_EQUAL(database.MaxNumDescriptors(), 0);
  BOOST_CHECK_EQUAL(database.NumMatches(), 0);
  BOOST_CHECK_EQUAL(database.NumMatchedImagePairs(), 0);
  BOOST_CHECK_EQUAL(database.NumVerifiedImagePairs(), 0);
}

BOOST_AUTO_TEST_CASE(TestImagePairToPairId) {
  BOOST_CHECK_EQUAL(Database::ImagePairToPairId(0, 0), 0);
  BOOST_CHECK_EQUAL(Database::ImagePairToPairId(0, 1), 1);
  BOOST_CHECK_EQUAL(Database::ImagePairToPairId(0, 2), 2);
  BOOST_CHECK_EQUAL(Database::ImagePairToPairId(0, 3), 3);
  BOOST_CHECK_EQUAL(Database::ImagePairToPairId(1, 2),
                    Database::kMaxNumImages + 2);
  for (image_t i = 0; i < 20; ++i) {
    for (image_t j = 0; j < 20; ++j) {
      const image_pair_t pair_id = Database::ImagePairToPairId(i, j);
      image_t image_id1;
      image_t image_id2;
      Database::PairIdToImagePair(pair_id, &image_id1, &image_id2);
      if (i < j) {
        BOOST_CHECK_EQUAL(i, image_id1);
        BOOST_CHECK_EQUAL(j, image_id2);
      } else {
        BOOST_CHECK_EQUAL(i, image_id2);
        BOOST_CHECK_EQUAL(j, image_id1);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(TestSwapImagePair) {
  BOOST_CHECK(!Database::SwapImagePair(0, 0));
  BOOST_CHECK(!Database::SwapImagePair(0, 1));
  BOOST_CHECK(Database::SwapImagePair(1, 0));
  BOOST_CHECK(!Database::SwapImagePair(1, 1));
}

BOOST_AUTO_TEST_CASE(TestCamera) {
  Database database(kMemoryDatabasePath);
  BOOST_CHECK_EQUAL(database.NumCameras(), 0);
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.SetCameraId(database.WriteCamera(camera));
  BOOST_CHECK_EQUAL(database.NumCameras(), 1);
  BOOST_CHECK_EQUAL(database.ExistsCamera(camera.CameraId()), true);
  BOOST_CHECK_EQUAL(database.ReadCamera(camera.CameraId()).CameraId(),
                    camera.CameraId());
  BOOST_CHECK_EQUAL(database.ReadCamera(camera.CameraId()).ModelId(),
                    camera.ModelId());
  BOOST_CHECK_EQUAL(database.ReadCamera(camera.CameraId()).FocalLength(),
                    camera.FocalLength());
  BOOST_CHECK_EQUAL(database.ReadCamera(camera.CameraId()).PrincipalPointX(),
                    camera.PrincipalPointX());
  BOOST_CHECK_EQUAL(database.ReadCamera(camera.CameraId()).PrincipalPointY(),
                    camera.PrincipalPointY());
  camera.SetFocalLength(2.0);
  database.UpdateCamera(camera);
  BOOST_CHECK_EQUAL(database.ReadCamera(camera.CameraId()).FocalLength(),
                    camera.FocalLength());
  Camera camera2 = camera;
  camera2.SetCameraId(camera.CameraId() + 1);
  database.WriteCamera(camera2, true);
  BOOST_CHECK_EQUAL(database.NumCameras(), 2);
  BOOST_CHECK_EQUAL(database.ExistsCamera(camera.CameraId()), true);
  BOOST_CHECK_EQUAL(database.ExistsCamera(camera2.CameraId()), true);
  BOOST_CHECK_EQUAL(database.ReadAllCameras().size(), 2);
  BOOST_CHECK_EQUAL(database.ReadAllCameras()[0].CameraId(), camera.CameraId());
  BOOST_CHECK_EQUAL(database.ReadAllCameras()[1].CameraId(),
                    camera2.CameraId());
}

BOOST_AUTO_TEST_CASE(TestImage) {
  Database database(kMemoryDatabasePath);
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.SetCameraId(database.WriteCamera(camera));
  BOOST_CHECK_EQUAL(database.NumImages(), 0);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.CameraId());
  image.SetQvecPrior(Eigen::Vector4d(0.1, 0.2, 0.3, 0.4));
  image.SetTvecPrior(Eigen::Vector3d(0.1, 0.2, 0.3));
  image.SetImageId(database.WriteImage(image));
  BOOST_CHECK_EQUAL(database.NumImages(), 1);
  BOOST_CHECK_EQUAL(database.ExistsImage(image.ImageId()), true);
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).ImageId(),
                    image.ImageId());
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).CameraId(),
                    image.CameraId());
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).QvecPrior(0),
                    image.QvecPrior(0));
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).QvecPrior(1),
                    image.QvecPrior(1));
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).QvecPrior(2),
                    image.QvecPrior(2));
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).QvecPrior(3),
                    image.QvecPrior(3));
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).TvecPrior(0),
                    image.TvecPrior(0));
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).TvecPrior(1),
                    image.TvecPrior(1));
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).TvecPrior(2),
                    image.TvecPrior(2));
  image.TvecPrior(0) += 2;
  database.UpdateImage(image);
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).ImageId(),
                    image.ImageId());
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).CameraId(),
                    image.CameraId());
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).QvecPrior(0),
                    image.QvecPrior(0));
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).QvecPrior(1),
                    image.QvecPrior(1));
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).QvecPrior(2),
                    image.QvecPrior(2));
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).QvecPrior(3),
                    image.QvecPrior(3));
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).TvecPrior(0),
                    image.TvecPrior(0));
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).TvecPrior(1),
                    image.TvecPrior(1));
  BOOST_CHECK_EQUAL(database.ReadImage(image.ImageId()).TvecPrior(2),
                    image.TvecPrior(2));
  Image image2 = image;
  image2.SetName("test2");
  image2.SetImageId(image.ImageId() + 1);
  database.WriteImage(image2, true);
  BOOST_CHECK_EQUAL(database.NumImages(), 2);
  BOOST_CHECK_EQUAL(database.ExistsImage(image.ImageId()), true);
  BOOST_CHECK_EQUAL(database.ExistsImage(image2.ImageId()), true);
  BOOST_CHECK_EQUAL(database.ReadAllImages().size(), 2);
}

BOOST_AUTO_TEST_CASE(TestKeypoints) {
  Database database(kMemoryDatabasePath);
  Camera camera;
  camera.SetCameraId(database.WriteCamera(camera));
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.CameraId());
  image.SetImageId(database.WriteImage(image));
  BOOST_CHECK_EQUAL(database.NumKeypoints(), 0);
  BOOST_CHECK_EQUAL(database.NumKeypointsForImage(image.ImageId()), 0);
  const FeatureKeypoints keypoints = FeatureKeypoints(10);
  database.WriteKeypoints(image.ImageId(), keypoints);
  const FeatureKeypoints keypoints_read =
      database.ReadKeypoints(image.ImageId());
  BOOST_CHECK_EQUAL(keypoints.size(), keypoints_read.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    BOOST_CHECK_EQUAL(keypoints[i].x, keypoints_read[i].x);
    BOOST_CHECK_EQUAL(keypoints[i].y, keypoints_read[i].y);
    BOOST_CHECK_EQUAL(keypoints[i].scale, keypoints_read[i].scale);
    BOOST_CHECK_EQUAL(keypoints[i].orientation, keypoints_read[i].orientation);
  }
  BOOST_CHECK_EQUAL(database.NumKeypoints(), 10);
  BOOST_CHECK_EQUAL(database.MaxNumKeypoints(), 10);
  BOOST_CHECK_EQUAL(database.NumKeypointsForImage(image.ImageId()), 10);
  const FeatureKeypoints keypoints2 = FeatureKeypoints(20);
  image.SetName("test2");
  image.SetImageId(database.WriteImage(image));
  database.WriteKeypoints(image.ImageId(), keypoints2);
  BOOST_CHECK_EQUAL(database.NumKeypoints(), 30);
  BOOST_CHECK_EQUAL(database.MaxNumKeypoints(), 20);
  BOOST_CHECK_EQUAL(database.NumKeypointsForImage(image.ImageId()), 20);
}

BOOST_AUTO_TEST_CASE(TestDescriptors) {
  Database database(kMemoryDatabasePath);
  Camera camera;
  camera.SetCameraId(database.WriteCamera(camera));
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.CameraId());
  image.SetImageId(database.WriteImage(image));
  BOOST_CHECK_EQUAL(database.NumDescriptors(), 0);
  BOOST_CHECK_EQUAL(database.NumDescriptorsForImage(image.ImageId()), 0);
  const FeatureDescriptors descriptors = FeatureDescriptors::Random(10, 128);
  database.WriteDescriptors(image.ImageId(), descriptors);
  const FeatureDescriptors descriptors_read =
      database.ReadDescriptors(image.ImageId());
  BOOST_CHECK_EQUAL(descriptors.rows(), descriptors_read.rows());
  BOOST_CHECK_EQUAL(descriptors.cols(), descriptors_read.cols());
  for (FeatureDescriptors::Index r = 0; r < descriptors.rows(); ++r) {
    for (FeatureDescriptors::Index c = 0; c < descriptors.cols(); ++c) {
      BOOST_CHECK_EQUAL(descriptors(r, c), descriptors_read(r, c));
    }
  }
  BOOST_CHECK_EQUAL(database.NumDescriptors(), 10);
  BOOST_CHECK_EQUAL(database.MaxNumDescriptors(), 10);
  BOOST_CHECK_EQUAL(database.NumDescriptorsForImage(image.ImageId()), 10);
  const FeatureDescriptors descriptors2 = FeatureDescriptors(20, 128);
  image.SetName("test2");
  image.SetImageId(database.WriteImage(image));
  database.WriteDescriptors(image.ImageId(), descriptors2);
  BOOST_CHECK_EQUAL(database.NumDescriptors(), 30);
  BOOST_CHECK_EQUAL(database.MaxNumDescriptors(), 20);
  BOOST_CHECK_EQUAL(database.NumDescriptorsForImage(image.ImageId()), 20);
}

BOOST_AUTO_TEST_CASE(TestMatches) {
  Database database(kMemoryDatabasePath);
  const image_t image_id1 = 1;
  const image_t image_id2 = 2;
  const FeatureMatches matches = FeatureMatches(1000);
  database.WriteMatches(image_id1, image_id2, matches);
  const FeatureMatches matches_read =
      database.ReadMatches(image_id1, image_id2);
  BOOST_CHECK_EQUAL(matches.size(), matches_read.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    BOOST_CHECK_EQUAL(matches[i].point2D_idx1, matches_read[i].point2D_idx1);
    BOOST_CHECK_EQUAL(matches[i].point2D_idx2, matches_read[i].point2D_idx2);
  }
  BOOST_CHECK_EQUAL(database.ReadAllMatches().size(), 1);
  BOOST_CHECK_EQUAL(database.ReadAllMatches()[0].first,
                    Database::ImagePairToPairId(image_id1, image_id2));
  BOOST_CHECK_EQUAL(database.NumMatches(), 1000);
  database.DeleteMatches(image_id1, image_id2);
  BOOST_CHECK_EQUAL(database.NumMatches(), 0);
  database.WriteMatches(image_id1, image_id2, matches);
  BOOST_CHECK_EQUAL(database.NumMatches(), 1000);
  database.ClearMatches();
  BOOST_CHECK_EQUAL(database.NumMatches(), 0);
}

BOOST_AUTO_TEST_CASE(TestInlierMatches) {
  Database database(kMemoryDatabasePath);
  const image_t image_id1 = 1;
  const image_t image_id2 = 2;
  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches = FeatureMatches(1000);
  database.WriteInlierMatches(image_id1, image_id2, two_view_geometry);
  const TwoViewGeometry two_view_geometry_read =
      database.ReadInlierMatches(image_id1, image_id2);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(),
                    two_view_geometry_read.inlier_matches.size());
  for (size_t i = 0; i < two_view_geometry_read.inlier_matches.size(); ++i) {
    BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[i].point2D_idx1,
                      two_view_geometry_read.inlier_matches[i].point2D_idx1);
    BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[i].point2D_idx2,
                      two_view_geometry_read.inlier_matches[i].point2D_idx2);
  }
  std::vector<image_pair_t> image_pair_ids;
  std::vector<TwoViewGeometry> two_view_geometries;
  database.ReadAllInlierMatches(&image_pair_ids, &two_view_geometries);
  BOOST_CHECK_EQUAL(image_pair_ids.size(), 1);
  BOOST_CHECK_EQUAL(two_view_geometries.size(), 1);
  BOOST_CHECK_EQUAL(image_pair_ids[0],
                    Database::ImagePairToPairId(image_id1, image_id2));
  std::vector<std::pair<image_t, image_t>> image_pairs;
  std::vector<int> num_inliers;
  database.ReadInlierMatchesGraph(&image_pairs, &num_inliers);
  BOOST_CHECK_EQUAL(image_pairs.size(), 1);
  BOOST_CHECK_EQUAL(num_inliers.size(), 1);
  BOOST_CHECK_EQUAL(image_pairs[0].first, image_id1);
  BOOST_CHECK_EQUAL(image_pairs[0].second, image_id2);
  BOOST_CHECK_EQUAL(num_inliers[0], two_view_geometry.inlier_matches.size());
  BOOST_CHECK_EQUAL(database.NumInlierMatches(), 1000);
  database.DeleteInlierMatches(image_id1, image_id2);
  BOOST_CHECK_EQUAL(database.NumInlierMatches(), 0);
  database.WriteInlierMatches(image_id1, image_id2, two_view_geometry);
  BOOST_CHECK_EQUAL(database.NumInlierMatches(), 1000);
  database.ClearInlierMatches();
  BOOST_CHECK_EQUAL(database.NumInlierMatches(), 0);
}
