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

#define TEST_NAME "base/reconstruction"
#include "util/testing.h"

#include "base/pose.h"
#include "base/reconstruction.h"

using namespace colmap;

void GenerateReconstruction(const image_t num_images,
                            Reconstruction* reconstruction,
                            SceneGraph* scene_graph) {
  const size_t kNumPoints2D = 10;

  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction->AddCamera(camera);

  for (image_t image_id = 1; image_id <= num_images; ++image_id) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(
        std::vector<Eigen::Vector2d>(kNumPoints2D, Eigen::Vector2d::Zero()));
    reconstruction->AddImage(image);
    reconstruction->RegisterImage(image_id);
    scene_graph->AddImage(image_id, kNumPoints2D);
  }

  reconstruction->SetUp(scene_graph);
}

BOOST_AUTO_TEST_CASE(TestEmpty) {
  Reconstruction reconstruction;
  BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), 0);
}

BOOST_AUTO_TEST_CASE(TestAddCamera) {
  Reconstruction reconstruction;
  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1, 1, 1);
  reconstruction.AddCamera(camera);
  BOOST_CHECK(reconstruction.ExistsCamera(camera.CameraId()));
  BOOST_CHECK_EQUAL(reconstruction.Camera(camera.CameraId()).CameraId(),
                    camera.CameraId());
  BOOST_CHECK_EQUAL(reconstruction.Cameras().count(camera.CameraId()), 1);
  BOOST_CHECK_EQUAL(reconstruction.Cameras().size(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), 0);
}

BOOST_AUTO_TEST_CASE(TestAddImage) {
  Reconstruction reconstruction;
  Image image;
  image.SetImageId(1);
  reconstruction.AddImage(image);
  BOOST_CHECK(reconstruction.ExistsImage(1));
  BOOST_CHECK_EQUAL(reconstruction.Image(1).ImageId(), 1);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).IsRegistered(), false);
  BOOST_CHECK_EQUAL(reconstruction.Images().count(1), 1);
  BOOST_CHECK_EQUAL(reconstruction.Images().size(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumImages(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), 0);
}

BOOST_AUTO_TEST_CASE(TestAddPoint3D) {
  Reconstruction reconstruction;
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK(reconstruction.ExistsPoint3D(point3D_id));
  BOOST_CHECK_EQUAL(reconstruction.Point3D(point3D_id).Track().Length(), 0);
  BOOST_CHECK_EQUAL(reconstruction.Points3D().count(point3D_id), 1);
  BOOST_CHECK_EQUAL(reconstruction.Points3D().size(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), 0);
  BOOST_CHECK_EQUAL(reconstruction.Point3DIds().count(point3D_id), 1);
}

BOOST_AUTO_TEST_CASE(TestAddObservation) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(1, &reconstruction, &scene_graph);
  Track track;
  track.AddElement(1, 0);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), track);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumPoints3D(), 1);
  BOOST_CHECK(reconstruction.Image(1).Point2D(0).HasPoint3D());
  BOOST_CHECK(!reconstruction.Image(1).Point2D(1).HasPoint3D());
  BOOST_CHECK_EQUAL(reconstruction.Point3D(point3D_id).Track().Length(), 1);
  reconstruction.AddObservation(point3D_id, TrackElement(1, 1));
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumPoints3D(), 2);
  BOOST_CHECK(reconstruction.Image(1).Point2D(0).HasPoint3D());
  BOOST_CHECK(reconstruction.Image(1).Point2D(1).HasPoint3D());
  BOOST_CHECK_EQUAL(reconstruction.Point3D(point3D_id).Track().Length(), 2);
}

BOOST_AUTO_TEST_CASE(TestMergePoints3D) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(2, &reconstruction, &scene_graph);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 0), Track());
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  reconstruction.Point3D(point3D_id1).Color() =
      Eigen::Matrix<uint8_t, 3, 1>(0, 0, 0);
  const point3D_t point3D_id2 =
      reconstruction.AddPoint3D(Eigen::Vector3d(1, 1, 1), Track());
  reconstruction.AddObservation(point3D_id2, TrackElement(1, 1));
  reconstruction.AddObservation(point3D_id2, TrackElement(2, 1));
  reconstruction.Point3D(point3D_id2).Color() =
      Eigen::Matrix<uint8_t, 3, 1>(20, 20, 20);
  const point3D_t merged_point3D_id =
      reconstruction.MergePoints3D(point3D_id1, point3D_id2);
  BOOST_CHECK(!reconstruction.ExistsPoint3D(point3D_id1));
  BOOST_CHECK(!reconstruction.ExistsPoint3D(point3D_id2));
  BOOST_CHECK(reconstruction.ExistsPoint3D(merged_point3D_id));
  BOOST_CHECK_EQUAL(reconstruction.Image(1).Point2D(0).Point3DId(),
                    merged_point3D_id);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).Point2D(1).Point3DId(),
                    merged_point3D_id);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).Point2D(0).Point3DId(),
                    merged_point3D_id);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).Point2D(1).Point3DId(),
                    merged_point3D_id);
  BOOST_CHECK(reconstruction.Point3D(merged_point3D_id)
                  .XYZ()
                  .isApprox(Eigen::Vector3d(0.5, 0.5, 0.5)));
  BOOST_CHECK_EQUAL(reconstruction.Point3D(merged_point3D_id).Color(),
                    Eigen::Vector3ub(10, 10, 10));
}

BOOST_AUTO_TEST_CASE(TestDeletePoint3D) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(1, &reconstruction, &scene_graph);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id, TrackElement(1, 0));
  reconstruction.DeletePoint3D(point3D_id);
  BOOST_CHECK(!reconstruction.ExistsPoint3D(point3D_id));
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumPoints3D(), 0);
}

BOOST_AUTO_TEST_CASE(TestDeleteObservation) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(2, &reconstruction, &scene_graph);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 0), Track());
  reconstruction.AddObservation(point3D_id, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id, TrackElement(1, 1));
  reconstruction.AddObservation(point3D_id, TrackElement(1, 2));
  reconstruction.DeleteObservation(1, 0);
  BOOST_CHECK_EQUAL(reconstruction.Point3D(point3D_id).Track().Length(), 2);
  BOOST_CHECK(!reconstruction.Image(point3D_id).Point2D(0).HasPoint3D());
  reconstruction.DeleteObservation(1, 1);
  BOOST_CHECK(!reconstruction.ExistsPoint3D(point3D_id));
  BOOST_CHECK(!reconstruction.Image(point3D_id).Point2D(1).HasPoint3D());
  BOOST_CHECK(!reconstruction.Image(point3D_id).Point2D(2).HasPoint3D());
}

BOOST_AUTO_TEST_CASE(TestRegisterImage) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(1, &reconstruction, &scene_graph);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 1);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).IsRegistered(), true);
  BOOST_CHECK(reconstruction.IsImageRegistered(1));
  reconstruction.RegisterImage(1);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 1);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).IsRegistered(), true);
  BOOST_CHECK(reconstruction.IsImageRegistered(1));
  reconstruction.DeRegisterImage(1);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).IsRegistered(), false);
  BOOST_CHECK(!reconstruction.IsImageRegistered(1));
}

BOOST_AUTO_TEST_CASE(TestNormalize) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(3, &reconstruction, &scene_graph);
  reconstruction.Image(1).Tvec(2) = -10.0;
  reconstruction.Image(2).Tvec(2) = 0.0;
  reconstruction.Image(3).Tvec(2) = 10.0;
  reconstruction.DeRegisterImage(1);
  reconstruction.DeRegisterImage(2);
  reconstruction.DeRegisterImage(3);
  reconstruction.Normalize();
  BOOST_CHECK_LT(std::abs(reconstruction.Image(1).Tvec(2) + 10), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(2).Tvec(2)), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(3).Tvec(2) - 10), 1e-6);
  reconstruction.RegisterImage(1);
  reconstruction.RegisterImage(2);
  reconstruction.RegisterImage(3);
  reconstruction.Normalize();
  BOOST_CHECK_LT(std::abs(reconstruction.Image(1).Tvec(2) + 5), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(2).Tvec(2)), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(3).Tvec(2) - 5), 1e-6);
  reconstruction.Normalize(5);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(1).Tvec(2) + 2.5), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(2).Tvec(2)), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(3).Tvec(2) - 2.5), 1e-6);
  reconstruction.Normalize(10, 0.0, 1.0);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(1).Tvec(2) + 5), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(2).Tvec(2)), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(3).Tvec(2) - 5), 1e-6);
  reconstruction.Normalize(20);
  Image image;
  image.SetImageId(4);
  reconstruction.AddImage(image);
  reconstruction.RegisterImage(4);
  image.SetImageId(5);
  reconstruction.AddImage(image);
  reconstruction.RegisterImage(5);
  image.SetImageId(6);
  reconstruction.AddImage(image);
  reconstruction.RegisterImage(6);
  image.SetImageId(7);
  reconstruction.AddImage(image);
  reconstruction.RegisterImage(7);
  reconstruction.Image(4).Tvec(2) = -7.5;
  reconstruction.Image(5).Tvec(2) = -5.0;
  reconstruction.Image(6).Tvec(2) = 5.0;
  reconstruction.Image(7).Tvec(2) = 7.5;
  reconstruction.RegisterImage(7);
  reconstruction.Normalize(10, 0.0, 1.0);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(1).Tvec(2) + 5), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(2).Tvec(2)), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(3).Tvec(2) - 5), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(4).Tvec(2) + 3.75), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(5).Tvec(2) + 2.5), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(6).Tvec(2) - 2.5), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(7).Tvec(2) - 3.75), 1e-6);
}

BOOST_AUTO_TEST_CASE(TestTransform) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(3, &reconstruction, &scene_graph);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(1, 1, 1), Track());
  reconstruction.AddObservation(point3D_id, TrackElement(1, 1));
  reconstruction.AddObservation(point3D_id, TrackElement(2, 1));
  reconstruction.Transform(2, ComposeIdentityQuaternion(),
                           Eigen::Vector3d(0, 1, 2));
  BOOST_CHECK_EQUAL(reconstruction.Image(1).ProjectionCenter(),
                    Eigen::Vector3d(0, 1, 2));
  BOOST_CHECK_EQUAL(reconstruction.Point3D(point3D_id).XYZ(),
                    Eigen::Vector3d(2, 3, 4));
}

BOOST_AUTO_TEST_CASE(TestFindImageWithName) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(2, &reconstruction, &scene_graph);
  BOOST_CHECK_EQUAL(reconstruction.FindImageWithName("image1"),
                    &reconstruction.Image(1));
  BOOST_CHECK_EQUAL(reconstruction.FindImageWithName("image2"),
                    &reconstruction.Image(2));
  BOOST_CHECK(reconstruction.FindImageWithName("image3") == nullptr);
}

BOOST_AUTO_TEST_CASE(TestFilterPoints3D) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(2, &reconstruction, &scene_graph);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(0.0, 0.0, std::unordered_set<point3D_t>{});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(0.0, 0.0,
                                std::unordered_set<point3D_t>{point3D_id1 + 1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(0.0, 0.0,
                                std::unordered_set<point3D_t>{point3D_id1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id2 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id2, TrackElement(1, 0));
  reconstruction.FilterPoints3D(0.0, 0.0,
                                std::unordered_set<point3D_t>{point3D_id2});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id3 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.5, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id3, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id3, TrackElement(2, 0));
  reconstruction.FilterPoints3D(0.0, 0.0,
                                std::unordered_set<point3D_t>{point3D_id3});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(0.0, 1e-3,
                                std::unordered_set<point3D_t>{point3D_id3});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id4 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.6, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id4, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id4, TrackElement(2, 0));
  reconstruction.FilterPoints3D(0.1, 0.0,
                                std::unordered_set<point3D_t>{point3D_id4});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(0.09, 0.0,
                                std::unordered_set<point3D_t>{point3D_id4});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
}

BOOST_AUTO_TEST_CASE(TestFilterPoints3DInImages) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(2, &reconstruction, &scene_graph);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(0.0, 0.0,
                                        std::unordered_set<image_t>{});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(0.0, 0.0,
                                        std::unordered_set<image_t>{1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.FilterPoints3DInImages(0.0, 0.0,
                                        std::unordered_set<image_t>{2});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(0.0, 0.0,
                                        std::unordered_set<image_t>{1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id3 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.5, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id3, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id3, TrackElement(2, 0));
  reconstruction.FilterPoints3DInImages(0.0, 0.0,
                                        std::unordered_set<image_t>{1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(0.0, 1e-3,
                                        std::unordered_set<image_t>{1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id4 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.6, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id4, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id4, TrackElement(2, 0));
  reconstruction.FilterPoints3DInImages(0.1, 0.0,
                                        std::unordered_set<image_t>{1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(0.09, 0.0,
                                        std::unordered_set<image_t>{1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
}

BOOST_AUTO_TEST_CASE(TestFilterAllPoints) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(2, &reconstruction, &scene_graph);
  reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterAllPoints3D(0.0, 0.0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id2 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id2, TrackElement(1, 0));
  reconstruction.FilterAllPoints3D(0.0, 0.0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id3 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.5, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id3, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id3, TrackElement(2, 0));
  reconstruction.FilterAllPoints3D(0.0, 0.0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterAllPoints3D(0.0, 1e-3);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id4 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.6, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id4, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id4, TrackElement(2, 0));
  reconstruction.FilterAllPoints3D(0.1, 0.0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterAllPoints3D(0.09, 0.0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
}

BOOST_AUTO_TEST_CASE(TestFilterObservationsWithNegativeDepth) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(2, &reconstruction, &scene_graph);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 1), Track());
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterObservationsWithNegativeDepth();
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.Point3D(point3D_id1).XYZ(2) = 0.001;
  reconstruction.FilterObservationsWithNegativeDepth();
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.Point3D(point3D_id1).XYZ(2) = 0.0;
  reconstruction.FilterObservationsWithNegativeDepth();
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.Point3D(point3D_id1).XYZ(2) = 0.001;
  reconstruction.FilterObservationsWithNegativeDepth();
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.Point3D(point3D_id1).XYZ(2) = 0.0;
  reconstruction.FilterObservationsWithNegativeDepth();
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
}

BOOST_AUTO_TEST_CASE(TestFilterImages) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(4, &reconstruction, &scene_graph);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  reconstruction.AddObservation(point3D_id1, TrackElement(3, 0));
  reconstruction.FilterImages(0.0, 10.0, 1.0);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 3);
  reconstruction.DeleteObservation(3, 0);
  reconstruction.FilterImages(0.0, 10.0, 1.0);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 2);
  reconstruction.FilterImages(0.0, 0.9, 1.0);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 0);
}

BOOST_AUTO_TEST_CASE(TestComputeNumObservations) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(2, &reconstruction, &scene_graph);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK_EQUAL(reconstruction.ComputeNumObservations(), 0);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  BOOST_CHECK_EQUAL(reconstruction.ComputeNumObservations(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 1));
  BOOST_CHECK_EQUAL(reconstruction.ComputeNumObservations(), 2);
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  BOOST_CHECK_EQUAL(reconstruction.ComputeNumObservations(), 3);
}

BOOST_AUTO_TEST_CASE(TestComputeMeanTrackLength) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(2, &reconstruction, &scene_graph);
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanTrackLength(), 0);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanTrackLength(), 0);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanTrackLength(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 1));
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanTrackLength(), 2);
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanTrackLength(), 3);
}

BOOST_AUTO_TEST_CASE(TestComputeMeanObservationsPerRegImage) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(2, &reconstruction, &scene_graph);
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanObservationsPerRegImage(), 0);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanObservationsPerRegImage(), 0);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanObservationsPerRegImage(), 0.5);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 1));
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanObservationsPerRegImage(), 1.0);
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanObservationsPerRegImage(), 1.5);
}

BOOST_AUTO_TEST_CASE(TestComputeMeanReprojectionError) {
  Reconstruction reconstruction;
  SceneGraph scene_graph;
  GenerateReconstruction(2, &reconstruction, &scene_graph);
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanReprojectionError(), 0);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanReprojectionError(), 0);
  reconstruction.Point3D(point3D_id1).SetError(0.0);
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanReprojectionError(), 0);
  reconstruction.Point3D(point3D_id1).SetError(1.0);
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanReprojectionError(), 1);
  reconstruction.Point3D(point3D_id1).SetError(2.0);
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanReprojectionError(), 2.0);
}
