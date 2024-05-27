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

#include "colmap/sfm/observation_manager.h"

#include <memory>

#include <gtest/gtest.h>

namespace colmap {
namespace {

void GenerateReconstruction(const image_t num_images,
                            Reconstruction& reconstruction) {
  const size_t kNumPoints2D = 10;

  Camera camera = Camera::CreateFromModelName(1, "PINHOLE", 1, 1, 1);
  reconstruction.AddCamera(camera);

  for (image_t image_id = 1; image_id <= num_images; ++image_id) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(camera.camera_id);
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(
        std::vector<Eigen::Vector2d>(kNumPoints2D, Eigen::Vector2d::Zero()));
    image.SetRegistered(true);
    reconstruction.AddImage(std::move(image));
  }
}

TEST(ObservationManager, FilterPoints3D) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, reconstruction);
  ObservationManager obs_manager(reconstruction);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  obs_manager.FilterPoints3D(0.0, 0.0, std::unordered_set<point3D_t>{});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  obs_manager.FilterPoints3D(
      0.0, 0.0, std::unordered_set<point3D_t>{point3D_id1 + 1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  obs_manager.FilterPoints3D(
      0.0, 0.0, std::unordered_set<point3D_t>{point3D_id1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id2 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id2, TrackElement(1, 0));
  obs_manager.FilterPoints3D(
      0.0, 0.0, std::unordered_set<point3D_t>{point3D_id2});
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id3 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.5, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id3, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id3, TrackElement(2, 0));
  obs_manager.FilterPoints3D(
      0.0, 0.0, std::unordered_set<point3D_t>{point3D_id3});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  obs_manager.FilterPoints3D(
      0.0, 1e-3, std::unordered_set<point3D_t>{point3D_id3});
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id4 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.6, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id4, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id4, TrackElement(2, 0));
  obs_manager.FilterPoints3D(
      0.1, 0.0, std::unordered_set<point3D_t>{point3D_id4});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  obs_manager.FilterPoints3D(
      0.09, 0.0, std::unordered_set<point3D_t>{point3D_id4});
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

TEST(ObservationManager, FilterPoints3DInImages) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, reconstruction);
  ObservationManager obs_manager(reconstruction);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  obs_manager.FilterPoints3DInImages(0.0, 0.0, std::unordered_set<image_t>{});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  obs_manager.FilterPoints3DInImages(0.0, 0.0, std::unordered_set<image_t>{1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  obs_manager.FilterPoints3DInImages(0.0, 0.0, std::unordered_set<image_t>{2});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  obs_manager.FilterPoints3DInImages(0.0, 0.0, std::unordered_set<image_t>{1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id3 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.5, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id3, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id3, TrackElement(2, 0));
  obs_manager.FilterPoints3DInImages(0.0, 0.0, std::unordered_set<image_t>{1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  obs_manager.FilterPoints3DInImages(0.0, 1e-3, std::unordered_set<image_t>{1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id4 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.6, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id4, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id4, TrackElement(2, 0));
  obs_manager.FilterPoints3DInImages(0.1, 0.0, std::unordered_set<image_t>{1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  obs_manager.FilterPoints3DInImages(0.09, 0.0, std::unordered_set<image_t>{1});
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

TEST(ObservationManager, FilterAllPoints) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, reconstruction);
  ObservationManager obs_manager(reconstruction);
  reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  obs_manager.FilterAllPoints3D(0.0, 0.0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id2 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id2, TrackElement(1, 0));
  obs_manager.FilterAllPoints3D(0.0, 0.0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id3 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.5, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id3, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id3, TrackElement(2, 0));
  obs_manager.FilterAllPoints3D(0.0, 0.0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  obs_manager.FilterAllPoints3D(0.0, 1e-3);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id4 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.6, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id4, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id4, TrackElement(2, 0));
  obs_manager.FilterAllPoints3D(0.1, 0.0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  obs_manager.FilterAllPoints3D(0.09, 0.0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

TEST(ObservationManager, FilterObservationsWithNegativeDepth) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, reconstruction);
  ObservationManager obs_manager(reconstruction);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 1), Track());
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  obs_manager.FilterObservationsWithNegativeDepth();
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.Point3D(point3D_id1).xyz(2) = 0.001;
  obs_manager.FilterObservationsWithNegativeDepth();
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.Point3D(point3D_id1).xyz(2) = 0.0;
  obs_manager.FilterObservationsWithNegativeDepth();
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.Point3D(point3D_id1).xyz(2) = 0.001;
  obs_manager.FilterObservationsWithNegativeDepth();
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  reconstruction.Point3D(point3D_id1).xyz(2) = 0.0;
  obs_manager.FilterObservationsWithNegativeDepth();
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

TEST(ObservationManager, FilterImages) {
  Reconstruction reconstruction;
  GenerateReconstruction(4, reconstruction);
  ObservationManager obs_manager(reconstruction);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  reconstruction.AddObservation(point3D_id1, TrackElement(3, 0));
  obs_manager.FilterImages(0.0, 10.0, 1.0);
  EXPECT_EQ(reconstruction.NumRegImages(), 3);
  reconstruction.DeleteObservation(3, 0);
  obs_manager.FilterImages(0.0, 10.0, 1.0);
  EXPECT_EQ(reconstruction.NumRegImages(), 2);
  obs_manager.FilterImages(0.0, 0.9, 1.0);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
}

TEST(ObservationManager, NumVisiblePoints3D) {
  Reconstruction reconstruction;
  const image_t kImageId1 = 1;
  const image_t kImageId2 = 2;
  const camera_t kCameraId = 1;
  const Camera camera = Camera::CreateFromModelId(kCameraId,
                                                  CameraModelId::kPinhole,
                                                  /*focal_length=*/10,
                                                  /*width=*/10,
                                                  /*height=*/10);
  reconstruction.AddCamera(camera);
  Image image;
  image.SetImageId(kImageId1);
  image.SetCameraId(kCameraId);
  image.SetPoints2D(std::vector<Eigen::Vector2d>(10));
  reconstruction.AddImage(image);
  image.SetImageId(kImageId2);
  reconstruction.AddImage(image);
  auto correspondence_graph = std::make_shared<CorrespondenceGraph>();
  correspondence_graph->AddImage(kImageId1, 10);
  correspondence_graph->AddImage(kImageId2, 10);
  FeatureMatches matches;
  for (size_t i = 0; i < 10; ++i) {
    matches.emplace_back(i, i);
  }
  correspondence_graph->AddCorrespondences(kImageId1, kImageId2, matches);
  correspondence_graph->Finalize();
  ObservationManager obs_manager(reconstruction, correspondence_graph);

  EXPECT_EQ(obs_manager.NumObservations(kImageId1), 10);
  EXPECT_EQ(obs_manager.NumCorrespondences(kImageId1), 10);

  EXPECT_EQ(obs_manager.NumVisiblePoints3D(kImageId1), 0);
  obs_manager.IncrementCorrespondenceHasPoint3D(kImageId1, 0);
  EXPECT_EQ(obs_manager.NumVisiblePoints3D(kImageId1), 1);
  obs_manager.IncrementCorrespondenceHasPoint3D(kImageId1, 0);
  obs_manager.IncrementCorrespondenceHasPoint3D(kImageId1, 1);
  EXPECT_EQ(obs_manager.NumVisiblePoints3D(kImageId1), 2);
  obs_manager.DecrementCorrespondenceHasPoint3D(kImageId1, 0);
  EXPECT_EQ(obs_manager.NumVisiblePoints3D(kImageId1), 2);
  obs_manager.DecrementCorrespondenceHasPoint3D(kImageId1, 0);
  EXPECT_EQ(obs_manager.NumVisiblePoints3D(kImageId1), 1);
  obs_manager.DecrementCorrespondenceHasPoint3D(kImageId1, 1);
  EXPECT_EQ(obs_manager.NumVisiblePoints3D(kImageId1), 0);
}

TEST(ObservationManager, Point3DVisibilityScore) {
  Reconstruction reconstruction;
  const image_t kImageId1 = 1;
  const camera_t kCameraId = 1;
  const image_t kImageId2 = 2;
  const Camera camera = Camera::CreateFromModelId(kCameraId,
                                                  CameraModelId::kPinhole,
                                                  /*focal_length=*/4,
                                                  /*width=*/4,
                                                  /*height=*/4);
  reconstruction.AddCamera(camera);
  Image image;
  image.SetImageId(kImageId1);
  image.SetCameraId(kCameraId);
  std::vector<Eigen::Vector2d> points2D;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      points2D.emplace_back(i, j);
    }
  }
  image.SetPoints2D(points2D);
  reconstruction.AddImage(image);
  image.SetImageId(kImageId2);
  reconstruction.AddImage(image);

  auto correspondence_graph = std::make_shared<CorrespondenceGraph>();
  correspondence_graph->AddImage(kImageId1, 16);
  correspondence_graph->AddImage(kImageId2, 16);
  FeatureMatches matches;
  for (size_t i = 0; i < 16; ++i) {
    matches.emplace_back(i, i);
  }
  correspondence_graph->AddCorrespondences(kImageId1, kImageId2, matches);
  correspondence_graph->Finalize();
  ObservationManager obs_manager(reconstruction, correspondence_graph);

  EXPECT_EQ(obs_manager.NumObservations(kImageId1), 16);
  EXPECT_EQ(obs_manager.NumCorrespondences(kImageId1), 16);

  Eigen::Matrix<size_t, Eigen::Dynamic, 1> scores(
      ObservationManager::kNumPoint3DVisibilityPyramidLevels, 1);
  for (int i = 1; i <= ObservationManager::kNumPoint3DVisibilityPyramidLevels;
       ++i) {
    scores(i - 1) = (1 << i) * (1 << i);
  }
  EXPECT_EQ(obs_manager.Point3DVisibilityScore(kImageId1), 0);
  obs_manager.IncrementCorrespondenceHasPoint3D(kImageId1, 0);
  EXPECT_EQ(obs_manager.Point3DVisibilityScore(kImageId1), scores.sum());
  obs_manager.IncrementCorrespondenceHasPoint3D(kImageId1, 0);
  EXPECT_EQ(obs_manager.Point3DVisibilityScore(kImageId1), scores.sum());
  obs_manager.IncrementCorrespondenceHasPoint3D(kImageId1, 1);
  EXPECT_EQ(obs_manager.Point3DVisibilityScore(kImageId1),
            scores.sum() + scores.bottomRows(scores.size() - 1).sum());
  obs_manager.IncrementCorrespondenceHasPoint3D(kImageId1, 1);
  obs_manager.IncrementCorrespondenceHasPoint3D(kImageId1, 1);
  obs_manager.IncrementCorrespondenceHasPoint3D(kImageId1, 4);
  EXPECT_EQ(obs_manager.Point3DVisibilityScore(kImageId1),
            scores.sum() + 2 * scores.bottomRows(scores.size() - 1).sum());
  obs_manager.IncrementCorrespondenceHasPoint3D(kImageId1, 4);
  obs_manager.IncrementCorrespondenceHasPoint3D(kImageId1, 5);
  EXPECT_EQ(obs_manager.Point3DVisibilityScore(kImageId1),
            scores.sum() + 3 * scores.bottomRows(scores.size() - 1).sum());
  obs_manager.DecrementCorrespondenceHasPoint3D(kImageId1, 0);
  EXPECT_EQ(obs_manager.Point3DVisibilityScore(kImageId1),
            scores.sum() + 3 * scores.bottomRows(scores.size() - 1).sum());
  obs_manager.DecrementCorrespondenceHasPoint3D(kImageId1, 0);
  EXPECT_EQ(obs_manager.Point3DVisibilityScore(kImageId1),
            scores.sum() + 2 * scores.bottomRows(scores.size() - 1).sum());
  obs_manager.IncrementCorrespondenceHasPoint3D(kImageId1, 2);
  EXPECT_EQ(obs_manager.Point3DVisibilityScore(kImageId1),
            2 * scores.sum() + 2 * scores.bottomRows(scores.size() - 1).sum());
}

}  // namespace
}  // namespace colmap
