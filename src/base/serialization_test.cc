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

#define TEST_NAME "base/serialization"

#include <Eigen/Core>
#include <boost/mpl/list.hpp>

#include "base/serialization.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"
#include "util/testing.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(WriteReadToInvalidBinaryFile) {
  constexpr const char* kInvalidFilePath = "invalid/path/for/testing";
  int data = 42;
  BOOST_CHECK_THROW(WriteToBinaryFile(kInvalidFilePath, data),
                    yas::io_exception);
  BOOST_CHECK_THROW(ReadFromBinaryFile(kInvalidFilePath, &data),
                    yas::io_exception);
}

BOOST_AUTO_TEST_CASE(WriteReadToValidBinaryFile) {
  std::string temporary_filename = boost::filesystem::unique_path().string();
  const int data = 42;

  // Using a valid file should not throw.
  BOOST_CHECK_NO_THROW(WriteToBinaryFile(temporary_filename, data));
  int dataread;
  BOOST_CHECK_NO_THROW(ReadFromBinaryFile(temporary_filename, &dataread));
  BOOST_CHECK_EQUAL(data, dataread);
}

template <typename T>
void WriteAndReadBinaryDataFromBuffer(const T& data, T* data_readback) {
  yas::mem_ostream output_stream;
  internal::WriteToBinaryStream(output_stream, data);
  internal::ReadFromBinaryStream(
      yas::mem_istream(output_stream.get_intrusive_buffer()), data_readback);
}

// Serialization of matrices is more flexible than user-defined types (e.g. one
// can store a (3, 4) block and read it out into a regular dynamic Matrix).
// Because of this, we provide a helper function to reflect this flexibility.
template <typename T, typename U>
void WriteAndReadBinaryDataFromBuffer(const Eigen::MatrixBase<T>& data,
                                      Eigen::MatrixBase<U>* data_readback) {
  yas::mem_ostream output_stream;
  internal::WriteToBinaryStream(output_stream, data);
  internal::ReadFromBinaryStream(
      yas::mem_istream(output_stream.get_intrusive_buffer()), data_readback);
}

template <typename Scalar, int Rows, int Cols>
void EigenStaticMatrixTest() {
  Eigen::Matrix<Scalar, Rows, Cols> matrix, matrix_readback;
  matrix.setRandom();
  WriteAndReadBinaryDataFromBuffer(matrix, &matrix_readback);
  BOOST_CHECK(matrix.isApprox(matrix_readback));
  matrix *= Scalar(4);
  WriteAndReadBinaryDataFromBuffer(matrix, &matrix_readback);
  BOOST_CHECK(matrix.isApprox(matrix_readback));
  // Accept expressions.
  WriteAndReadBinaryDataFromBuffer(matrix * Scalar(7), &matrix_readback);
  matrix *= Scalar(7);
  BOOST_CHECK(matrix.isApprox(matrix_readback));
}

typedef boost::mpl::list<float, double, int, uint8_t> ScalarTypes;
BOOST_AUTO_TEST_CASE_TEMPLATE(EigenStaticMatrices, Scalar, ScalarTypes) {
  EigenStaticMatrixTest<Scalar, 2, 2>();
  EigenStaticMatrixTest<Scalar, 2, 4>();
  EigenStaticMatrixTest<Scalar, 4, 2>();
  EigenStaticMatrixTest<Scalar, 5, 5>();
  EigenStaticMatrixTest<Scalar, 1, 15>();
  EigenStaticMatrixTest<Scalar, 14, 2>();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(
    EigenStaticExpressions_WriteExpressionReadMatrix, Scalar, ScalarTypes) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix;
  matrix.resize(50, 50);
  matrix.setRandom();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix_readback;
  WriteAndReadBinaryDataFromBuffer(matrix.template block<3, 4>(0, 0),
                                   &matrix_readback);
  BOOST_CHECK_EQUAL(matrix_readback.rows(), 3);
  BOOST_CHECK_EQUAL(matrix_readback.cols(), 4);
  const auto block = matrix.template block<3, 4>(0, 0);
  BOOST_CHECK(block.isApprox(matrix_readback));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(
    EigenStaticExpressions_WriteExpressionReadExpression, Scalar, ScalarTypes) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix;
  matrix.resize(50, 50);
  matrix.setRandom();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix_readback;
  // In this version, we can write and read using Eigen expressions, but the
  // underlying memory must be pre-allocated.
  matrix_readback.resize(10, 10);
  auto read_block = matrix_readback.template block<3, 4>(0, 0);
  const auto block = matrix.template block<3, 4>(0, 0);
  WriteAndReadBinaryDataFromBuffer(block, &read_block);
  BOOST_CHECK_EQUAL(read_block.rows(), 3);
  BOOST_CHECK_EQUAL(read_block.cols(), 4);
  BOOST_CHECK(block.isApprox(read_block));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(EigenDynamicMatrices, Scalar, ScalarTypes) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix, matrix_readback;
  matrix.resize(50, 50);
  matrix.setRandom();
  WriteAndReadBinaryDataFromBuffer(matrix, &matrix_readback);
  BOOST_CHECK(matrix.isApprox(matrix_readback));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(
    EigenDynamicExpressions_WriteExpressionReadMatrix, Scalar, ScalarTypes) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix;
  matrix.resize(50, 50);
  matrix.setRandom();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix_readback;
  WriteAndReadBinaryDataFromBuffer(matrix.block(0, 0, 3, 4),
                                   &matrix_readback);
  BOOST_CHECK_EQUAL(matrix_readback.rows(), 3);
  BOOST_CHECK_EQUAL(matrix_readback.cols(), 4);
  const auto block = matrix.block(0, 0, 3, 4);
  BOOST_CHECK(block.isApprox(matrix_readback));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(
    EigenDynamicExpressions_WriteExpressionReadExpression, Scalar, ScalarTypes) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix;
  matrix.resize(50, 50);
  matrix.setRandom();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix_readback;
  matrix_readback.resize(10, 10);
  auto read_block = matrix_readback.block(0, 0, 3, 4);
  const auto block = matrix.block(0, 0, 3, 4);
  WriteAndReadBinaryDataFromBuffer(block, &read_block);
  BOOST_CHECK_EQUAL(read_block.rows(), 3);
  BOOST_CHECK_EQUAL(read_block.cols(), 4);
  BOOST_CHECK(block.isApprox(read_block));
}

BOOST_AUTO_TEST_CASE(TestPoint2D) {
  Point2D point2D, point2D_readback;
  point2D.SetXY({0.1, -2.4});
  point2D.SetPoint3DId(1234);
  WriteAndReadBinaryDataFromBuffer(point2D, &point2D_readback);
  BOOST_CHECK(point2D.XY().isApprox(point2D_readback.XY()));
  BOOST_CHECK_EQUAL(point2D.Point3DId(), point2D_readback.Point3DId());
}

BOOST_AUTO_TEST_CASE(TestTrackElement) {
  TrackElement element, element_readback;
  element.image_id = 123;
  element.point2D_idx = 321;
  WriteAndReadBinaryDataFromBuffer(element, &element_readback);
  BOOST_CHECK_EQUAL(element.image_id, element_readback.image_id);
  BOOST_CHECK_EQUAL(element.point2D_idx, element_readback.point2D_idx);
}

BOOST_AUTO_TEST_CASE(TestTrack) {
  Track track, track_readback;
  track.SetElements({{123, 321}, {4242, 2424}, {9939, 4839}});
  WriteAndReadBinaryDataFromBuffer(track, &track_readback);
  BOOST_CHECK_EQUAL(track.Length(), track_readback.Length());
  for (size_t i = 0u; i < track.Length(); ++i) {
    BOOST_CHECK_EQUAL(track.Element(i).image_id,
                      track_readback.Element(i).image_id);
    BOOST_CHECK_EQUAL(track.Element(i).point2D_idx,
                      track_readback.Element(i).point2D_idx);
  }
}

BOOST_AUTO_TEST_CASE(TestPoint3D) {
  Point3D point3D, point3D_readback;
  point3D.XYZ() << 1., 12., 34.;
  point3D.Color() << 124, 99, 250;
  point3D.SetError(1e-3);
  // point3D.Track() -> empty, since it was already tested above.
  WriteAndReadBinaryDataFromBuffer(point3D, &point3D_readback);
  BOOST_CHECK_EQUAL(point3D.XYZ(), point3D_readback.XYZ());
  BOOST_CHECK_EQUAL(point3D.Color(), point3D_readback.Color());
  BOOST_CHECK_EQUAL(point3D.Error(), point3D_readback.Error());
}

BOOST_AUTO_TEST_CASE(TestCamera) {
  Camera camera, camera_readback;
  camera.SetCameraId(1912);
  camera.SetModelId(2);
  camera.SetWidth(800);
  camera.SetHeight(600);
  camera.SetParams({1., 2., 3., 4.});
  camera.SetPriorFocalLength(true);
  WriteAndReadBinaryDataFromBuffer(camera, &camera_readback);
  BOOST_CHECK_EQUAL(camera.CameraId(), camera_readback.CameraId());
  BOOST_CHECK_EQUAL(camera.ModelId(), camera_readback.ModelId());
  BOOST_CHECK_EQUAL(camera.Width(), camera_readback.Width());
  BOOST_CHECK_EQUAL(camera.Height(), camera_readback.Height());
  BOOST_CHECK_EQUAL(camera.NumParams(), camera_readback.NumParams());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      camera.Params().begin(), camera.Params().end(),
      camera_readback.Params().begin(), camera_readback.Params().end());
  BOOST_CHECK_EQUAL(camera.HasPriorFocalLength(),
                    camera_readback.HasPriorFocalLength());
}

BOOST_AUTO_TEST_CASE(TestVisibilityPyramid) {
  VisibilityPyramid pyramid(8u, 100u, 200u);
  VisibilityPyramid pyramid_readback;
  WriteAndReadBinaryDataFromBuffer(pyramid, &pyramid_readback);
  BOOST_CHECK_EQUAL(pyramid.NumLevels(), pyramid_readback.NumLevels());
  BOOST_CHECK_EQUAL(pyramid.Width(), pyramid_readback.Width());
  BOOST_CHECK_EQUAL(pyramid.Height(), pyramid_readback.Height());
  BOOST_CHECK_EQUAL(pyramid.MaxScore(), pyramid_readback.MaxScore());
  BOOST_CHECK_EQUAL(pyramid.Score(), pyramid_readback.Score());
  pyramid.SetPoint(0, 0);
  pyramid_readback.SetPoint(0, 0);
  BOOST_CHECK_EQUAL(pyramid.Score(), pyramid_readback.Score());
}

void CheckImagesEqual(const Image& image_a, const Image& image_b) {
  BOOST_CHECK_EQUAL(image_a.ImageId(), image_b.ImageId());
  BOOST_CHECK_EQUAL(image_a.Name(), image_b.Name());
  BOOST_CHECK_EQUAL(image_a.CameraId(), image_b.CameraId());
  BOOST_CHECK_EQUAL(image_a.IsRegistered(), image_b.IsRegistered());
  BOOST_CHECK(image_a.Tvec().isApprox(image_b.Tvec()));
  BOOST_CHECK(image_a.Qvec().isApprox(image_b.Qvec()));
  BOOST_CHECK_EQUAL(image_a.HasQvecPrior(), image_b.HasQvecPrior());
  if (image_a.HasQvecPrior()) {
    BOOST_CHECK(image_a.QvecPrior().isApprox(image_b.QvecPrior()));
  }
  BOOST_CHECK_EQUAL(image_a.HasTvecPrior(), image_b.HasTvecPrior());
  if (image_a.HasTvecPrior()) {
    BOOST_CHECK(image_a.TvecPrior().isApprox(image_b.TvecPrior()));
  }
  BOOST_CHECK_EQUAL(image_a.NumObservations(), image_b.NumObservations());
  BOOST_CHECK_EQUAL(image_a.NumCorrespondences(),
                    image_b.NumCorrespondences());
  BOOST_CHECK_EQUAL(image_a.NumPoints2D(), image_b.NumPoints2D());
  BOOST_CHECK_EQUAL(image_a.NumPoints3D(), image_b.NumPoints3D());
  BOOST_CHECK_EQUAL(image_a.NumVisiblePoints3D(),
                    image_b.NumVisiblePoints3D());
  BOOST_CHECK_EQUAL(image_a.Point3DVisibilityScore(),
                    image_b.Point3DVisibilityScore());
}

BOOST_AUTO_TEST_CASE(TestImage) {
  Camera camera;
  camera.SetCameraId(1912);
  camera.SetModelId(2);
  camera.SetWidth(800);
  camera.SetHeight(600);
  camera.SetParams({1., 2., 3., 4.});
  camera.SetPriorFocalLength(true);

  Image image, image_readback;
  image.SetImageId(13);
  image.SetName("best_camera");
  image.SetCameraId(1912);
  image.SetUp(camera);
  image.SetRegistered(true);
  image.SetTvec({0.1, 2.4, 0.});
  image.SetTvecPrior({0.0, 2.0, 0.5});
  image.SetQvec({1.0, 0.9, 2., -1.});
  image.SetQvecPrior({1.1, 1.1, 2.1, -1.2});
  image.SetPoints2D({{0., 0.}, {1., 1.}, {2., 2.}, {3., 3.}, {4., 4.}});
  image.SetPoint3DForPoint2D(0, 1230);
  image.SetPoint3DForPoint2D(3, 301);
  image.SetNumObservations(10);
  image.IncrementCorrespondenceHasPoint3D(0);
  WriteAndReadBinaryDataFromBuffer(image, &image_readback);
  CheckImagesEqual(image, image_readback);
  BOOST_CHECK(image_readback.HasPoint3D(1230));
  BOOST_CHECK(image_readback.HasPoint3D(301));
  BOOST_CHECK(image_readback.IsPoint3DVisible(0));
  BOOST_CHECK(!image_readback.IsPoint3DVisible(3));
}

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

BOOST_AUTO_TEST_CASE(TestReconstruction) {
  Reconstruction reconstruction, reconstruction_readback;
  SceneGraph scene_graph;
  GenerateReconstruction(10, &reconstruction, &scene_graph);
  WriteAndReadBinaryDataFromBuffer(reconstruction, &reconstruction_readback);
  for (const auto& camera : reconstruction.Cameras()) {
    BOOST_CHECK(reconstruction_readback.Cameras().find(camera.first) !=
                reconstruction_readback.Cameras().end());
    const Camera& camera_readback =
      reconstruction_readback.Cameras().at(camera.first);
    BOOST_CHECK_EQUAL(camera.second.ModelId(), camera_readback.ModelId());
    BOOST_CHECK_EQUAL(camera.second.ModelName(), camera_readback.ModelName());
    BOOST_CHECK_EQUAL(camera.second.FocalLengthX(),
                      camera_readback.FocalLengthX());
    BOOST_CHECK_EQUAL(camera.second.FocalLengthY(),
                      camera_readback.FocalLengthY());
    BOOST_CHECK_EQUAL(camera.second.Width(), camera_readback.Width());
    BOOST_CHECK_EQUAL(camera.second.Height(), camera_readback.Height());
  }

  for (const auto& image : reconstruction.Images()) {
    BOOST_CHECK(reconstruction_readback.Images().find(image.first) !=
                reconstruction_readback.Images().end());
    const Image& image_readback =
      reconstruction_readback.Images().at(image.first);
    CheckImagesEqual(image.second, image_readback);
  }

  for (const auto& point3D : reconstruction.Points3D()) {
    BOOST_CHECK(reconstruction_readback.Points3D().find(point3D.first) !=
                reconstruction_readback.Points3D().end());
    const Point3D& point3D_readback =
      reconstruction_readback.Points3D().at(point3D.first);
    BOOST_CHECK_EQUAL(point3D.second.XYZ(), point3D_readback.XYZ());
    BOOST_CHECK_EQUAL(point3D.second.Color(), point3D_readback.Color());
    BOOST_CHECK_EQUAL(point3D.second.Error(), point3D_readback.Error());
  }

  BOOST_CHECK_EQUAL_COLLECTIONS(
      reconstruction.RegImageIds().begin(), reconstruction.RegImageIds().end(),
      reconstruction_readback.RegImageIds().begin(),
      reconstruction_readback.RegImageIds().end());
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(),
                    reconstruction_readback.NumPoints3D());
}

BOOST_AUTO_TEST_CASE(TestSceneGraphCorrespondence) {
  SceneGraph::Correspondence correspondence, correspondence_readback;
  correspondence.point2D_idx = 12;
  correspondence.image_id = 123;
  WriteAndReadBinaryDataFromBuffer(correspondence, &correspondence_readback);
  BOOST_CHECK_EQUAL(correspondence.image_id, correspondence_readback.image_id);
  BOOST_CHECK_EQUAL(correspondence.point2D_idx,
                    correspondence_readback.point2D_idx);
}

void BuildDefaultTwoViewSceneGraph(SceneGraph* scene_graph_ptr,
                                   FeatureMatches* matches_ptr) {
  SceneGraph& scene_graph = *CHECK_NOTNULL(scene_graph_ptr);
  scene_graph.AddImage(0, 10);
  scene_graph.AddImage(1, 10);

  FeatureMatches& matches = *CHECK_NOTNULL(matches_ptr);
  matches.resize(4);
  matches[0].point2D_idx1 = 0;
  matches[0].point2D_idx2 = 0;
  matches[1].point2D_idx1 = 1;
  matches[1].point2D_idx2 = 2;
  matches[2].point2D_idx1 = 3;
  matches[2].point2D_idx2 = 7;
  matches[3].point2D_idx1 = 4;
  matches[3].point2D_idx2 = 8;
  scene_graph.AddCorrespondences(0, 1, matches);
}

void CheckDefaultTwoViewSceneGraph(const SceneGraph& scene_graph,
                                   const FeatureMatches& matches) {
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(0), 4);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesForImage(1), 4);
  const image_pair_t pair_id = Database::ImagePairToPairId(0, 1);
  BOOST_CHECK_EQUAL(scene_graph.NumCorrespondencesBetweenImages().size(), 1);
  BOOST_CHECK_EQUAL(
      scene_graph.NumCorrespondencesBetweenImages().at(pair_id), 4);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 0).size(), 1);
  BOOST_CHECK(scene_graph.HasCorrespondences(0, 0));
  BOOST_CHECK(scene_graph.IsTwoViewObservation(0, 0));
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 0).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 0).size(), 1);
  BOOST_CHECK(scene_graph.HasCorrespondences(1, 0));
  BOOST_CHECK(scene_graph.IsTwoViewObservation(1, 0));
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 0).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 1).size(), 1);
  BOOST_CHECK(scene_graph.HasCorrespondences(0, 1));
  BOOST_CHECK(scene_graph.IsTwoViewObservation(0, 1));
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 1).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 1).at(0).point2D_idx, 2);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 2).size(), 1);
  BOOST_CHECK(scene_graph.HasCorrespondences(1, 2));
  BOOST_CHECK(scene_graph.IsTwoViewObservation(1, 2));
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 2).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 2).at(0).point2D_idx, 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 4).size(), 1);
  BOOST_CHECK(scene_graph.HasCorrespondences(0, 3));
  BOOST_CHECK(scene_graph.IsTwoViewObservation(0, 4));
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 3).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 3).at(0).point2D_idx, 7);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 4).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(0, 4).at(0).point2D_idx, 8);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 7).size(), 1);
  BOOST_CHECK(scene_graph.HasCorrespondences(1, 7));
  BOOST_CHECK(scene_graph.IsTwoViewObservation(1, 7));
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 7).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 7).at(0).point2D_idx, 3);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 8).size(), 1);
  BOOST_CHECK(scene_graph.HasCorrespondences(1, 8));
  BOOST_CHECK(scene_graph.IsTwoViewObservation(1, 8));
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 8).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(scene_graph.FindCorrespondences(1, 8).at(0).point2D_idx, 4);
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(scene_graph.FindTransitiveCorrespondences(0, i, 0).size(),
                      0);
    BOOST_CHECK_EQUAL(
        scene_graph.FindCorrespondences(0, i).size(),
        scene_graph.FindTransitiveCorrespondences(0, i, 1).size());
    BOOST_CHECK_EQUAL(
        scene_graph.FindCorrespondences(0, i).size(),
        scene_graph.FindTransitiveCorrespondences(0, i, 2).size());
    BOOST_CHECK_EQUAL(scene_graph.FindTransitiveCorrespondences(1, i, 0).size(),
                      0);
    BOOST_CHECK_EQUAL(
        scene_graph.FindCorrespondences(1, i).size(),
        scene_graph.FindTransitiveCorrespondences(1, i, 1).size());
    BOOST_CHECK_EQUAL(
        scene_graph.FindCorrespondences(1, i).size(),
        scene_graph.FindTransitiveCorrespondences(1, i, 2).size());
  }
  const auto corrs01 = scene_graph.FindCorrespondencesBetweenImages(0, 1);
  const auto corrs10 = scene_graph.FindCorrespondencesBetweenImages(1, 0);
  BOOST_CHECK_EQUAL(corrs01.size(), matches.size());
  BOOST_CHECK_EQUAL(corrs10.size(), matches.size());
  for (size_t i = 0; i < corrs01.size(); ++i) {
    BOOST_CHECK_EQUAL(corrs01[i].first, corrs10[i].second);
    BOOST_CHECK_EQUAL(corrs01[i].second, corrs10[i].first);
    BOOST_CHECK_EQUAL(matches[i].point2D_idx1, corrs01[i].first);
    BOOST_CHECK_EQUAL(matches[i].point2D_idx2, corrs01[i].second);
  }
}

BOOST_AUTO_TEST_CASE(TestSceneGraph) {
  SceneGraph scene_graph, scene_graph_readback;
  FeatureMatches matches;
  BuildDefaultTwoViewSceneGraph(&scene_graph, &matches);
  WriteAndReadBinaryDataFromBuffer(scene_graph, &scene_graph_readback);
  CheckDefaultTwoViewSceneGraph(scene_graph_readback, matches);
}
