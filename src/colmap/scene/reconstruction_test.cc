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

#include "colmap/scene/reconstruction.h"

#include "colmap/geometry/sim3.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/reconstruction_io_text.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/sensor/models.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void ExpectValidPtrs(const Reconstruction& reconstruction) {
  for (const auto& frame : reconstruction.Frames()) {
    EXPECT_TRUE(frame.second.HasRigPtr());
    auto& rig = reconstruction.Rig(frame.second.RigId());
    EXPECT_EQ(frame.second.RigPtr(), &rig);
  }
  for (const auto& image : reconstruction.Images()) {
    EXPECT_TRUE(image.second.HasCameraPtr());
    auto& camera = reconstruction.Camera(image.second.CameraId());
    EXPECT_EQ(image.second.CameraPtr(), &camera);
    EXPECT_TRUE(image.second.HasFramePtr());
    auto& frame = reconstruction.Frame(image.second.FrameId());
    EXPECT_EQ(image.second.FramePtr(), &frame);
  }
}

void ExpectEqualSerialization(const Reconstruction& reconstruction1,
                              const Reconstruction& reconstruction2) {
  // compare rigs
  std::stringstream stream1_rigs, stream2_rigs;
  WriteRigsText(reconstruction1, stream1_rigs);
  WriteRigsText(reconstruction2, stream2_rigs);
  EXPECT_EQ(stream1_rigs.str(), stream2_rigs.str());

  // compare cameras
  std::stringstream stream1_cameras, stream2_cameras;
  WriteCamerasText(reconstruction1, stream1_cameras);
  WriteCamerasText(reconstruction2, stream2_cameras);
  EXPECT_EQ(stream1_cameras.str(), stream2_cameras.str());

  // compare frames
  std::stringstream stream1_frames, stream2_frames;
  WriteFramesText(reconstruction1, stream1_frames);
  WriteFramesText(reconstruction2, stream2_frames);
  EXPECT_EQ(stream1_frames.str(), stream2_frames.str());

  // compare images
  std::stringstream stream1_images, stream2_images;
  WriteImagesText(reconstruction1, stream1_images);
  WriteImagesText(reconstruction2, stream2_images);
  EXPECT_EQ(stream1_images.str(), stream2_images.str());

  // compare point3ds
  std::stringstream stream1_points3D, stream2_points3D;
  WritePoints3DText(reconstruction1, stream1_points3D);
  WritePoints3DText(reconstruction2, stream2_points3D);
  EXPECT_EQ(stream1_points3D.str(), stream2_points3D.str());
}

void GenerateReconstruction(const image_t num_images,
                            Reconstruction* reconstruction) {
  const size_t kNumPoints2D = 10;

  Camera camera = Camera::CreateFromModelName(1, "PINHOLE", 1, 1, 1);
  reconstruction->AddCamera(camera);
  Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(camera.SensorId());
  reconstruction->AddRig(rig);

  for (image_t image_id = 1; image_id <= num_images; ++image_id) {
    Frame frame;
    frame.SetFrameId(image_id);
    frame.SetRigId(rig.RigId());
    frame.AddDataId(data_t(camera.SensorId(), image_id));
    frame.SetRigFromWorld(Rigid3d());
    reconstruction->AddFrame(frame);
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(camera.camera_id);
    image.SetFrameId(frame.FrameId());
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(
        std::vector<Eigen::Vector2d>(kNumPoints2D, Eigen::Vector2d::Zero()));
    reconstruction->AddImage(image);
  }
}

TEST(Reconstruction, Empty) {
  Reconstruction reconstruction;
  EXPECT_EQ(reconstruction.NumRigs(), 0);
  EXPECT_EQ(reconstruction.NumCameras(), 0);
  EXPECT_EQ(reconstruction.NumFrames(), 0);
  EXPECT_EQ(reconstruction.NumRegFrames(), 0);
  EXPECT_EQ(reconstruction.NumImages(), 0);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

TEST(Reconstruction, ConstructCopy) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_rigs = 3;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 8;
  synthetic_dataset_options.num_points3D = 21;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const Reconstruction reconstruction_copy(reconstruction);
  EXPECT_THAT(reconstruction, ReconstructionEq(reconstruction_copy));
  ExpectEqualSerialization(reconstruction, reconstruction_copy);
  ExpectValidPtrs(reconstruction);
  ExpectValidPtrs(reconstruction_copy);
}

TEST(Reconstruction, AssignCopy) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_rigs = 3;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 8;
  synthetic_dataset_options.num_points3D = 21;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  Reconstruction reconstruction_copy;
  reconstruction_copy = reconstruction;
  EXPECT_THAT(reconstruction, ReconstructionEq(reconstruction_copy));
  ExpectEqualSerialization(reconstruction, reconstruction_copy);
  ExpectValidPtrs(reconstruction);
  ExpectValidPtrs(reconstruction_copy);
}

TEST(Reconstruction, Print) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 3;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  std::ostringstream stream;
  stream << reconstruction;
  EXPECT_EQ(stream.str(),
            "Reconstruction(num_rigs=1, num_cameras=1, num_frames=2, "
            "num_reg_frames=2, num_images=2, num_points3D=3)");
}

TEST(Reconstruction, AddRig) {
  Reconstruction reconstruction;
  Rig rig;
  rig.SetRigId(1);
  const Camera camera =
      Camera::CreateFromModelId(1, SimplePinholeCameraModel::model_id, 1, 1, 1);
  rig.AddRefSensor(camera.SensorId());
  EXPECT_ANY_THROW(reconstruction.AddRig(rig));
  reconstruction.AddCamera(camera);
  reconstruction.AddRig(rig);
  EXPECT_TRUE(reconstruction.ExistsRig(rig.RigId()));
  EXPECT_EQ(reconstruction.Rig(rig.RigId()).RigId(), rig.RigId());
  EXPECT_EQ(reconstruction.Rigs().count(rig.RigId()), 1);
  EXPECT_EQ(reconstruction.Rigs().size(), 1);
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumCameras(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), 0);
  EXPECT_EQ(reconstruction.NumRegFrames(), 0);
  EXPECT_EQ(reconstruction.NumImages(), 0);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

TEST(Reconstruction, AddCamera) {
  Reconstruction reconstruction;
  Camera camera =
      Camera::CreateFromModelId(1, SimplePinholeCameraModel::model_id, 1, 1, 1);
  reconstruction.AddCamera(camera);
  EXPECT_TRUE(reconstruction.ExistsCamera(camera.camera_id));
  EXPECT_EQ(reconstruction.Camera(camera.camera_id).camera_id,
            camera.camera_id);
  EXPECT_EQ(reconstruction.Cameras().count(camera.camera_id), 1);
  EXPECT_EQ(reconstruction.Cameras().size(), 1);
  EXPECT_EQ(reconstruction.NumRigs(), 0);
  EXPECT_EQ(reconstruction.NumCameras(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), 0);
  EXPECT_EQ(reconstruction.NumRegFrames(), 0);
  EXPECT_EQ(reconstruction.NumImages(), 0);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

TEST(Reconstruction, AddCameraWithTrivialRig) {
  Reconstruction reconstruction;
  Camera camera =
      Camera::CreateFromModelId(1, SimplePinholeCameraModel::model_id, 1, 1, 1);
  reconstruction.AddCameraWithTrivialRig(camera);
  EXPECT_TRUE(reconstruction.ExistsCamera(camera.camera_id));
  EXPECT_EQ(reconstruction.Camera(camera.camera_id).camera_id,
            camera.camera_id);
  EXPECT_EQ(reconstruction.Cameras().count(camera.camera_id), 1);
  EXPECT_EQ(reconstruction.Cameras().size(), 1);
  EXPECT_TRUE(reconstruction.ExistsRig(camera.camera_id));
  EXPECT_EQ(reconstruction.Rig(camera.camera_id).RigId(), camera.camera_id);
  EXPECT_EQ(reconstruction.Rig(camera.camera_id).NumSensors(), 1);
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumCameras(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), 0);
  EXPECT_EQ(reconstruction.NumRegFrames(), 0);
  EXPECT_EQ(reconstruction.NumImages(), 0);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

TEST(Reconstruction, AddFrame) {
  Reconstruction reconstruction;
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1, 1, 1);
  reconstruction.AddCamera(camera);
  Rig rig;
  rig.SetRigId(1);
  Frame frame;
  frame.SetFrameId(1);
  frame.SetRigId(rig.RigId());
  frame.AddDataId(data_t(camera.SensorId(), 1));
  try {
    reconstruction.AddFrame(frame);
  } catch (const std::exception& e) {
    EXPECT_THAT(std::string(e.what()),
                testing::HasSubstr("Rig with ID 1 does not exist"));
  }
  reconstruction.AddRig(rig);
  try {
    reconstruction.AddFrame(frame);
  } catch (const std::exception& e) {
    EXPECT_THAT(
        std::string(e.what()),
        testing::HasSubstr("Check failed: rig.HasSensor(data_id.sensor_id)"));
  }
  EXPECT_ANY_THROW(reconstruction.AddFrame(frame));
  reconstruction.Rig(frame.RigId()).AddRefSensor(camera.SensorId());
  reconstruction.AddFrame(frame);
  EXPECT_TRUE(reconstruction.ExistsFrame(1));
  EXPECT_EQ(reconstruction.Frame(1).FrameId(), 1);
  EXPECT_EQ(reconstruction.Frames().count(1), 1);
  EXPECT_EQ(reconstruction.Frames().size(), 1);
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumCameras(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), 1);
  EXPECT_EQ(reconstruction.NumRegFrames(), 0);
  EXPECT_EQ(reconstruction.NumImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  reconstruction.Frame(1).SetRigFromWorld(Rigid3d());
  reconstruction.RegisterFrame(1);
  EXPECT_EQ(reconstruction.NumRegFrames(), 1);
  ExpectValidPtrs(reconstruction);
}

TEST(Reconstruction, AddImageWrongFrameCorrespondence) {
  Reconstruction reconstruction;
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1, 1, 1);
  reconstruction.AddCamera(camera);
  Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(camera.SensorId());
  reconstruction.AddRig(rig);
  Frame frame;
  frame.SetFrameId(1);
  frame.SetRigId(rig.RigId());
  Image image;
  image.SetCameraId(camera.camera_id);
  image.SetImageId(1);
  image.SetFrameId(frame.FrameId());
  frame.AddDataId(image.DataId());
  reconstruction.AddFrame(frame);
  reconstruction.AddImage(image);
  image.SetImageId(2);
  EXPECT_ANY_THROW(reconstruction.AddImage(image));
}

TEST(Reconstruction, AddImage) {
  Reconstruction reconstruction;
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1, 1, 1);
  reconstruction.AddCamera(camera);
  Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(camera.SensorId());
  reconstruction.AddRig(rig);
  Frame frame;
  frame.SetFrameId(1);
  frame.SetRigId(rig.RigId());
  Image image;
  image.SetCameraId(camera.camera_id);
  image.SetImageId(1);
  image.SetFrameId(frame.FrameId());
  try {
    reconstruction.AddImage(image);
  } catch (const std::exception& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr("Frame with ID 1 does not exist"));
  }
  reconstruction.AddFrame(frame);
  try {
    reconstruction.AddImage(image);
  } catch (const std::exception& e) {
    EXPECT_THAT(
        e.what(),
        testing::HasSubstr("Check failed: frame.HasDataId(image.DataId())"));
  }
  reconstruction.Frame(frame.FrameId()).AddDataId(image.DataId());
  reconstruction.AddImage(image);
  EXPECT_TRUE(reconstruction.ExistsImage(1));
  EXPECT_EQ(reconstruction.Image(1).ImageId(), 1);
  EXPECT_FALSE(reconstruction.Image(1).HasPose());
  EXPECT_EQ(reconstruction.Images().count(1), 1);
  EXPECT_EQ(reconstruction.Images().size(), 1);
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumCameras(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), 1);
  EXPECT_EQ(reconstruction.NumRegFrames(), 0);
  EXPECT_EQ(reconstruction.NumImages(), 1);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  reconstruction.Image(1).FramePtr()->SetRigFromWorld(Rigid3d());
  reconstruction.RegisterFrame(1);
  EXPECT_EQ(reconstruction.NumRegFrames(), 1);
  EXPECT_EQ(reconstruction.NumRegImages(), 1);
  ExpectValidPtrs(reconstruction);
}

TEST(Reconstruction, AddImageWithTrivialFrame) {
  Reconstruction reconstruction;
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1, 1, 1);
  reconstruction.AddCameraWithTrivialRig(camera);
  Image image;
  image.SetCameraId(camera.camera_id);
  image.SetImageId(1);
  reconstruction.AddImageWithTrivialFrame(image);

  EXPECT_TRUE(reconstruction.ExistsImage(1));
  EXPECT_EQ(reconstruction.Image(1).ImageId(), 1);
  EXPECT_EQ(reconstruction.Image(1).FrameId(), 1);
  EXPECT_FALSE(reconstruction.Image(1).HasPose());
  EXPECT_EQ(reconstruction.Images().count(1), 1);
  EXPECT_EQ(reconstruction.Images().size(), 1);
  EXPECT_TRUE(reconstruction.ExistsFrame(1));
  EXPECT_EQ(reconstruction.Frame(1).NumDataIds(), 1);
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumCameras(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), 1);
  EXPECT_EQ(reconstruction.NumRegFrames(), 0);
  EXPECT_EQ(reconstruction.NumImages(), 1);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  reconstruction.Image(1).FramePtr()->SetRigFromWorld(Rigid3d());
  reconstruction.RegisterFrame(1);
  EXPECT_EQ(reconstruction.NumRegFrames(), 1);
  EXPECT_EQ(reconstruction.NumRegImages(), 1);
  ExpectValidPtrs(reconstruction);
}

TEST(Reconstruction, AddImageWithTrivialFrameExistsNonTrivialRig) {
  Reconstruction reconstruction;
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1, 1, 1);
  reconstruction.AddCameraWithTrivialRig(camera);
  camera.camera_id = 2;
  reconstruction.Rig(1).AddSensor(camera.SensorId());
  THROW_CHECK_EQ(reconstruction.Rig(1).NumSensors(), 2);

  Image image;
  image.SetImageId(1);
  // The rig has multiple cameras
  image.SetCameraId(1);
  EXPECT_ANY_THROW(reconstruction.AddImageWithTrivialFrame(image));
  // No rig with id 2 found in the reconstruction
  image.SetCameraId(2);
  EXPECT_ANY_THROW(reconstruction.AddImageWithTrivialFrame(image));
}

TEST(Reconstruction, AddImageWithTrivialFrameSetCamFromWorld) {
  Reconstruction reconstruction;
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1, 1, 1);
  reconstruction.AddCameraWithTrivialRig(camera);
  Image image;
  image.SetCameraId(camera.camera_id);
  image.SetImageId(1);
  reconstruction.AddImageWithTrivialFrame(image, Rigid3d());
  EXPECT_TRUE(reconstruction.ExistsImage(1));
  EXPECT_EQ(reconstruction.Image(1).ImageId(), 1);
  EXPECT_EQ(reconstruction.Image(1).FrameId(), 1);
  EXPECT_TRUE(reconstruction.Image(1).HasPose());
  EXPECT_EQ(reconstruction.Images().count(1), 1);
  EXPECT_EQ(reconstruction.Images().size(), 1);
  EXPECT_TRUE(reconstruction.ExistsFrame(1));
  EXPECT_EQ(reconstruction.Frame(1).NumDataIds(), 1);
  EXPECT_TRUE(reconstruction.Frame(1).HasPose());
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumCameras(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), 1);
  EXPECT_EQ(reconstruction.NumRegFrames(), 1);
  EXPECT_EQ(reconstruction.NumImages(), 1);
  EXPECT_EQ(reconstruction.NumRegImages(), 1);
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  ExpectValidPtrs(reconstruction);
}

TEST(Reconstruction, RegImageIds) {
  Reconstruction reconstruction;

  const Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1, 1, 1);
  reconstruction.AddCamera(camera);

  Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(camera.SensorId());
  reconstruction.AddRig(rig);

  Image image1;
  image1.SetCameraId(camera.camera_id);
  image1.SetImageId(1);
  image1.SetFrameId(1);
  Image image2;
  image2.SetCameraId(camera.camera_id);
  image2.SetImageId(2);
  image2.SetFrameId(1);

  Frame frame;
  frame.SetFrameId(1);
  frame.SetRigId(rig.RigId());
  frame.AddDataId(image1.DataId());
  frame.AddDataId(image2.DataId());
  reconstruction.AddFrame(frame);
  reconstruction.Frame(frame.FrameId()).SetRigFromWorld(Rigid3d());

  reconstruction.RegisterFrame(frame.FrameId());

  // Throws because no image was added.
  EXPECT_ANY_THROW(reconstruction.RegImageIds());
  EXPECT_EQ(reconstruction.NumRegFrames(), 1);
  EXPECT_EQ(reconstruction.NumRegImages(), 2);

  // Throws because second image was not added.
  reconstruction.AddImage(image1);
  EXPECT_ANY_THROW(reconstruction.RegImageIds());
  EXPECT_EQ(reconstruction.NumRegFrames(), 1);
  EXPECT_EQ(reconstruction.NumRegImages(), 2);

  reconstruction.AddImage(image2);
  EXPECT_EQ(reconstruction.NumRegFrames(), 1);
  EXPECT_EQ(reconstruction.NumRegImages(), 2);
  EXPECT_THAT(reconstruction.RegImageIds(), testing::ElementsAre(1, 2));

  // Registering a frame twice is a no-op.
  reconstruction.RegisterFrame(frame.FrameId());
  EXPECT_EQ(reconstruction.NumRegFrames(), 1);
  EXPECT_EQ(reconstruction.NumRegImages(), 2);
  EXPECT_THAT(reconstruction.RegImageIds(), testing::ElementsAre(1, 2));

  reconstruction.DeRegisterFrame(frame.FrameId());
  EXPECT_EQ(reconstruction.NumRegFrames(), 0);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_THAT(reconstruction.RegImageIds(), testing::IsEmpty());

  // De-registering a frame twice is a no-op.
  reconstruction.DeRegisterFrame(frame.FrameId());
  EXPECT_EQ(reconstruction.NumRegFrames(), 0);
  EXPECT_EQ(reconstruction.NumRegImages(), 0);
  EXPECT_THAT(reconstruction.RegImageIds(), testing::IsEmpty());
}

TEST(Reconstruction, AddPoint3D) {
  Reconstruction reconstruction;
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_TRUE(reconstruction.ExistsPoint3D(point3D_id));
  EXPECT_EQ(reconstruction.Point3D(point3D_id).track.Length(), 0);
  EXPECT_EQ(reconstruction.Points3D().count(point3D_id), 1);
  EXPECT_EQ(reconstruction.Points3D().size(), 1);
  EXPECT_EQ(reconstruction.NumRigs(), 0);
  EXPECT_EQ(reconstruction.NumCameras(), 0);
  EXPECT_EQ(reconstruction.NumFrames(), 0);
  EXPECT_EQ(reconstruction.NumImages(), 0);
  EXPECT_EQ(reconstruction.NumRegFrames(), 0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 1);
  EXPECT_EQ(reconstruction.Point3DIds().count(point3D_id), 1);

  Reconstruction reconstruction2;
  GenerateReconstruction(2, &reconstruction2);
  struct Point3D point3D;
  point3D.xyz = Eigen::Vector3d(1.0, 2.0, 3.0);
  point3D.track.AddElement(1, 0);
  point3D.track.AddElement(2, 1);
  reconstruction2.AddPoint3D(5, point3D);
  EXPECT_EQ(reconstruction2.Point3D(5).track.Length(), 2);
  EXPECT_TRUE(reconstruction2.Image(1).Point2D(0).HasPoint3D());
  EXPECT_TRUE(reconstruction2.Image(2).Point2D(1).HasPoint3D());
  EXPECT_EQ(reconstruction2.NumRigs(), 1);
  EXPECT_EQ(reconstruction2.NumCameras(), 1);
  EXPECT_EQ(reconstruction2.NumFrames(), 2);
  EXPECT_EQ(reconstruction2.NumImages(), 2);
  EXPECT_EQ(reconstruction2.NumRegFrames(), 2);
  EXPECT_EQ(reconstruction2.NumPoints3D(), 1);
  EXPECT_EQ(reconstruction2.Point3DIds().count(5), 1);
}

TEST(Reconstruction, AddObservation) {
  Reconstruction reconstruction;
  GenerateReconstruction(3, &reconstruction);
  Track track;
  track.AddElement(1, 0);
  track.AddElement(2, 1);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), track);
  EXPECT_EQ(reconstruction.Image(1).NumPoints3D(), 1);
  EXPECT_TRUE(reconstruction.Image(1).Point2D(0).HasPoint3D());
  EXPECT_FALSE(reconstruction.Image(1).Point2D(1).HasPoint3D());
  EXPECT_EQ(reconstruction.Image(2).NumPoints3D(), 1);
  EXPECT_FALSE(reconstruction.Image(2).Point2D(0).HasPoint3D());
  EXPECT_TRUE(reconstruction.Image(2).Point2D(1).HasPoint3D());
  EXPECT_EQ(reconstruction.Point3D(point3D_id).track.Length(), 2);
  reconstruction.AddObservation(point3D_id, TrackElement(3, 2));
  EXPECT_EQ(reconstruction.Image(3).NumPoints3D(), 1);
  EXPECT_TRUE(reconstruction.Image(3).Point2D(2).HasPoint3D());
  EXPECT_EQ(reconstruction.Point3D(point3D_id).track.Length(), 3);
}

TEST(Reconstruction, MergePoints3D) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 0), Track());
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  reconstruction.Point3D(point3D_id1).color =
      Eigen::Matrix<uint8_t, 3, 1>(0, 0, 0);
  const point3D_t point3D_id2 =
      reconstruction.AddPoint3D(Eigen::Vector3d(1, 1, 1), Track());
  reconstruction.AddObservation(point3D_id2, TrackElement(1, 1));
  reconstruction.AddObservation(point3D_id2, TrackElement(2, 1));
  reconstruction.Point3D(point3D_id2).color =
      Eigen::Matrix<uint8_t, 3, 1>(20, 20, 20);
  const point3D_t merged_point3D_id =
      reconstruction.MergePoints3D(point3D_id1, point3D_id2);
  EXPECT_FALSE(reconstruction.ExistsPoint3D(point3D_id1));
  EXPECT_FALSE(reconstruction.ExistsPoint3D(point3D_id2));
  EXPECT_TRUE(reconstruction.ExistsPoint3D(merged_point3D_id));
  EXPECT_EQ(reconstruction.Image(1).Point2D(0).point3D_id, merged_point3D_id);
  EXPECT_EQ(reconstruction.Image(1).Point2D(1).point3D_id, merged_point3D_id);
  EXPECT_EQ(reconstruction.Image(2).Point2D(0).point3D_id, merged_point3D_id);
  EXPECT_EQ(reconstruction.Image(2).Point2D(1).point3D_id, merged_point3D_id);
  EXPECT_TRUE(reconstruction.Point3D(merged_point3D_id)
                  .xyz.isApprox(Eigen::Vector3d(0.5, 0.5, 0.5)));
  EXPECT_EQ(reconstruction.Point3D(merged_point3D_id).color,
            Eigen::Vector3ub(10, 10, 10));
}

TEST(Reconstruction, DeletePoint3D) {
  Reconstruction reconstruction;
  GenerateReconstruction(1, &reconstruction);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id, TrackElement(1, 0));
  reconstruction.DeletePoint3D(point3D_id);
  EXPECT_FALSE(reconstruction.ExistsPoint3D(point3D_id));
  EXPECT_EQ(reconstruction.Image(1).NumPoints3D(), 0);
}

TEST(Reconstruction, DeleteObservation) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 0), Track());
  reconstruction.AddObservation(point3D_id, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id, TrackElement(1, 1));
  reconstruction.AddObservation(point3D_id, TrackElement(1, 2));
  reconstruction.DeleteObservation(1, 0);
  EXPECT_EQ(reconstruction.Point3D(point3D_id).track.Length(), 2);
  EXPECT_FALSE(reconstruction.Image(point3D_id).Point2D(0).HasPoint3D());
  reconstruction.DeleteObservation(1, 1);
  EXPECT_FALSE(reconstruction.ExistsPoint3D(point3D_id));
  EXPECT_FALSE(reconstruction.Image(point3D_id).Point2D(1).HasPoint3D());
  EXPECT_FALSE(reconstruction.Image(point3D_id).Point2D(2).HasPoint3D());
}

TEST(Reconstruction, SetRigsAndFrames) {
  Reconstruction reconstruction;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_rigs = 3;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 8;
  synthetic_dataset_options.num_points3D = 21;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction, database.get());
  for (const auto& [frame_id, _] : reconstruction.Frames()) {
    reconstruction.DeRegisterFrame(frame_id);
  }
  const Reconstruction orig_reconstruction = reconstruction;
  reconstruction.SetRigsAndFrames(database->ReadAllRigs(),
                                  database->ReadAllFrames());
  EXPECT_THAT(reconstruction, ReconstructionEq(orig_reconstruction));
  ExpectEqualSerialization(reconstruction, orig_reconstruction);
}

TEST(Reconstruction, SetRigsAndFramesResetsNumRegImages) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const size_t num_reg_images_before = reconstruction.NumRegImages();
  EXPECT_GT(num_reg_images_before, 0);
  // Copy rigs and frames (with poses) from the reconstruction to re-apply.
  std::vector<class Rig> rigs;
  for (const auto& [_, rig] : reconstruction.Rigs()) {
    rigs.push_back(rig);
  }
  std::vector<class Frame> frames;
  for (auto [_, frame] : reconstruction.Frames()) {
    frame.ResetRigPtr();
    frames.push_back(std::move(frame));
  }
  const size_t num_rigs_before = reconstruction.NumRigs();
  const size_t num_frames_before = reconstruction.NumFrames();
  const size_t num_reg_frames_before = reconstruction.NumRegFrames();
  // Call SetRigsAndFrames while frames are still registered. Previously this
  // would double-count num_reg_images_ because it was not reset to zero.
  reconstruction.SetRigsAndFrames(std::move(rigs), std::move(frames));
  // Verify num_reg_images_ is not double-counted.
  EXPECT_EQ(reconstruction.NumRegImages(), num_reg_images_before);
  // Verify rigs, frames, and registered frames are preserved.
  EXPECT_EQ(reconstruction.NumRigs(), num_rigs_before);
  EXPECT_EQ(reconstruction.NumFrames(), num_frames_before);
  EXPECT_EQ(reconstruction.NumRegFrames(), num_reg_frames_before);
  // Verify every registered frame still has a pose.
  for (const auto& frame_id : reconstruction.RegFrameIds()) {
    EXPECT_TRUE(reconstruction.Frame(frame_id).HasPose());
  }
  // Verify image-to-frame pointers are correctly re-wired.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    EXPECT_TRUE(image.HasFrameId());
    EXPECT_TRUE(image.HasFramePtr());
    EXPECT_EQ(image.FramePtr(), &reconstruction.Frame(image.FrameId()));
  }
}

TEST(Reconstruction, RegisterFrame) {
  Reconstruction reconstruction;
  GenerateReconstruction(1, &reconstruction);
  EXPECT_EQ(reconstruction.NumRegFrames(), 1);
  EXPECT_TRUE(reconstruction.Image(1).HasPose());
  EXPECT_TRUE(reconstruction.Frame(1).HasPose());
  reconstruction.RegisterFrame(1);
  EXPECT_EQ(reconstruction.NumRegFrames(), 1);
  EXPECT_TRUE(reconstruction.Image(1).HasPose());
  EXPECT_TRUE(reconstruction.Frame(1).HasPose());
  reconstruction.RegisterFrame(1);
  EXPECT_EQ(reconstruction.NumRegFrames(), 1);
  EXPECT_TRUE(reconstruction.Image(1).HasPose());
  EXPECT_TRUE(reconstruction.Frame(1).HasPose());
  reconstruction.DeRegisterFrame(1);
  EXPECT_EQ(reconstruction.NumRegFrames(), 0);
  EXPECT_FALSE(reconstruction.Image(1).HasPose());
  EXPECT_FALSE(reconstruction.Frame(1).HasPose());
  reconstruction.DeRegisterFrame(1);
  EXPECT_EQ(reconstruction.NumRegFrames(), 0);
  EXPECT_FALSE(reconstruction.Image(1).HasPose());
  EXPECT_FALSE(reconstruction.Frame(1).HasPose());
}

TEST(Reconstruction, Normalize) {
  Reconstruction reconstruction;
  GenerateReconstruction(7, &reconstruction);
  for (const auto& [frame_id, _] : reconstruction.Frames()) {
    reconstruction.DeRegisterFrame(frame_id);
  }
  Sim3d tform = reconstruction.Normalize(/*fixed_scale=*/false);
  EXPECT_EQ(tform.scale(), 1);
  EXPECT_EQ(tform.rotation().coeffs(), Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(tform.translation(), Eigen::Vector3d::Zero());
  reconstruction.Frame(1).SetRigFromWorld(Rigid3d());
  reconstruction.Frame(2).SetRigFromWorld(Rigid3d());
  reconstruction.Frame(3).SetRigFromWorld(Rigid3d());
  reconstruction.Frame(1).RigFromWorld().translation().z() = -20.0;
  reconstruction.Frame(2).RigFromWorld().translation().z() = -10.0;
  reconstruction.Frame(3).RigFromWorld().translation().z() = 0.0;
  reconstruction.RegisterFrame(1);
  reconstruction.RegisterFrame(2);
  reconstruction.RegisterFrame(3);
  reconstruction.Normalize(/*fixed_scale=*/true);
  EXPECT_NEAR(
      reconstruction.Image(1).CamFromWorld().translation().z(), -10, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(2).CamFromWorld().translation().z(), 0, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(3).CamFromWorld().translation().z(), 10, 1e-6);
  reconstruction.Normalize(/*fixed_scale=*/false);
  EXPECT_NEAR(
      reconstruction.Image(1).CamFromWorld().translation().z(), -5, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(2).CamFromWorld().translation().z(), 0, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(3).CamFromWorld().translation().z(), 5, 1e-6);
  reconstruction.Normalize(/*fixed_scale=*/false, 5);
  EXPECT_NEAR(
      reconstruction.Image(1).CamFromWorld().translation().z(), -2.5, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(2).CamFromWorld().translation().z(), 0, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(3).CamFromWorld().translation().z(), 2.5, 1e-6);
  reconstruction.Normalize(/*fixed_scale=*/false, 10, 0.0, 1.0);
  EXPECT_NEAR(
      reconstruction.Image(1).CamFromWorld().translation().z(), -5, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(2).CamFromWorld().translation().z(), 0, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(3).CamFromWorld().translation().z(), 5, 1e-6);
  tform = reconstruction.Normalize(/*fixed_scale=*/false, 20);
  EXPECT_NEAR(
      reconstruction.Image(1).CamFromWorld().translation().z(), -10, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(2).CamFromWorld().translation().z(), 0, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(3).CamFromWorld().translation().z(), 10, 1e-6);
  reconstruction.Transform(Inverse(tform));
  EXPECT_NEAR(
      reconstruction.Image(1).CamFromWorld().translation().z(), -5, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(2).CamFromWorld().translation().z(), 0, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(3).CamFromWorld().translation().z(), 5, 1e-6);
  reconstruction.Transform(tform);
  EXPECT_NEAR(
      reconstruction.Image(1).CamFromWorld().translation().z(), -10, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(2).CamFromWorld().translation().z(), 0, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(3).CamFromWorld().translation().z(), 10, 1e-6);
  reconstruction.Image(4).FramePtr()->SetRigFromWorld(Rigid3d());
  reconstruction.Image(5).FramePtr()->SetRigFromWorld(Rigid3d());
  reconstruction.Image(6).FramePtr()->SetRigFromWorld(Rigid3d());
  reconstruction.Image(7).FramePtr()->SetRigFromWorld(Rigid3d());
  reconstruction.Image(4).FramePtr()->RigFromWorld().translation().z() = -7.5;
  reconstruction.Image(5).FramePtr()->RigFromWorld().translation().z() = -5.0;
  reconstruction.Image(6).FramePtr()->RigFromWorld().translation().z() = 5.0;
  reconstruction.Image(7).FramePtr()->RigFromWorld().translation().z() = 7.5;
  reconstruction.RegisterFrame(4);
  reconstruction.RegisterFrame(5);
  reconstruction.RegisterFrame(6);
  reconstruction.RegisterFrame(7);
  reconstruction.Normalize(/*fixed_scale=*/false, 10, 0.0, 1.0);
  EXPECT_NEAR(
      reconstruction.Image(1).CamFromWorld().translation().z(), -5, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(2).CamFromWorld().translation().z(), 0, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(3).CamFromWorld().translation().z(), 5, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(4).CamFromWorld().translation().z(), -3.75, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(5).CamFromWorld().translation().z(), -2.5, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(6).CamFromWorld().translation().z(), 2.5, 1e-6);
  EXPECT_NEAR(
      reconstruction.Image(7).CamFromWorld().translation().z(), 3.75, 1e-6);
}

TEST(Reconstruction, ComputeBoundsAndCentroidEmpty) {
  Reconstruction reconstruction;
  const Eigen::Vector3d centroid = reconstruction.ComputeCentroid(0.0, 1.0);
  const Eigen::AlignedBox3d bbox = reconstruction.ComputeBoundingBox(0.0, 1.0);
  EXPECT_NEAR(centroid(0), 0, 1e-6);
  EXPECT_NEAR(centroid(1), 0, 1e-6);
  EXPECT_NEAR(centroid(2), 0, 1e-6);
  EXPECT_NEAR(bbox.min().x(), 0, 1e-6);
  EXPECT_NEAR(bbox.min().y(), 0, 1e-6);
  EXPECT_NEAR(bbox.min().z(), 0, 1e-6);
  EXPECT_NEAR(bbox.max().x(), 0, 1e-6);
  EXPECT_NEAR(bbox.max().y(), 0, 1e-6);
  EXPECT_NEAR(bbox.max().z(), 0, 1e-6);
}

TEST(Reconstruction, ComputeBoundsAndCentroid) {
  Reconstruction reconstruction;
  reconstruction.AddPoint3D(Eigen::Vector3d(3.0, 0.0, 0.0), Track());
  reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 3.0, 0.0), Track());
  reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 3.0), Track());
  const Eigen::Vector3d centroid = reconstruction.ComputeCentroid(0.0, 1.0);
  const Eigen::AlignedBox3d bbox = reconstruction.ComputeBoundingBox(0.0, 1.0);
  EXPECT_NEAR(centroid(0), 1.0, 1e-6);
  EXPECT_NEAR(centroid(1), 1.0, 1e-6);
  EXPECT_NEAR(centroid(2), 1.0, 1e-6);
  EXPECT_NEAR(bbox.min().x(), 0, 1e-6);
  EXPECT_NEAR(bbox.min().y(), 0, 1e-6);
  EXPECT_NEAR(bbox.min().z(), 0, 1e-6);
  EXPECT_NEAR(bbox.max().x(), 3.0, 1e-6);
  EXPECT_NEAR(bbox.max().y(), 3.0, 1e-6);
  EXPECT_NEAR(bbox.max().z(), 3.0, 1e-6);
}

TEST(Reconstruction, Crop) {
  Reconstruction reconstruction;
  GenerateReconstruction(3, &reconstruction);
  point3D_t point_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 0.0), Track());
  reconstruction.AddObservation(point_id, TrackElement(1, 1));
  point_id = reconstruction.AddPoint3D(Eigen::Vector3d(0.5, 0.5, 0.0), Track());
  reconstruction.AddObservation(point_id, TrackElement(1, 2));
  point_id = reconstruction.AddPoint3D(Eigen::Vector3d(1.0, 1.0, 0.0), Track());
  reconstruction.AddObservation(point_id, TrackElement(2, 3));
  point_id = reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 0.5), Track());
  reconstruction.AddObservation(point_id, TrackElement(2, 4));
  point_id = reconstruction.AddPoint3D(Eigen::Vector3d(0.5, 0.5, 1.0), Track());
  reconstruction.AddObservation(point_id, TrackElement(3, 5));

  // Check correct reconstruction setup
  EXPECT_EQ(reconstruction.NumCameras(), 1);
  EXPECT_EQ(reconstruction.NumImages(), 3);
  EXPECT_EQ(reconstruction.NumRegFrames(), 3);
  EXPECT_EQ(reconstruction.NumPoints3D(), 5);

  // Test emtpy reconstruction after cropping.
  const Reconstruction cropped1 = reconstruction.Crop(Eigen::AlignedBox3d(
      Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(-0.5, -0.5, -0.5)));
  EXPECT_EQ(cropped1.NumCameras(), 1);
  EXPECT_EQ(cropped1.NumImages(), 3);
  EXPECT_EQ(cropped1.NumRegFrames(), 0);
  EXPECT_EQ(cropped1.NumPoints3D(), 0);

  // Test reconstruction with contents after cropping
  const Reconstruction cropped2 = reconstruction.Crop(Eigen::AlignedBox3d(
      Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.75, 0.75, 0.75)));
  EXPECT_EQ(cropped2.NumCameras(), 1);
  EXPECT_EQ(cropped2.NumImages(), 3);
  EXPECT_EQ(cropped2.NumRegFrames(), 2);
  EXPECT_EQ(cropped2.NumPoints3D(), 3);
  EXPECT_TRUE(cropped2.Image(1).HasPose());
  EXPECT_TRUE(cropped2.Image(2).HasPose());
  EXPECT_FALSE(cropped2.Image(3).HasPose());
}

TEST(Reconstruction, Transform) {
  Reconstruction reconstruction;
  GenerateReconstruction(3, &reconstruction);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(1, 1, 1), Track());
  reconstruction.AddObservation(point3D_id, TrackElement(1, 1));
  reconstruction.AddObservation(point3D_id, TrackElement(2, 1));
  reconstruction.Transform(
      Sim3d(2, Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 1, 2)));
  EXPECT_EQ(reconstruction.Image(1).ProjectionCenter(),
            Eigen::Vector3d(0, 1, 2));
  EXPECT_EQ(reconstruction.Point3D(point3D_id).xyz, Eigen::Vector3d(2, 3, 4));
}

TEST(Reconstruction, FindImageWithName) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  EXPECT_EQ(reconstruction.FindImageWithName("image1"),
            &reconstruction.Image(1));
  EXPECT_EQ(reconstruction.FindImageWithName("image2"),
            &reconstruction.Image(2));
  EXPECT_TRUE(reconstruction.FindImageWithName("image3") == nullptr);
}

TEST(Reconstruction, FindCommonRegImageIds) {
  Reconstruction reconstruction1;
  GenerateReconstruction(5, &reconstruction1);
  Reconstruction reconstruction2;
  GenerateReconstruction(5, &reconstruction2);
  reconstruction1.DeRegisterFrame(1);
  reconstruction1.Image(2).SetName("foo");
  reconstruction2.DeRegisterFrame(3);
  reconstruction2.Image(4).SetName("bar");
  const auto common_image_ids =
      reconstruction1.FindCommonRegImageIds(reconstruction2);
  ASSERT_EQ(common_image_ids.size(), 1);
  EXPECT_EQ(common_image_ids[0].first, 5);
  EXPECT_EQ(common_image_ids[0].second, 5);
  EXPECT_EQ(common_image_ids,
            reconstruction2.FindCommonRegImageIds(reconstruction1));
}

TEST(Reconstruction, ComputeNumObservations) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_EQ(reconstruction.ComputeNumObservations(), 0);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  EXPECT_EQ(reconstruction.ComputeNumObservations(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 1));
  EXPECT_EQ(reconstruction.ComputeNumObservations(), 2);
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  EXPECT_EQ(reconstruction.ComputeNumObservations(), 3);
}

TEST(Reconstruction, ComputeMeanTrackLength) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  EXPECT_EQ(reconstruction.ComputeMeanTrackLength(), 0);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_EQ(reconstruction.ComputeMeanTrackLength(), 0);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  EXPECT_EQ(reconstruction.ComputeMeanTrackLength(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 1));
  EXPECT_EQ(reconstruction.ComputeMeanTrackLength(), 2);
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  EXPECT_EQ(reconstruction.ComputeMeanTrackLength(), 3);
}

TEST(Reconstruction, ComputeMeanObservationsPerRegImage) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  EXPECT_EQ(reconstruction.ComputeMeanObservationsPerRegImage(), 0);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_EQ(reconstruction.ComputeMeanObservationsPerRegImage(), 0);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  EXPECT_EQ(reconstruction.ComputeMeanObservationsPerRegImage(), 0.5);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 1));
  EXPECT_EQ(reconstruction.ComputeMeanObservationsPerRegImage(), 1.0);
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  EXPECT_EQ(reconstruction.ComputeMeanObservationsPerRegImage(), 1.5);
}

TEST(Reconstruction, ComputeMeanReprojectionError) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  EXPECT_EQ(reconstruction.ComputeMeanReprojectionError(), 0);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  EXPECT_EQ(reconstruction.ComputeMeanReprojectionError(), 0);
  reconstruction.Point3D(point3D_id1).error = 0.0;
  EXPECT_EQ(reconstruction.ComputeMeanReprojectionError(), 0);
  reconstruction.Point3D(point3D_id1).error = 1.0;
  EXPECT_EQ(reconstruction.ComputeMeanReprojectionError(), 1);
  reconstruction.Point3D(point3D_id1).error = 2.0;
  EXPECT_EQ(reconstruction.ComputeMeanReprojectionError(), 2.0);
}

TEST(Reconstruction, UpdatePoint3DErrors) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  EXPECT_EQ(reconstruction.ComputeMeanReprojectionError(), 0);
  Track track;
  track.AddElement(1, 0);
  reconstruction.Image(1).Point2D(0).xy = Eigen::Vector2d(0.5, 0.5);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 1), track);
  EXPECT_EQ(reconstruction.Point3D(point3D_id).error, -1);
  reconstruction.UpdatePoint3DErrors();
  EXPECT_EQ(reconstruction.Point3D(point3D_id).error, 0);
  reconstruction.Point3D(point3D_id).xyz = Eigen::Vector3d(0, 1, 1);
  reconstruction.UpdatePoint3DErrors();
  EXPECT_EQ(reconstruction.Point3D(point3D_id).error, 1);
}

TEST(Reconstruction, DeleteAllPoints2DAndPoints3D) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 20;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  reconstruction.DeleteAllPoints2DAndPoints3D();
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
  ExpectValidPtrs(reconstruction);
}

TEST(Reconstruction, TranscribeImageIdsToDatabase) {
  const std::vector<std::string> kImageNames = {
      "test_image1.jpg", "test_image2.jpg", "test_image3.jpg"};

  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1, 1, 1);
  database->WriteCamera(camera, /*use_camera_id=*/true);

  // Write images to database.
  Image db_image1;
  db_image1.SetName(kImageNames.at(0));
  db_image1.SetCameraId(camera.camera_id);
  db_image1.SetImageId(database->WriteImage(db_image1));
  Image db_image2;
  db_image2.SetName(kImageNames.at(1));
  db_image2.SetCameraId(camera.camera_id);
  db_image2.SetImageId(database->WriteImage(db_image2));
  Image db_image3;
  db_image3.SetName(kImageNames.at(2));
  db_image3.SetCameraId(camera.camera_id);
  db_image3.SetImageId(database->WriteImage(db_image3));

  // Create a reconstruction with different image IDs but same names.
  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(camera.SensorId());
  reconstruction.AddRig(rig);

  const std::vector<image_t> recon_image_ids = {100, 200, 300};
  for (size_t i = 0; i < 3; ++i) {
    const image_t image_id = recon_image_ids.at(i);

    Frame frame;
    frame.SetFrameId(image_id);
    frame.SetRigId(rig.RigId());
    frame.AddDataId(data_t(camera.SensorId(), image_id));
    frame.SetRigFromWorld(Rigid3d());
    reconstruction.AddFrame(frame);

    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(camera.camera_id);
    image.SetFrameId(frame.FrameId());
    image.SetName(kImageNames.at(i));
    image.SetPoints2D(
        std::vector<Eigen::Vector2d>(10, Eigen::Vector2d::Zero()));
    reconstruction.AddImage(image);
  }

  // Add a 3D point with observations to test track updates.
  Track track;
  track.AddElement(recon_image_ids.at(0), 0);
  track.AddElement(recon_image_ids.at(1), 1);
  track.AddElement(recon_image_ids.at(2), 2);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), track);

  reconstruction.TranscribeImageIdsToDatabase(*database);

  // Verify image IDs were updated to match database.
  EXPECT_TRUE(reconstruction.ExistsImage(db_image1.ImageId()));
  EXPECT_TRUE(reconstruction.ExistsImage(db_image2.ImageId()));
  EXPECT_TRUE(reconstruction.ExistsImage(db_image3.ImageId()));
  EXPECT_FALSE(reconstruction.ExistsImage(recon_image_ids.at(0)));
  EXPECT_FALSE(reconstruction.ExistsImage(recon_image_ids.at(1)));
  EXPECT_FALSE(reconstruction.ExistsImage(recon_image_ids.at(2)));

  // Verify image names are preserved.
  EXPECT_EQ(reconstruction.Image(db_image1.ImageId()).Name(),
            kImageNames.at(0));
  EXPECT_EQ(reconstruction.Image(db_image2.ImageId()).Name(),
            kImageNames.at(1));
  EXPECT_EQ(reconstruction.Image(db_image3.ImageId()).Name(),
            kImageNames.at(2));

  // Verify frame data IDs were updated.
  EXPECT_TRUE(reconstruction.Frame(recon_image_ids.at(0))
                  .HasDataId(db_image1.DataId()));
  EXPECT_TRUE(reconstruction.Frame(recon_image_ids.at(1))
                  .HasDataId(db_image2.DataId()));
  EXPECT_TRUE(reconstruction.Frame(recon_image_ids.at(2))
                  .HasDataId(db_image3.DataId()));

  // Verify track elements were updated.
  const Track& updated_track = reconstruction.Point3D(point3D_id).track;
  EXPECT_EQ(updated_track.Length(), 3);
  std::vector<image_t> track_image_ids;
  for (const auto& track_el : updated_track.Elements()) {
    track_image_ids.push_back(track_el.image_id);
  }
  EXPECT_THAT(
      track_image_ids,
      testing::UnorderedElementsAre(
          db_image1.ImageId(), db_image2.ImageId(), db_image3.ImageId()));
}

TEST(Reconstruction, IsValid) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  Track track;
  track.AddElement(1, 0);
  track.AddElement(2, 1);
  reconstruction.AddPoint3D(Eigen::Vector3d::Random(), track);
  EXPECT_TRUE(reconstruction.IsValid());

  // Test empty frame pointer for image.
  {
    Reconstruction reconstruction_copy(reconstruction);
    reconstruction_copy.Image(1).ResetFramePtr();
    EXPECT_FALSE(reconstruction_copy.IsValid());
  }

  // Test breaking track consistency by directly modifying point3D track.
  {
    Reconstruction reconstruction_copy(reconstruction);
    reconstruction_copy.Point3D(1).track.SetElement(0, TrackElement(1, 5));
    EXPECT_FALSE(reconstruction_copy.IsValid());
  }

  // Test registered frame without pose.
  {
    Reconstruction reconstruction_copy(reconstruction);
    reconstruction_copy.Frame(1).ResetPose();
    EXPECT_FALSE(reconstruction_copy.IsValid());
  }
}

TEST(Reconstruction, DeRegisterFrame) {
  Reconstruction reconstruction;
  GenerateReconstruction(3, &reconstruction);
  Track track;
  track.AddElement(1, 0);
  track.AddElement(2, 0);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), track);

  EXPECT_EQ(reconstruction.NumRegFrames(), 3);
  EXPECT_EQ(reconstruction.NumRegImages(), 3);

  reconstruction.DeRegisterFrame(1);
  EXPECT_EQ(reconstruction.NumRegFrames(), 2);
  EXPECT_EQ(reconstruction.NumRegImages(), 2);
  EXPECT_TRUE(reconstruction.ExistsFrame(1));
  EXPECT_FALSE(reconstruction.Frame(1).HasPose());
  // The 3D point had observations in images 1 and 2; after de-registering
  // frame 1, the point should be deleted (track becomes too short).
  EXPECT_FALSE(reconstruction.ExistsPoint3D(point3D_id));

  // De-registering an already de-registered frame is a no-op (with warning)
  reconstruction.DeRegisterFrame(1);
  EXPECT_EQ(reconstruction.NumRegFrames(), 2);
}

TEST(Reconstruction, TearDown) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 10;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  // De-register one frame to create an unregistered frame
  const auto reg_frames_before = reconstruction.NumRegFrames();
  const auto frame_ids = reconstruction.RegFrameIds();
  reconstruction.DeRegisterFrame(frame_ids[0]);
  EXPECT_EQ(reconstruction.NumRegFrames(), reg_frames_before - 1);

  // TearDown should remove the unregistered frame and its images
  const auto num_frames_before = reconstruction.NumFrames();
  reconstruction.TearDown();
  EXPECT_LT(reconstruction.NumFrames(), num_frames_before);
  EXPECT_EQ(reconstruction.NumFrames(), reconstruction.NumRegFrames());
  ExpectValidPtrs(reconstruction);
}

TEST(Reconstruction, ConvertToPLY) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  reconstruction.AddPoint3D(
      Eigen::Vector3d(1, 2, 3), Track(), Eigen::Vector3ub(255, 0, 0));
  reconstruction.AddPoint3D(
      Eigen::Vector3d(4, 5, 6), Track(), Eigen::Vector3ub(0, 255, 0));

  const std::vector<PlyPoint> ply_points = reconstruction.ConvertToPLY();
  EXPECT_EQ(ply_points.size(), 2);

  // Decompose into individual EXPECT calls by locating both expected points.
  const auto it_p1 =
      std::find_if(ply_points.begin(), ply_points.end(), [](const PlyPoint& p) {
        return std::abs(p.x - 1.0) < 1e-6;
      });
  const auto it_p2 =
      std::find_if(ply_points.begin(), ply_points.end(), [](const PlyPoint& p) {
        return std::abs(p.x - 4.0) < 1e-6;
      });

  ASSERT_NE(it_p1, ply_points.end());
  ASSERT_NE(it_p2, ply_points.end());

  EXPECT_NEAR(it_p1->y, 2.0, 1e-6);
  EXPECT_NEAR(it_p1->z, 3.0, 1e-6);
  EXPECT_EQ(it_p1->r, 255);
  EXPECT_EQ(it_p1->g, 0);
  EXPECT_EQ(it_p1->b, 0);

  EXPECT_NEAR(it_p2->y, 5.0, 1e-6);
  EXPECT_NEAR(it_p2->z, 6.0, 1e-6);
  EXPECT_EQ(it_p2->r, 0);
  EXPECT_EQ(it_p2->g, 255);
  EXPECT_EQ(it_p2->b, 0);
}

TEST(Reconstruction, ImportPLYFromVector) {
  Reconstruction reconstruction;
  std::vector<PlyPoint> ply_points(2);
  ply_points[0].x = 1;
  ply_points[0].y = 2;
  ply_points[0].z = 3;
  ply_points[0].r = 100;
  ply_points[0].g = 150;
  ply_points[0].b = 200;
  ply_points[1].x = 4;
  ply_points[1].y = 5;
  ply_points[1].z = 6;
  ply_points[1].r = 50;
  ply_points[1].g = 60;
  ply_points[1].b = 70;

  reconstruction.ImportPLY(ply_points);
  EXPECT_EQ(reconstruction.NumPoints3D(), 2);

  // Verify the points were imported correctly by locating both expected points.
  const auto points3D = reconstruction.Points3D();

  const auto it_p1 = std::find_if(
      points3D.begin(), points3D.end(), [](const auto& id_point3D) {
        const auto& point3D = id_point3D.second;
        return std::abs(point3D.xyz(0) - 1.0) < 1e-6;
      });
  const auto it_p2 = std::find_if(
      points3D.begin(), points3D.end(), [](const auto& id_point3D) {
        const auto& point3D = id_point3D.second;
        return std::abs(point3D.xyz(0) - 4.0) < 1e-6;
      });

  ASSERT_NE(it_p1, points3D.end());
  ASSERT_NE(it_p2, points3D.end());

  EXPECT_NEAR(it_p1->second.xyz(1), 2.0, 1e-6);
  EXPECT_NEAR(it_p1->second.xyz(2), 3.0, 1e-6);
  EXPECT_EQ(it_p1->second.color(0), 100);
  EXPECT_EQ(it_p1->second.color(1), 150);
  EXPECT_EQ(it_p1->second.color(2), 200);

  EXPECT_NEAR(it_p2->second.xyz(1), 5.0, 1e-6);
  EXPECT_NEAR(it_p2->second.xyz(2), 6.0, 1e-6);
  EXPECT_EQ(it_p2->second.color(0), 50);
  EXPECT_EQ(it_p2->second.color(1), 60);
  EXPECT_EQ(it_p2->second.color(2), 70);
}

TEST(Reconstruction, Point3DIds) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  const point3D_t p1 =
      reconstruction.AddPoint3D(Eigen::Vector3d(1, 2, 3), Track());
  const point3D_t p2 =
      reconstruction.AddPoint3D(Eigen::Vector3d(4, 5, 6), Track());

  auto ids = reconstruction.Point3DIds();
  EXPECT_EQ(ids.size(), 2);
  EXPECT_EQ(ids.count(p1), 1);
  EXPECT_EQ(ids.count(p2), 1);
}

TEST(Reconstruction, ReadWriteTextRoundtrip) {
  SetPRNGSeed(0);
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 5;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto test_dir = CreateTestDir();
  reconstruction.WriteText(test_dir);

  Reconstruction loaded;
  loaded.ReadText(test_dir);

  EXPECT_THAT(loaded, ReconstructionEq(reconstruction));
  ExpectValidPtrs(loaded);
}

TEST(Reconstruction, ReadWriteBinaryRoundtrip) {
  SetPRNGSeed(0);
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 5;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto test_dir = CreateTestDir();
  reconstruction.WriteBinary(test_dir);

  Reconstruction loaded;
  loaded.ReadBinary(test_dir);

  EXPECT_THAT(loaded, ReconstructionEq(reconstruction));
  ExpectValidPtrs(loaded);
}

TEST(Reconstruction, ReadAutoDetectFormat) {
  SetPRNGSeed(0);
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 3;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  // Write binary and verify Read auto-detects binary format
  {
    const auto test_dir = CreateTestDir();
    reconstruction.WriteBinary(test_dir);

    Reconstruction loaded;
    loaded.Read(test_dir);

    EXPECT_THAT(loaded, ReconstructionEq(reconstruction));
    ExpectValidPtrs(loaded);
  }

  // Write text and verify Read auto-detects text format
  {
    const auto test_dir = CreateTestDir();
    reconstruction.WriteText(test_dir);

    Reconstruction loaded;
    loaded.Read(test_dir);

    EXPECT_THAT(loaded, ReconstructionEq(reconstruction));
    ExpectValidPtrs(loaded);
  }
}

TEST(Reconstruction, CreateImageDirs) {
  Reconstruction reconstruction;
  GenerateReconstruction(2, &reconstruction);
  reconstruction.Image(1).SetName("subdir1/image1.jpg");
  reconstruction.Image(2).SetName("subdir2/subdir3/image2.jpg");

  const auto test_dir = CreateTestDir();
  reconstruction.CreateImageDirs(test_dir);

  EXPECT_TRUE(std::filesystem::exists(test_dir / "subdir1"));
  EXPECT_TRUE(std::filesystem::exists(test_dir / "subdir2" / "subdir3"));
}

void WriteSolidColorImages(const Reconstruction& reconstruction,
                           const std::filesystem::path& image_path,
                           const BitmapColor<uint8_t>& color) {
  for (const auto& image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    const auto& camera = reconstruction.Camera(image.CameraId());
    Bitmap bitmap(camera.width, camera.height, /*as_rgb=*/true);
    bitmap.Fill(color);
    bitmap.Write(image_path / image.Name());
  }
}

TEST(Reconstruction, ExtractColorsForAllImages) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 3;
  options.num_points3D = 10;
  options.num_points2D_without_point3D = 2;
  SynthesizeDataset(options, &reconstruction);

  const auto image_path = CreateTestDir() / "images";
  CreateDirIfNotExists(image_path);
  const BitmapColor<uint8_t> kColor(20, 40, 220);
  WriteSolidColorImages(reconstruction, image_path, kColor);

  // Delete one image file so extraction must handle the missing file.
  const auto first_image_id = *reconstruction.RegImageIds().begin();
  const auto& first_image = reconstruction.Image(first_image_id);
  std::filesystem::remove(image_path / first_image.Name());

  reconstruction.ExtractColorsForAllImages(image_path, /*num_threads=*/2);

  for (const auto& point3D_id : reconstruction.Point3DIds()) {
    const auto& color = reconstruction.Point3D(point3D_id).color;
    EXPECT_EQ(color, Eigen::Vector3ub(kColor.r, kColor.g, kColor.b));
  }
}

}  // namespace
}  // namespace colmap
