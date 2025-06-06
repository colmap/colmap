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

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point2d.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/reconstruction_io.h"
#include "colmap/scene/reconstruction_io_utils.h"
#include "colmap/scene/track.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/types.h"

#include <fstream>

namespace colmap {

void ReadRigsBinary(Reconstruction& reconstruction, std::istream& stream) {
  THROW_CHECK(stream.good());

  const size_t num_rigs = ReadBinaryLittleEndian<uint64_t>(&stream);
  for (size_t i = 0; i < num_rigs; ++i) {
    Rig rig;

    rig.SetRigId(ReadBinaryLittleEndian<rig_t>(&stream));
    const uint32_t num_sensors = ReadBinaryLittleEndian<uint32_t>(&stream);

    if (num_sensors > 0) {
      sensor_t ref_sensor_id;
      ref_sensor_id.type =
          static_cast<SensorType>(ReadBinaryLittleEndian<int>(&stream));
      ref_sensor_id.id = ReadBinaryLittleEndian<uint32_t>(&stream);
      rig.AddRefSensor(ref_sensor_id);
    }

    if (num_sensors > 1) {
      for (uint32_t j = 0; j < num_sensors - 1; ++j) {
        sensor_t sensor_id;
        sensor_id.type =
            static_cast<SensorType>(ReadBinaryLittleEndian<int>(&stream));
        sensor_id.id = ReadBinaryLittleEndian<uint32_t>(&stream);

        const bool has_pose = ReadBinaryLittleEndian<uint8_t>(&stream);
        std::optional<Rigid3d> sensor_from_rig;
        if (has_pose) {
          sensor_from_rig = Rigid3d();
          sensor_from_rig->rotation.w() =
              ReadBinaryLittleEndian<double>(&stream);
          sensor_from_rig->rotation.x() =
              ReadBinaryLittleEndian<double>(&stream);
          sensor_from_rig->rotation.y() =
              ReadBinaryLittleEndian<double>(&stream);
          sensor_from_rig->rotation.z() =
              ReadBinaryLittleEndian<double>(&stream);
          sensor_from_rig->translation.x() =
              ReadBinaryLittleEndian<double>(&stream);
          sensor_from_rig->translation.y() =
              ReadBinaryLittleEndian<double>(&stream);
          sensor_from_rig->translation.z() =
              ReadBinaryLittleEndian<double>(&stream);
        }

        rig.AddSensor(sensor_id, sensor_from_rig);
      }
    }

    reconstruction.AddRig(std::move(rig));
  }
}

void ReadRigsBinary(Reconstruction& reconstruction, const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadRigsBinary(reconstruction, file);
}

void ReadCamerasBinary(Reconstruction& reconstruction, std::istream& stream) {
  THROW_CHECK(stream.good());

  const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&stream);
  for (size_t i = 0; i < num_cameras; ++i) {
    struct Camera camera;
    camera.camera_id = ReadBinaryLittleEndian<camera_t>(&stream);
    camera.model_id =
        static_cast<CameraModelId>(ReadBinaryLittleEndian<int>(&stream));
    camera.width = ReadBinaryLittleEndian<uint64_t>(&stream);
    camera.height = ReadBinaryLittleEndian<uint64_t>(&stream);
    camera.params.resize(CameraModelNumParams(camera.model_id), 0.);
    ReadBinaryLittleEndian<double>(&stream, &camera.params);
    THROW_CHECK(camera.VerifyParams());
    reconstruction.AddCamera(std::move(camera));
  }
}

void ReadCamerasBinary(Reconstruction& reconstruction,
                       const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadCamerasBinary(reconstruction, file);
}

void ReadFramesBinary(Reconstruction& reconstruction, std::istream& stream) {
  THROW_CHECK(stream.good());

  const uint64_t num_frames = ReadBinaryLittleEndian<uint64_t>(&stream);
  for (uint64_t i = 0; i < num_frames; ++i) {
    Frame frame;

    frame.SetFrameId(ReadBinaryLittleEndian<frame_t>(&stream));
    frame.SetRigId(ReadBinaryLittleEndian<rig_t>(&stream));

    Rigid3d rig_from_world;
    rig_from_world.rotation.w() = ReadBinaryLittleEndian<double>(&stream);
    rig_from_world.rotation.x() = ReadBinaryLittleEndian<double>(&stream);
    rig_from_world.rotation.y() = ReadBinaryLittleEndian<double>(&stream);
    rig_from_world.rotation.z() = ReadBinaryLittleEndian<double>(&stream);
    rig_from_world.translation.x() = ReadBinaryLittleEndian<double>(&stream);
    rig_from_world.translation.y() = ReadBinaryLittleEndian<double>(&stream);
    rig_from_world.translation.z() = ReadBinaryLittleEndian<double>(&stream);
    frame.SetRigFromWorld(rig_from_world);

    const uint32_t num_data_ids = ReadBinaryLittleEndian<uint32_t>(&stream);
    for (uint32_t j = 0; j < num_data_ids; ++j) {
      data_t data_id;
      data_id.sensor_id.type =
          static_cast<SensorType>(ReadBinaryLittleEndian<int>(&stream));
      data_id.sensor_id.id = ReadBinaryLittleEndian<uint32_t>(&stream);
      data_id.id = ReadBinaryLittleEndian<uint64_t>(&stream);
      frame.AddDataId(data_id);
    }

    reconstruction.AddFrame(std::move(frame));
  }
}

void ReadFramesBinary(Reconstruction& reconstruction, const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadFramesBinary(reconstruction, file);
}

void ReadImagesBinary(Reconstruction& reconstruction, std::istream& stream) {
  THROW_CHECK(stream.good());

  // Handle backwards-compatibility for when we didn't have rigs and frames.
  const bool is_legacy_reconstruction =
      reconstruction.NumRigs() == 0 && reconstruction.NumFrames() == 0;
  if (is_legacy_reconstruction) {
    CreateOneRigPerCamera(reconstruction);
  }

  const std::unordered_map<image_t, Frame*> image_to_frame =
      ExtractImageToFramePtr(reconstruction);

  std::vector<Eigen::Vector2d> points2D;
  std::vector<point3D_t> point3D_ids;

  const size_t num_reg_images = ReadBinaryLittleEndian<uint64_t>(&stream);
  for (size_t i = 0; i < num_reg_images; ++i) {
    class Image image;

    image.SetImageId(ReadBinaryLittleEndian<image_t>(&stream));

    Rigid3d cam_from_world;
    cam_from_world.rotation.w() = ReadBinaryLittleEndian<double>(&stream);
    cam_from_world.rotation.x() = ReadBinaryLittleEndian<double>(&stream);
    cam_from_world.rotation.y() = ReadBinaryLittleEndian<double>(&stream);
    cam_from_world.rotation.z() = ReadBinaryLittleEndian<double>(&stream);
    cam_from_world.translation.x() = ReadBinaryLittleEndian<double>(&stream);
    cam_from_world.translation.y() = ReadBinaryLittleEndian<double>(&stream);
    cam_from_world.translation.z() = ReadBinaryLittleEndian<double>(&stream);

    image.SetCameraId(ReadBinaryLittleEndian<camera_t>(&stream));

    if (is_legacy_reconstruction) {
      CreateFrameForImage(image, cam_from_world, reconstruction);
      image.SetFrameId(image.ImageId());
      image.SetFramePtr(&reconstruction.Frame(image.ImageId()));
    } else {
      Frame* frame = image_to_frame.at(image.ImageId());
      image.SetFrameId(frame->FrameId());
      image.SetFramePtr(frame);
    }

    char name_char;
    do {
      stream.read(&name_char, 1);
      if (name_char != '\0') {
        image.Name() += name_char;
      }
    } while (name_char != '\0');

    const size_t num_points2D = ReadBinaryLittleEndian<uint64_t>(&stream);

    points2D.clear();
    points2D.reserve(num_points2D);
    point3D_ids.clear();
    point3D_ids.reserve(num_points2D);
    for (size_t j = 0; j < num_points2D; ++j) {
      const double x = ReadBinaryLittleEndian<double>(&stream);
      const double y = ReadBinaryLittleEndian<double>(&stream);
      points2D.emplace_back(x, y);
      point3D_ids.push_back(ReadBinaryLittleEndian<point3D_t>(&stream));
    }

    image.SetPoints2D(points2D);

    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      if (point3D_ids[point2D_idx] != kInvalidPoint3DId) {
        image.SetPoint3DForPoint2D(point2D_idx, point3D_ids[point2D_idx]);
      }
    }

    reconstruction.AddImage(std::move(image));
  }
}

void ReadImagesBinary(Reconstruction& reconstruction, const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadImagesBinary(reconstruction, file);
}

void ReadPoints3DBinary(Reconstruction& reconstruction, std::istream& stream) {
  THROW_CHECK(stream.good());

  const size_t num_points3D = ReadBinaryLittleEndian<uint64_t>(&stream);
  for (size_t i = 0; i < num_points3D; ++i) {
    struct Point3D point3D;

    const point3D_t point3D_id = ReadBinaryLittleEndian<point3D_t>(&stream);

    point3D.xyz(0) = ReadBinaryLittleEndian<double>(&stream);
    point3D.xyz(1) = ReadBinaryLittleEndian<double>(&stream);
    point3D.xyz(2) = ReadBinaryLittleEndian<double>(&stream);
    point3D.color(0) = ReadBinaryLittleEndian<uint8_t>(&stream);
    point3D.color(1) = ReadBinaryLittleEndian<uint8_t>(&stream);
    point3D.color(2) = ReadBinaryLittleEndian<uint8_t>(&stream);
    point3D.error = ReadBinaryLittleEndian<double>(&stream);

    const size_t track_length = ReadBinaryLittleEndian<uint64_t>(&stream);
    for (size_t j = 0; j < track_length; ++j) {
      const image_t image_id = ReadBinaryLittleEndian<image_t>(&stream);
      const point2D_t point2D_idx = ReadBinaryLittleEndian<point2D_t>(&stream);
      point3D.track.AddElement(image_id, point2D_idx);
    }
    point3D.track.Compress();

    reconstruction.AddPoint3D(point3D_id, std::move(point3D));
  }
}

void ReadPoints3DBinary(Reconstruction& reconstruction,
                        const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadPoints3DBinary(reconstruction, file);
}

void WriteRigsBinary(const Reconstruction& reconstruction,
                     std::ostream& stream) {
  THROW_CHECK(stream.good());

  WriteBinaryLittleEndian<uint64_t>(&stream, reconstruction.NumRigs());

  for (const rig_t rig_id : ExtractSortedIds(reconstruction.Rigs())) {
    const Rig& rig = reconstruction.Rig(rig_id);
    WriteBinaryLittleEndian<rig_t>(&stream, rig_id);
    WriteBinaryLittleEndian<uint32_t>(&stream, rig.NumSensors());
    if (rig.NumSensors() > 0) {
      WriteBinaryLittleEndian<int>(&stream,
                                   static_cast<int>(rig.RefSensorId().type));
      WriteBinaryLittleEndian<uint32_t>(&stream, rig.RefSensorId().id);
    }
    for (const auto& [sensor_id, sensor_from_rig] : rig.Sensors()) {
      WriteBinaryLittleEndian<int>(&stream, static_cast<int>(sensor_id.type));
      WriteBinaryLittleEndian<uint32_t>(&stream, sensor_id.id);
      WriteBinaryLittleEndian<uint8_t>(&stream,
                                       sensor_from_rig.has_value() ? 1 : 0);
      if (sensor_from_rig.has_value()) {
        WriteBinaryLittleEndian<double>(&stream, sensor_from_rig->rotation.w());
        WriteBinaryLittleEndian<double>(&stream, sensor_from_rig->rotation.x());
        WriteBinaryLittleEndian<double>(&stream, sensor_from_rig->rotation.y());
        WriteBinaryLittleEndian<double>(&stream, sensor_from_rig->rotation.z());
        WriteBinaryLittleEndian<double>(&stream,
                                        sensor_from_rig->translation.x());
        WriteBinaryLittleEndian<double>(&stream,
                                        sensor_from_rig->translation.y());
        WriteBinaryLittleEndian<double>(&stream,
                                        sensor_from_rig->translation.z());
      }
    }
  }
}

void WriteRigsBinary(const Reconstruction& reconstruction,
                     const std::string& path) {
  std::ofstream file(path, std::ios::trunc | std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  WriteRigsBinary(reconstruction, file);
}

void WriteCamerasBinary(const Reconstruction& reconstruction,
                        std::ostream& stream) {
  THROW_CHECK(stream.good());

  WriteBinaryLittleEndian<uint64_t>(&stream, reconstruction.NumCameras());

  for (const camera_t camera_id : ExtractSortedIds(reconstruction.Cameras())) {
    const Camera& camera = reconstruction.Camera(camera_id);
    WriteBinaryLittleEndian<camera_t>(&stream, camera_id);
    WriteBinaryLittleEndian<int>(&stream, static_cast<int>(camera.model_id));
    WriteBinaryLittleEndian<uint64_t>(&stream, camera.width);
    WriteBinaryLittleEndian<uint64_t>(&stream, camera.height);
    for (const double param : camera.params) {
      WriteBinaryLittleEndian<double>(&stream, param);
    }
  }
}

void WriteCamerasBinary(const Reconstruction& reconstruction,
                        const std::string& path) {
  std::ofstream file(path, std::ios::trunc | std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  WriteCamerasBinary(reconstruction, file);
}

void WriteFramesBinary(const Reconstruction& reconstruction,
                       std::ostream& stream) {
  THROW_CHECK(stream.good());

  const std::vector<frame_t> frame_ids = ExtractSortedIds<frame_t, Frame>(
      reconstruction.Frames(),
      [](const Frame& frame) { return frame.HasPose(); });

  WriteBinaryLittleEndian<uint64_t>(&stream, frame_ids.size());

  for (const frame_t frame_id : frame_ids) {
    const Frame& frame = reconstruction.Frame(frame_id);

    WriteBinaryLittleEndian<frame_t>(&stream, frame_id);
    WriteBinaryLittleEndian<rig_t>(&stream, frame.RigId());

    const Rigid3d& rig_from_world = frame.RigFromWorld();
    WriteBinaryLittleEndian<double>(&stream, rig_from_world.rotation.w());
    WriteBinaryLittleEndian<double>(&stream, rig_from_world.rotation.x());
    WriteBinaryLittleEndian<double>(&stream, rig_from_world.rotation.y());
    WriteBinaryLittleEndian<double>(&stream, rig_from_world.rotation.z());
    WriteBinaryLittleEndian<double>(&stream, rig_from_world.translation.x());
    WriteBinaryLittleEndian<double>(&stream, rig_from_world.translation.y());
    WriteBinaryLittleEndian<double>(&stream, rig_from_world.translation.z());

    const std::set<data_t>& data_ids = frame.DataIds();
    WriteBinaryLittleEndian<uint32_t>(&stream, data_ids.size());
    for (const data_t& data_id : data_ids) {
      WriteBinaryLittleEndian<int>(&stream,
                                   static_cast<int>(data_id.sensor_id.type));
      WriteBinaryLittleEndian<uint32_t>(&stream, data_id.sensor_id.id);
      WriteBinaryLittleEndian<uint64_t>(&stream, data_id.id);
    }
  }
}

void WriteFramesBinary(const Reconstruction& reconstruction,
                       const std::string& path) {
  std::ofstream file(path, std::ios::trunc | std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  WriteFramesBinary(reconstruction, file);
}

void WriteImagesBinary(const Reconstruction& reconstruction,
                       std::ostream& stream) {
  THROW_CHECK(stream.good());

  WriteBinaryLittleEndian<uint64_t>(&stream, reconstruction.NumRegImages());

  for (const image_t image_id : reconstruction.RegImageIds()) {
    const Image& image = reconstruction.Image(image_id);

    WriteBinaryLittleEndian<image_t>(&stream, image_id);

    const Rigid3d& cam_from_world = image.CamFromWorld();
    WriteBinaryLittleEndian<double>(&stream, cam_from_world.rotation.w());
    WriteBinaryLittleEndian<double>(&stream, cam_from_world.rotation.x());
    WriteBinaryLittleEndian<double>(&stream, cam_from_world.rotation.y());
    WriteBinaryLittleEndian<double>(&stream, cam_from_world.rotation.z());
    WriteBinaryLittleEndian<double>(&stream, cam_from_world.translation.x());
    WriteBinaryLittleEndian<double>(&stream, cam_from_world.translation.y());
    WriteBinaryLittleEndian<double>(&stream, cam_from_world.translation.z());

    WriteBinaryLittleEndian<camera_t>(&stream, image.CameraId());

    const std::string name = image.Name() + '\0';
    stream.write(name.c_str(), name.size());

    WriteBinaryLittleEndian<uint64_t>(&stream, image.NumPoints2D());
    for (const Point2D& point2D : image.Points2D()) {
      WriteBinaryLittleEndian<double>(&stream, point2D.xy(0));
      WriteBinaryLittleEndian<double>(&stream, point2D.xy(1));
      WriteBinaryLittleEndian<point3D_t>(&stream, point2D.point3D_id);
    }
  }
}

void WriteImagesBinary(const Reconstruction& reconstruction,
                       const std::string& path) {
  std::ofstream file(path, std::ios::trunc | std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  WriteImagesBinary(reconstruction, file);
}

void WritePoints3DBinary(const Reconstruction& reconstruction,
                         std::ostream& stream) {
  THROW_CHECK(stream.good());

  WriteBinaryLittleEndian<uint64_t>(&stream, reconstruction.NumPoints3D());

  for (const point3D_t point3D_id :
       ExtractSortedIds(reconstruction.Points3D())) {
    const Point3D& point3D = reconstruction.Point3D(point3D_id);

    WriteBinaryLittleEndian<point3D_t>(&stream, point3D_id);
    WriteBinaryLittleEndian<double>(&stream, point3D.xyz(0));
    WriteBinaryLittleEndian<double>(&stream, point3D.xyz(1));
    WriteBinaryLittleEndian<double>(&stream, point3D.xyz(2));
    WriteBinaryLittleEndian<uint8_t>(&stream, point3D.color(0));
    WriteBinaryLittleEndian<uint8_t>(&stream, point3D.color(1));
    WriteBinaryLittleEndian<uint8_t>(&stream, point3D.color(2));
    WriteBinaryLittleEndian<double>(&stream, point3D.error);

    WriteBinaryLittleEndian<uint64_t>(&stream, point3D.track.Length());
    for (const TrackElement& track_el : point3D.track.Elements()) {
      WriteBinaryLittleEndian<image_t>(&stream, track_el.image_id);
      WriteBinaryLittleEndian<point2D_t>(&stream, track_el.point2D_idx);
    }
  }
}

void WritePoints3DBinary(const Reconstruction& reconstruction,
                         const std::string& path) {
  std::ofstream file(path, std::ios::trunc | std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  WritePoints3DBinary(reconstruction, file);
}

}  // namespace colmap
