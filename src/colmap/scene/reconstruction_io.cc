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

#include "colmap/scene/reconstruction_io.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point2d.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/track.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/ply.h"
#include "colmap/util/types.h"

#include <fstream>

namespace colmap {
namespace {

// Helper method to extract sorted camera, image, point3D identifiers.
// We sort the identifiers before writing to the stream, such that we produce
// deterministic output independent of standard library dependent ordering of
// the unordered map container.
template <typename ID_TYPE, typename DATA_TYPE>
std::vector<ID_TYPE> ExtractSortedIds(
    const std::unordered_map<ID_TYPE, DATA_TYPE>& data) {
  std::vector<ID_TYPE> ids;
  ids.reserve(data.size());
  for (const auto& [id, _] : data) {
    ids.push_back(id);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

}  // namespace

void ReadCamerasText(Reconstruction& reconstruction, std::istream& stream) {
  THROW_CHECK(stream.good());

  std::string line;
  std::string item;

  while (std::getline(stream, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream line_stream(line);

    struct Camera camera;

    // ID
    std::getline(line_stream, item, ' ');
    camera.camera_id = std::stoul(item);

    // MODEL
    std::getline(line_stream, item, ' ');
    camera.model_id = CameraModelNameToId(item);

    // WIDTH
    std::getline(line_stream, item, ' ');
    camera.width = std::stoll(item);

    // HEIGHT
    std::getline(line_stream, item, ' ');
    camera.height = std::stoll(item);

    // PARAMS
    camera.params.reserve(CameraModelNumParams(camera.model_id));
    while (!line_stream.eof()) {
      std::getline(line_stream, item, ' ');
      camera.params.push_back(std::stold(item));
    }

    THROW_CHECK(camera.VerifyParams());
    reconstruction.AddCamera(std::move(camera));
  }
}

void ReadCamerasText(Reconstruction& reconstruction, const std::string& path) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadCamerasText(reconstruction, file);
}

void ReadImagesText(Reconstruction& reconstruction, std::istream& stream) {
  THROW_CHECK(stream.good());

  std::string line;
  std::string item;

  std::vector<Eigen::Vector2d> points2D;
  std::vector<point3D_t> point3D_ids;

  while (std::getline(stream, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream line_stream1(line);

    // ID
    std::getline(line_stream1, item, ' ');
    const image_t image_id = std::stoul(item);

    class Image image;
    image.SetImageId(image_id);

    Rigid3d cam_from_world;

    std::getline(line_stream1, item, ' ');
    cam_from_world.rotation.w() = std::stold(item);

    std::getline(line_stream1, item, ' ');
    cam_from_world.rotation.x() = std::stold(item);

    std::getline(line_stream1, item, ' ');
    cam_from_world.rotation.y() = std::stold(item);

    std::getline(line_stream1, item, ' ');
    cam_from_world.rotation.z() = std::stold(item);

    std::getline(line_stream1, item, ' ');
    cam_from_world.translation.x() = std::stold(item);

    std::getline(line_stream1, item, ' ');
    cam_from_world.translation.y() = std::stold(item);

    std::getline(line_stream1, item, ' ');
    cam_from_world.translation.z() = std::stold(item);

    image.SetCamFromWorld(cam_from_world);

    // CAMERA_ID
    std::getline(line_stream1, item, ' ');
    image.SetCameraId(std::stoul(item));

    // NAME
    std::getline(line_stream1, item, ' ');
    image.SetName(item);

    // POINTS2D
    if (!std::getline(stream, line)) {
      break;
    }

    StringTrim(&line);
    std::stringstream line_stream2(line);

    points2D.clear();
    point3D_ids.clear();

    if (!line.empty()) {
      while (!line_stream2.eof()) {
        Eigen::Vector2d point;

        std::getline(line_stream2, item, ' ');
        point.x() = std::stold(item);

        std::getline(line_stream2, item, ' ');
        point.y() = std::stold(item);

        points2D.push_back(point);

        std::getline(line_stream2, item, ' ');
        if (item == "-1") {
          point3D_ids.push_back(kInvalidPoint3DId);
        } else {
          point3D_ids.push_back(std::stoll(item));
        }
      }
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

void ReadImagesText(Reconstruction& reconstruction, const std::string& path) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadImagesText(reconstruction, file);
}

void ReadPoints3DText(Reconstruction& reconstruction, std::istream& stream) {
  THROW_CHECK(stream.good());

  std::string line;
  std::string item;

  while (std::getline(stream, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream line_stream(line);

    // ID
    std::getline(line_stream, item, ' ');
    const point3D_t point3D_id = std::stoll(item);

    struct Point3D point3D;

    // XYZ
    std::getline(line_stream, item, ' ');
    point3D.xyz(0) = std::stold(item);

    std::getline(line_stream, item, ' ');
    point3D.xyz(1) = std::stold(item);

    std::getline(line_stream, item, ' ');
    point3D.xyz(2) = std::stold(item);

    // Color
    std::getline(line_stream, item, ' ');
    point3D.color(0) = static_cast<uint8_t>(std::stoi(item));

    std::getline(line_stream, item, ' ');
    point3D.color(1) = static_cast<uint8_t>(std::stoi(item));

    std::getline(line_stream, item, ' ');
    point3D.color(2) = static_cast<uint8_t>(std::stoi(item));

    // ERROR
    std::getline(line_stream, item, ' ');
    point3D.error = std::stold(item);

    // TRACK
    while (!line_stream.eof()) {
      TrackElement track_el;

      std::getline(line_stream, item, ' ');
      StringTrim(&item);
      if (item.empty()) {
        break;
      }
      track_el.image_id = std::stoul(item);

      std::getline(line_stream, item, ' ');
      track_el.point2D_idx = std::stoul(item);

      point3D.track.AddElement(track_el);
    }

    point3D.track.Compress();

    reconstruction.AddPoint3D(point3D_id, std::move(point3D));
  }
}

void ReadPoints3DText(Reconstruction& reconstruction, const std::string& path) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadPoints3DText(reconstruction, file);
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

void ReadImagesBinary(Reconstruction& reconstruction, std::istream& stream) {
  THROW_CHECK(stream.good());

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
    image.SetCamFromWorld(cam_from_world);

    image.SetCameraId(ReadBinaryLittleEndian<camera_t>(&stream));

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

void WriteCamerasText(const Reconstruction& reconstruction,
                      std::ostream& stream) {
  THROW_CHECK(stream.good());

  // Ensure that we don't loose any precision by storing in text.
  stream.precision(17);

  stream << "# Camera list with one line of data per camera:" << std::endl;
  stream << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" << std::endl;
  stream << "# Number of cameras: " << reconstruction.NumCameras() << std::endl;

  for (const camera_t camera_id : ExtractSortedIds(reconstruction.Cameras())) {
    const Camera& camera = reconstruction.Camera(camera_id);

    std::ostringstream line;
    line.precision(17);

    line << camera_id << " ";
    line << camera.ModelName() << " ";
    line << camera.width << " ";
    line << camera.height << " ";

    for (const double param : camera.params) {
      line << param << " ";
    }

    std::string line_string = line.str();
    line_string = line_string.substr(0, line_string.size() - 1);

    stream << line_string << std::endl;
  }
}

void WriteCamerasText(const Reconstruction& reconstruction,
                      const std::string& path) {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  WriteCamerasText(reconstruction, file);
}

void WriteImagesText(const Reconstruction& reconstruction,
                     std::ostream& stream) {
  THROW_CHECK(stream.good());

  // Ensure that we don't loose any precision by storing in text.
  stream.precision(17);

  stream << "# Image list with two lines of data per image:" << std::endl;
  stream << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, "
            "NAME"
         << std::endl;
  stream << "#   POINTS2D[] as (X, Y, POINT3D_ID)" << std::endl;
  stream << "# Number of images: " << reconstruction.NumRegImages()
         << ", mean observations per image: "
         << reconstruction.ComputeMeanObservationsPerRegImage() << std::endl;

  std::ostringstream line;
  line.precision(17);

  for (const image_t image_id : reconstruction.RegImageIds()) {
    const Image& image = reconstruction.Image(image_id);

    line.str("");
    line.clear();

    line << image_id << " ";

    const Rigid3d& cam_from_world = image.CamFromWorld();
    line << cam_from_world.rotation.w() << " ";
    line << cam_from_world.rotation.x() << " ";
    line << cam_from_world.rotation.y() << " ";
    line << cam_from_world.rotation.z() << " ";
    line << cam_from_world.translation.x() << " ";
    line << cam_from_world.translation.y() << " ";
    line << cam_from_world.translation.z() << " ";

    line << image.CameraId() << " ";

    line << image.Name();

    stream << line.str() << std::endl;

    line.str("");
    line.clear();

    for (const Point2D& point2D : image.Points2D()) {
      line << point2D.xy(0) << " ";
      line << point2D.xy(1) << " ";
      if (point2D.HasPoint3D()) {
        line << point2D.point3D_id << " ";
      } else {
        line << -1 << " ";
      }
    }
    if (image.NumPoints2D() > 0) {
      line.seekp(-1, std::ios_base::end);
    }
    stream << line.str() << std::endl;
  }
}

void WriteImagesText(const Reconstruction& reconstruction,
                     const std::string& path) {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  WriteImagesText(reconstruction, file);
}

void WritePoints3DText(const Reconstruction& reconstruction,
                       std::ostream& stream) {
  THROW_CHECK(stream.good());

  // Ensure that we don't loose any precision by storing in text.
  stream.precision(17);

  stream << "# 3D point list with one line of data per point:" << std::endl;
  stream << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)"
         << std::endl;
  stream << "# Number of points: " << reconstruction.NumPoints3D()
         << ", mean track length: " << reconstruction.ComputeMeanTrackLength()
         << std::endl;

  for (const point3D_t point3D_id :
       ExtractSortedIds(reconstruction.Points3D())) {
    const Point3D& point3D = reconstruction.Point3D(point3D_id);

    stream << point3D_id << " ";
    stream << point3D.xyz(0) << " ";
    stream << point3D.xyz(1) << " ";
    stream << point3D.xyz(2) << " ";
    stream << static_cast<int>(point3D.color(0)) << " ";
    stream << static_cast<int>(point3D.color(1)) << " ";
    stream << static_cast<int>(point3D.color(2)) << " ";
    stream << point3D.error << " ";

    std::ostringstream line;
    line.precision(17);

    for (const auto& track_el : point3D.track.Elements()) {
      line << track_el.image_id << " ";
      line << track_el.point2D_idx << " ";
    }

    std::string line_string = line.str();
    line_string = line_string.substr(0, line_string.size() - 1);

    stream << line_string << std::endl;
  }
}

void WritePoints3DText(const Reconstruction& reconstruction,
                       const std::string& path) {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  WritePoints3DText(reconstruction, file);
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

bool ExportNVM(const Reconstruction& reconstruction,
               const std::string& path,
               bool skip_distortion) {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  // Ensure that we don't lose any precision by storing in text.
  file.precision(17);

  // White space added for compatibility with Meshlab.
  file << "NVM_V3 " << std::endl << " " << std::endl;
  file << reconstruction.NumRegImages() << "  " << std::endl;

  std::unordered_map<image_t, size_t> image_id_to_idx_;
  size_t image_idx = 0;

  for (const auto image_id : reconstruction.RegImageIds()) {
    const class Image& image = reconstruction.Image(image_id);
    const struct Camera& camera = reconstruction.Camera(image.CameraId());

    double k;
    if (skip_distortion ||
        camera.model_id == SimplePinholeCameraModel::model_id ||
        camera.model_id == PinholeCameraModel::model_id) {
      k = 0.0;
    } else if (camera.model_id == SimpleRadialCameraModel::model_id) {
      k = -1 * camera.params[SimpleRadialCameraModel::extra_params_idxs[0]];
    } else {
      LOG(WARNING) << "NVM only supports `SIMPLE_RADIAL` "
                      "and pinhole camera models."
                   << std::endl;
      return false;
    }

    const Eigen::Vector3d proj_center = image.ProjectionCenter();

    file << image.Name() << " ";
    file << camera.MeanFocalLength() << " ";
    file << image.CamFromWorld().rotation.w() << " ";
    file << image.CamFromWorld().rotation.x() << " ";
    file << image.CamFromWorld().rotation.y() << " ";
    file << image.CamFromWorld().rotation.z() << " ";
    file << proj_center.x() << " ";
    file << proj_center.y() << " ";
    file << proj_center.z() << " ";
    file << k << " ";
    file << 0 << std::endl;

    image_id_to_idx_[image_id] = image_idx;
    image_idx += 1;
  }

  file << std::endl << reconstruction.NumPoints3D() << std::endl;

  for (const auto& point3D : reconstruction.Points3D()) {
    file << point3D.second.xyz(0) << " ";
    file << point3D.second.xyz(1) << " ";
    file << point3D.second.xyz(2) << " ";
    file << static_cast<int>(point3D.second.color(0)) << " ";
    file << static_cast<int>(point3D.second.color(1)) << " ";
    file << static_cast<int>(point3D.second.color(2)) << " ";

    std::ostringstream line;

    std::unordered_set<image_t> image_ids;
    for (const auto& track_el : point3D.second.track.Elements()) {
      // Make sure that each point only has a single observation per image,
      // since VisualSfM does not support with multiple observations.
      if (image_ids.count(track_el.image_id) == 0) {
        const class Image& image = reconstruction.Image(track_el.image_id);
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);
        line << image_id_to_idx_[track_el.image_id] << " ";
        line << track_el.point2D_idx << " ";
        line << point2D.xy(0) << " ";
        line << point2D.xy(1) << " ";
        image_ids.insert(track_el.image_id);
      }
    }

    std::string line_string = line.str();
    line_string = line_string.substr(0, line_string.size() - 1);

    file << image_ids.size() << " ";
    file << line_string << std::endl;
  }

  return true;
}

bool ExportCam(const Reconstruction& reconstruction,
               const std::string& path,
               bool skip_distortion) {
  reconstruction.CreateImageDirs(path);
  for (const auto image_id : reconstruction.RegImageIds()) {
    std::string name, ext;
    const class Image& image = reconstruction.Image(image_id);
    const struct Camera& camera = reconstruction.Camera(image.CameraId());

    SplitFileExtension(image.Name(), &name, &ext);
    name = JoinPaths(path, name.append(".cam"));
    std::ofstream file(name, std::ios::trunc);

    THROW_CHECK_FILE_OPEN(file, name);

    // Ensure that we don't lose any precision by storing in text.
    file.precision(17);

    double k1, k2;
    if (skip_distortion ||
        camera.model_id == SimplePinholeCameraModel::model_id ||
        camera.model_id == PinholeCameraModel::model_id) {
      k1 = 0.0;
      k2 = 0.0;
    } else if (camera.model_id == SimpleRadialCameraModel::model_id) {
      k1 = camera.params[SimpleRadialCameraModel::extra_params_idxs[0]];
      k2 = 0.0;
    } else if (camera.model_id == RadialCameraModel::model_id) {
      k1 = camera.params[RadialCameraModel::extra_params_idxs[0]];
      k2 = camera.params[RadialCameraModel::extra_params_idxs[1]];
    } else {
      LOG(WARNING) << "CAM only supports `SIMPLE_RADIAL`, `RADIAL`, "
                      "and pinhole camera models."
                   << std::endl;
      return false;
    }

    // If both k1 and k2 values are non-zero, then the CAM format assumes
    // a Bundler-like radial distortion model, which converts well from
    // COLMAP. However, if k2 is zero, then a different model is used
    // that does not translate as well, so we avoid setting k2 to zero.
    if (k1 != 0.0 && k2 == 0.0) {
      k2 = 1e-10;
    }

    const double fx = camera.FocalLengthX();
    const double fy = camera.FocalLengthY();
    double focal_length;
    if (camera.width * fy < camera.height * fx) {
      focal_length = fy / camera.height;
    } else {
      focal_length = fx / camera.width;
    }

    const Eigen::Matrix3d R = image.CamFromWorld().rotation.toRotationMatrix();
    file << image.CamFromWorld().translation.x() << " "
         << image.CamFromWorld().translation.y() << " "
         << image.CamFromWorld().translation.z() << " " << R(0, 0) << " "
         << R(0, 1) << " " << R(0, 2) << " " << R(1, 0) << " " << R(1, 1) << " "
         << R(1, 2) << " " << R(2, 0) << " " << R(2, 1) << " " << R(2, 2)
         << std::endl;
    file << focal_length << " " << k1 << " " << k2 << " " << fy / fx << " "
         << camera.PrincipalPointX() / camera.width << " "
         << camera.PrincipalPointY() / camera.height << std::endl;
  }

  return true;
}

bool ExportRecon3D(const Reconstruction& reconstruction,
                   const std::string& path,
                   bool skip_distortion) {
  std::string base_path = EnsureTrailingSlash(StringReplace(path, "\\", "/"));
  CreateDirIfNotExists(base_path);
  base_path = base_path.append("Recon/");
  CreateDirIfNotExists(base_path);
  std::string synth_path = base_path + "synth_0.out";
  std::string image_list_path = base_path + "urd-images.txt";
  std::string image_map_path = base_path + "imagemap_0.txt";

  std::ofstream synth_file(synth_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(synth_file, synth_path);
  std::ofstream image_list_file(image_list_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(image_list_file, image_list_path);
  std::ofstream image_map_file(image_map_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(image_map_file, image_map_path);

  // Ensure that we don't lose any precision by storing in text.
  synth_file.precision(17);

  // Write header info
  synth_file << "colmap 1.0" << std::endl;
  synth_file << reconstruction.NumRegImages() << " "
             << reconstruction.NumPoints3D() << std::endl;

  std::unordered_map<image_t, size_t> image_id_to_idx_;
  size_t image_idx = 0;

  // Write image/camera info
  for (const auto image_id : reconstruction.RegImageIds()) {
    const class Image& image = reconstruction.Image(image_id);
    const struct Camera& camera = reconstruction.Camera(image.CameraId());

    double k1, k2;
    if (skip_distortion ||
        camera.model_id == SimplePinholeCameraModel::model_id ||
        camera.model_id == PinholeCameraModel::model_id) {
      k1 = 0.0;
      k2 = 0.0;
    } else if (camera.model_id == SimpleRadialCameraModel::model_id) {
      k1 = -1 * camera.params[SimpleRadialCameraModel::extra_params_idxs[0]];
      k2 = 0.0;
    } else if (camera.model_id == RadialCameraModel::model_id) {
      k1 = -1 * camera.params[RadialCameraModel::extra_params_idxs[0]];
      k2 = -1 * camera.params[RadialCameraModel::extra_params_idxs[1]];
    } else {
      LOG(WARNING) << "Recon3D only supports `SIMPLE_RADIAL`, "
                      "`RADIAL`, and pinhole camera models.";
      return false;
    }

    const double scale = 1.0 / (double)std::max(camera.width, camera.height);
    synth_file << scale * camera.MeanFocalLength() << " " << k1 << " " << k2
               << std::endl;
    synth_file << image.CamFromWorld().rotation.toRotationMatrix() << std::endl;
    synth_file << image.CamFromWorld().translation.transpose() << std::endl;

    image_id_to_idx_[image_id] = image_idx;
    image_list_file << image.Name() << std::endl
                    << camera.width << " " << camera.height << std::endl;
    image_map_file << image_idx << std::endl;

    image_idx += 1;
  }
  image_list_file.close();
  image_map_file.close();

  // Write point info
  for (const auto& point3D : reconstruction.Points3D()) {
    auto& p = point3D.second;
    synth_file << p.xyz(0) << " " << p.xyz(1) << " " << p.xyz(2) << std::endl;
    synth_file << static_cast<int>(p.color(0)) << " "
               << static_cast<int>(p.color(1)) << " "
               << static_cast<int>(p.color(2)) << std::endl;

    std::ostringstream line;

    std::unordered_set<image_t> image_ids;
    for (const auto& track_el : p.track.Elements()) {
      // Make sure that each point only has a single observation per image,
      // since VisualSfM does not support with multiple observations.
      if (image_ids.count(track_el.image_id) == 0) {
        const class Image& image = reconstruction.Image(track_el.image_id);
        const struct Camera& camera = reconstruction.Camera(image.CameraId());
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);

        const double scale =
            1.0 / (double)std::max(camera.width, camera.height);

        line << image_id_to_idx_[track_el.image_id] << " ";
        line << track_el.point2D_idx << " ";
        // Use a scale of -1.0 to mark as invalid as it is not needed currently
        line << "-1.0 ";
        line << (point2D.xy(0) - camera.PrincipalPointX()) * scale << " ";
        line << (point2D.xy(1) - camera.PrincipalPointY()) * scale << " ";
        image_ids.insert(track_el.image_id);
      }
    }

    std::string line_string = line.str();
    line_string = line_string.substr(0, line_string.size() - 1);

    synth_file << image_ids.size() << " ";
    synth_file << line_string << std::endl;
  }
  synth_file.close();

  return true;
}

bool ExportBundler(const Reconstruction& reconstruction,
                   const std::string& path,
                   const std::string& list_path,
                   bool skip_distortion) {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  std::ofstream list_file(list_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(list_file, list_path);

  // Ensure that we don't lose any precision by storing in text.
  file.precision(17);

  file << "# Bundle file v0.3" << std::endl;

  file << reconstruction.NumRegImages() << " " << reconstruction.NumPoints3D()
       << std::endl;

  std::unordered_map<image_t, size_t> image_id_to_idx_;
  size_t image_idx = 0;

  for (const image_t image_id : reconstruction.RegImageIds()) {
    const class Image& image = reconstruction.Image(image_id);
    const struct Camera& camera = reconstruction.Camera(image.CameraId());

    double k1, k2;
    if (skip_distortion ||
        camera.model_id == SimplePinholeCameraModel::model_id ||
        camera.model_id == PinholeCameraModel::model_id) {
      k1 = 0.0;
      k2 = 0.0;
    } else if (camera.model_id == SimpleRadialCameraModel::model_id) {
      k1 = camera.params[SimpleRadialCameraModel::extra_params_idxs[0]];
      k2 = 0.0;
    } else if (camera.model_id == RadialCameraModel::model_id) {
      k1 = camera.params[RadialCameraModel::extra_params_idxs[0]];
      k2 = camera.params[RadialCameraModel::extra_params_idxs[1]];
    } else {
      LOG(WARNING) << "Bundler only supports `SIMPLE_RADIAL`, "
                      "`RADIAL`, and pinhole camera models."
                   << std::endl;
      return false;
    }

    file << camera.MeanFocalLength() << " " << k1 << " " << k2 << std::endl;

    const Eigen::Matrix3d R = image.CamFromWorld().rotation.toRotationMatrix();
    file << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << std::endl;
    file << -R(1, 0) << " " << -R(1, 1) << " " << -R(1, 2) << std::endl;
    file << -R(2, 0) << " " << -R(2, 1) << " " << -R(2, 2) << std::endl;

    file << image.CamFromWorld().translation.x() << " ";
    file << -image.CamFromWorld().translation.y() << " ";
    file << -image.CamFromWorld().translation.z() << std::endl;

    list_file << image.Name() << std::endl;

    image_id_to_idx_[image_id] = image_idx;
    image_idx += 1;
  }

  for (const auto& point3D : reconstruction.Points3D()) {
    file << point3D.second.xyz(0) << " ";
    file << point3D.second.xyz(1) << " ";
    file << point3D.second.xyz(2) << std::endl;

    file << static_cast<int>(point3D.second.color(0)) << " ";
    file << static_cast<int>(point3D.second.color(1)) << " ";
    file << static_cast<int>(point3D.second.color(2)) << std::endl;

    std::ostringstream line;

    line << point3D.second.track.Length() << " ";

    for (const auto& track_el : point3D.second.track.Elements()) {
      const class Image& image = reconstruction.Image(track_el.image_id);
      const struct Camera& camera = reconstruction.Camera(image.CameraId());

      // Bundler output assumes image coordinate system origin
      // in the lower left corner of the image with the center of
      // the lower left pixel being (0, 0). Our coordinate system
      // starts in the upper left corner with the center of the
      // upper left pixel being (0.5, 0.5).

      const Point2D& point2D = image.Point2D(track_el.point2D_idx);

      line << image_id_to_idx_.at(track_el.image_id) << " ";
      line << track_el.point2D_idx << " ";
      line << point2D.xy(0) - camera.PrincipalPointX() << " ";
      line << camera.PrincipalPointY() - point2D.xy(1) << " ";
    }

    std::string line_string = line.str();
    line_string = line_string.substr(0, line_string.size() - 1);

    file << line_string << std::endl;
  }

  return true;
}

void ExportPLY(const Reconstruction& reconstruction, const std::string& path) {
  const auto ply_points = reconstruction.ConvertToPLY();

  const bool kWriteNormal = false;
  const bool kWriteRGB = true;
  WriteBinaryPlyPoints(path, ply_points, kWriteNormal, kWriteRGB);
}

void ExportVRML(const Reconstruction& reconstruction,
                const std::string& images_path,
                const std::string& points3D_path,
                const double image_scale,
                const Eigen::Vector3d& image_rgb) {
  std::ofstream images_file(images_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(images_file, images_path);

  const double six = image_scale * 0.15;
  const double siy = image_scale * 0.1;

  std::vector<Eigen::Vector3d> points;
  points.emplace_back(-six, -siy, six * 1.0 * 2.0);
  points.emplace_back(+six, -siy, six * 1.0 * 2.0);
  points.emplace_back(+six, +siy, six * 1.0 * 2.0);
  points.emplace_back(-six, +siy, six * 1.0 * 2.0);
  points.emplace_back(0, 0, 0);
  points.emplace_back(-six / 3.0, -siy / 3.0, six * 1.0 * 2.0);
  points.emplace_back(+six / 3.0, -siy / 3.0, six * 1.0 * 2.0);
  points.emplace_back(+six / 3.0, +siy / 3.0, six * 1.0 * 2.0);
  points.emplace_back(-six / 3.0, +siy / 3.0, six * 1.0 * 2.0);

  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (!reconstruction.IsImageRegistered(image_id)) {
      continue;
    }

    images_file << "Shape{\n";
    images_file << " appearance Appearance {\n";
    images_file << "  material DEF Default-ffRffGffB Material {\n";
    images_file << "  ambientIntensity 0\n";
    images_file << "  diffuseColor "
                << " " << image_rgb(0) << " " << image_rgb(1) << " "
                << image_rgb(2) << "\n";
    images_file << "  emissiveColor 0.1 0.1 0.1 } }\n";
    images_file << " geometry IndexedFaceSet {\n";
    images_file << " solid FALSE \n";
    images_file << " colorPerVertex TRUE \n";
    images_file << " ccw TRUE \n";

    images_file << " coord Coordinate {\n";
    images_file << " point [\n";

    // Move camera base model to camera pose.
    const Eigen::Matrix3x4d world_from_cam =
        Inverse(image.CamFromWorld()).ToMatrix();
    for (size_t i = 0; i < points.size(); i++) {
      const Eigen::Vector3d point = world_from_cam * points[i].homogeneous();
      images_file << point(0) << " " << point(1) << " " << point(2) << "\n";
    }

    images_file << " ] }\n";

    images_file << "color Color {color [\n";
    for (size_t p = 0; p < points.size(); p++) {
      images_file << " " << image_rgb(0) << " " << image_rgb(1) << " "
                  << image_rgb(2) << "\n";
    }

    images_file << "\n] }\n";

    images_file << "coordIndex [\n";
    images_file << " 0, 1, 2, 3, -1\n";
    images_file << " 5, 6, 4, -1\n";
    images_file << " 6, 7, 4, -1\n";
    images_file << " 7, 8, 4, -1\n";
    images_file << " 8, 5, 4, -1\n";
    images_file << " \n] \n";

    images_file << " texCoord TextureCoordinate { point [\n";
    images_file << "  1 1,\n";
    images_file << "  0 1,\n";
    images_file << "  0 0,\n";
    images_file << "  1 0,\n";
    images_file << "  0 0,\n";
    images_file << "  0 0,\n";
    images_file << "  0 0,\n";
    images_file << "  0 0,\n";
    images_file << "  0 0,\n";

    images_file << " ] }\n";
    images_file << "} }\n";
  }

  // Write 3D points

  std::ofstream points3D_file(points3D_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(points3D_file, points3D_path);

  points3D_file << "#VRML V2.0 utf8\n";
  points3D_file << "Background { skyColor [1.0 1.0 1.0] } \n";
  points3D_file << "Shape{ appearance Appearance {\n";
  points3D_file << " material Material {emissiveColor 1 1 1} }\n";
  points3D_file << " geometry PointSet {\n";
  points3D_file << " coord Coordinate {\n";
  points3D_file << "  point [\n";

  for (const auto& point3D : reconstruction.Points3D()) {
    points3D_file << point3D.second.xyz(0) << ", ";
    points3D_file << point3D.second.xyz(1) << ", ";
    points3D_file << point3D.second.xyz(2) << std::endl;
  }

  points3D_file << " ] }\n";
  points3D_file << " color Color { color [\n";

  for (const auto& point3D : reconstruction.Points3D()) {
    points3D_file << point3D.second.color(0) / 255.0 << ", ";
    points3D_file << point3D.second.color(1) / 255.0 << ", ";
    points3D_file << point3D.second.color(2) / 255.0 << std::endl;
  }

  points3D_file << " ] } } }\n";
}

}  // namespace colmap
