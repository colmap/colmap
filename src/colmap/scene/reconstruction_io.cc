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

#include "colmap/scene/reconstruction_io.h"

#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/ply.h"
#include "colmap/util/types.h"

#include <fstream>

namespace colmap {

bool ExportNVM(const Reconstruction& reconstruction,
               const std::string& path,
               bool skip_distortion) {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  // Ensure that we don't lose any precision by storing in text.
  file.precision(17);

  // White space added for compatibility with Meshlab.
  file << "NVM_V3 \n" << " \n";
  file << reconstruction.NumRegImages() << "  \n";

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
                      "and pinhole camera models.\n";
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
    file << 0 << '\n';

    image_id_to_idx_[image_id] = image_idx;
    image_idx += 1;
  }

  file << '\n' << reconstruction.NumPoints3D() << '\n';

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
    file << line_string << '\n';
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
                      "and pinhole camera models.\n";
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
         << '\n';
    file << focal_length << " " << k1 << " " << k2 << " " << fy / fx << " "
         << camera.PrincipalPointX() / camera.width << " "
         << camera.PrincipalPointY() / camera.height << '\n';
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
  synth_file << "colmap 1.0\n";
  synth_file << reconstruction.NumRegImages() << " "
             << reconstruction.NumPoints3D() << '\n';

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
               << '\n';
    synth_file << image.CamFromWorld().rotation.toRotationMatrix() << '\n';
    synth_file << image.CamFromWorld().translation.transpose() << '\n';

    image_id_to_idx_[image_id] = image_idx;
    image_list_file << image.Name() << '\n'
                    << camera.width << " " << camera.height << '\n';
    image_map_file << image_idx << '\n';

    image_idx += 1;
  }
  image_list_file.close();
  image_map_file.close();

  // Write point info
  for (const auto& point3D : reconstruction.Points3D()) {
    auto& p = point3D.second;
    synth_file << p.xyz(0) << " " << p.xyz(1) << " " << p.xyz(2) << '\n';
    synth_file << static_cast<int>(p.color(0)) << " "
               << static_cast<int>(p.color(1)) << " "
               << static_cast<int>(p.color(2)) << '\n';

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
    synth_file << line_string << '\n';
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

  file << "# Bundle file v0.3\n";

  file << reconstruction.NumRegImages() << " " << reconstruction.NumPoints3D()
       << '\n';

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
                      "`RADIAL`, and pinhole camera models.\n";
      return false;
    }

    file << camera.MeanFocalLength() << " " << k1 << " " << k2 << '\n';

    const Eigen::Matrix3d R = image.CamFromWorld().rotation.toRotationMatrix();
    file << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << '\n';
    file << -R(1, 0) << " " << -R(1, 1) << " " << -R(1, 2) << '\n';
    file << -R(2, 0) << " " << -R(2, 1) << " " << -R(2, 2) << '\n';

    file << image.CamFromWorld().translation.x() << " ";
    file << -image.CamFromWorld().translation.y() << " ";
    file << -image.CamFromWorld().translation.z() << '\n';

    list_file << image.Name() << '\n';

    image_id_to_idx_[image_id] = image_idx;
    image_idx += 1;
  }

  for (const auto& point3D : reconstruction.Points3D()) {
    file << point3D.second.xyz(0) << " ";
    file << point3D.second.xyz(1) << " ";
    file << point3D.second.xyz(2) << '\n';

    file << static_cast<int>(point3D.second.color(0)) << " ";
    file << static_cast<int>(point3D.second.color(1)) << " ";
    file << static_cast<int>(point3D.second.color(2)) << '\n';

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

    file << line_string << '\n';
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
    if (!image.HasPose()) {
      continue;
    }

    images_file << "Shape{\n";
    images_file << " appearance Appearance {\n";
    images_file << "  material DEF Default-ffRffGffB Material {\n";
    images_file << "  ambientIntensity 0\n";
    images_file << "  diffuseColor "
                << " " << image_rgb(0) << " " << image_rgb(1) << " "
                << image_rgb(2) << '\n';
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
      images_file << point(0) << " " << point(1) << " " << point(2) << '\n';
    }

    images_file << " ] }\n";

    images_file << "color Color {color [\n";
    for (size_t p = 0; p < points.size(); p++) {
      images_file << " " << image_rgb(0) << " " << image_rgb(1) << " "
                  << image_rgb(2) << '\n';
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
    points3D_file << point3D.second.xyz(2) << '\n';
  }

  points3D_file << " ] }\n";
  points3D_file << " color Color { color [\n";

  for (const auto& point3D : reconstruction.Points3D()) {
    points3D_file << point3D.second.color(0) / 255.0 << ", ";
    points3D_file << point3D.second.color(1) / 255.0 << ", ";
    points3D_file << point3D.second.color(2) / 255.0 << '\n';
  }

  points3D_file << " ] } } }\n";
}

}  // namespace colmap
