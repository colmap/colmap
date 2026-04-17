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

#include "colmap/exe/sphere.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/track.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/sensor/models.h"
#include "colmap/sensor/rig.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/string.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {
namespace {

// Cube face rotations. Convention (COLMAP camera frame: +X right, +Y down,
// +Z forward):
//   R_sphere_from_face maps a face-cam-frame ray onto the sphere-cam-frame.
//   The face camera's +Z axis points along the face's direction on the
//   sphere, with a sensible "up" convention (sphere -Y stays up for F, B, L,
//   R; reoriented for U, D).
struct FaceSpec {
  const char* name;
  Eigen::Matrix3d r_sphere_from_face;
};

std::array<FaceSpec, 6> CubeFaceSpecs() {
  std::array<FaceSpec, 6> faces{};
  // F: face +Z -> sphere +Z.
  faces[0] = {"F", Eigen::Matrix3d::Identity()};
  // B: face +Z -> sphere -Z. 180° yaw.
  faces[1].name = "B";
  faces[1].r_sphere_from_face << -1, 0, 0, 0, 1, 0, 0, 0, -1;
  // L: face +Z -> sphere -X.
  faces[2].name = "L";
  faces[2].r_sphere_from_face << 0, 0, -1, 0, 1, 0, 1, 0, 0;
  // R: face +Z -> sphere +X.
  faces[3].name = "R";
  faces[3].r_sphere_from_face << 0, 0, 1, 0, 1, 0, -1, 0, 0;
  // U: face +Z -> sphere -Y.
  faces[4].name = "U";
  faces[4].r_sphere_from_face << 1, 0, 0, 0, 0, -1, 0, 1, 0;
  // D: face +Z -> sphere +Y.
  faces[5].name = "D";
  faces[5].r_sphere_from_face << 1, 0, 0, 0, 0, 1, 0, -1, 0;
  return faces;
}

// Render one cube face from an equirectangular (ERP) bitmap by bilinear
// resampling. Uses COLMAP camera-frame conventions (+X right, +Y down, +Z
// forward). face_size is the output side in pixels; focal_px is the face
// pinhole focal length in pixels.
Bitmap RenderCubeFace(const Bitmap& erp,
                      const Eigen::Matrix3d& r_sphere_from_face,
                      int face_size,
                      double focal_px) {
  const double W = erp.Width();
  const double H = erp.Height();
  const double cx = face_size / 2.0;
  const double cy = face_size / 2.0;

  Bitmap face(face_size, face_size, /*as_rgb=*/true);
  for (int v = 0; v < face_size; ++v) {
    for (int u = 0; u < face_size; ++u) {
      // Face-cam-frame ray (unit length) for pixel (u + 0.5, v + 0.5).
      const double x = (u + 0.5 - cx) / focal_px;
      const double y = (v + 0.5 - cy) / focal_px;
      const double z = 1.0;
      const double norm = std::sqrt(x * x + y * y + z * z);
      const Eigen::Vector3d ray_face(x / norm, y / norm, z / norm);
      const Eigen::Vector3d ray_sphere = r_sphere_from_face * ray_face;

      // Sphere-frame ray -> ERP pixel. +Z forward, +X right, +Y down.
      const double theta = std::atan2(ray_sphere.x(), ray_sphere.z());
      const double phi = std::asin(-ray_sphere.y());
      double erp_x = (theta / (2.0 * M_PI) + 0.5) * W;
      const double erp_y = (0.5 - phi / M_PI) * H;

      // Wrap longitude, clamp latitude.
      erp_x = std::fmod(erp_x, W);
      if (erp_x < 0) erp_x += W;

      const std::optional<BitmapColor<float>> sample =
          erp.InterpolateBilinear(erp_x, erp_y);
      if (sample) {
        face.SetPixel(u,
                      v,
                      BitmapColor<uint8_t>(
                          static_cast<uint8_t>(std::clamp(
                              std::lround(sample->r), 0L, 255L)),
                          static_cast<uint8_t>(std::clamp(
                              std::lround(sample->g), 0L, 255L)),
                          static_cast<uint8_t>(std::clamp(
                              std::lround(sample->b), 0L, 255L))));
      }
    }
  }
  return face;
}

// Quaternion from 3x3 rotation matrix. Equivalent to
// Eigen::Quaterniond(R) but avoids a dependency on the specific
// constructor behavior for non-normalized matrices.
Eigen::Quaterniond QuaternionFromMatrix(const Eigen::Matrix3d& r) {
  return Eigen::Quaterniond(r).normalized();
}

}  // namespace

int RunSphereToCubic(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::filesystem::path output_image_path;
  int face_size = 1024;
  double fov_deg = 90.0;
  int jpeg_quality = 95;

  OptionManager options;
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path,
                            "Path to input SPHERE sparse model.");
  options.AddRequiredOption("output_path", &output_path,
                            "Output path for cubic sparse model.");
  options.AddRequiredOption(
      "output_image_path", &output_image_path,
      "Output directory for rendered cube face images.");
  options.AddDefaultOption("face_size", &face_size,
                           "Output face size in pixels (square).");
  options.AddDefaultOption("fov_deg", &fov_deg,
                           "Output face FOV in degrees.");
  options.AddDefaultOption("jpeg_quality", &jpeg_quality,
                           "JPEG quality [1, 100] for rendered faces.");
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  CreateDirIfNotExists(output_path);
  CreateDirIfNotExists(output_image_path);

  LOG_HEADING1("Reading reconstruction");
  Reconstruction reconstruction;
  reconstruction.Read(input_path);
  LOG(INFO) << StringPrintf("=> Reconstruction with %d images and %d points",
                            reconstruction.NumImages(),
                            reconstruction.NumPoints3D());

  const auto faces = CubeFaceSpecs();
  const double focal_px =
      face_size / (2.0 * std::tan(DegToRad(fov_deg) / 2.0));
  const double face_cx = face_size / 2.0;
  const double face_cy = face_size / 2.0;

  // Single shared PINHOLE camera for every rendered face, wrapped in a
  // trivial one-camera rig (COLMAP 4.x images live inside a Frame, and the
  // Frame lives inside a Rig, so even a singleton camera needs this setup).
  Reconstruction cubic;
  Camera pinhole_camera;
  pinhole_camera.camera_id = 1;
  pinhole_camera.model_id = CameraModelId::kPinhole;
  pinhole_camera.width = face_size;
  pinhole_camera.height = face_size;
  pinhole_camera.params = {focal_px, focal_px, face_cx, face_cy};
  pinhole_camera.has_prior_focal_length = true;
  cubic.AddCamera(pinhole_camera);

  constexpr rig_t kRigId = 1;
  const sensor_t pinhole_sensor_id(SensorType::CAMERA, pinhole_camera.camera_id);
  {
    Rig pinhole_rig;
    pinhole_rig.SetRigId(kRigId);
    pinhole_rig.AddRefSensor(pinhole_sensor_id);
    cubic.AddRig(std::move(pinhole_rig));
  }

  // For each SPHERE image, render 6 cube face bitmaps and register them as
  // new Image entries with the appropriate poses.
  //
  // Mapping of sphere image_id -> per-face new image_id:
  //   new_id = sphere_image_id * 6 + face_idx + 1
  // This keeps the mapping injective and reversible.
  const std::string image_dir = options.image_path->string();

  std::unordered_map<image_t, std::array<image_t, 6>>
      sphere_to_face_image_ids;

  int num_rendered = 0;
  int num_skipped_non_sphere = 0;
  for (const image_t sphere_image_id : reconstruction.RegImageIds()) {
    const Image& sphere_image = reconstruction.Image(sphere_image_id);
    const Camera& sphere_camera =
        reconstruction.Camera(sphere_image.CameraId());
    if (sphere_camera.model_id != CameraModelId::kSphere) {
      ++num_skipped_non_sphere;
      continue;
    }

    Bitmap erp;
    const std::filesystem::path erp_path =
        std::filesystem::path(image_dir) / sphere_image.Name();
    if (!erp.Read(erp_path.string())) {
      LOG(WARNING) << "Failed to read " << erp_path;
      continue;
    }

    // Sphere-cam <- world rotation and camera center in world.
    const Rigid3d sphere_cam_from_world = sphere_image.CamFromWorld();
    const Eigen::Matrix3d r_sphere_cam_from_world =
        sphere_cam_from_world.rotation().toRotationMatrix();
    const Eigen::Vector3d proj_center =
        r_sphere_cam_from_world.transpose() *
        (-sphere_cam_from_world.translation());

    std::array<image_t, 6> face_image_ids{};
    for (size_t face_idx = 0; face_idx < faces.size(); ++face_idx) {
      const FaceSpec& face = faces[face_idx];

      Bitmap face_bitmap =
          RenderCubeFace(erp, face.r_sphere_from_face, face_size, focal_px);
      face_bitmap.SetJpegQuality(jpeg_quality);

      const std::filesystem::path face_stem =
          std::filesystem::path(sphere_image.Name()).stem();
      const std::string face_filename =
          face_stem.string() + "_" + face.name + ".jpg";
      const std::filesystem::path face_full_path =
          output_image_path / face_filename;
      face_bitmap.Write(face_full_path.string());

      // Face-cam <- world: R_wc_face = R_sphere_from_face.T @ R_sphere_cam_from_world.
      const Eigen::Matrix3d r_face_cam_from_world =
          face.r_sphere_from_face.transpose() * r_sphere_cam_from_world;
      const Eigen::Vector3d t_face_cam_from_world =
          -r_face_cam_from_world * proj_center;

      const image_t new_image_id =
          sphere_image_id * faces.size() + face_idx + 1;
      const frame_t new_frame_id = new_image_id;

      // Each face image gets its own single-camera frame (one sensor per
      // frame, same rig reused across all frames).
      Frame face_frame;
      face_frame.SetFrameId(new_frame_id);
      face_frame.SetRigId(kRigId);
      face_frame.AddDataId(data_t(pinhole_sensor_id, new_image_id));
      face_frame.SetRigFromWorld(
          Rigid3d(QuaternionFromMatrix(r_face_cam_from_world),
                  t_face_cam_from_world));
      cubic.AddFrame(std::move(face_frame));
      cubic.RegisterFrame(new_frame_id);

      Image face_image;
      face_image.SetImageId(new_image_id);
      face_image.SetCameraId(pinhole_camera.camera_id);
      face_image.SetFrameId(new_frame_id);
      face_image.SetName(face_filename);
      face_image.SetPoints2D(std::vector<struct Point2D>{});
      cubic.AddImage(std::move(face_image));
      face_image_ids[face_idx] = new_image_id;
    }
    sphere_to_face_image_ids[sphere_image_id] = face_image_ids;

    ++num_rendered;
    if (num_rendered % 10 == 0) {
      LOG(INFO) << "  rendered " << num_rendered << " panoramas";
    }
  }
  LOG(INFO) << StringPrintf(
      "=> %d sphere images -> %d face images (skipped %d non-SPHERE)",
      num_rendered,
      num_rendered * static_cast<int>(faces.size()),
      num_skipped_non_sphere);

  // Reproject 3D points: for each track element (sphere_image_id, kp_idx),
  // project the 3D point into each cube face and add an observation to the
  // first face whose image bounds contain the projection.
  int num_points_kept = 0;
  int num_observations_added = 0;
  int num_observations_dropped = 0;
  for (const auto& [point_id, sphere_point] : reconstruction.Points3D()) {
    const Eigen::Vector3d xyz = sphere_point.xyz;
    Track new_track;

    for (const TrackElement& elem : sphere_point.track.Elements()) {
      const auto face_images_it =
          sphere_to_face_image_ids.find(elem.image_id);
      if (face_images_it == sphere_to_face_image_ids.end()) {
        ++num_observations_dropped;
        continue;
      }

      const Image& sphere_image = reconstruction.Image(elem.image_id);
      const Rigid3d& sphere_cam_from_world = sphere_image.CamFromWorld();
      const Eigen::Vector3d point_in_sphere_cam =
          sphere_cam_from_world.rotation() * xyz +
          sphere_cam_from_world.translation();

      image_t chosen_face_image_id = kInvalidImageId;
      Eigen::Vector2d chosen_face_uv;
      for (size_t face_idx = 0; face_idx < faces.size(); ++face_idx) {
        const Eigen::Vector3d point_in_face_cam =
            faces[face_idx].r_sphere_from_face.transpose() *
            point_in_sphere_cam;
        if (point_in_face_cam.z() <= 1e-6) {
          continue;
        }
        const double u = focal_px * point_in_face_cam.x() / point_in_face_cam.z() + face_cx;
        const double v = focal_px * point_in_face_cam.y() / point_in_face_cam.z() + face_cy;
        if (u >= 0 && u < face_size && v >= 0 && v < face_size) {
          chosen_face_image_id = face_images_it->second[face_idx];
          chosen_face_uv = Eigen::Vector2d(u, v);
          break;
        }
      }
      if (chosen_face_image_id == kInvalidImageId) {
        ++num_observations_dropped;
        continue;
      }

      Image& face_image = cubic.Image(chosen_face_image_id);
      const point2D_t new_point2D_idx =
          static_cast<point2D_t>(face_image.Points2D().size());
      struct Point2D new_point2D;
      new_point2D.xy = chosen_face_uv;
      new_point2D.point3D_id = point_id;
      face_image.Points2D().push_back(new_point2D);
      new_track.AddElement(chosen_face_image_id, new_point2D_idx);
      ++num_observations_added;
    }

    if (new_track.Length() >= 2) {
      Point3D new_point;
      new_point.xyz = xyz;
      new_point.color = sphere_point.color;
      new_point.error = sphere_point.error;
      new_point.track = std::move(new_track);
      cubic.AddPoint3D(point_id, std::move(new_point));
      ++num_points_kept;
    }
  }

  LOG(INFO) << StringPrintf(
      "=> reprojection: added %d observations, dropped %d, kept %d/%d points",
      num_observations_added,
      num_observations_dropped,
      num_points_kept,
      reconstruction.NumPoints3D());

  LOG_HEADING1("Writing cubic reconstruction");
  cubic.Write(output_path);
  LOG(INFO) << "=> wrote sparse model to " << output_path;
  LOG(INFO) << "=> wrote rendered faces to " << output_image_path;

  return EXIT_SUCCESS;
}

}  // namespace colmap
