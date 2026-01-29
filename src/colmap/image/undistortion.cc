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

#include "colmap/image/undistortion.h"

#include "colmap/image/warp.h"
#include "colmap/math/math.h"
#include "colmap/sensor/models.h"

namespace colmap {

Camera UndistortCamera(const UndistortCameraOptions& options,
                       const Camera& camera) {
  THROW_CHECK_GE(options.blank_pixels, 0);
  THROW_CHECK_LE(options.blank_pixels, 1);
  THROW_CHECK_GT(options.min_scale, 0.0);
  THROW_CHECK_LE(options.min_scale, options.max_scale);
  THROW_CHECK_NE(options.max_image_size, 0);
  THROW_CHECK_GE(options.roi_min_x, 0.0);
  THROW_CHECK_GE(options.roi_min_y, 0.0);
  THROW_CHECK_LE(options.roi_max_x, 1.0);
  THROW_CHECK_LE(options.roi_max_y, 1.0);
  THROW_CHECK_LT(options.roi_min_x, options.roi_max_x);
  THROW_CHECK_LT(options.roi_min_y, options.roi_max_y);

  Camera undistorted_camera;
  undistorted_camera.model_id = PinholeCameraModel::model_id;
  undistorted_camera.width = camera.width;
  undistorted_camera.height = camera.height;
  undistorted_camera.params.resize(PinholeCameraModel::num_params, 0);

  // Copy focal length parameters.
  const span<const size_t> focal_length_idxs = camera.FocalLengthIdxs();
  THROW_CHECK_LE(focal_length_idxs.size(), 2)
      << "Not more than two focal length parameters supported.";
  undistorted_camera.SetFocalLengthX(camera.FocalLengthX());
  undistorted_camera.SetFocalLengthY(camera.FocalLengthY());

  // Copy principal point parameters.
  undistorted_camera.SetPrincipalPointX(camera.PrincipalPointX());
  undistorted_camera.SetPrincipalPointY(camera.PrincipalPointY());

  // Modify undistorted camera parameters based on ROI if enabled
  size_t roi_min_x = 0;
  size_t roi_min_y = 0;
  size_t roi_max_x = camera.width;
  size_t roi_max_y = camera.height;

  const bool roi_enabled = options.roi_min_x > 0.0 || options.roi_min_y > 0.0 ||
                           options.roi_max_x < 1.0 || options.roi_max_y < 1.0;

  if (roi_enabled) {
    roi_min_x = static_cast<size_t>(
        std::round(options.roi_min_x * static_cast<double>(camera.width)));
    roi_min_y = static_cast<size_t>(
        std::round(options.roi_min_y * static_cast<double>(camera.height)));
    roi_max_x = static_cast<size_t>(
        std::round(options.roi_max_x * static_cast<double>(camera.width)));
    roi_max_y = static_cast<size_t>(
        std::round(options.roi_max_y * static_cast<double>(camera.height)));

    // Make sure that the roi is valid.
    roi_min_x = std::min(roi_min_x, camera.width - 1);
    roi_min_y = std::min(roi_min_y, camera.height - 1);
    roi_max_x = std::max(roi_max_x, roi_min_x + 1);
    roi_max_y = std::max(roi_max_y, roi_min_y + 1);

    undistorted_camera.width = roi_max_x - roi_min_x;
    undistorted_camera.height = roi_max_y - roi_min_y;

    undistorted_camera.SetPrincipalPointX(camera.PrincipalPointX() -
                                          static_cast<double>(roi_min_x));
    undistorted_camera.SetPrincipalPointY(camera.PrincipalPointY() -
                                          static_cast<double>(roi_min_y));
  }

  // Scale in order to match the boundary of the undistorted image.
  if (roi_enabled || (camera.model_id != SimplePinholeCameraModel::model_id &&
                      camera.model_id != PinholeCameraModel::model_id)) {
    // Determine min/max coordinates along top / bottom image border.

    double left_min_x = std::numeric_limits<double>::max();
    double left_max_x = std::numeric_limits<double>::lowest();
    double right_min_x = std::numeric_limits<double>::max();
    double right_max_x = std::numeric_limits<double>::lowest();

    for (size_t y = roi_min_y; y < roi_max_y; ++y) {
      // Left border.
      if (const std::optional<Eigen::Vector2d> cam_point1 =
              camera.CamFromImg(Eigen::Vector2d(0.5, y + 0.5));
          cam_point1.has_value()) {
        if (const std::optional<Eigen::Vector2d> undistorted_point1 =
                undistorted_camera.ImgFromCam(cam_point1->homogeneous());
            undistorted_point1) {
          left_min_x = std::min(left_min_x, undistorted_point1->x());
          left_max_x = std::max(left_max_x, undistorted_point1->x());
        }
      }
      // Right border.
      if (const std::optional<Eigen::Vector2d> cam_point2 =
              camera.CamFromImg(Eigen::Vector2d(camera.width - 0.5, y + 0.5));
          cam_point2.has_value()) {
        if (const std::optional<Eigen::Vector2d> undistorted_point2 =
                undistorted_camera.ImgFromCam(cam_point2->homogeneous());
            undistorted_point2) {
          right_min_x = std::min(right_min_x, undistorted_point2->x());
          right_max_x = std::max(right_max_x, undistorted_point2->x());
        }
      }
    }

    // Determine min, max coordinates along left / right image border.

    double top_min_y = std::numeric_limits<double>::max();
    double top_max_y = std::numeric_limits<double>::lowest();
    double bottom_min_y = std::numeric_limits<double>::max();
    double bottom_max_y = std::numeric_limits<double>::lowest();

    for (size_t x = roi_min_x; x < roi_max_x; ++x) {
      // Top border.
      if (const std::optional<Eigen::Vector2d> cam_point1 =
              camera.CamFromImg(Eigen::Vector2d(x + 0.5, 0.5));
          cam_point1) {
        if (const std::optional<Eigen::Vector2d> undistorted_point1 =
                undistorted_camera.ImgFromCam(cam_point1->homogeneous());
            undistorted_point1) {
          top_min_y = std::min(top_min_y, undistorted_point1->y());
          top_max_y = std::max(top_max_y, undistorted_point1->y());
        }
      }
      // Bottom border.
      if (const std::optional<Eigen::Vector2d> cam_point2 =
              camera.CamFromImg(Eigen::Vector2d(x + 0.5, camera.height - 0.5));
          cam_point2) {
        if (const std::optional<Eigen::Vector2d> undistorted_point2 =
                undistorted_camera.ImgFromCam(cam_point2->homogeneous());
            undistorted_point2) {
          bottom_min_y = std::min(bottom_min_y, undistorted_point2->y());
          bottom_max_y = std::max(bottom_max_y, undistorted_point2->y());
        }
      }
    }

    const double cx = undistorted_camera.PrincipalPointX();
    const double cy = undistorted_camera.PrincipalPointY();

    // Scale such that undistorted image contains all pixels of distorted image.
    const double min_scale_x =
        std::min(cx / (cx - left_min_x),
                 (undistorted_camera.width - 0.5 - cx) / (right_max_x - cx));
    const double min_scale_y =
        std::min(cy / (cy - top_min_y),
                 (undistorted_camera.height - 0.5 - cy) / (bottom_max_y - cy));

    // Scale such that there are no blank pixels in undistorted image.
    const double max_scale_x =
        std::max(cx / (cx - left_max_x),
                 (undistorted_camera.width - 0.5 - cx) / (right_min_x - cx));
    const double max_scale_y =
        std::max(cy / (cy - top_max_y),
                 (undistorted_camera.height - 0.5 - cy) / (bottom_min_y - cy));

    // Interpolate scale according to blank_pixels.
    double scale_x = 1.0 / (min_scale_x * options.blank_pixels +
                            max_scale_x * (1.0 - options.blank_pixels));
    double scale_y = 1.0 / (min_scale_y * options.blank_pixels +
                            max_scale_y * (1.0 - options.blank_pixels));

    // Clip the scaling factors.
    scale_x = Clamp(scale_x, options.min_scale, options.max_scale);
    scale_y = Clamp(scale_y, options.min_scale, options.max_scale);

    // Scale undistorted camera dimensions.
    const size_t orig_undistorted_camera_width = undistorted_camera.width;
    const size_t orig_undistorted_camera_height = undistorted_camera.height;
    undistorted_camera.width =
        static_cast<size_t>(std::max(1.0, scale_x * undistorted_camera.width));
    undistorted_camera.height =
        static_cast<size_t>(std::max(1.0, scale_y * undistorted_camera.height));

    // Scale the principal point according to the new dimensions of the camera.
    undistorted_camera.SetPrincipalPointX(
        undistorted_camera.PrincipalPointX() *
        static_cast<double>(undistorted_camera.width) /
        static_cast<double>(orig_undistorted_camera_width));
    undistorted_camera.SetPrincipalPointY(
        undistorted_camera.PrincipalPointY() *
        static_cast<double>(undistorted_camera.height) /
        static_cast<double>(orig_undistorted_camera_height));
  }

  if (options.max_image_size > 0) {
    const double max_image_scale_x =
        options.max_image_size / static_cast<double>(undistorted_camera.width);
    const double max_image_scale_y =
        options.max_image_size / static_cast<double>(undistorted_camera.height);
    const double max_image_scale =
        std::min(max_image_scale_x, max_image_scale_y);
    if (max_image_scale < 1.0) {
      undistorted_camera.Rescale(max_image_scale);
    }
  }

  return undistorted_camera;
}

void UndistortImage(const UndistortCameraOptions& options,
                    const Bitmap& distorted_bitmap,
                    const Camera& distorted_camera,
                    Bitmap* undistorted_bitmap,
                    Camera* undistorted_camera) {
  THROW_CHECK_EQ(distorted_camera.width, distorted_bitmap.Width());
  THROW_CHECK_EQ(distorted_camera.height, distorted_bitmap.Height());

  *undistorted_camera = UndistortCamera(options, distorted_camera);

  WarpImageBetweenCameras(distorted_camera,
                          *undistorted_camera,
                          distorted_bitmap,
                          undistorted_bitmap);

  distorted_bitmap.CloneMetadata(undistorted_bitmap);
}

void UndistortReconstruction(const UndistortCameraOptions& options,
                             Reconstruction* reconstruction) {
  const std::unordered_map<camera_t, Camera> distorted_cameras =
      reconstruction->Cameras();
  for (const auto& camera : distorted_cameras) {
    if (camera.second.IsUndistorted()) {
      continue;
    }
    reconstruction->Camera(camera.first) =
        UndistortCamera(options, camera.second);
  }

  for (const auto& distorted_image : reconstruction->Images()) {
    Image& image = reconstruction->Image(distorted_image.first);
    const Camera& distorted_camera = distorted_cameras.at(image.CameraId());
    const Camera& undistorted_camera = *image.CameraPtr();
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      auto& point2D = image.Point2D(point2D_idx);
      const std::optional<Eigen::Vector2d> cam_point =
          distorted_camera.CamFromImg(point2D.xy);
      if (!cam_point) {
        point2D.xy =
            Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN());
      } else {
        const std::optional<Eigen::Vector2d> undistorted_point =
            undistorted_camera.ImgFromCam(cam_point->homogeneous());
        if (undistorted_point) {
          point2D.xy = *undistorted_point;
        } else {
          point2D.xy = Eigen::Vector2d::Constant(
              std::numeric_limits<double>::quiet_NaN());
        }
      }
    }
  }
}

void RectifyStereoCameras(const Camera& camera1,
                          const Camera& camera2,
                          const Rigid3d& cam2_from_cam1,
                          Eigen::Matrix3d* H1,
                          Eigen::Matrix3d* H2,
                          Eigen::Matrix4d* Q) {
  THROW_CHECK(camera1.model_id == SimplePinholeCameraModel::model_id ||
              camera1.model_id == PinholeCameraModel::model_id);
  THROW_CHECK(camera2.model_id == SimplePinholeCameraModel::model_id ||
              camera2.model_id == PinholeCameraModel::model_id);

  // Compute the average rotation between the first and the second camera.
  Eigen::AngleAxisd half_cam2_from_cam1(cam2_from_cam1.rotation());
  half_cam2_from_cam1.angle() *= -0.5;

  Eigen::Matrix3d R2 = half_cam2_from_cam1.toRotationMatrix();
  Eigen::Matrix3d R1 = R2.transpose();

  // Determine the translation, such that it coincides with the X-axis.
  Eigen::Vector3d t = R2 * cam2_from_cam1.translation();

  Eigen::Vector3d x_unit_vector(1, 0, 0);
  if (t.transpose() * x_unit_vector < 0) {
    x_unit_vector *= -1;
  }

  const Eigen::Vector3d rotation_axis = t.cross(x_unit_vector);

  Eigen::Matrix3d R_x;
  if (rotation_axis.norm() < std::numeric_limits<double>::epsilon()) {
    R_x = Eigen::Matrix3d::Identity();
  } else {
    const double angle = std::acos(std::abs(t.transpose() * x_unit_vector) /
                                   (t.norm() * x_unit_vector.norm()));
    R_x = Eigen::AngleAxisd(angle, rotation_axis.normalized());
  }

  // Apply the X-axis correction.
  R1 = R_x * R1;
  R2 = R_x * R2;
  t = R_x * t;

  // Determine the intrinsic calibration matrix.
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  K(0, 0) = std::min(camera1.MeanFocalLength(), camera2.MeanFocalLength());
  K(1, 1) = K(0, 0);
  K(0, 2) = camera1.PrincipalPointX();
  K(1, 2) = (camera1.PrincipalPointY() + camera2.PrincipalPointY()) / 2;

  // Compose the homographies.
  *H1 = K * R1 * camera1.CalibrationMatrix().inverse();
  *H2 = K * R2 * camera2.CalibrationMatrix().inverse();

  // Determine the inverse projection matrix that transforms disparity values
  // to 3D world coordinates: [x, y, disparity, 1] * Q = [X, Y, Z, 1] * w.
  *Q = Eigen::Matrix4d::Identity();
  (*Q)(3, 0) = -K(1, 2);
  (*Q)(3, 1) = -K(0, 2);
  (*Q)(3, 2) = K(0, 0);
  (*Q)(2, 3) = -1 / t(0);
  (*Q)(3, 3) = 0;
}

void RectifyAndUndistortStereoImages(const UndistortCameraOptions& options,
                                     const Bitmap& distorted_image1,
                                     const Bitmap& distorted_image2,
                                     const Camera& distorted_camera1,
                                     const Camera& distorted_camera2,
                                     const Rigid3d& cam2_from_cam1,
                                     Bitmap* undistorted_image1,
                                     Bitmap* undistorted_image2,
                                     Camera* undistorted_camera,
                                     Eigen::Matrix4d* Q) {
  THROW_CHECK_EQ(distorted_camera1.width, distorted_image1.Width());
  THROW_CHECK_EQ(distorted_camera1.height, distorted_image1.Height());
  THROW_CHECK_EQ(distorted_camera2.width, distorted_image2.Width());
  THROW_CHECK_EQ(distorted_camera2.height, distorted_image2.Height());

  *undistorted_camera = UndistortCamera(options, distorted_camera1);
  *undistorted_image1 = Bitmap(static_cast<int>(undistorted_camera->width),
                               static_cast<int>(undistorted_camera->height),
                               distorted_image1.IsRGB());
  distorted_image1.CloneMetadata(undistorted_image1);

  *undistorted_image2 = Bitmap(static_cast<int>(undistorted_camera->width),
                               static_cast<int>(undistorted_camera->height),
                               distorted_image2.IsRGB());
  distorted_image2.CloneMetadata(undistorted_image2);

  Eigen::Matrix3d H1;
  Eigen::Matrix3d H2;
  RectifyStereoCameras(
      *undistorted_camera, *undistorted_camera, cam2_from_cam1, &H1, &H2, Q);

  WarpImageWithHomographyBetweenCameras(H1.inverse(),
                                        distorted_camera1,
                                        *undistorted_camera,
                                        distorted_image1,
                                        undistorted_image1);
  WarpImageWithHomographyBetweenCameras(H2.inverse(),
                                        distorted_camera2,
                                        *undistorted_camera,
                                        distorted_image2,
                                        undistorted_image2);
}

}  // namespace colmap
