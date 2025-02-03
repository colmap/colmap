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

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/point2d.h"
#include "colmap/scene/visibility_pyramid.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <string>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// Class that holds information about an image. An image is the product of one
// camera shot at a certain location (parameterized as the pose). An image may
// share a camera with multiple other images, if its intrinsics are the same.
class Image {
 public:
  Image();

  // Access the unique identifier of the image.
  inline image_t ImageId() const;
  inline void SetImageId(image_t image_id);

  // Access the name of the image.
  inline const std::string& Name() const;
  inline std::string& Name();
  inline void SetName(const std::string& name);

  // Access the unique identifier of the camera. Note that multiple images
  // might share the same camera.
  inline camera_t CameraId() const;
  inline void SetCameraId(camera_t camera_id);
  // Check whether identifier of camera has been set.
  inline bool HasCameraId() const;

  // Access to the underlying, shared camera object.
  // This is typically only set when the image was added to a reconstruction.
  inline struct Camera* CameraPtr() const;
  inline void SetCameraPtr(struct Camera* camera);
  inline void ResetCameraPtr();
  inline bool HasCameraPtr() const;

  // Check if image is registered.
  inline bool IsRegistered() const;
  inline void SetRegistered(bool registered);

  // Get the number of image points.
  inline point2D_t NumPoints2D() const;

  // Get the number of triangulations, i.e. the number of points that
  // are part of a 3D point track.
  inline point2D_t NumPoints3D() const;

  // [Optional] The corresponding frame of the image
  inline frame_t FrameId() const;
  inline void SetFrameId(frame_t frame_id);
  inline const std::shared_ptr<class Frame>& Frame() const;
  inline void SetFrame(std::shared_ptr<class Frame> frame);
  inline bool HasFrame() const;
  // Check if the cam_from_world needs to be composited with rig calibration.
  inline bool HasNonTrivialFrame() const;

  // World to camera pose.
  // Access cam_from_world transformation with a constant reference. While
  // calling this method, the cam_from_world will also be synced from the
  // transformations in the frame (rig) if needed
  inline const Rigid3d& CamFromWorld() const;
  // Access cam_from_world as a reference to do in-place update and
  // optimization. Will throw an error if the image has a non trivial frame
  // (rig) attached to it.
  inline Rigid3d& CamFromWorld();

  // Access the coordinates of image points.
  inline const struct Point2D& Point2D(point2D_t point2D_idx) const;
  inline struct Point2D& Point2D(point2D_t point2D_idx);
  inline const std::vector<struct Point2D>& Points2D() const;
  inline std::vector<struct Point2D>& Points2D();
  void SetPoints2D(const std::vector<Eigen::Vector2d>& points);
  void SetPoints2D(const std::vector<struct Point2D>& points);

  // Set the point as triangulated, i.e. it is part of a 3D point track.
  void SetPoint3DForPoint2D(point2D_t point2D_idx, point3D_t point3D_id);

  // Set the point as not triangulated, i.e. it is not part of a 3D point track.
  void ResetPoint3DForPoint2D(point2D_t point2D_idx);

  // Check whether one of the image points is part of the 3D point track.
  bool HasPoint3D(point3D_t point3D_id) const;

  // Extract the projection center in world space.
  Eigen::Vector3d ProjectionCenter() const;

  // Extract the viewing direction of the image.
  Eigen::Vector3d ViewingDirection() const;

  // Reproject the 3D point onto the image in pixels (throws if the camera
  // object was not set). Return false if the 3D point is behind the camera.
  std::pair<bool, Eigen::Vector2d> ProjectPoint(
      const Eigen::Vector3d& point3D) const;

 private:
  // Identifier of the image, if not specified `kInvalidImageId`.
  image_t image_id_;

  // The name of the image, i.e. the relative path.
  std::string name_;

  // The identifier of the associated camera. Note that multiple images might
  // share the same camera. If not specified `kInvalidCameraId`.
  camera_t camera_id_;
  struct Camera* camera_ptr_;

  // Whether the image is successfully registered in the reconstruction.
  bool registered_;

  // The number of 3D points the image observes, i.e. the sum of its `points2D`
  // where `point3D_id != kInvalidPoint3DId`.
  point2D_t num_points3D_;

  // [Optional] The corresponding frame (rig) of the image
  frame_t frame_id_;
  std::shared_ptr<class Frame> frame_ = nullptr;

  // The pose of the image, defined as the transformation from world to camera.
  // Only useful when the corresponding frame (rig) does not exist.
  mutable Rigid3d cam_from_world_;

  // All image points, including points that are not part of a 3D point track.
  std::vector<struct Point2D> points2D_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

image_t Image::ImageId() const { return image_id_; }

void Image::SetImageId(const image_t image_id) { image_id_ = image_id; }

const std::string& Image::Name() const { return name_; }

std::string& Image::Name() { return name_; }

void Image::SetName(const std::string& name) { name_ = name; }

inline camera_t Image::CameraId() const { return camera_id_; }

inline void Image::SetCameraId(const camera_t camera_id) {
  THROW_CHECK_NE(camera_id, kInvalidCameraId);
  THROW_CHECK(!HasCameraPtr());
  camera_id_ = camera_id;
}

inline bool Image::HasCameraId() const {
  return camera_id_ != kInvalidCameraId;
}

inline struct Camera* Image::CameraPtr() const {
  return THROW_CHECK_NOTNULL(camera_ptr_);
}

inline void Image::SetCameraPtr(struct Camera* camera) {
  THROW_CHECK_NOTNULL(camera);
  THROW_CHECK_NE(camera->camera_id, kInvalidCameraId);
  if (!HasCameraPtr()) {
    THROW_CHECK_EQ(camera->camera_id, camera_id_);
    camera_ptr_ = camera;
  } else {  // switch to new camera
    camera_id_ = camera->camera_id;
    camera_ptr_ = camera;
  }
}

inline void Image::ResetCameraPtr() { camera_ptr_ = nullptr; }

inline bool Image::HasCameraPtr() const { return camera_ptr_ != nullptr; }

bool Image::IsRegistered() const { return registered_; }

void Image::SetRegistered(const bool registered) { registered_ = registered; }

point2D_t Image::NumPoints2D() const {
  return static_cast<point2D_t>(points2D_.size());
}

point2D_t Image::NumPoints3D() const { return num_points3D_; }

frame_t Image::FrameId() const { return frame_id_; }

void Image::SetFrameId(frame_t frame_id) { frame_id_ = frame_id; }

const std::shared_ptr<class Frame>& Image::Frame() const { return frame_; }

void Image::SetFrame(std::shared_ptr<class Frame> frame) {
  frame_ = std::move(frame);
}

bool Image::HasFrame() const { return frame_ != nullptr; }

bool Image::HasNonTrivialFrame() const {
  return HasFrame() && frame_->HasRigCalibration() &&
         !frame_->RigCalibration()->IsReference(
             std::make_pair(SensorType::Camera, CameraId()));
}

const Rigid3d& Image::CamFromWorld() const {
  if (HasNonTrivialFrame()) {
    // sync cam from world
    sensor_t sensor_id = std::make_pair(SensorType::Camera, CameraId());
    cam_from_world_ = frame_->SensorFromWorld(sensor_id);
    return cam_from_world_;
  } else if (HasFrame()) {
    return frame_->SensorFromWorld();
  } else
    return cam_from_world_;
}

Rigid3d& Image::CamFromWorld() {
  if (HasNonTrivialFrame())
    LOG(FATAL_THROW)
        << "No reference available for cam_from_world transformation, since "
           "composition with rig calibration is needed";
  if (HasFrame()) {
    return frame_->SensorFromWorld();
  } else
    return cam_from_world_;
}

const struct Point2D& Image::Point2D(const point2D_t point2D_idx) const {
  return points2D_.at(point2D_idx);
}

struct Point2D& Image::Point2D(const point2D_t point2D_idx) {
  return points2D_.at(point2D_idx);
}

const std::vector<struct Point2D>& Image::Points2D() const { return points2D_; }

std::vector<struct Point2D>& Image::Points2D() { return points2D_; }

}  // namespace colmap
