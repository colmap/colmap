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

#pragma once

#include "colmap/geometry/gps.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/point2d.h"
#include "colmap/scene/visibility_pyramid.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// Class that holds information about an image. An image is the product of one
// camera exposure at a certain location (parameterized as the pose). An image
// may share a camera with multiple other images, if its intrinsics are the
// same.
class Image {
 public:
  Image();

  // Copy construct/assign.
  // Initialize a new Frame object if the image has a trivial frame.
  Image(const Image& other);
  Image& operator=(const Image& other);
  // Move construct/assign.
  Image(Image&& other) = default;
  Image& operator=(Image&& other) = default;

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

  inline data_t DataId() const;

  // Access to the underlying, shared camera object.
  // This is typically only set when the image was added to a reconstruction.
  inline struct Camera* CameraPtr() const;
  inline void SetCameraPtr(struct Camera* camera);
  inline void ResetCameraPtr();
  inline bool HasCameraPtr() const;

  // Get the number of image points.
  inline point2D_t NumPoints2D() const;

  // Get the number of triangulations, i.e. the number of points that
  // are part of a 3D point track.
  inline point2D_t NumPoints3D() const;

  // Access the unique identifier of the frame.
  inline camera_t FrameId() const;
  inline void SetFrameId(frame_t frame_id);
  // Check whether identifier of Frame has been set.
  inline bool HasFrameId() const;

  // [Optional] The corresponding frame of the image.
  inline class Frame* FramePtr() const;
  inline void SetFramePtr(class Frame* frame);
  inline void ResetFramePtr();
  inline bool HasFramePtr() const;
  // Check if cam_from_world needs to be composed with sensor_from_rig pose.
  inline bool HasTrivialFrame() const;

  // Composition of sensor_from_rig and rig_from_world transformations.
  // If the corresponding frame is trivial, this is equal to rig_from_world.
  inline Rigid3d CamFromWorld() const;
  inline bool HasPose() const;

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
  // object was not set). Return null if the 3D point is behind the camera.
  std::optional<Eigen::Vector2d> ProjectPoint(
      const Eigen::Vector3d& point3D) const;

  inline bool operator==(const Image& other) const;
  inline bool operator!=(const Image& other) const;

 private:
  // Identifier of the image, if not specified `kInvalidImageId`.
  image_t image_id_;

  // The name of the image, i.e. the relative path.
  std::string name_;

  // The identifier of the associated camera. Note that multiple images might
  // share the same camera. If not specified `kInvalidCameraId`.
  camera_t camera_id_;
  struct Camera* camera_ptr_;

  // The corresponding frame of the image. Note that multiple images might
  // share the same frame. If not specified `kInvalidFrameId`.
  frame_t frame_id_;
  class Frame* frame_ptr_;

  // The number of 3D points the image observes, i.e. the sum of its `points2D`
  // where `point3D_id != kInvalidPoint3DId`.
  point2D_t num_points3D_;

  // All image points, including points that are not part of a 3D point track.
  std::vector<struct Point2D> points2D_;
};

std::ostream& operator<<(std::ostream& stream, const Image& image);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

image_t Image::ImageId() const { return image_id_; }

void Image::SetImageId(const image_t image_id) { image_id_ = image_id; }

const std::string& Image::Name() const { return name_; }

std::string& Image::Name() { return name_; }

void Image::SetName(const std::string& name) { name_ = name; }

camera_t Image::CameraId() const { return camera_id_; }

void Image::SetCameraId(const camera_t camera_id) {
  THROW_CHECK_NE(camera_id, kInvalidCameraId);
  THROW_CHECK(!HasCameraPtr());
  camera_id_ = camera_id;
}

bool Image::HasCameraId() const { return camera_id_ != kInvalidCameraId; }

data_t Image::DataId() const {
  return data_t(sensor_t(SensorType::CAMERA, camera_id_), image_id_);
}

struct Camera* Image::CameraPtr() const {
  return THROW_CHECK_NOTNULL(camera_ptr_);
}

void Image::SetCameraPtr(struct Camera* camera) {
  THROW_CHECK_NOTNULL(camera);
  THROW_CHECK_NE(camera->camera_id, kInvalidCameraId);
  if (!HasCameraPtr()) {
    THROW_CHECK_EQ(camera->camera_id, camera_id_);
    camera_ptr_ = camera;
  } else {
    camera_id_ = camera->camera_id;
    camera_ptr_ = camera;
  }
}

void Image::ResetCameraPtr() { camera_ptr_ = nullptr; }

bool Image::HasCameraPtr() const { return camera_ptr_ != nullptr; }

frame_t Image::FrameId() const { return frame_id_; }

void Image::SetFrameId(const frame_t frame_id) {
  THROW_CHECK_NE(frame_id, kInvalidFrameId);
  THROW_CHECK(!HasFramePtr());
  frame_id_ = frame_id;
}

bool Image::HasFrameId() const { return frame_id_ != kInvalidFrameId; }

class Frame* Image::FramePtr() const { return THROW_CHECK_NOTNULL(frame_ptr_); }

void Image::SetFramePtr(class Frame* frame) {
  THROW_CHECK_NOTNULL(frame);
  THROW_CHECK_NE(frame->FrameId(), kInvalidFrameId);
  if (!HasFramePtr()) {
    THROW_CHECK_EQ(frame->FrameId(), frame_id_);
    frame_ptr_ = frame;
  } else {
    frame_id_ = frame->FrameId();
    frame_ptr_ = frame;
  }
}

void Image::ResetFramePtr() { frame_ptr_ = nullptr; }

bool Image::HasFramePtr() const { return frame_ptr_ != nullptr; }

bool Image::HasTrivialFrame() const {
  return THROW_CHECK_NOTNULL(frame_ptr_)
      ->RigPtr()
      ->IsRefSensor(sensor_t(SensorType::CAMERA, camera_id_));
}

point2D_t Image::NumPoints2D() const {
  return static_cast<point2D_t>(points2D_.size());
}

point2D_t Image::NumPoints3D() const { return num_points3D_; }

Rigid3d Image::CamFromWorld() const {
  return THROW_CHECK_NOTNULL(frame_ptr_)
      ->SensorFromWorld(sensor_t(SensorType::CAMERA, camera_id_));
}

bool Image::HasPose() const {
  if (frame_ptr_ == nullptr) {
    return false;
  } else {
    return frame_ptr_->HasPose();
  }
}

const struct Point2D& Image::Point2D(const point2D_t point2D_idx) const {
  return points2D_.at(point2D_idx);
}

struct Point2D& Image::Point2D(const point2D_t point2D_idx) {
  return points2D_.at(point2D_idx);
}

const std::vector<struct Point2D>& Image::Points2D() const { return points2D_; }

std::vector<struct Point2D>& Image::Points2D() { return points2D_; }

bool Image::operator==(const Image& other) const {
  const bool result = image_id_ == other.image_id_ &&          //
                      camera_id_ == other.camera_id_ &&        //
                      frame_id_ == other.frame_id_ &&          //
                      name_ == other.name_ &&                  //
                      num_points3D_ == other.num_points3D_ &&  //
                      HasPose() == other.HasPose() &&          //
                      points2D_ == other.points2D_;
  if (!HasPose()) {
    return result;
  } else {
    return result &&
           frame_ptr_->RigFromWorld() == other.frame_ptr_->RigFromWorld();
  }
}

bool Image::operator!=(const Image& other) const { return !(*this == other); }

}  // namespace colmap
