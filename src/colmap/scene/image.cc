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

#include "colmap/scene/image.h"

#include "colmap/geometry/pose.h"
#include "colmap/scene/projection.h"

namespace colmap {

Image::Image()
    : image_id_(kInvalidImageId),
      name_(""),
      camera_id_(kInvalidCameraId),
      camera_ptr_(nullptr),
      num_points3D_(0) {}

void Image::SetPoints2D(const std::vector<Eigen::Vector2d>& points) {
  THROW_CHECK(points2D_.empty());
  points2D_.resize(points.size());
  for (point2D_t point2D_idx = 0; point2D_idx < points.size(); ++point2D_idx) {
    points2D_[point2D_idx].xy = points[point2D_idx];
  }
}

void Image::SetPoints2D(const std::vector<struct Point2D>& points) {
  THROW_CHECK(points2D_.empty());
  points2D_ = points;
  num_points3D_ = 0;
  for (const auto& point2D : points2D_) {
    if (point2D.HasPoint3D()) {
      num_points3D_ += 1;
    }
  }
}

void Image::SetPoint3DForPoint2D(const point2D_t point2D_idx,
                                 const point3D_t point3D_id) {
  THROW_CHECK_NE(point3D_id, kInvalidPoint3DId);
  struct Point2D& point2D = points2D_.at(point2D_idx);
  if (!point2D.HasPoint3D()) {
    num_points3D_ += 1;
  }
  point2D.point3D_id = point3D_id;
}

void Image::ResetPoint3DForPoint2D(const point2D_t point2D_idx) {
  struct Point2D& point2D = points2D_.at(point2D_idx);
  if (point2D.HasPoint3D()) {
    point2D.point3D_id = kInvalidPoint3DId;
    num_points3D_ -= 1;
  }
}

bool Image::HasPoint3D(const point3D_t point3D_id) const {
  return std::find_if(points2D_.begin(),
                      points2D_.end(),
                      [point3D_id](const struct Point2D& point2D) {
                        return point2D.point3D_id == point3D_id;
                      }) != points2D_.end();
}

Eigen::Vector3d Image::ProjectionCenter() const {
  return CamFromWorld().rotation.inverse() * -CamFromWorld().translation;
}

Eigen::Vector3d Image::ViewingDirection() const {
  return CamFromWorld().rotation.toRotationMatrix().row(2);
}

std::pair<bool, Eigen::Vector2d> Image::ProjectPoint(
    const Eigen::Vector3d& point3D) const {
  THROW_CHECK(HasCameraPtr());
  const Eigen::Vector3d point3D_in_cam = CamFromWorld() * point3D;
  if (point3D_in_cam.z() < std::numeric_limits<double>::epsilon()) {
    return {false, Eigen::Vector2d()};
  }
  return {true, camera_ptr_->ImgFromCam(point3D_in_cam.hnormalized())};
}

std::ostream& operator<<(std::ostream& stream, const Image& image) {
  stream << "Image(image_id="
         << (image.ImageId() != kInvalidImageId
                 ? std::to_string(image.ImageId())
                 : "Invalid");
  if (!image.HasCameraPtr()) {
    stream << ", camera_id="
           << (image.HasCameraId() ? std::to_string(image.CameraId())
                                   : "Invalid");
  } else {
    stream << ", camera=Camera(camera_id=" << std::to_string(image.CameraId())
           << ")";
  }
  stream << ", name=\"" << image.Name() << "\""
         << ", has_pose=" << image.HasPose()
         << ", triangulated=" << image.NumPoints3D() << "/"
         << image.NumPoints2D() << ")";
  return stream;
}

}  // namespace colmap
