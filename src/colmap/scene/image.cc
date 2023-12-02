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
namespace {

static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

const int Image::kNumPoint3DVisibilityPyramidLevels = 6;

Image::Image()
    : image_id_(kInvalidImageId),
      name_(""),
      camera_id_(kInvalidCameraId),
      registered_(false),
      num_points3D_(0),
      num_observations_(0),
      num_correspondences_(0),
      num_visible_points3D_(0),
      cam_from_world_prior_(Eigen::Quaterniond(kNaN, kNaN, kNaN, kNaN),
                            Eigen::Vector3d(kNaN, kNaN, kNaN)) {}

void Image::SetUp(const struct Camera& camera) {
  CHECK_EQ(camera_id_, camera.camera_id);
  point3D_visibility_pyramid_ = VisibilityPyramid(
      kNumPoint3DVisibilityPyramidLevels, camera.width, camera.height);
}

void Image::TearDown() {
  point3D_visibility_pyramid_ = VisibilityPyramid(0, 0, 0);
}

void Image::SetPoints2D(const std::vector<Eigen::Vector2d>& points) {
  CHECK(points2D_.empty());
  points2D_.resize(points.size());
  num_correspondences_have_point3D_.resize(points.size(), 0);
  for (point2D_t point2D_idx = 0; point2D_idx < points.size(); ++point2D_idx) {
    points2D_[point2D_idx].xy = points[point2D_idx];
  }
}

void Image::SetPoints2D(const std::vector<struct Point2D>& points) {
  CHECK(points2D_.empty());
  points2D_ = points;
  num_correspondences_have_point3D_.resize(points.size(), 0);
  num_points3D_ = 0;
  for (const auto& point2D : points2D_) {
    if (point2D.HasPoint3D()) {
      num_points3D_ += 1;
    }
  }
}

void Image::SetPoint3DForPoint2D(const point2D_t point2D_idx,
                                 const point3D_t point3D_id) {
  CHECK_NE(point3D_id, kInvalidPoint3DId);
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

void Image::IncrementCorrespondenceHasPoint3D(const point2D_t point2D_idx) {
  const struct Point2D& point2D = points2D_.at(point2D_idx);

  num_correspondences_have_point3D_[point2D_idx] += 1;
  if (num_correspondences_have_point3D_[point2D_idx] == 1) {
    num_visible_points3D_ += 1;
  }

  point3D_visibility_pyramid_.SetPoint(point2D.xy(0), point2D.xy(1));

  assert(num_visible_points3D_ <= num_observations_);
}

void Image::DecrementCorrespondenceHasPoint3D(const point2D_t point2D_idx) {
  const struct Point2D& point2D = points2D_.at(point2D_idx);

  num_correspondences_have_point3D_[point2D_idx] -= 1;
  if (num_correspondences_have_point3D_[point2D_idx] == 0) {
    num_visible_points3D_ -= 1;
  }

  point3D_visibility_pyramid_.ResetPoint(point2D.xy(0), point2D.xy(1));

  assert(num_visible_points3D_ <= num_observations_);
}

Eigen::Vector3d Image::ProjectionCenter() const {
  return cam_from_world_.rotation.inverse() * -cam_from_world_.translation;
}

Eigen::Vector3d Image::ViewingDirection() const {
  return cam_from_world_.rotation.toRotationMatrix().row(2);
}

}  // namespace colmap
