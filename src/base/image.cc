// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "base/image.h"

#include "base/pose.h"
#include "base/projection.h"

namespace colmap {
namespace {

static const double kNaN = std::numeric_limits<double>::quiet_NaN();

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
      qvec_(1.0, 0.0, 0.0, 0.0),
      tvec_(0.0, 0.0, 0.0),
      qvec_prior_(kNaN, kNaN, kNaN, kNaN),
      tvec_prior_(kNaN, kNaN, kNaN) {}

void Image::SetUp(const class Camera& camera) {
  CHECK_EQ(camera_id_, camera.CameraId());
  point3D_visibility_pyramid_ = VisibilityPyramid(
      kNumPoint3DVisibilityPyramidLevels, camera.Width(), camera.Height());
}

void Image::TearDown() {
  point3D_visibility_pyramid_ = VisibilityPyramid(0, 0, 0);
}

void Image::SetPoints2D(const std::vector<Eigen::Vector2d>& points) {
  CHECK(points2D_.empty());
  points2D_.resize(points.size());
  num_correspondences_have_point3D_.resize(points.size(), 0);
  for (point2D_t point2D_idx = 0; point2D_idx < points.size(); ++point2D_idx) {
    points2D_[point2D_idx].SetXY(points[point2D_idx]);
  }
}

void Image::SetPoints2D(const std::vector<class Point2D>& points) {
  CHECK(points2D_.empty());
  points2D_ = points;
  num_correspondences_have_point3D_.resize(points.size(), 0);
}

void Image::SetPoint3DForPoint2D(const point2D_t point2D_idx,
                                 const point3D_t point3D_id) {
  CHECK_NE(point3D_id, kInvalidPoint3DId);
  class Point2D& point2D = points2D_.at(point2D_idx);
  if (!point2D.HasPoint3D()) {
    num_points3D_ += 1;
  }
  point2D.SetPoint3DId(point3D_id);
}

void Image::ResetPoint3DForPoint2D(const point2D_t point2D_idx) {
  class Point2D& point2D = points2D_.at(point2D_idx);
  if (point2D.HasPoint3D()) {
    point2D.SetPoint3DId(kInvalidPoint3DId);
    num_points3D_ -= 1;
  }
}

bool Image::HasPoint3D(const point3D_t point3D_id) const {
  return std::find_if(points2D_.begin(), points2D_.end(),
                      [point3D_id](const class Point2D& point2D) {
                        return point2D.Point3DId() == point3D_id;
                      }) != points2D_.end();
}

void Image::IncrementCorrespondenceHasPoint3D(const point2D_t point2D_idx) {
  const class Point2D& point2D = points2D_.at(point2D_idx);

  num_correspondences_have_point3D_[point2D_idx] += 1;
  if (num_correspondences_have_point3D_[point2D_idx] == 1) {
    num_visible_points3D_ += 1;
  }

  point3D_visibility_pyramid_.SetPoint(point2D.X(), point2D.Y());

  assert(num_visible_points3D_ <= num_observations_);
}

void Image::DecrementCorrespondenceHasPoint3D(const point2D_t point2D_idx) {
  const class Point2D& point2D = points2D_.at(point2D_idx);

  num_correspondences_have_point3D_[point2D_idx] -= 1;
  if (num_correspondences_have_point3D_[point2D_idx] == 0) {
    num_visible_points3D_ -= 1;
  }

  point3D_visibility_pyramid_.ResetPoint(point2D.X(), point2D.Y());

  assert(num_visible_points3D_ <= num_observations_);
}

void Image::NormalizeQvec() { qvec_ = NormalizeQuaternion(qvec_); }

Eigen::Matrix3x4d Image::ProjectionMatrix() const {
  return ComposeProjectionMatrix(qvec_, tvec_);
}

Eigen::Matrix3x4d Image::InverseProjectionMatrix() const {
  return InvertProjectionMatrix(ComposeProjectionMatrix(qvec_, tvec_));
}

Eigen::Matrix3d Image::RotationMatrix() const {
  return QuaternionToRotationMatrix(qvec_);
}

Eigen::Vector3d Image::ProjectionCenter() const {
  return ProjectionCenterFromParameters(qvec_, tvec_);
}

Eigen::Vector3d Image::ViewingDirection() const {
  return RotationMatrix().row(2);
}

}  // namespace colmap
