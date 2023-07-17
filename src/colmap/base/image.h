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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include "colmap/base/camera.h"
#include "colmap/base/point2d.h"
#include "colmap/base/visibility_pyramid.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
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

  // Setup / tear down the image and necessary internal data structures before
  // and after being used in reconstruction.
  void SetUp(const Camera& camera);
  void TearDown();

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
  inline bool HasCamera() const;

  // Check if image is registered.
  inline bool IsRegistered() const;
  inline void SetRegistered(bool registered);

  // Get the number of image points.
  inline point2D_t NumPoints2D() const;

  // Get the number of triangulations, i.e. the number of points that
  // are part of a 3D point track.
  inline point2D_t NumPoints3D() const;

  // Get the number of observations, i.e. the number of image points that
  // have at least one correspondence to another image.
  inline point2D_t NumObservations() const;
  inline void SetNumObservations(point2D_t num_observations);

  // Get the number of correspondences for all image points.
  inline point2D_t NumCorrespondences() const;
  inline void SetNumCorrespondences(point2D_t num_observations);

  // Get the number of observations that see a triangulated point, i.e. the
  // number of image points that have at least one correspondence to a
  // triangulated point in another image.
  inline point2D_t NumVisiblePoints3D() const;

  // Get the score of triangulated observations. In contrast to
  // `NumVisiblePoints3D`, this score also captures the distribution
  // of triangulated observations in the image. This is useful to select
  // the next best image in incremental reconstruction, because a more
  // uniform distribution of observations results in more robust registration.
  inline size_t Point3DVisibilityScore() const;

  // World to camera pose.
  inline const Rigid3d& CamFromWorld() const;
  inline Rigid3d& CamFromWorld();

  // World to camera pose prior, e.g. given by EXIF gyroscope tag.
  inline const Rigid3d& CamFromWorldPrior() const;
  inline Rigid3d& CamFromWorldPrior();

  // Access the coordinates of image points.
  inline const class Point2D& Point2D(point2D_t point2D_idx) const;
  inline class Point2D& Point2D(point2D_t point2D_idx);
  inline const std::vector<class Point2D>& Points2D() const;
  inline std::vector<class Point2D>& Points2D();
  void SetPoints2D(const std::vector<Eigen::Vector2d>& points);
  void SetPoints2D(const std::vector<class Point2D>& points);

  // Set the point as triangulated, i.e. it is part of a 3D point track.
  void SetPoint3DForPoint2D(point2D_t point2D_idx, point3D_t point3D_id);

  // Set the point as not triangulated, i.e. it is not part of a 3D point track.
  void ResetPoint3DForPoint2D(point2D_t point2D_idx);

  // Check whether an image point has a correspondence to an image point in
  // another image that has a 3D point.
  inline bool IsPoint3DVisible(point2D_t point2D_idx) const;

  // Check whether one of the image points is part of the 3D point track.
  bool HasPoint3D(point3D_t point3D_id) const;

  // Indicate that another image has a point that is triangulated and has
  // a correspondence to this image point. Note that this must only be called
  // after calling `SetUp`.
  void IncrementCorrespondenceHasPoint3D(point2D_t point2D_idx);

  // Indicate that another image has a point that is not triangulated any more
  // and has a correspondence to this image point. This assumes that
  // `IncrementCorrespondenceHasPoint3D` was called for the same image point
  // and correspondence before. Note that this must only be called
  // after calling `SetUp`.
  void DecrementCorrespondenceHasPoint3D(point2D_t point2D_idx);

  // Extract the projection center in world space.
  Eigen::Vector3d ProjectionCenter() const;

  // Extract the viewing direction of the image.
  Eigen::Vector3d ViewingDirection() const;

  // The number of levels in the 3D point multi-resolution visibility pyramid.
  static const int kNumPoint3DVisibilityPyramidLevels;

 private:
  // Identifier of the image, if not specified `kInvalidImageId`.
  image_t image_id_;

  // The name of the image, i.e. the relative path.
  std::string name_;

  // The identifier of the associated camera. Note that multiple images might
  // share the same camera. If not specified `kInvalidCameraId`.
  camera_t camera_id_;

  // Whether the image is successfully registered in the reconstruction.
  bool registered_;

  // The number of 3D points the image observes, i.e. the sum of its `points2D`
  // where `point3D_id != kInvalidPoint3DId`.
  point2D_t num_points3D_;

  // The number of image points that have at least one correspondence to
  // another image.
  point2D_t num_observations_;

  // The sum of correspondences per image point.
  point2D_t num_correspondences_;

  // The number of 2D points, which have at least one corresponding 2D point in
  // another image that is part of a 3D point track, i.e. the sum of `points2D`
  // where `num_tris > 0`.
  point2D_t num_visible_points3D_;

  // The pose of the image, defined as the transformation from world to camera.
  Rigid3d cam_from_world_;

  // The pose prior of the image, e.g. extracted from EXIF tags.
  Rigid3d cam_from_world_prior_;

  // All image points, including points that are not part of a 3D point track.
  std::vector<class Point2D> points2D_;

  // Per image point, the number of correspondences that have a 3D point.
  std::vector<point2D_t> num_correspondences_have_point3D_;

  // Data structure to compute the distribution of triangulated correspondences
  // in the image. Note that this structure is only usable after `SetUp`.
  VisibilityPyramid point3D_visibility_pyramid_;
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
  CHECK_NE(camera_id, kInvalidCameraId);
  camera_id_ = camera_id;
}

inline bool Image::HasCamera() const { return camera_id_ != kInvalidCameraId; }

bool Image::IsRegistered() const { return registered_; }

void Image::SetRegistered(const bool registered) { registered_ = registered; }

point2D_t Image::NumPoints2D() const {
  return static_cast<point2D_t>(points2D_.size());
}

point2D_t Image::NumPoints3D() const { return num_points3D_; }

point2D_t Image::NumObservations() const { return num_observations_; }

void Image::SetNumObservations(const point2D_t num_observations) {
  num_observations_ = num_observations;
}

point2D_t Image::NumCorrespondences() const { return num_correspondences_; }

void Image::SetNumCorrespondences(const point2D_t num_correspondences) {
  num_correspondences_ = num_correspondences;
}

point2D_t Image::NumVisiblePoints3D() const { return num_visible_points3D_; }

size_t Image::Point3DVisibilityScore() const {
  return point3D_visibility_pyramid_.Score();
}

const Rigid3d& Image::CamFromWorld() const { return cam_from_world_; }

Rigid3d& Image::CamFromWorld() { return cam_from_world_; }

const Rigid3d& Image::CamFromWorldPrior() const {
  return cam_from_world_prior_;
}

Rigid3d& Image::CamFromWorldPrior() { return cam_from_world_prior_; }

const class Point2D& Image::Point2D(const point2D_t point2D_idx) const {
  return points2D_.at(point2D_idx);
}

class Point2D& Image::Point2D(const point2D_t point2D_idx) {
  return points2D_.at(point2D_idx);
}

const std::vector<class Point2D>& Image::Points2D() const { return points2D_; }

std::vector<class Point2D>& Image::Points2D() { return points2D_; }

bool Image::IsPoint3DVisible(const point2D_t point2D_idx) const {
  return num_correspondences_have_point3D_.at(point2D_idx) > 0;
}

}  // namespace colmap
