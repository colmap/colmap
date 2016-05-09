// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#include "ui/colormaps.h"

#include "base/camera_models.h"
#include "base/pose.h"
#include "util/math.h"

namespace colmap {

float JetColormap::Red(const float gray) { return Base(gray - 0.25f); }

float JetColormap::Green(const float gray) { return Base(gray); }

float JetColormap::Blue(const float gray) { return Base(gray + 0.25f); }

float JetColormap::Base(const float val) {
  if (val <= 0.125f) {
    return 0.0f;
  } else if (val <= 0.375f) {
    return Interpolate(2.0f * val - 1.0f, 0.0f, -0.75f, 1.0f, -0.25f);
  } else if (val <= 0.625f) {
    return 1.0f;
  } else if (val <= 0.87f) {
    return Interpolate(2.0f * val - 1.0f, 1.0f, 0.25f, 0.0f, 0.75f);
  } else {
    return 0.0f;
  }
}

float JetColormap::Interpolate(const float val, const float y0, const float x0,
                               const float y1, const float x1) {
  return (val - x0) * (y1 - y0) / (x1 - x0) + y0;
}

PointColormapBase::PointColormapBase()
    : scale(1.0f),
      min(0.0f),
      max(0.0f),
      range(0.0f),
      min_q(0.0f),
      max_q(1.0f) {}

void PointColormapBase::UpdateScale(std::vector<float>* values) {
  if (values->empty()) {
    min = 0.0f;
    max = 0.0f;
    range = 0.0f;
  } else {
    std::sort(values->begin(), values->end());
    min = (*values)[static_cast<size_t>(min_q * (values->size() - 1))];
    max = (*values)[static_cast<size_t>(max_q * (values->size() - 1))];
    range = max - min;
  }
}

float PointColormapBase::AdjustScale(const float gray) {
  if (range == 0.0f) {
    return 0.0f;
  } else {
    const float gray_clipped = std::min(std::max(gray, min), max);
    const float gray_scaled = (gray_clipped - min) / range;
    return std::pow(gray_scaled, scale);
  }
}

void PointColormapPhotometric::Prepare(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& points3D,
    std::vector<image_t>& reg_image_ids) {}

Eigen::Vector3f PointColormapPhotometric::ComputeColor(
    const point3D_t point3D_id, const Point3D& point3D) {
  return Eigen::Vector3f(point3D.Color(0) / 255.0f, point3D.Color(1) / 255.0f,
                         point3D.Color(2) / 255.0f);
}

void PointColormapError::Prepare(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& points3D,
    std::vector<image_t>& reg_image_ids) {
  std::vector<float> errors;
  errors.reserve(points3D.size());

  for (const auto& point3D : points3D) {
    errors.push_back(static_cast<float>(point3D.second.Error()));
  }

  UpdateScale(&errors);
}

Eigen::Vector3f PointColormapError::ComputeColor(const point3D_t point3D_id,
                                                 const Point3D& point3D) {
  const float gray = AdjustScale(static_cast<float>(point3D.Error()));
  return Eigen::Vector3f(JetColormap::Red(gray), JetColormap::Green(gray),
                         JetColormap::Blue(gray));
}

void PointColormapTrackLen::Prepare(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& points3D,
    std::vector<image_t>& reg_image_ids) {
  std::vector<float> track_lengths;
  track_lengths.reserve(points3D.size());

  for (const auto& point3D : points3D) {
    track_lengths.push_back(point3D.second.Track().Length());
  }

  UpdateScale(&track_lengths);
}

Eigen::Vector3f PointColormapTrackLen::ComputeColor(const point3D_t point3D_id,
                                                    const Point3D& point3D) {
  const float gray = AdjustScale(point3D.Track().Length());
  return Eigen::Vector3f(JetColormap::Red(gray), JetColormap::Green(gray),
                         JetColormap::Blue(gray));
}

void PointColormapGroundResolution::Prepare(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& points3D,
    std::vector<image_t>& reg_image_ids) {
  std::vector<float> resolutions;
  resolutions.reserve(points3D.size());

  std::unordered_map<camera_t, float> focal_lengths;
  std::unordered_map<camera_t, Eigen::Vector2f> principal_points;
  for (const auto& camera : cameras) {
    focal_lengths[camera.first] =
        static_cast<float>(camera.second.MeanFocalLength());
    principal_points[camera.first] =
        Eigen::Vector2f(static_cast<float>(camera.second.PrincipalPointX()),
                        static_cast<float>(camera.second.PrincipalPointY()));
  }

  std::unordered_map<image_t, Eigen::Vector3f> proj_centers;
  for (const auto& image : images) {
    proj_centers[image.first] = image.second.ProjectionCenter().cast<float>();
  }

  for (const auto& point3D : points3D) {
    float min_resolution = std::numeric_limits<float>::max();

    const Eigen::Vector3f xyz = point3D.second.XYZ().cast<float>();

    for (const auto track_el : point3D.second.Track().Elements()) {
      const auto& image = images[track_el.image_id];
      const float focal_length = focal_lengths[image.CameraId()];
      const float focal_length2 = focal_length * focal_length;
      const Eigen::Vector2f& pp = principal_points[image.CameraId()];

      const Eigen::Vector2f xy =
          image.Point2D(track_el.point2D_idx).XY().cast<float>() - pp;

      // Distance from principal point to observation on image plane
      const float pixel_radius1 = xy.norm();

      const float x1 = xy(0) + (xy(0) < 0 ? -1.0f : 1.0f);
      const float y1 = xy(1) + (xy(1) < 0 ? -1.0f : 1.0f);
      const float pixel_radius2 = std::sqrt(x1 * x1 + y1 * y1);

      // Distance from camera center to observation on image plane
      const float pixel_dist1 =
          std::sqrt(pixel_radius1 * pixel_radius1 + focal_length2);
      const float pixel_dist2 =
          std::sqrt(pixel_radius2 * pixel_radius2 + focal_length2);

      // Distance from 3D point to camera center
      const float dist = (xyz - proj_centers[track_el.image_id]).norm();

      // Perpendicular distance from 3D point to principal axis
      const float r1 = pixel_radius1 * dist / pixel_dist1;
      const float r2 = pixel_radius2 * dist / pixel_dist2;
      const float dr = r2 - r1;

      // Ground resolution of observation, use "minus" to highlight
      // high resolution
      const float resolution = -dr * dr;

      if (std::isfinite(resolution)) {
        min_resolution = std::min(resolution, min_resolution);
      }
    }

    resolutions.push_back(min_resolution);
    resolutions_[point3D.first] = min_resolution;
  }

  UpdateScale(&resolutions);
}

Eigen::Vector3f PointColormapGroundResolution::ComputeColor(
    const point3D_t point3D_id, const Point3D& point3D) {
  const float gray = AdjustScale(resolutions_[point3D_id]);
  return Eigen::Vector3f(JetColormap::Red(gray), JetColormap::Green(gray),
                         JetColormap::Blue(gray));
}

}  // namespace colmap
