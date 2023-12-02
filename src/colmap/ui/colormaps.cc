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

#include "colmap/ui/colormaps.h"

#include "colmap/sensor/bitmap.h"

namespace colmap {

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

Eigen::Vector4f PointColormapPhotometric::ComputeColor(
    const point3D_t point3D_id, const Point3D& point3D) {
  return Eigen::Vector4f(point3D.color(0) / 255.0f,
                         point3D.color(1) / 255.0f,
                         point3D.color(2) / 255.0f,
                         1.0f);
}

void PointColormapError::Prepare(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& points3D,
    std::vector<image_t>& reg_image_ids) {
  std::vector<float> errors;
  errors.reserve(points3D.size());

  for (const auto& point3D : points3D) {
    errors.push_back(static_cast<float>(point3D.second.error));
  }

  UpdateScale(&errors);
}

Eigen::Vector4f PointColormapError::ComputeColor(const point3D_t point3D_id,
                                                 const Point3D& point3D) {
  const float gray = AdjustScale(static_cast<float>(point3D.error));
  return Eigen::Vector4f(JetColormap::Red(gray),
                         JetColormap::Green(gray),
                         JetColormap::Blue(gray),
                         1.0f);
}

void PointColormapTrackLen::Prepare(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& points3D,
    std::vector<image_t>& reg_image_ids) {
  std::vector<float> track_lengths;
  track_lengths.reserve(points3D.size());

  for (const auto& point3D : points3D) {
    track_lengths.push_back(point3D.second.track.Length());
  }

  UpdateScale(&track_lengths);
}

Eigen::Vector4f PointColormapTrackLen::ComputeColor(const point3D_t point3D_id,
                                                    const Point3D& point3D) {
  const float gray = AdjustScale(point3D.track.Length());
  return Eigen::Vector4f(JetColormap::Red(gray),
                         JetColormap::Green(gray),
                         JetColormap::Blue(gray),
                         1.0f);
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

    const Eigen::Vector3f xyz = point3D.second.xyz.cast<float>();

    for (const auto& track_el : point3D.second.track.Elements()) {
      const auto& image = images[track_el.image_id];
      const float focal_length = focal_lengths[image.CameraId()];
      const float focal_length2 = focal_length * focal_length;
      const Eigen::Vector2f& pp = principal_points[image.CameraId()];

      const Eigen::Vector2f xy =
          image.Point2D(track_el.point2D_idx).xy.cast<float>() - pp;

      // Distance from principal point to observation on image plane.
      const float pixel_radius1 = xy.norm();

      const float x1 = xy(0) + (xy(0) < 0 ? -1.0f : 1.0f);
      const float y1 = xy(1) + (xy(1) < 0 ? -1.0f : 1.0f);
      const float pixel_radius2 = std::sqrt(x1 * x1 + y1 * y1);

      // Distance from camera center to observation on image plane.
      const float pixel_dist1 =
          std::sqrt(pixel_radius1 * pixel_radius1 + focal_length2);
      const float pixel_dist2 =
          std::sqrt(pixel_radius2 * pixel_radius2 + focal_length2);

      // Distance from 3D point to camera center.
      const float dist = (xyz - proj_centers[track_el.image_id]).norm();

      // Perpendicular distance from 3D point to principal axis
      const float r1 = pixel_radius1 * dist / pixel_dist1;
      const float r2 = pixel_radius2 * dist / pixel_dist2;
      const float dr = r2 - r1;

      // Ground resolution of observation, use "minus" to highlight
      // high resolution.
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

Eigen::Vector4f PointColormapGroundResolution::ComputeColor(
    const point3D_t point3D_id, const Point3D& point3D) {
  const float gray = AdjustScale(resolutions_[point3D_id]);
  return Eigen::Vector4f(JetColormap::Red(gray),
                         JetColormap::Green(gray),
                         JetColormap::Blue(gray),
                         1.0f);
}

const Eigen::Vector4f ImageColormapBase::kDefaultPlaneColor = {
    1.0f, 0.1f, 0.0f, 0.6f};
const Eigen::Vector4f ImageColormapBase::kDefaultFrameColor = {
    0.8f, 0.1f, 0.0f, 1.0f};

ImageColormapBase::ImageColormapBase() {}

void ImageColormapUniform::Prepare(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& points3D,
    std::vector<image_t>& reg_image_ids) {}

void ImageColormapUniform::ComputeColor(const Image& image,
                                        Eigen::Vector4f* plane_color,
                                        Eigen::Vector4f* frame_color) {
  *plane_color = uniform_plane_color;
  *frame_color = uniform_frame_color;
}

void ImageColormapNameFilter::Prepare(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& points3D,
    std::vector<image_t>& reg_image_ids) {}

void ImageColormapNameFilter::AddColorForWord(
    const std::string& word,
    const Eigen::Vector4f& plane_color,
    const Eigen::Vector4f& frame_color) {
  image_name_colors_.emplace_back(word,
                                  std::make_pair(plane_color, frame_color));
}

void ImageColormapNameFilter::ComputeColor(const Image& image,
                                           Eigen::Vector4f* plane_color,
                                           Eigen::Vector4f* frame_color) {
  for (const auto& image_name_color : image_name_colors_) {
    if (StringContains(image.Name(), image_name_color.first)) {
      *plane_color = image_name_color.second.first;
      *frame_color = image_name_color.second.second;
      return;
    }
  }

  *plane_color = kDefaultPlaneColor;
  *frame_color = kDefaultFrameColor;
}

}  // namespace colmap
