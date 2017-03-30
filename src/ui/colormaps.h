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

#ifndef COLMAP_SRC_UI_COLORMAPS_H_
#define COLMAP_SRC_UI_COLORMAPS_H_

#include <Eigen/Core>

#include "base/reconstruction.h"
#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

// Base class for 3D point color mapping.
class PointColormapBase {
 public:
  PointColormapBase();

  virtual void Prepare(EIGEN_STL_UMAP(camera_t, Camera) & cameras,
                       EIGEN_STL_UMAP(image_t, Image) & images,
                       EIGEN_STL_UMAP(point3D_t, Point3D) & points3D,
                       std::vector<image_t>& reg_image_ids) = 0;

  virtual Eigen::Vector3f ComputeColor(const point3D_t point3D_id,
                                       const Point3D& point3D) = 0;

  void UpdateScale(std::vector<float>* values);
  float AdjustScale(const float gray);

  float scale;
  float min;
  float max;
  float range;
  float min_q;
  float max_q;
};

// Map color according to RGB value from image.
class PointColormapPhotometric : public PointColormapBase {
 public:
  void Prepare(EIGEN_STL_UMAP(camera_t, Camera) & cameras,
               EIGEN_STL_UMAP(image_t, Image) & images,
               EIGEN_STL_UMAP(point3D_t, Point3D) & points3D,
               std::vector<image_t>& reg_image_ids);

  Eigen::Vector3f ComputeColor(const point3D_t point3D_id,
                               const Point3D& point3D);
};

// Map color according to error.
class PointColormapError : public PointColormapBase {
 public:
  void Prepare(EIGEN_STL_UMAP(camera_t, Camera) & cameras,
               EIGEN_STL_UMAP(image_t, Image) & images,
               EIGEN_STL_UMAP(point3D_t, Point3D) & points3D,
               std::vector<image_t>& reg_image_ids);

  Eigen::Vector3f ComputeColor(const point3D_t point3D_id,
                               const Point3D& point3D);
};

// Map color according to track length.
class PointColormapTrackLen : public PointColormapBase {
 public:
  void Prepare(EIGEN_STL_UMAP(camera_t, Camera) & cameras,
               EIGEN_STL_UMAP(image_t, Image) & images,
               EIGEN_STL_UMAP(point3D_t, Point3D) & points3D,
               std::vector<image_t>& reg_image_ids);

  Eigen::Vector3f ComputeColor(const point3D_t point3D_id,
                               const Point3D& point3D);
};

// Map color according to ground-resolution.
class PointColormapGroundResolution : public PointColormapBase {
 public:
  void Prepare(EIGEN_STL_UMAP(camera_t, Camera) & cameras,
               EIGEN_STL_UMAP(image_t, Image) & images,
               EIGEN_STL_UMAP(point3D_t, Point3D) & points3D,
               std::vector<image_t>& reg_image_ids);

  Eigen::Vector3f ComputeColor(const point3D_t point3D_id,
                               const Point3D& point3D);

 private:
  std::unordered_map<point3D_t, float> resolutions_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_COLORMAPS_H_
