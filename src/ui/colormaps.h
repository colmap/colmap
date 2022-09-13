// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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
  virtual ~PointColormapBase() = default;

  virtual void Prepare(EIGEN_STL_UMAP(camera_t, Camera) & cameras,
                       EIGEN_STL_UMAP(image_t, Image) & images,
                       EIGEN_STL_UMAP(point3D_t, Point3D) & points3D,
                       std::vector<image_t>& reg_image_ids) = 0;

  virtual Eigen::Vector4f ComputeColor(const point3D_t point3D_id,
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
               std::vector<image_t>& reg_image_ids) override;

  Eigen::Vector4f ComputeColor(const point3D_t point3D_id,
                               const Point3D& point3D) override;
};

// Map color according to error.
class PointColormapError : public PointColormapBase {
 public:
  void Prepare(EIGEN_STL_UMAP(camera_t, Camera) & cameras,
               EIGEN_STL_UMAP(image_t, Image) & images,
               EIGEN_STL_UMAP(point3D_t, Point3D) & points3D,
               std::vector<image_t>& reg_image_ids) override;

  Eigen::Vector4f ComputeColor(const point3D_t point3D_id,
                               const Point3D& point3D) override;
};

// Map color according to track length.
class PointColormapTrackLen : public PointColormapBase {
 public:
  void Prepare(EIGEN_STL_UMAP(camera_t, Camera) & cameras,
               EIGEN_STL_UMAP(image_t, Image) & images,
               EIGEN_STL_UMAP(point3D_t, Point3D) & points3D,
               std::vector<image_t>& reg_image_ids) override;

  Eigen::Vector4f ComputeColor(const point3D_t point3D_id,
                               const Point3D& point3D) override;
};

// Map color according to ground-resolution.
class PointColormapGroundResolution : public PointColormapBase {
 public:
  void Prepare(EIGEN_STL_UMAP(camera_t, Camera) & cameras,
               EIGEN_STL_UMAP(image_t, Image) & images,
               EIGEN_STL_UMAP(point3D_t, Point3D) & points3D,
               std::vector<image_t>& reg_image_ids) override;

  Eigen::Vector4f ComputeColor(const point3D_t point3D_id,
                               const Point3D& point3D) override;

 private:
  std::unordered_map<point3D_t, float> resolutions_;
};

// Base class for image color mapping.
class ImageColormapBase {
 public:
  ImageColormapBase();
  virtual ~ImageColormapBase() = default;

  virtual void Prepare(EIGEN_STL_UMAP(camera_t, Camera) & cameras,
                       EIGEN_STL_UMAP(image_t, Image) & images,
                       EIGEN_STL_UMAP(point3D_t, Point3D) & points3D,
                       std::vector<image_t>& reg_image_ids) = 0;

  virtual void ComputeColor(const Image& image, Eigen::Vector4f* plane_color,
                            Eigen::Vector4f* frame_color) = 0;

  const static Eigen::Vector4f kDefaultPlaneColor;
  const static Eigen::Vector4f kDefaultFrameColor;
};

// Use uniform color for all images.
class ImageColormapUniform : public ImageColormapBase {
 public:
  void Prepare(EIGEN_STL_UMAP(camera_t, Camera) & cameras,
               EIGEN_STL_UMAP(image_t, Image) & images,
               EIGEN_STL_UMAP(point3D_t, Point3D) & points3D,
               std::vector<image_t>& reg_image_ids) override;

  void ComputeColor(const Image& image, Eigen::Vector4f* plane_color,
                    Eigen::Vector4f* frame_color) override;

  Eigen::Vector4f uniform_plane_color = kDefaultPlaneColor;
  Eigen::Vector4f uniform_frame_color = kDefaultFrameColor;
};

// Use color for images with specific words in their name.
class ImageColormapNameFilter : public ImageColormapBase {
 public:
  void Prepare(EIGEN_STL_UMAP(camera_t, Camera) & cameras,
               EIGEN_STL_UMAP(image_t, Image) & images,
               EIGEN_STL_UMAP(point3D_t, Point3D) & points3D,
               std::vector<image_t>& reg_image_ids) override;

  void AddColorForWord(const std::string& word,
                       const Eigen::Vector4f& plane_color,
                       const Eigen::Vector4f& frame_color);

  void ComputeColor(const Image& image, Eigen::Vector4f* plane_color,
                    Eigen::Vector4f* frame_color) override;

 private:
  // The plane and frame colors for different words.
  std::vector<
      std::pair<std::string, std::pair<Eigen::Vector4f, Eigen::Vector4f>>>
      image_name_colors_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_COLORMAPS_H_
