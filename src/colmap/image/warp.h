// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/camera.h"
#include "colmap/sensor/bitmap.h"

namespace colmap {

struct WarpImageOptions {
  enum class Interpolation {
    kNearestNeighbor,
    kBilinear,
  };

  // Interpolation method for sampling the source image.
  Interpolation interpolation = Interpolation::kBilinear;

  // Minimum target-to-source dimension ratio for warping directly at target
  // resolution. Smaller ratios warp at source resolution and resize the result
  // to avoid aliasing.
  double direct_warp_min_scale = 0.5;
};

// Warp source image to target image by projecting the pixels of the target
// image up to infinity and projecting it down into the source image
// (i.e. an inverse mapping). The function allocates the target image. If the
// minimum target-to-source dimension ratio is at least
// `options.direct_warp_min_scale`, the image is warped directly at target
// resolution. Otherwise, it is warped at source resolution and resized to
// avoid aliasing.
void WarpImageBetweenCameras(const WarpImageOptions& options,
                             const Camera& source_camera,
                             const Camera& target_camera,
                             const Bitmap& source_image,
                             Bitmap* target_image);

// Warp an image with the given homography, where H defines the pixel mapping
// from the target to source image. Note that the pixel centers are assumed to
// have coordinates (0.5, 0.5).
void WarpImageWithHomography(const Eigen::Matrix3d& H,
                             const Bitmap& source_image,
                             Bitmap* target_image);

// First, warp source image to target image by projecting the pixels of the
// target image up to infinity and projecting it down into the source image
// (i.e. an inverse mapping). Second, warp the coordinates from the first
// warping with the given homography. The function allocates the target image
// and uses the same direct-warp threshold as `WarpImageBetweenCameras`.
void WarpImageWithHomographyBetweenCameras(const WarpImageOptions& options,
                                           const Eigen::Matrix3d& H,
                                           const Camera& source_camera,
                                           const Camera& target_camera,
                                           const Bitmap& source_image,
                                           Bitmap* target_image);

// Resample row-major image using bilinear interpolation.
void ResampleImageBilinear(const float* data,
                           int rows,
                           int cols,
                           int new_rows,
                           int new_cols,
                           float* resampled);

// Smooth row-major image using a Gaussian filter kernel.
void SmoothImage(const float* data,
                 int rows,
                 int cols,
                 float sigma_r,
                 float sigma_c,
                 float* smoothed);

// Downsample row-major image by first smoothing and then resampling.
void DownsampleImage(const float* data,
                     int rows,
                     int cols,
                     int new_rows,
                     int new_cols,
                     float* downsampled);

}  // namespace colmap
