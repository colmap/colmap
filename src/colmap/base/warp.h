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

#ifndef COLMAP_SRC_BASE_WARP_H_
#define COLMAP_SRC_BASE_WARP_H_

#include "colmap/base/camera.h"
#include "colmap/util/bitmap.h"

namespace colmap {

// Warp source image to target image by projecting the pixels of the target
// image up to infinity and projecting it down into the source image
// (i.e. an inverse mapping). The function allocates the target image.
void WarpImageBetweenCameras(const Camera& source_camera,
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
// warping with the given homography. The function allocates the target image.
void WarpImageWithHomographyBetweenCameras(const Eigen::Matrix3d& H,
                                           const Camera& source_camera,
                                           const Camera& target_camera,
                                           const Bitmap& source_image,
                                           Bitmap* target_image);

// Resample row-major image using bilinear interpolation.
void ResampleImageBilinear(const float* data,
                           const int rows,
                           const int cols,
                           const int new_rows,
                           const int new_cols,
                           float* resampled);

// Smooth row-major image using a Gaussian filter kernel.
void SmoothImage(const float* data,
                 const int rows,
                 const int cols,
                 const float sigma_r,
                 const float sigma_c,
                 float* smoothed);

// Downsample row-major image by first smoothing and then resampling.
void DownsampleImage(const float* data,
                     const int rows,
                     const int cols,
                     const int new_rows,
                     const int new_cols,
                     float* downsampled);

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_WARP_H_
