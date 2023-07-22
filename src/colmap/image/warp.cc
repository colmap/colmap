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

#include "colmap/image/warp.h"

#include "colmap/util/logging.h"

#include "lib/VLFeat/imopv.h"

#include <Eigen/Geometry>

namespace colmap {
namespace {

float GetPixelConstantBorder(const float* data,
                             const int rows,
                             const int cols,
                             const int row,
                             const int col) {
  if (row >= 0 && col >= 0 && row < rows && col < cols) {
    return data[row * cols + col];
  } else {
    return 0;
  }
}

}  // namespace

void WarpImageBetweenCameras(const Camera& source_camera,
                             const Camera& target_camera,
                             const Bitmap& source_image,
                             Bitmap* target_image) {
  CHECK_EQ(source_camera.Width(), source_image.Width());
  CHECK_EQ(source_camera.Height(), source_image.Height());
  CHECK_NOTNULL(target_image);

  target_image->Allocate(static_cast<int>(source_camera.Width()),
                         static_cast<int>(source_camera.Height()),
                         source_image.IsRGB());

  // To avoid aliasing, perform the warping in the source resolution and
  // then rescale the image at the end.
  Camera scaled_target_camera = target_camera;
  if (target_camera.Width() != source_camera.Width() ||
      target_camera.Height() != source_camera.Height()) {
    scaled_target_camera.Rescale(source_camera.Width(), source_camera.Height());
  }

  Eigen::Vector2d image_point;
  for (int y = 0; y < target_image->Height(); ++y) {
    image_point.y() = y + 0.5;
    for (int x = 0; x < target_image->Width(); ++x) {
      image_point.x() = x + 0.5;

      // Camera models assume that the upper left pixel center is (0.5, 0.5).
      const Eigen::Vector2d cam_point =
          scaled_target_camera.CamFromImg(image_point);
      const Eigen::Vector2d source_point = source_camera.ImgFromCam(cam_point);

      BitmapColor<float> color;
      if (source_image.InterpolateBilinear(
              source_point.x() - 0.5, source_point.y() - 0.5, &color)) {
        target_image->SetPixel(x, y, color.Cast<uint8_t>());
      } else {
        target_image->SetPixel(x, y, BitmapColor<uint8_t>(0));
      }
    }
  }

  if (target_camera.Width() != source_camera.Width() ||
      target_camera.Height() != source_camera.Height()) {
    target_image->Rescale(target_camera.Width(), target_camera.Height());
  }
}

void WarpImageWithHomography(const Eigen::Matrix3d& H,
                             const Bitmap& source_image,
                             Bitmap* target_image) {
  CHECK_NOTNULL(target_image);
  CHECK_GT(target_image->Width(), 0);
  CHECK_GT(target_image->Height(), 0);
  CHECK_EQ(source_image.IsRGB(), target_image->IsRGB());

  Eigen::Vector3d target_pixel(0, 0, 1);
  for (int y = 0; y < target_image->Height(); ++y) {
    target_pixel.y() = y + 0.5;
    for (int x = 0; x < target_image->Width(); ++x) {
      target_pixel.x() = x + 0.5;

      const Eigen::Vector2d source_pixel = (H * target_pixel).hnormalized();

      BitmapColor<float> color;
      if (source_image.InterpolateBilinear(
              source_pixel.x() - 0.5, source_pixel.y() - 0.5, &color)) {
        target_image->SetPixel(x, y, color.Cast<uint8_t>());
      } else {
        target_image->SetPixel(x, y, BitmapColor<uint8_t>(0));
      }
    }
  }
}

void WarpImageWithHomographyBetweenCameras(const Eigen::Matrix3d& H,
                                           const Camera& source_camera,
                                           const Camera& target_camera,
                                           const Bitmap& source_image,
                                           Bitmap* target_image) {
  CHECK_EQ(source_camera.Width(), source_image.Width());
  CHECK_EQ(source_camera.Height(), source_image.Height());
  CHECK_NOTNULL(target_image);

  target_image->Allocate(static_cast<int>(source_camera.Width()),
                         static_cast<int>(source_camera.Height()),
                         source_image.IsRGB());

  // To avoid aliasing, perform the warping in the source resolution and
  // then rescale the image at the end.
  Camera scaled_target_camera = target_camera;
  if (target_camera.Width() != source_camera.Width() ||
      target_camera.Height() != source_camera.Height()) {
    scaled_target_camera.Rescale(source_camera.Width(), source_camera.Height());
  }

  Eigen::Vector3d image_point(0, 0, 1);
  for (int y = 0; y < target_image->Height(); ++y) {
    image_point.y() = y + 0.5;
    for (int x = 0; x < target_image->Width(); ++x) {
      image_point.x() = x + 0.5;

      // Camera models assume that the upper left pixel center is (0.5, 0.5).
      const Eigen::Vector3d warped_point = H * image_point;
      const Eigen::Vector2d cam_point =
          target_camera.CamFromImg(warped_point.hnormalized());
      const Eigen::Vector2d source_point = source_camera.ImgFromCam(cam_point);

      BitmapColor<float> color;
      if (source_image.InterpolateBilinear(
              source_point.x() - 0.5, source_point.y() - 0.5, &color)) {
        target_image->SetPixel(x, y, color.Cast<uint8_t>());
      } else {
        target_image->SetPixel(x, y, BitmapColor<uint8_t>(0));
      }
    }
  }

  if (target_camera.Width() != source_camera.Width() ||
      target_camera.Height() != source_camera.Height()) {
    target_image->Rescale(target_camera.Width(), target_camera.Height());
  }
}

void ResampleImageBilinear(const float* data,
                           const int rows,
                           const int cols,
                           const int new_rows,
                           const int new_cols,
                           float* resampled) {
  CHECK_NOTNULL(data);
  CHECK_NOTNULL(resampled);
  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  CHECK_GT(new_rows, 0);
  CHECK_GT(new_cols, 0);

  const float scale_r = static_cast<float>(rows) / static_cast<float>(new_rows);
  const float scale_c = static_cast<float>(cols) / static_cast<float>(new_cols);

  for (int r = 0; r < new_rows; ++r) {
    const float r_i = (r + 0.5f) * scale_r - 0.5f;
    const int r_i_min = std::floor(r_i);
    const int r_i_max = r_i_min + 1;
    const float d_r_min = r_i - r_i_min;
    const float d_r_max = r_i_max - r_i;

    for (int c = 0; c < new_cols; ++c) {
      const float c_i = (c + 0.5f) * scale_c - 0.5f;
      const int c_i_min = std::floor(c_i);
      const int c_i_max = c_i_min + 1;
      const float d_c_min = c_i - c_i_min;
      const float d_c_max = c_i_max - c_i;

      // Interpolation in column direction.
      const float value1 =
          d_c_max * GetPixelConstantBorder(data, rows, cols, r_i_min, c_i_min) +
          d_c_min * GetPixelConstantBorder(data, rows, cols, r_i_min, c_i_max);
      const float value2 =
          d_c_max * GetPixelConstantBorder(data, rows, cols, r_i_max, c_i_min) +
          d_c_min * GetPixelConstantBorder(data, rows, cols, r_i_max, c_i_max);

      // Interpolation in row direction.
      resampled[r * new_cols + c] = d_r_max * value1 + d_r_min * value2;
    }
  }
}

void SmoothImage(const float* data,
                 const int rows,
                 const int cols,
                 const float sigma_r,
                 const float sigma_c,
                 float* smoothed) {
  CHECK_NOTNULL(data);
  CHECK_NOTNULL(smoothed);
  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  CHECK_GT(sigma_r, 0);
  CHECK_GT(sigma_c, 0);
  vl_imsmooth_f(smoothed, cols, data, cols, rows, cols, sigma_c, sigma_r);
}

void DownsampleImage(const float* data,
                     const int rows,
                     const int cols,
                     const int new_rows,
                     const int new_cols,
                     float* downsampled) {
  CHECK_NOTNULL(data);
  CHECK_NOTNULL(downsampled);
  CHECK_LE(new_rows, rows);
  CHECK_LE(new_cols, cols);
  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  CHECK_GT(new_rows, 0);
  CHECK_GT(new_cols, 0);

  const float scale_c = static_cast<float>(cols) / static_cast<float>(new_cols);
  const float scale_r = static_cast<float>(rows) / static_cast<float>(new_rows);

  const float kSigmaScale = 0.5f;
  const float sigma_c = std::max(std::numeric_limits<float>::epsilon(),
                                 kSigmaScale * (scale_c - 1));
  const float sigma_r = std::max(std::numeric_limits<float>::epsilon(),
                                 kSigmaScale * (scale_r - 1));

  std::vector<float> smoothed(rows * cols);
  SmoothImage(data, rows, cols, sigma_r, sigma_c, smoothed.data());

  ResampleImageBilinear(
      smoothed.data(), rows, cols, new_rows, new_cols, downsampled);
}

}  // namespace colmap
