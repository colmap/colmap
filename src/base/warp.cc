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

#include "base/warp.h"

#include "ext/VLFeat/imopv.h"
#include "util/logging.h"

namespace colmap {
namespace {

float GetPixelConstantBorder(const float* data, const int rows, const int cols,
                             const int row, const int col) {
  if (row >= 0 && col >= 0 && row < rows && col < cols) {
    return data[row * cols + col];
  } else {
    return 0;
  }
}

}  // namespace

void WarpImageBetweenCameras(const Camera& source_camera,
                             const Camera& target_camera,
                             const Bitmap& source_image, Bitmap* target_image) {
  CHECK_EQ(source_camera.Width(), source_image.Width());
  CHECK_EQ(source_camera.Height(), source_image.Height());
  CHECK_NOTNULL(target_image);

  target_image->Allocate(static_cast<int>(target_camera.Width()),
                         static_cast<int>(target_camera.Height()),
                         source_image.IsRGB());

  Eigen::Vector2d image_point;
  for (int y = 0; y < target_image->Height(); ++y) {
    image_point.y() = y + 0.5;
    for (int x = 0; x < target_image->Width(); ++x) {
      image_point.x() = x + 0.5;
      // Camera models assume that the upper left pixel center is (0.5, 0.5).
      const Eigen::Vector2d world_point =
          target_camera.ImageToWorld(image_point);
      const Eigen::Vector2d source_point =
          source_camera.WorldToImage(world_point);

      BitmapColor<float> color;
      if (source_image.InterpolateBilinear(source_point.x() - 0.5,
                                           source_point.y() - 0.5, &color)) {
        target_image->SetPixel(x, y, color.Cast<uint8_t>());
      } else {
        target_image->SetPixel(x, y, BitmapColor<uint8_t>(0, 0, 0));
      }
    }
  }
}

void WarpImageWithHomography(const Eigen::Matrix3d& H,
                             const Bitmap& source_image, Bitmap* target_image) {
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
      if (source_image.InterpolateBilinear(source_pixel.x() - 0.5,
                                           source_pixel.y() - 0.5, &color)) {
        target_image->SetPixel(x, y, color.Cast<uint8_t>());
      } else {
        target_image->SetPixel(x, y, BitmapColor<uint8_t>(0, 0, 0));
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

  target_image->Allocate(static_cast<int>(target_camera.Width()),
                         static_cast<int>(target_camera.Height()),
                         source_image.IsRGB());

  Eigen::Vector3d image_point(0, 0, 1);
  for (int y = 0; y < target_image->Height(); ++y) {
    image_point.y() = y + 0.5;
    for (int x = 0; x < target_image->Width(); ++x) {
      image_point.x() = x + 0.5;

      // Camera models assume that the upper left pixel center is (0.5, 0.5).
      const Eigen::Vector3d warped_point = H * image_point;
      const Eigen::Vector2d world_point =
          target_camera.ImageToWorld(warped_point.hnormalized());
      const Eigen::Vector2d source_point =
          source_camera.WorldToImage(world_point);

      BitmapColor<float> color;
      if (source_image.InterpolateBilinear(source_point.x() - 0.5,
                                           source_point.y() - 0.5, &color)) {
        target_image->SetPixel(x, y, color.Cast<uint8_t>());
      } else {
        target_image->SetPixel(x, y, BitmapColor<uint8_t>(0, 0, 0));
      }
    }
  }
}

void ResampleImageBilinear(const float* data, const int rows, const int cols,
                           const int new_rows, const int new_cols,
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

void SmoothImage(const float* data, const int rows, const int cols,
                 const float sigma_r, const float sigma_c, float* smoothed) {
  CHECK_NOTNULL(data);
  CHECK_NOTNULL(smoothed);
  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  CHECK_GT(sigma_r, 0);
  CHECK_GT(sigma_c, 0);
  vl_imsmooth_f(smoothed, cols, data, cols, rows, cols, sigma_c, sigma_r);
}

void DownsampleImage(const float* data, const int rows, const int cols,
                     const int new_rows, const int new_cols,
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

  const float kSigmaScale = 0.25f;
  const float sigma_c = kSigmaScale * scale_c;
  const float sigma_r = kSigmaScale * scale_r;

  std::vector<float> smoothed(rows * cols);
  SmoothImage(data, rows, cols, sigma_r, sigma_c, smoothed.data());

  ResampleImageBilinear(smoothed.data(), rows, cols, new_rows, new_cols,
                        downsampled);
}

}  // namespace colmap
