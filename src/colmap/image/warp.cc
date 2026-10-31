// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/image/warp.h"

#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include "thirdparty/VLFeat/imopv.h"

#include <algorithm>
#include <cstdint>
#include <vector>

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

template <WarpImageOptions::Interpolation interpolation>
std::optional<BitmapColor<uint8_t>> InterpolatePixel(
    const Bitmap& image, const Eigen::Vector2d& point) {
  const double x = point.x() - 0.5;
  const double y = point.y() - 0.5;
  if constexpr (interpolation ==
                WarpImageOptions::Interpolation::kNearestNeighbor) {
    return image.InterpolateNearestNeighbor(x, y);
  } else {
    static_assert(interpolation == WarpImageOptions::Interpolation::kBilinear);
    const std::optional<BitmapColor<float>> color =
        image.InterpolateBilinear(x, y);
    return color ? std::make_optional(color->Cast<uint8_t>()) : std::nullopt;
  }
}

bool ShouldWarpDirectly(const Camera& source_camera,
                        const Camera& target_camera,
                        const WarpImageOptions& options) {
  THROW_CHECK_GE(options.direct_warp_min_scale, 0);
  THROW_CHECK_GT(source_camera.width, 0);
  THROW_CHECK_GT(source_camera.height, 0);
  THROW_CHECK_GT(target_camera.width, 0);
  THROW_CHECK_GT(target_camera.height, 0);
  if (target_camera.width == source_camera.width &&
      target_camera.height == source_camera.height) {
    return true;
  }
  const double scale_x = static_cast<double>(target_camera.width) /
                         static_cast<double>(source_camera.width);
  const double scale_y = static_cast<double>(target_camera.height) /
                         static_cast<double>(source_camera.height);
  return std::min(scale_x, scale_y) >= options.direct_warp_min_scale;
}

// Unproject a scanline to normalized camera coordinates, dispatching on the
// camera model once per row. If `img_points` is null, unprojects pixel centers
// `(x + 0.5, y)`; otherwise unprojects `img_points[x]` where `valid[x]` holds.
void CamFromImgRow(const Camera& camera,
                   const double y,
                   const int width,
                   const Eigen::Vector2d* img_points,
                   uint8_t* valid,
                   Eigen::Vector2d* cam_points) {
  const double* params = camera.params.data();
  switch (camera.model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                   \
  case CameraModel::model_id:                                            \
    if (img_points == nullptr) {                                         \
      for (int x = 0; x < width; ++x) {                                  \
        valid[x] = CameraModel::CamFromImg(                              \
            params, x + 0.5, y, &cam_points[x].x(), &cam_points[x].y()); \
      }                                                                  \
    } else {                                                             \
      for (int x = 0; x < width; ++x) {                                  \
        if (valid[x]) {                                                  \
          valid[x] = CameraModel::CamFromImg(params,                     \
                                             img_points[x].x(),          \
                                             img_points[x].y(),          \
                                             &cam_points[x].x(),         \
                                             &cam_points[x].y());        \
        }                                                                \
      }                                                                  \
    }                                                                    \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

// Project normalized camera coordinates (u, v, 1) along a scanline to image
// coordinates, dispatching on the camera model once per row.
void ImgFromCamRow(const Camera& camera,
                   const int width,
                   const Eigen::Vector2d* cam_points,
                   uint8_t* valid,
                   Eigen::Vector2d* img_points) {
  const double* params = camera.params.data();
  switch (camera.model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                          \
  case CameraModel::model_id:                                   \
    for (int x = 0; x < width; ++x) {                           \
      if (valid[x]) {                                           \
        valid[x] = CameraModel::ImgFromCam(params,              \
                                           cam_points[x].x(),   \
                                           cam_points[x].y(),   \
                                           1.0,                 \
                                           &img_points[x].x(),  \
                                           &img_points[x].y()); \
      }                                                         \
    }                                                           \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

template <WarpImageOptions::Interpolation interpolation>
void WarpBetweenCamerasImpl(const WarpImageOptions& options,
                            const Eigen::Matrix3d* H,
                            const Camera& source_camera,
                            const Camera& target_camera,
                            const Bitmap& source_image,
                            Bitmap* target_image) {
  THROW_CHECK_EQ(source_camera.width, source_image.Width());
  THROW_CHECK_EQ(source_camera.height, source_image.Height());
  THROW_CHECK_NOTNULL(target_image);
  THROW_CHECK_EQ(source_camera.params.size(),
                 CameraModelNumParams(source_camera.model_id));
  THROW_CHECK_EQ(target_camera.params.size(),
                 CameraModelNumParams(target_camera.model_id));

  const bool warp_directly =
      ShouldWarpDirectly(source_camera, target_camera, options);
  Camera warp_target_camera = target_camera;
  Eigen::Matrix3d warp_H;
  if (H != nullptr) {
    warp_H = *H;
    if (!warp_directly) {
      // The intermediate image is at source resolution, but H maps pixels in
      // the target camera's coordinate system.
      const Eigen::Vector3d target_from_intermediate_scale(
          static_cast<double>(target_camera.width) /
              static_cast<double>(source_camera.width),
          static_cast<double>(target_camera.height) /
              static_cast<double>(source_camera.height),
          1.0);
      warp_H = *H * target_from_intermediate_scale.asDiagonal();
    }
  } else if (!warp_directly) {
    warp_target_camera.Rescale(source_camera.width, source_camera.height);
  }
  *target_image = Bitmap(static_cast<int>(warp_directly ? target_camera.width
                                                        : source_camera.width),
                         static_cast<int>(warp_directly ? target_camera.height
                                                        : source_camera.height),
                         source_image.IsRGB());

  const int width = target_image->Width();
  const int height = target_image->Height();
  std::vector<Eigen::Vector2d> img_points(width);
  std::vector<Eigen::Vector2d> cam_points(width);
  std::vector<uint8_t> valid(width);
  for (int y = 0; y < height; ++y) {
    // Camera models assume that the upper left pixel center is (0.5, 0.5).
    const double y_center = y + 0.5;
    if (H != nullptr) {
      for (int x = 0; x < width; ++x) {
        const Eigen::Vector3d warped_point =
            warp_H * Eigen::Vector3d(x + 0.5, y_center, 1.0);
        valid[x] = warped_point.z() != 0;
        if (valid[x]) {
          img_points[x] = warped_point.hnormalized();
        }
      }
    }
    CamFromImgRow(warp_target_camera,
                  y_center,
                  width,
                  H != nullptr ? img_points.data() : nullptr,
                  valid.data(),
                  cam_points.data());
    ImgFromCamRow(source_camera,
                  width,
                  cam_points.data(),
                  valid.data(),
                  img_points.data());
    for (int x = 0; x < width; ++x) {
      if (!valid[x]) {
        continue;
      }
      if (const auto color =
              InterpolatePixel<interpolation>(source_image, img_points[x])) {
        target_image->SetPixel(x, y, *color);
      }
    }
  }

  if (!warp_directly) {
    target_image->Rescale(target_camera.width, target_camera.height);
  }
}

void WarpBetweenCameras(const WarpImageOptions& options,
                        const Eigen::Matrix3d* H,
                        const Camera& source_camera,
                        const Camera& target_camera,
                        const Bitmap& source_image,
                        Bitmap* target_image) {
  switch (options.interpolation) {
    case WarpImageOptions::Interpolation::kNearestNeighbor:
      return WarpBetweenCamerasImpl<
          WarpImageOptions::Interpolation::kNearestNeighbor>(
          options, H, source_camera, target_camera, source_image, target_image);
    case WarpImageOptions::Interpolation::kBilinear:
      return WarpBetweenCamerasImpl<WarpImageOptions::Interpolation::kBilinear>(
          options, H, source_camera, target_camera, source_image, target_image);
  }
  LOG(FATAL_THROW) << "Invalid warp image interpolation mode: "
                   << static_cast<int>(options.interpolation);
}

}  // namespace

void WarpImageBetweenCameras(const WarpImageOptions& options,
                             const Camera& source_camera,
                             const Camera& target_camera,
                             const Bitmap& source_image,
                             Bitmap* target_image) {
  WarpBetweenCameras(options,
                     /*H=*/nullptr,
                     source_camera,
                     target_camera,
                     source_image,
                     target_image);
}

void WarpImageWithHomography(const Eigen::Matrix3d& H,
                             const Bitmap& source_image,
                             Bitmap* target_image) {
  THROW_CHECK_NOTNULL(target_image);
  THROW_CHECK_GT(target_image->Width(), 0);
  THROW_CHECK_GT(target_image->Height(), 0);
  THROW_CHECK_EQ(source_image.IsRGB(), target_image->IsRGB());

  Eigen::Vector3d target_pixel(0, 0, 1);
  for (int y = 0; y < target_image->Height(); ++y) {
    target_pixel.y() = y + 0.5;
    for (int x = 0; x < target_image->Width(); ++x) {
      target_pixel.x() = x + 0.5;

      const Eigen::Vector2d source_pixel = (H * target_pixel).hnormalized();

      if (const auto color = source_image.InterpolateBilinear(
              source_pixel.x() - 0.5, source_pixel.y() - 0.5)) {
        target_image->SetPixel(x, y, color->Cast<uint8_t>());
      } else {
        target_image->SetPixel(x, y, BitmapColor<uint8_t>(0));
      }
    }
  }
}

void WarpImageWithHomographyBetweenCameras(const WarpImageOptions& options,
                                           const Eigen::Matrix3d& H,
                                           const Camera& source_camera,
                                           const Camera& target_camera,
                                           const Bitmap& source_image,
                                           Bitmap* target_image) {
  WarpBetweenCameras(
      options, &H, source_camera, target_camera, source_image, target_image);
}

void ResampleImageBilinear(const float* data,
                           const int rows,
                           const int cols,
                           const int new_rows,
                           const int new_cols,
                           float* resampled) {
  THROW_CHECK_NOTNULL(data);
  THROW_CHECK_NOTNULL(resampled);
  THROW_CHECK_GT(rows, 0);
  THROW_CHECK_GT(cols, 0);
  THROW_CHECK_GT(new_rows, 0);
  THROW_CHECK_GT(new_cols, 0);

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
  THROW_CHECK_NOTNULL(data);
  THROW_CHECK_NOTNULL(smoothed);
  THROW_CHECK_GT(rows, 0);
  THROW_CHECK_GT(cols, 0);
  THROW_CHECK_GT(sigma_r, 0);
  THROW_CHECK_GT(sigma_c, 0);
  vl_imsmooth_f(smoothed, cols, data, cols, rows, cols, sigma_c, sigma_r);
}

void DownsampleImage(const float* data,
                     const int rows,
                     const int cols,
                     const int new_rows,
                     const int new_cols,
                     float* downsampled) {
  THROW_CHECK_NOTNULL(data);
  THROW_CHECK_NOTNULL(downsampled);
  THROW_CHECK_LE(new_rows, rows);
  THROW_CHECK_LE(new_cols, cols);
  THROW_CHECK_GT(rows, 0);
  THROW_CHECK_GT(cols, 0);
  THROW_CHECK_GT(new_rows, 0);
  THROW_CHECK_GT(new_cols, 0);

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
