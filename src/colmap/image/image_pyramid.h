// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/sensor/bitmap.h"

#include <optional>
#include <vector>

namespace colmap {

// Single level of a grayscale 8-bit image pyramid.
// Coordinates use COLMAP's convention where pixel [ix, iy] is centered at
// (ix + 0.5, iy + 0.5) in level coordinates. A full-resolution COLMAP image
// coordinate (full_x, full_y) maps to continuous level pixel index
//   u = full_x * scale_x - 0.5,  v = full_y * scale_y - 0.5.
struct GrayscaleImageLevel {
  Bitmap bitmap;
  double scale_x = 1.0;
  double scale_y = 1.0;

  // Bilinearly interpolate intensity and optional analytical derivatives w.r.t.
  // full-resolution COLMAP image coordinates (full_x, full_y).
  // Returns std::nullopt if the mapped level index (u, v) lies outside the
  // valid interpolation domain [0, bitmap.Width() - 1) x
  // [0, bitmap.Height() - 1).
  inline std::optional<double> InterpolateBilinear(double full_x,
                                                   double full_y,
                                                   double* dx = nullptr,
                                                   double* dy = nullptr) const;
};

// Multi-scale power-of-two grayscale image pyramid built by separable even-tap
// binomial downsampling (preserving sub-pixel alignment across levels even for
// odd image dimensions).
class GrayscaleImagePyramid {
 public:
  GrayscaleImagePyramid() = default;

  // Build a pyramid with up to `num_levels` levels from a Bitmap (converted to
  // grayscale if RGB). If `max_image_size > 0` and max(width, height) exceeds
  // it, the base level is first rescaled to fit while preserving the mapping
  // from original full-resolution camera coordinates.
  GrayscaleImagePyramid(const Bitmap& bitmap,
                        int num_levels,
                        int max_image_size = -1);

  int NumLevels() const { return static_cast<int>(levels_.size()); }
  bool IsEmpty() const { return levels_.empty(); }

  const GrayscaleImageLevel& Level(int level) const;

  // Select the pyramid level whose pixel spacing best matches
  // `full_res_spacing` (in full-resolution pixels) to minimize aliasing.
  int SelectLevelForSpacing(double full_res_spacing) const;

 private:
  std::vector<GrayscaleImageLevel> levels_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

std::optional<double> GrayscaleImageLevel::InterpolateBilinear(
    const double full_x, const double full_y, double* dx, double* dy) const {
  const std::optional<double> value = bitmap.InterpolateBilinearGrey(
      full_x * scale_x - 0.5, full_y * scale_y - 0.5, dx, dy);
  if (value.has_value()) {
    if (dx != nullptr) {
      *dx *= scale_x;
    }
    if (dy != nullptr) {
      *dy *= scale_y;
    }
  }
  return value;
}

}  // namespace colmap
