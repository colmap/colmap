// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define COLMAP_RNG_HD __host__ __device__
#else
#define COLMAP_RNG_HD
#endif

namespace colmap {
namespace mvs {

// Integer layout only. Caller validates positive dimensions, coordinate bounds,
// and overflow before allocation. Coordinates are in the ORIGINAL image.
// Corners have exactly one slot, shared by adjacent sweep orientations.
struct PatchMatchRngPerimeter {
  const size_t width;
  const size_t height;

  COLMAP_RNG_HD constexpr size_t Count() const {
    if (width == 1) return height;
    if (height == 1) return width;
    return 2 * (width - 1) + 2 * (height - 1);
  }

  // Preconditions: row/col is on the perimeter, within bounds.
  COLMAP_RNG_HD constexpr size_t Index(size_t row, size_t col) const {
    if (width == 1) return row;
    if (height == 1 || row == 0) return col;
    if (col == width - 1) return width + row - 1;
    if (row == height - 1) return 2 * width + height - 3 - col;
    return 2 * width + 2 * height - 4 - row;
  }

  // Inverse mapping for one-thread-per-state perimeter initialization.
  COLMAP_RNG_HD constexpr void Coordinates(size_t index,
                                           size_t* row,
                                           size_t* col) const {
    if (width == 1) {
      *row = index;
      *col = 0;
      return;
    }
    if (height == 1 || index < width) {
      *row = 0;
      *col = index;
      return;
    }
    if (index < width + height - 1) {
      *row = index - width + 1;
      *col = width - 1;
      return;
    }
    if (index < 2 * width + height - 2) {
      *row = height - 1;
      *col = 2 * width + height - 3 - index;
      return;
    }
    *row = 2 * width + 2 * height - 4 - index;
    *col = 0;
  }

  // col is in the CURRENT orientation, rotation is in [0,3] CCW quarter turns.
  // No state permutation or RNG call is needed at rotation boundaries.
  COLMAP_RNG_HD constexpr size_t SweepIndex(unsigned rotation,
                                            size_t col) const {
    switch (rotation) {
      case 0:
        return Index(0, col);
      case 1:
        return Index(col, width - 1);
      case 2:
        return Index(height - 1, width - 1 - col);
      default:
        return Index(height - 1 - col, 0);
    }
  }
};

}  // namespace mvs
}  // namespace colmap
#undef COLMAP_RNG_HD
