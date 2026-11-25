// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/image/image_pyramid.h"

#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace colmap {

namespace {

inline int Binomial4Tap(int a, int b, int c, int d) {
  return a + 3 * (b + c) + d;
}

// Horizontal 2x-decimated pass of the [1, 3, 3, 1] binomial filter. Outputs
// unnormalized sums (max 8 * 255 = 2040, fitting uint16_t) with clamp-to-edge
// boundary handling; caller guarantees src_width >= 4 (so dst_width >= 2).
void FilterRow4Tap(const uint8_t* src,
                   const int src_width,
                   uint16_t* dst,
                   const int dst_width) {
  // Peel border pixels so the interior loop is branch-free (left tap -1 clamps
  // to 0; right tap sx + 2 only overflows when src_width is even).
  dst[0] = static_cast<uint16_t>(Binomial4Tap(src[0], src[0], src[1], src[2]));
  for (int x = 1; x < dst_width - 1; ++x) {
    const int sx = 2 * x;
    dst[x] = static_cast<uint16_t>(
        Binomial4Tap(src[sx - 1], src[sx], src[sx + 1], src[sx + 2]));
  }
  const int sx = 2 * (dst_width - 1);
  dst[dst_width - 1] = static_cast<uint16_t>(Binomial4Tap(
      src[sx - 1], src[sx], src[sx + 1], src[std::min(sx + 2, src_width - 1)]));
}

// Implemented directly in fixed-point integer arithmetic instead of delegating
// to Bitmap::Rescale / OpenImageIO::ImageBufAlgo::resize: OIIO's generic float
// triangle resize matches the same 2x [1, 3, 3, 1] / 8 weights but is ~200x
// slower on high-resolution images (~3s vs ~15ms for a 5-level 24MP pyramid).
GrayscaleImageLevel Downsample2xBinomial(const GrayscaleImageLevel& prev) {
  const int prev_width = prev.bitmap.Width();
  const int prev_height = prev.bitmap.Height();
  const int next_width = prev_width / 2;
  const int next_height = prev_height / 2;

  GrayscaleImageLevel next;
  next.bitmap = Bitmap(next_width, next_height, /*as_rgb=*/false);
  // Separable 4-tap binomial filter [1, 3, 3, 1] / 8 is centered halfway
  // between pixels 2*i and 2*i + 1, whose full-res midpoint is
  // (2*i + 1) / prev.scale == (i + 0.5) / (prev.scale * 0.5). Halving scale
  // therefore preserves exact sub-pixel alignment even for odd dimensions.
  next.scale_x = prev.scale_x * 0.5;
  next.scale_y = prev.scale_y * 0.5;

  const uint8_t* prev_data = prev.bitmap.RowMajorData().data();
  uint8_t* next_data = next.bitmap.RowMajorData().data();

  // Stream horizontally filtered rows through a 4-row L1-resident ring buffer
  // instead of allocating a full intermediate image. Each output row y reads
  // clamped source rows in [2*y - 1, 2*y + 2] (at most 4 consecutive rows), so
  // indexing by sy % 4 never overwrites a row still needed by row y or later.
  constexpr int kTaps = 4;
  std::vector<uint16_t> ring(static_cast<size_t>(kTaps) * next_width);
  auto ring_row = [&](int sy) -> uint16_t* {
    return ring.data() + static_cast<size_t>(sy % kTaps) * next_width;
  };

  int next_src_row = 0;
  for (int y = 0; y < next_height; ++y) {
    const int hi = std::min(2 * y + 2, prev_height - 1);
    while (next_src_row <= hi) {
      FilterRow4Tap(prev_data + static_cast<size_t>(next_src_row) * prev_width,
                    prev_width,
                    ring_row(next_src_row),
                    next_width);
      ++next_src_row;
    }

    const uint16_t* r0 = ring_row(std::max(2 * y - 1, 0));
    const uint16_t* r1 = ring_row(2 * y);
    const uint16_t* r2 = ring_row(2 * y + 1);
    const uint16_t* r3 = ring_row(hi);
    uint8_t* dst = next_data + static_cast<size_t>(y) * next_width;
    // Vertical pass: total separable weight is 8 * 8 = 64 (+32 rounds nearest).
    for (int x = 0; x < next_width; ++x) {
      dst[x] = static_cast<uint8_t>(
          (Binomial4Tap(r0[x], r1[x], r2[x], r3[x]) + 32) >> 6);
    }
  }
  return next;
}

}  // namespace

GrayscaleImagePyramid::GrayscaleImagePyramid(const Bitmap& bitmap,
                                             const int num_levels,
                                             const int max_image_size) {
  THROW_CHECK(!bitmap.IsEmpty());
  THROW_CHECK_GE(num_levels, 1);

  levels_.reserve(std::min(num_levels, 32));

  Bitmap grey = bitmap.CloneAsGrey();
  if (max_image_size > 0) {
    grey.Thumbnail(max_image_size);
  }

  GrayscaleImageLevel base;
  base.scale_x =
      static_cast<double>(grey.Width()) / static_cast<double>(bitmap.Width());
  base.scale_y =
      static_cast<double>(grey.Height()) / static_cast<double>(bitmap.Height());
  base.bitmap = std::move(grey);
  levels_.push_back(std::move(base));

  for (int lvl = 1; lvl < num_levels; ++lvl) {
    const GrayscaleImageLevel& prev = levels_.back();
    if (prev.bitmap.Width() < 4 || prev.bitmap.Height() < 4) {
      break;
    }
    levels_.push_back(Downsample2xBinomial(prev));
  }
}

const GrayscaleImageLevel& GrayscaleImagePyramid::Level(const int level) const {
  THROW_CHECK_GE(level, 0);
  THROW_CHECK_LT(level, static_cast<int>(levels_.size()));
  return levels_[level];
}

int GrayscaleImagePyramid::SelectLevelForSpacing(
    const double full_res_spacing) const {
  THROW_CHECK(!levels_.empty());
  const double base_scale = 0.5 * (levels_[0].scale_x + levels_[0].scale_y);
  const double spacing_in_base = full_res_spacing * base_scale;
  if (!(spacing_in_base > 1.0)) {
    return 0;
  }
  if (!std::isfinite(spacing_in_base)) {
    return static_cast<int>(levels_.size()) - 1;
  }
  const int target = static_cast<int>(std::llround(std::log2(spacing_in_base)));
  return std::min(target, static_cast<int>(levels_.size()) - 1);
}

}  // namespace colmap
