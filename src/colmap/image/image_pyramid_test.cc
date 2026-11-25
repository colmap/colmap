// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/image/image_pyramid.h"

#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

namespace colmap {
namespace {

Bitmap MakeLinearRamp(const int width,
                      const int height,
                      const int ax,
                      const int ay,
                      const int c,
                      const bool as_rgb = false) {
  Bitmap bitmap(width, height, as_rgb);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      bitmap.SetPixel(x, y, BitmapColor<uint8_t>(ax * x + ay * y + c));
    }
  }
  return bitmap;
}

void ExpectAnalyticalGradientsMatchFiniteDiff(const GrayscaleImageLevel& level,
                                              const double px,
                                              const double py,
                                              const double dx,
                                              const double dy) {
  constexpr double kEps = 1e-4;
  const auto vp_x = level.InterpolateBilinear(px + kEps, py);
  const auto vm_x = level.InterpolateBilinear(px - kEps, py);
  ASSERT_TRUE(vp_x.has_value() && vm_x.has_value());
  EXPECT_NEAR(dx, (*vp_x - *vm_x) / (2.0 * kEps), 1e-6);
  const auto vp_y = level.InterpolateBilinear(px, py + kEps);
  const auto vm_y = level.InterpolateBilinear(px, py - kEps);
  ASSERT_TRUE(vp_y.has_value() && vm_y.has_value());
  EXPECT_NEAR(dy, (*vp_y - *vm_y) / (2.0 * kEps), 1e-6);
}

TEST(GrayscaleImagePyramid, LinearRampInterpolationAndGradients) {
  // Symmetric 4-tap binomial downsampling preserves linear functions in the
  // interior across even, odd, and anisotropically rescaled RGB pyramids.
  struct Case {
    int width, height, ax, ay, c, num_levels, max_image_size;
    bool as_rgb;
    double px, py;
  };
  for (const Case& tc : {Case{64, 48, 1, 2, 10, 4, -1, false, 21.35, 17.65},
                         Case{35, 27, 4, 2, 5, 3, -1, false, 19.0, 13.0},
                         Case{45, 30, 2, 3, 20, 3, 20, true, 24.0, 16.0}}) {
    const Bitmap bitmap =
        MakeLinearRamp(tc.width, tc.height, tc.ax, tc.ay, tc.c, tc.as_rgb);
    const GrayscaleImagePyramid pyramid(
        bitmap, tc.num_levels, tc.max_image_size);
    ASSERT_EQ(pyramid.NumLevels(), tc.num_levels);

    const int base_w = pyramid.Level(0).bitmap.Width();
    const int base_h = pyramid.Level(0).bitmap.Height();
    const double base_sx =
        static_cast<double>(base_w) / static_cast<double>(tc.width);
    const double base_sy =
        static_cast<double>(base_h) / static_cast<double>(tc.height);
    if (tc.max_image_size > 0) {
      EXPECT_EQ(base_w, 20);
      EXPECT_EQ(base_h, 13);
      EXPECT_NE(base_sx, base_sy);
    }

    for (int lvl = 0; lvl < pyramid.NumLevels(); ++lvl) {
      const auto& level = pyramid.Level(lvl);
      EXPECT_EQ(level.bitmap.Width(), base_w >> lvl);
      EXPECT_EQ(level.bitmap.Height(), base_h >> lvl);
      EXPECT_DOUBLE_EQ(level.scale_x, base_sx / (1 << lvl));
      EXPECT_DOUBLE_EQ(level.scale_y, base_sy / (1 << lvl));

      double dx = 0.0;
      double dy = 0.0;
      const auto val = level.InterpolateBilinear(tc.px, tc.py, &dx, &dy);
      ASSERT_TRUE(val.has_value());
      const double expected =
          tc.ax * (tc.px - 0.5) + tc.ay * (tc.py - 0.5) + tc.c;
      EXPECT_NEAR(*val, expected, 1.5);
      EXPECT_NEAR(dx, tc.ax, 0.25);
      EXPECT_NEAR(dy, tc.ay, 0.25);
      ExpectAnalyticalGradientsMatchFiniteDiff(level, tc.px, tc.py, dx, dy);
    }
  }
}

TEST(GrayscaleImagePyramid, BoundsSelectLevelEarlyStopAndPreconditions) {
  Bitmap bitmap(32, 32, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(100, 100, 100));
  const GrayscaleImagePyramid pyramid(
      bitmap,
      /*num_levels=*/std::numeric_limits<int>::max(),
      /*max_image_size=*/16);
  // 16 -> 8 -> 4 -> 2 (stops once width/height < 4).
  ASSERT_EQ(pyramid.NumLevels(), 4);
  EXPECT_EQ(pyramid.Level(0).bitmap.Width(), 16);
  EXPECT_EQ(pyramid.Level(3).bitmap.Width(), 2);
  EXPECT_DOUBLE_EQ(pyramid.Level(0).scale_x, 0.5);

  EXPECT_FALSE(pyramid.Level(0).InterpolateBilinear(0.1, 10.0).has_value());
  EXPECT_FALSE(pyramid.Level(0).InterpolateBilinear(31.9, 10.0).has_value());
  const auto val = pyramid.Level(0).InterpolateBilinear(16.0, 16.0);
  ASSERT_TRUE(val.has_value());
  EXPECT_DOUBLE_EQ(*val, 100.0);

  // Base scale is 0.5, so 2.0 full-res pixels == 1.0 base pixel -> level 0,
  // and rounding switches at 2 * sqrt(2) ~= 2.828 (2.8 -> 0, 2.9 -> 1).
  const double kInf = std::numeric_limits<double>::infinity();
  const double kNaN = std::numeric_limits<double>::quiet_NaN();
  for (const auto& [spacing, expected_lvl] :
       std::initializer_list<std::pair<double, int>>{{0.0, 0},
                                                     {-2.0, 0},
                                                     {kNaN, 0},
                                                     {-kInf, 0},
                                                     {1.0, 0},
                                                     {2.0, 0},
                                                     {2.8, 0},
                                                     {2.9, 1},
                                                     {4.0, 1},
                                                     {8.0, 2},
                                                     {16.0, 3},
                                                     {1024.0, 3},
                                                     {kInf, 3}}) {
    EXPECT_EQ(pyramid.SelectLevelForSpacing(spacing), expected_lvl);
  }

  // Rec. 709 luminance: round(0.2126*200 + 0.7152*100 + 0.0722*40) == 117.
  Bitmap colored(1, 1, /*as_rgb=*/true);
  colored.Fill(BitmapColor<uint8_t>(200, 100, 40));
  EXPECT_EQ(GrayscaleImagePyramid(colored, 1).Level(0).bitmap.GetPixel(0, 0)->r,
            117);

  // Precondition checks.
  Bitmap empty_bitmap;
  EXPECT_THROW(GrayscaleImagePyramid(empty_bitmap, 1), std::invalid_argument);
  EXPECT_THROW(GrayscaleImagePyramid(bitmap, 0), std::invalid_argument);
  EXPECT_THROW(pyramid.Level(-1), std::invalid_argument);
  EXPECT_THROW(pyramid.Level(pyramid.NumLevels()), std::invalid_argument);
  GrayscaleImagePyramid empty_pyramid;
  EXPECT_TRUE(empty_pyramid.IsEmpty());
  EXPECT_THROW(empty_pyramid.SelectLevelForSpacing(1.0), std::invalid_argument);
}

TEST(GrayscaleImagePyramid, BinomialWeightsAndBorders) {
  // Non-overlapping impulses at interior (4, 4) and clamped corners (0, 0),
  // (15, 15) verify exact 4-tap [1,3,3,1]/8 weights and border clamping.
  Bitmap impulse(16, 16, /*as_rgb=*/false);
  impulse.Fill(BitmapColor<uint8_t>(0));
  impulse.SetPixel(0, 0, BitmapColor<uint8_t>(255));
  impulse.SetPixel(4, 4, BitmapColor<uint8_t>(255));
  impulse.SetPixel(15, 15, BitmapColor<uint8_t>(255));
  const GrayscaleImagePyramid imp_pyr(impulse, /*num_levels=*/2);
  ASSERT_EQ(imp_pyr.NumLevels(), 2);
  const Bitmap& lvl1 = imp_pyr.Level(1).bitmap;
  // Interior (4, 4) contributes weight 3 to output index 2 (taps 3,4,5,6) and
  // weight 1 to output index 1 (taps 1,2,3,4).
  EXPECT_EQ(lvl1.GetPixel(2, 2)->r, static_cast<uint8_t>((255 * 9 + 32) >> 6));
  EXPECT_EQ(lvl1.GetPixel(1, 2)->r, static_cast<uint8_t>((255 * 3 + 32) >> 6));
  EXPECT_EQ(lvl1.GetPixel(2, 1)->r, static_cast<uint8_t>((255 * 3 + 32) >> 6));
  EXPECT_EQ(lvl1.GetPixel(1, 1)->r, static_cast<uint8_t>((255 * 1 + 32) >> 6));
  // Clamped corners (0, 0) and (15, 15) each accumulate weight (1 + 3)^2 = 16.
  EXPECT_EQ(lvl1.GetPixel(0, 0)->r, static_cast<uint8_t>((255 * 16 + 32) >> 6));
  EXPECT_EQ(lvl1.GetPixel(7, 7)->r, static_cast<uint8_t>((255 * 16 + 32) >> 6));

  // Odd-dimension trailing pixel (14, 14) on 15x15 is unclamped (weight 1*1).
  Bitmap odd_impulse(15, 15, /*as_rgb=*/false);
  odd_impulse.Fill(BitmapColor<uint8_t>(0));
  odd_impulse.SetPixel(14, 14, BitmapColor<uint8_t>(255));
  const GrayscaleImagePyramid odd_pyr(odd_impulse, /*num_levels=*/2);
  EXPECT_EQ(odd_pyr.Level(1).bitmap.GetPixel(6, 6)->r,
            static_cast<uint8_t>((255 + 32) >> 6));
}

}  // namespace
}  // namespace colmap
