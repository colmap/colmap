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

#define TEST_NAME "util/bitmap"
#include "util/testing.h"

#include "util/bitmap.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestBitmapColorEmpty) {
  BitmapColor<uint8_t> color;
  BOOST_CHECK_EQUAL(color.r, 0);
  BOOST_CHECK_EQUAL(color.g, 0);
  BOOST_CHECK_EQUAL(color.b, 0);
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(0));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(0, 0, 0));
}

BOOST_AUTO_TEST_CASE(TestBitmapGrayColor) {
  BitmapColor<uint8_t> color(5);
  BOOST_CHECK_EQUAL(color.r, 5);
  BOOST_CHECK_EQUAL(color.g, 5);
  BOOST_CHECK_EQUAL(color.b, 5);
}

BOOST_AUTO_TEST_CASE(TestBitmapColorCast) {
  BitmapColor<float> color1(1.1, 2.9, -3.0);
  BitmapColor<uint8_t> color2 = color1.Cast<uint8_t>();
  BOOST_CHECK_EQUAL(color2.r, 1);
  BOOST_CHECK_EQUAL(color2.g, 3);
  BOOST_CHECK_EQUAL(color2.b, 0);
}

BOOST_AUTO_TEST_CASE(TestEmpty) {
  Bitmap bitmap;
  BOOST_CHECK_EQUAL(bitmap.Width(), 0);
  BOOST_CHECK_EQUAL(bitmap.Height(), 0);
  BOOST_CHECK_EQUAL(bitmap.Channels(), 0);
  BOOST_CHECK_EQUAL(bitmap.IsRGB(), false);
  BOOST_CHECK_EQUAL(bitmap.IsGrey(), false);
}

BOOST_AUTO_TEST_CASE(TestAllocateRGB) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  BOOST_CHECK_EQUAL(bitmap.Width(), 100);
  BOOST_CHECK_EQUAL(bitmap.Height(), 100);
  BOOST_CHECK_EQUAL(bitmap.Channels(), 3);
  BOOST_CHECK_EQUAL(bitmap.IsRGB(), true);
  BOOST_CHECK_EQUAL(bitmap.IsGrey(), false);
}

BOOST_AUTO_TEST_CASE(TestAllocateGrey) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, false);
  BOOST_CHECK_EQUAL(bitmap.Width(), 100);
  BOOST_CHECK_EQUAL(bitmap.Height(), 100);
  BOOST_CHECK_EQUAL(bitmap.Channels(), 1);
  BOOST_CHECK_EQUAL(bitmap.IsRGB(), false);
  BOOST_CHECK_EQUAL(bitmap.IsGrey(), true);
}

BOOST_AUTO_TEST_CASE(TestDeallocate) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, false);
  bitmap.Deallocate();
  BOOST_CHECK_EQUAL(bitmap.Width(), 0);
  BOOST_CHECK_EQUAL(bitmap.Height(), 0);
  BOOST_CHECK_EQUAL(bitmap.Channels(), 0);
  BOOST_CHECK_EQUAL(bitmap.NumBytes(), 0);
  BOOST_CHECK_EQUAL(bitmap.IsRGB(), false);
  BOOST_CHECK_EQUAL(bitmap.IsGrey(), false);
}

BOOST_AUTO_TEST_CASE(TestBitsPerPixel) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  BOOST_CHECK_EQUAL(bitmap.BitsPerPixel(), 24);
  bitmap.Allocate(100, 100, false);
  BOOST_CHECK_EQUAL(bitmap.BitsPerPixel(), 8);
}

BOOST_AUTO_TEST_CASE(TestNumBytes) {
  Bitmap bitmap;
  BOOST_CHECK_EQUAL(bitmap.NumBytes(), 0);
  bitmap.Allocate(100, 100, true);
  BOOST_CHECK_EQUAL(bitmap.NumBytes(), 3 * 100 * 100);
  bitmap.Allocate(100, 100, false);
  BOOST_CHECK_EQUAL(bitmap.NumBytes(), 100 * 100);
}

BOOST_AUTO_TEST_CASE(TestConvertToRowMajorArrayRGB) {
  Bitmap bitmap;
  bitmap.Allocate(2, 2, true);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 1, BitmapColor<uint8_t>(1, 0, 0));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(2, 0, 0));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(3, 0, 0));
  const std::vector<uint8_t> array = bitmap.ConvertToRowMajorArray();
  BOOST_CHECK_EQUAL(array.size(), 12);
  BOOST_CHECK_EQUAL(array[0], 0);
  BOOST_CHECK_EQUAL(array[1], 0);
  BOOST_CHECK_EQUAL(array[2], 0);
  BOOST_CHECK_EQUAL(array[3], 0);
  BOOST_CHECK_EQUAL(array[4], 0);
  BOOST_CHECK_EQUAL(array[5], 2);
  BOOST_CHECK_EQUAL(array[6], 0);
  BOOST_CHECK_EQUAL(array[7], 0);
  BOOST_CHECK_EQUAL(array[8], 1);
  BOOST_CHECK_EQUAL(array[9], 0);
  BOOST_CHECK_EQUAL(array[10], 0);
  BOOST_CHECK_EQUAL(array[11], 3);
}

BOOST_AUTO_TEST_CASE(TestConvertToRowMajorArrayGrey) {
  Bitmap bitmap;
  bitmap.Allocate(2, 2, false);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 1, BitmapColor<uint8_t>(1, 0, 0));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(2, 0, 0));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(3, 0, 0));
  const std::vector<uint8_t> array = bitmap.ConvertToRowMajorArray();
  BOOST_CHECK_EQUAL(array.size(), 4);
  BOOST_CHECK_EQUAL(array[0], 0);
  BOOST_CHECK_EQUAL(array[1], 2);
  BOOST_CHECK_EQUAL(array[2], 1);
  BOOST_CHECK_EQUAL(array[3], 3);
}

BOOST_AUTO_TEST_CASE(TestConvertToColMajorArrayRGB) {
  Bitmap bitmap;
  bitmap.Allocate(2, 2, true);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 1, BitmapColor<uint8_t>(1, 0, 0));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(2, 0, 0));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(3, 0, 0));
  const std::vector<uint8_t> array = bitmap.ConvertToColMajorArray();
  BOOST_CHECK_EQUAL(array.size(), 12);
  BOOST_CHECK_EQUAL(array[0], 0);
  BOOST_CHECK_EQUAL(array[1], 0);
  BOOST_CHECK_EQUAL(array[2], 0);
  BOOST_CHECK_EQUAL(array[3], 0);
  BOOST_CHECK_EQUAL(array[4], 0);
  BOOST_CHECK_EQUAL(array[5], 0);
  BOOST_CHECK_EQUAL(array[6], 0);
  BOOST_CHECK_EQUAL(array[7], 0);
  BOOST_CHECK_EQUAL(array[8], 0);
  BOOST_CHECK_EQUAL(array[9], 1);
  BOOST_CHECK_EQUAL(array[10], 2);
  BOOST_CHECK_EQUAL(array[11], 3);
}

BOOST_AUTO_TEST_CASE(TestConvertToColMajorArrayGrey) {
  Bitmap bitmap;
  bitmap.Allocate(2, 2, false);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 1, BitmapColor<uint8_t>(1, 0, 0));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(2, 0, 0));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(3, 0, 0));
  const std::vector<uint8_t> array = bitmap.ConvertToColMajorArray();
  BOOST_CHECK_EQUAL(array.size(), 4);
  BOOST_CHECK_EQUAL(array[0], 0);
  BOOST_CHECK_EQUAL(array[1], 1);
  BOOST_CHECK_EQUAL(array[2], 2);
  BOOST_CHECK_EQUAL(array[3], 3);
}

BOOST_AUTO_TEST_CASE(TestGetAndSetPixelRGB) {
  Bitmap bitmap;
  bitmap.Allocate(1, 1, true);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(1, 2, 3));
  BitmapColor<uint8_t> color;
  BOOST_CHECK(bitmap.GetPixel(0, 0, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(1, 2, 3));
}

BOOST_AUTO_TEST_CASE(TestGetAndSetPixelGrey) {
  Bitmap bitmap;
  bitmap.Allocate(1, 1, false);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0, 2, 3));
  BitmapColor<uint8_t> color;
  BOOST_CHECK(bitmap.GetPixel(0, 0, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(1, 2, 3));
  BOOST_CHECK(bitmap.GetPixel(0, 0, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(1, 0, 0));
}

BOOST_AUTO_TEST_CASE(TestGetScanlineRGB) {
  Bitmap bitmap;
  bitmap.Allocate(3, 3, true);
  bitmap.Fill(BitmapColor<uint8_t>(1, 2, 3));
  for (size_t r = 0; r < 3; ++r) {
    const uint8_t* scanline = bitmap.GetScanline(r);
    for (size_t c = 0; c < 3; ++c) {
      BitmapColor<uint8_t> color;
      BOOST_CHECK(bitmap.GetPixel(r, c, &color));
      BOOST_CHECK_EQUAL(scanline[c * 3 + FI_RGBA_RED], color.r);
      BOOST_CHECK_EQUAL(scanline[c * 3 + FI_RGBA_GREEN], color.g);
      BOOST_CHECK_EQUAL(scanline[c * 3 + FI_RGBA_BLUE], color.b);
    }
  }
}

BOOST_AUTO_TEST_CASE(TestGetScanlineGrey) {
  Bitmap bitmap;
  bitmap.Allocate(3, 3, false);
  bitmap.Fill(BitmapColor<uint8_t>(1, 2, 3));
  for (size_t r = 0; r < 3; ++r) {
    const uint8_t* scanline = bitmap.GetScanline(r);
    for (size_t c = 0; c < 3; ++c) {
      BitmapColor<uint8_t> color;
      BOOST_CHECK(bitmap.GetPixel(r, c, &color));
      BOOST_CHECK_EQUAL(scanline[c], color.r);
    }
  }
}

BOOST_AUTO_TEST_CASE(TestFill) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  bitmap.Fill(BitmapColor<uint8_t>(1, 2, 3));
  for (int y = 0; y < bitmap.Height(); ++y) {
    for (int x = 0; x < bitmap.Width(); ++x) {
      BitmapColor<uint8_t> color;
      BOOST_CHECK(bitmap.GetPixel(x, y, &color));
      BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(1, 2, 3));
    }
  }
}

BOOST_AUTO_TEST_CASE(TestInterpolateNearestNeighbor) {
  Bitmap bitmap;
  bitmap.Allocate(11, 11, true);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(5, 5, BitmapColor<uint8_t>(1, 2, 3));
  BitmapColor<uint8_t> color;
  BOOST_CHECK(bitmap.InterpolateNearestNeighbor(5, 5, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(1, 2, 3));
  BOOST_CHECK(bitmap.InterpolateNearestNeighbor(5.4999, 5.4999, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(1, 2, 3));
  BOOST_CHECK(bitmap.InterpolateNearestNeighbor(5.5, 5.5, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(0, 0, 0));
  BOOST_CHECK(bitmap.InterpolateNearestNeighbor(4.5, 5.4999, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<uint8_t>(1, 2, 3));
}

BOOST_AUTO_TEST_CASE(TestInterpolateBilinear) {
  Bitmap bitmap;
  bitmap.Allocate(11, 11, true);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(5, 5, BitmapColor<uint8_t>(1, 2, 3));
  BitmapColor<float> color;
  BOOST_CHECK(bitmap.InterpolateBilinear(5, 5, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<float>(1, 2, 3));
  BOOST_CHECK(bitmap.InterpolateBilinear(5.5, 5, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<float>(0.5, 1, 1.5));
  BOOST_CHECK(bitmap.InterpolateBilinear(5.5, 5.5, &color));
  BOOST_CHECK_EQUAL(color, BitmapColor<float>(0.25, 0.5, 0.75));
}

BOOST_AUTO_TEST_CASE(TestSmoothRGB) {
  Bitmap bitmap;
  bitmap.Allocate(50, 50, true);
  for (int x = 0; x < 50; ++x) {
    for (int y = 0; y < 50; ++y) {
      bitmap.SetPixel(x, y,
                      BitmapColor<uint8_t>(y * 50 + x, y * 50 + x, y * 50 + x));
    }
  }
  bitmap.Smooth(1, 1);
  BOOST_CHECK_EQUAL(bitmap.Width(), 50);
  BOOST_CHECK_EQUAL(bitmap.Height(), 50);
  BOOST_CHECK_EQUAL(bitmap.Channels(), 3);
  for (int x = 0; x < 50; ++x) {
    for (int y = 0; y < 50; ++y) {
      BitmapColor<uint8_t> color;
      BOOST_CHECK(bitmap.GetPixel(x, y, &color));
      BOOST_CHECK_EQUAL(color.r, color.g);
      BOOST_CHECK_EQUAL(color.r, color.b);
    }
  }
}

BOOST_AUTO_TEST_CASE(TestSmoothGrey) {
  Bitmap bitmap;
  bitmap.Allocate(50, 50, false);
  for (int x = 0; x < 50; ++x) {
    for (int y = 0; y < 50; ++y) {
      bitmap.SetPixel(x, y,
                      BitmapColor<uint8_t>(y * 50 + x, y * 50 + x, y * 50 + x));
    }
  }
  bitmap.Smooth(1, 1);
  BOOST_CHECK_EQUAL(bitmap.Width(), 50);
  BOOST_CHECK_EQUAL(bitmap.Height(), 50);
  BOOST_CHECK_EQUAL(bitmap.Channels(), 1);
}

BOOST_AUTO_TEST_CASE(TestRescaleRGB) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  Bitmap bitmap1 = bitmap.Clone();
  bitmap1.Rescale(50, 25);
  BOOST_CHECK_EQUAL(bitmap1.Width(), 50);
  BOOST_CHECK_EQUAL(bitmap1.Height(), 25);
  BOOST_CHECK_EQUAL(bitmap1.Channels(), 3);
  Bitmap bitmap2 = bitmap.Clone();
  bitmap2.Rescale(150, 20);
  BOOST_CHECK_EQUAL(bitmap2.Width(), 150);
  BOOST_CHECK_EQUAL(bitmap2.Height(), 20);
  BOOST_CHECK_EQUAL(bitmap2.Channels(), 3);
}

BOOST_AUTO_TEST_CASE(TestRescaleGrey) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, false);
  Bitmap bitmap1 = bitmap.Clone();
  bitmap1.Rescale(50, 25);
  BOOST_CHECK_EQUAL(bitmap1.Width(), 50);
  BOOST_CHECK_EQUAL(bitmap1.Height(), 25);
  BOOST_CHECK_EQUAL(bitmap1.Channels(), 1);
  Bitmap bitmap2 = bitmap.Clone();
  bitmap2.Rescale(150, 20);
  BOOST_CHECK_EQUAL(bitmap2.Width(), 150);
  BOOST_CHECK_EQUAL(bitmap2.Height(), 20);
  BOOST_CHECK_EQUAL(bitmap2.Channels(), 1);
}

BOOST_AUTO_TEST_CASE(TestClone) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  const Bitmap cloned_bitmap = bitmap.Clone();
  BOOST_CHECK_EQUAL(cloned_bitmap.Width(), 100);
  BOOST_CHECK_EQUAL(cloned_bitmap.Height(), 100);
  BOOST_CHECK_EQUAL(cloned_bitmap.Channels(), 3);
  BOOST_CHECK_NE(bitmap.Data(), cloned_bitmap.Data());
}

BOOST_AUTO_TEST_CASE(TestCloneAsRGB) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, false);
  const Bitmap cloned_bitmap = bitmap.CloneAsRGB();
  BOOST_CHECK_EQUAL(cloned_bitmap.Width(), 100);
  BOOST_CHECK_EQUAL(cloned_bitmap.Height(), 100);
  BOOST_CHECK_EQUAL(cloned_bitmap.Channels(), 3);
  BOOST_CHECK_NE(bitmap.Data(), cloned_bitmap.Data());
}

BOOST_AUTO_TEST_CASE(TestCloneAsGrey) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  const Bitmap cloned_bitmap = bitmap.CloneAsGrey();
  BOOST_CHECK_EQUAL(cloned_bitmap.Width(), 100);
  BOOST_CHECK_EQUAL(cloned_bitmap.Height(), 100);
  BOOST_CHECK_EQUAL(cloned_bitmap.Channels(), 1);
  BOOST_CHECK_NE(bitmap.Data(), cloned_bitmap.Data());
}
