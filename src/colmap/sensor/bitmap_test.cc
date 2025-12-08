// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/sensor/bitmap.h"

#include "colmap/util/testing.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Bitmap, BitmapColorEmpty) {
  BitmapColor<uint8_t> color;
  EXPECT_EQ(color.r, 0);
  EXPECT_EQ(color.g, 0);
  EXPECT_EQ(color.b, 0);
  EXPECT_EQ(color, BitmapColor<uint8_t>(0));
  EXPECT_EQ(color, BitmapColor<uint8_t>(0, 0, 0));
}

TEST(Bitmap, BitmapGrayColor) {
  BitmapColor<uint8_t> color(5);
  EXPECT_EQ(color.r, 5);
  EXPECT_EQ(color.g, 5);
  EXPECT_EQ(color.b, 5);
}

TEST(Bitmap, BitmapRGBColor) {
  BitmapColor<uint8_t> color(1, 2, 3);
  EXPECT_EQ(color.r, 1);
  EXPECT_EQ(color.g, 2);
  EXPECT_EQ(color.b, 3);
}

TEST(Bitmap, BitmapColorCast) {
  BitmapColor<float> color1(1.1f, 2.9f, -3.0f);
  BitmapColor<uint8_t> color2 = color1.Cast<uint8_t>();
  EXPECT_EQ(color2.r, 1);
  EXPECT_EQ(color2.g, 3);
  EXPECT_EQ(color2.b, 0);
}

TEST(Bitmap, Empty) {
  Bitmap bitmap;
  EXPECT_EQ(bitmap.Width(), 0);
  EXPECT_EQ(bitmap.Height(), 0);
  EXPECT_EQ(bitmap.Channels(), 0);
  EXPECT_FALSE(bitmap.IsRGB());
  EXPECT_FALSE(bitmap.IsGrey());
  EXPECT_TRUE(bitmap.IsEmpty());
}

TEST(Bitmap, Print) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);
  std::ostringstream stream;
  stream << bitmap;
  EXPECT_EQ(stream.str(), "Bitmap(width=100, height=80, channels=3)");
}

TEST(Bitmap, AllocateRGB) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);
  EXPECT_EQ(bitmap.Width(), 100);
  EXPECT_EQ(bitmap.Height(), 80);
  EXPECT_EQ(bitmap.Channels(), 3);
  EXPECT_TRUE(bitmap.IsRGB());
  EXPECT_FALSE(bitmap.IsGrey());
  EXPECT_FALSE(bitmap.IsEmpty());
}

TEST(Bitmap, AllocateGrey) {
  Bitmap bitmap(100, 80, /*as_rgb=*/false);
  EXPECT_EQ(bitmap.Width(), 100);
  EXPECT_EQ(bitmap.Height(), 80);
  EXPECT_EQ(bitmap.Channels(), 1);
  EXPECT_FALSE(bitmap.IsRGB());
  EXPECT_TRUE(bitmap.IsGrey());
  EXPECT_FALSE(bitmap.IsEmpty());
}

TEST(Bitmap, MoveConstruct) {
  Bitmap bitmap(2, 1, /*as_rgb=*/true);
  Bitmap moved_bitmap(std::move(bitmap));
  EXPECT_EQ(moved_bitmap.Width(), 2);
  EXPECT_EQ(moved_bitmap.Height(), 1);
  EXPECT_EQ(moved_bitmap.Channels(), 3);
  // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(bitmap.Width(), 0);
  EXPECT_EQ(bitmap.Height(), 0);
  EXPECT_EQ(bitmap.Channels(), 0);
  EXPECT_EQ(bitmap.NumBytes(), 0);
  // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
}

TEST(Bitmap, MoveAssign) {
  Bitmap bitmap(2, 1, /*as_rgb=*/true);
  Bitmap moved_bitmap = std::move(bitmap);
  EXPECT_EQ(moved_bitmap.Width(), 2);
  EXPECT_EQ(moved_bitmap.Height(), 1);
  EXPECT_EQ(moved_bitmap.Channels(), 3);
  // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(bitmap.Width(), 0);
  EXPECT_EQ(bitmap.Height(), 0);
  EXPECT_EQ(bitmap.Channels(), 0);
  EXPECT_EQ(bitmap.NumBytes(), 0);
  // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
}

TEST(Bitmap, BitsPerPixel) {
  Bitmap bitmap(1, 1, /*as_rgb=*/true);
  EXPECT_EQ(bitmap.BitsPerPixel(), 24);
  bitmap = Bitmap(1, 1, /*as_rgb=*/false);
  EXPECT_EQ(bitmap.BitsPerPixel(), 8);
}

TEST(Bitmap, NumBytes) {
  Bitmap bitmap;
  EXPECT_EQ(bitmap.NumBytes(), 0);
  bitmap = Bitmap(100, 80, /*as_rgb=*/true);
  EXPECT_EQ(bitmap.NumBytes(), 3 * 100 * 80);
  bitmap = Bitmap(100, 80, /*as_rgb=*/false);
  EXPECT_EQ(bitmap.NumBytes(), 100 * 80);
}

TEST(Bitmap, RowMajorDataRGB) {
  Bitmap bitmap(2, 3, /*as_rgb=*/true);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 1, BitmapColor<uint8_t>(1, 0, 0));
  bitmap.SetPixel(0, 2, BitmapColor<uint8_t>(2, 0, 0));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(3, 0, 0));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(4, 0, 0));
  bitmap.SetPixel(1, 2, BitmapColor<uint8_t>(5, 0, 0));
  EXPECT_THAT(bitmap.RowMajorData(),
              testing::ElementsAre(
                  0, 0, 0, 3, 0, 0, 1, 0, 0, 4, 0, 0, 2, 0, 0, 5, 0, 0));
}

TEST(Bitmap, RowMajorDataGrey) {
  Bitmap bitmap(2, 3, /*as_rgb=*/false);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 1, BitmapColor<uint8_t>(1, 0, 0));
  bitmap.SetPixel(0, 2, BitmapColor<uint8_t>(2, 0, 0));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(3, 0, 0));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(4, 0, 0));
  bitmap.SetPixel(1, 2, BitmapColor<uint8_t>(5, 0, 0));
  EXPECT_THAT(bitmap.RowMajorData(), testing::ElementsAre(0, 3, 1, 4, 2, 5));
}

TEST(Bitmap, GetAndSetPixelRGB) {
  Bitmap bitmap(2, 3, /*as_rgb=*/true);
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(1, 2, 3));
  BitmapColor<uint8_t> color;
  EXPECT_TRUE(bitmap.GetPixel(1, 1, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(1, 2, 3));
}

TEST(Bitmap, GetAndSetPixelGrey) {
  Bitmap bitmap(2, 3, /*as_rgb=*/false);
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(0, 2, 3));
  BitmapColor<uint8_t> color;
  EXPECT_TRUE(bitmap.GetPixel(1, 1, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(1, 2, 3));
  EXPECT_TRUE(bitmap.GetPixel(1, 1, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(1, 1, 1));
}

TEST(Bitmap, Fill) {
  Bitmap bitmap(100, 100, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(1, 2, 3));
  for (int y = 0; y < bitmap.Height(); ++y) {
    for (int x = 0; x < bitmap.Width(); ++x) {
      BitmapColor<uint8_t> color;
      EXPECT_TRUE(bitmap.GetPixel(x, y, &color));
      EXPECT_EQ(color, BitmapColor<uint8_t>(1, 2, 3));
    }
  }
}

TEST(Bitmap, InterpolateNearestNeighbor) {
  Bitmap bitmap(11, 10, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(5, 4, BitmapColor<uint8_t>(1, 2, 3));
  BitmapColor<uint8_t> color;
  EXPECT_TRUE(bitmap.InterpolateNearestNeighbor(5, 4, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(1, 2, 3));
  EXPECT_TRUE(bitmap.InterpolateNearestNeighbor(5.4999, 4.4999, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(1, 2, 3));
  EXPECT_TRUE(bitmap.InterpolateNearestNeighbor(5.5, 4.5, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(0, 0, 0));
  EXPECT_TRUE(bitmap.InterpolateNearestNeighbor(4.5, 4.4999, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(1, 2, 3));
}

TEST(Bitmap, InterpolateBilinear) {
  Bitmap bitmap(11, 10, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(5, 4, BitmapColor<uint8_t>(1, 2, 3));
  BitmapColor<float> color;
  EXPECT_TRUE(bitmap.InterpolateBilinear(5, 4, &color));
  EXPECT_EQ(color, BitmapColor<float>(1, 2, 3));
  EXPECT_TRUE(bitmap.InterpolateBilinear(5.5, 4, &color));
  EXPECT_EQ(color, BitmapColor<float>(0.5, 1, 1.5));
  EXPECT_TRUE(bitmap.InterpolateBilinear(5.5, 4.5, &color));
  EXPECT_EQ(color, BitmapColor<float>(0.25, 0.5, 0.75));
}

TEST(Bitmap, RescaleRGB) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);
  Bitmap bitmap1 = bitmap.Clone();
  bitmap1.Rescale(50, 25);
  EXPECT_EQ(bitmap1.Width(), 50);
  EXPECT_EQ(bitmap1.Height(), 25);
  EXPECT_EQ(bitmap1.Channels(), 3);
  Bitmap bitmap2 = bitmap.Clone();
  bitmap2.Rescale(150, 20);
  EXPECT_EQ(bitmap2.Width(), 150);
  EXPECT_EQ(bitmap2.Height(), 20);
  EXPECT_EQ(bitmap2.Channels(), 3);
}

TEST(Bitmap, RescaleGrey) {
  Bitmap bitmap(100, 80, /*as_rgb=*/false);
  Bitmap bitmap1 = bitmap.Clone();
  bitmap1.Rescale(50, 25);
  EXPECT_EQ(bitmap1.Width(), 50);
  EXPECT_EQ(bitmap1.Height(), 25);
  EXPECT_EQ(bitmap1.Channels(), 1);
  Bitmap bitmap2 = bitmap.Clone();
  bitmap2.Rescale(150, 20);
  EXPECT_EQ(bitmap2.Width(), 150);
  EXPECT_EQ(bitmap2.Height(), 20);
  EXPECT_EQ(bitmap2.Channels(), 1);
}

TEST(Bitmap, Clone) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(10, 20, 30));
  const Bitmap cloned_bitmap = bitmap.Clone();
  EXPECT_EQ(cloned_bitmap.Width(), 100);
  EXPECT_EQ(cloned_bitmap.Height(), 80);
  EXPECT_EQ(cloned_bitmap.Channels(), 3);
  BitmapColor<uint8_t> color;
  EXPECT_TRUE(cloned_bitmap.GetPixel(0, 0, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(10, 20, 30));
}

TEST(Bitmap, CloneAsRGB) {
  Bitmap bitmap(100, 80, /*as_rgb=*/false);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(10, 0, 0));
  const Bitmap cloned_bitmap = bitmap.CloneAsRGB();
  EXPECT_EQ(cloned_bitmap.Width(), 100);
  EXPECT_EQ(cloned_bitmap.Height(), 80);
  EXPECT_EQ(cloned_bitmap.Channels(), 3);
  BitmapColor<uint8_t> color;
  EXPECT_TRUE(cloned_bitmap.GetPixel(0, 0, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(10, 10, 10));
}

TEST(Bitmap, CloneAsGrey) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(10, 20, 30));
  const Bitmap cloned_bitmap = bitmap.CloneAsGrey();
  EXPECT_EQ(cloned_bitmap.Width(), 100);
  EXPECT_EQ(cloned_bitmap.Height(), 80);
  EXPECT_EQ(cloned_bitmap.Channels(), 1);
  BitmapColor<uint8_t> color;
  EXPECT_TRUE(cloned_bitmap.GetPixel(0, 0, &color));
  EXPECT_EQ(color, BitmapColor<uint8_t>(19, 19, 19));
}

TEST(Bitmap, SetGetMetaData) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);
  const float kValue = 1.f;
  bitmap.SetMetaData("foobar", "float", &kValue);
  float value = 0.f;
  EXPECT_TRUE(bitmap.GetMetaData("foobar", "float", &value));
  EXPECT_EQ(value, kValue);
  EXPECT_FALSE(bitmap.GetMetaData("does_not_exist", "float", &value));
  EXPECT_FALSE(bitmap.GetMetaData("foobar", "int8", &value));
}

TEST(Bitmap, CloneMetaData) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);
  const float kValue = 1.f;
  bitmap.SetMetaData("foobar", "float", &kValue);

  Bitmap bitmap2(100, 80, /*as_rgb=*/true);
  float value = 0.f;
  EXPECT_FALSE(bitmap2.GetMetaData("foobar", "float", &value));
  bitmap.CloneMetadata(&bitmap2);
  EXPECT_TRUE(bitmap2.GetMetaData("foobar", "float", &value));
  EXPECT_EQ(value, kValue);
}

TEST(Bitmap, ExifCameraModel) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  std::string camera_model;
  EXPECT_FALSE(bitmap.ExifCameraModel(&camera_model));

  bitmap.SetMetaData("Make", "make");
  bitmap.SetMetaData("Model", "model");
  const float focal_length_in_35mm_film = 50.f;
  bitmap.SetMetaData(
      "Exif:FocalLengthIn35mmFilm", "float", &focal_length_in_35mm_film);

  EXPECT_TRUE(bitmap.ExifCameraModel(&camera_model));
  EXPECT_EQ(camera_model, "make-model-50.000000-100x80");
}

TEST(Bitmap, ExifFocalLengthIn35mm) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  double focal_length = 0.0;
  EXPECT_FALSE(bitmap.ExifFocalLength(&focal_length));

  const float focal_length_in_35mm_film = 70.f;
  bitmap.SetMetaData(
      "Exif:FocalLengthIn35mmFilm", "float", &focal_length_in_35mm_film);

  EXPECT_TRUE(bitmap.ExifFocalLength(&focal_length));
  EXPECT_NEAR(focal_length, 207.17, 0.1);
}

TEST(Bitmap, ExifFocalLengthWithPlane) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  double focal_length = 0.0;
  EXPECT_FALSE(bitmap.ExifFocalLength(&focal_length));

  const float kFocalLengthVal = 72.f;
  bitmap.SetMetaData("Exif:FocalLength", "float", &kFocalLengthVal);
  bitmap.SetMetaData("Make", "canon");
  bitmap.SetMetaData("Model", "eos1dsmarkiii");

  EXPECT_TRUE(bitmap.ExifFocalLength(&focal_length));
  EXPECT_EQ(focal_length, 200);
}

TEST(Bitmap, ExifFocalLengthWithDatabaseLookup) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  double focal_length = 0.0;
  EXPECT_FALSE(bitmap.ExifFocalLength(&focal_length));

  const float kFocalLengthVal = 120.f;
  bitmap.SetMetaData("Exif:FocalLength", "float", &kFocalLengthVal);
  const int kPixelXDim = 100;
  bitmap.SetMetaData("Exif:PixelXDimension", "int", &kPixelXDim);
  const float kPlaneXRes = 1.f;
  bitmap.SetMetaData("Exif:FocalPlaneXResolution", "float", &kPlaneXRes);
  const int kPlanResUnit = 4;
  bitmap.SetMetaData("Exif:FocalPlaneResolutionUnit", "int", &kPlanResUnit);

  EXPECT_TRUE(bitmap.ExifFocalLength(&focal_length));
  EXPECT_EQ(focal_length, 120);
}

TEST(Bitmap, ExifLatitude) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  double latitude = 0.0;
  EXPECT_FALSE(bitmap.ExifLatitude(&latitude));

  bitmap.SetMetaData("GPS:LatitudeRef", "N");
  const float kDegMinSec[3] = {46, 30, 900};
  bitmap.SetMetaData("GPS:Latitude", "point", kDegMinSec);

  EXPECT_TRUE(bitmap.ExifLatitude(&latitude));
  EXPECT_EQ(latitude, 46.75);

  bitmap.SetMetaData("GPS:LatitudeRef", "S");

  EXPECT_TRUE(bitmap.ExifLatitude(&latitude));
  EXPECT_EQ(latitude, -46.75);
}

TEST(Bitmap, ExifLongitude) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  double longitude = 0.0;
  EXPECT_FALSE(bitmap.ExifLongitude(&longitude));

  bitmap.SetMetaData("GPS:LongitudeRef", "W");
  const float kDegMinSec[3] = {92, 30, 900};
  bitmap.SetMetaData("GPS:Longitude", "point", kDegMinSec);

  EXPECT_TRUE(bitmap.ExifLongitude(&longitude));
  EXPECT_EQ(longitude, 92.75);

  bitmap.SetMetaData("GPS:LongitudeRef", "E");

  EXPECT_TRUE(bitmap.ExifLongitude(&longitude));
  EXPECT_EQ(longitude, -92.75);
}

TEST(Bitmap, ExifAltitude) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  double altitude = 0.0;
  EXPECT_FALSE(bitmap.ExifAltitude(&altitude));

  bitmap.SetMetaData("GPS:AltitudeRef", "0");
  const float kAltitudeVal = 123.456;
  bitmap.SetMetaData("GPS:Altitude", "float", &kAltitudeVal);

  EXPECT_TRUE(bitmap.ExifAltitude(&altitude));
  EXPECT_EQ(altitude, kAltitudeVal);

  bitmap.SetMetaData("GPS:AltitudeRef", "1");

  EXPECT_TRUE(bitmap.ExifAltitude(&altitude));
  EXPECT_EQ(altitude, -kAltitudeVal);
}

TEST(Bitmap, ExifGravity) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  double gravity[3] = {0.0, 0.0, 0.0};
  EXPECT_FALSE(bitmap.ExifGravity(gravity));

  // Set acceleration vector in sensor coordinates (x, y, z) in m/s^2
  // Accelerometer measures proper acceleration, so when device is stationary
  // and upright, it measures acceleration upward (opposite to gravity)
  const float kAccelerationVector[3] = {0.5f, -0.3f, 9.8f};
  bitmap.SetMetaData("Exif:AccelerationVector", "point", kAccelerationVector);

  EXPECT_TRUE(bitmap.ExifGravity(gravity));
  // Gravity should be negative of acceleration
  EXPECT_NEAR(gravity[0], -0.5, 1e-6);
  EXPECT_NEAR(gravity[1], 0.3, 1e-6);
  EXPECT_NEAR(gravity[2], -9.8, 1e-6);
}

TEST(Bitmap, ExifGravityFallback) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  double gravity[3] = {0.0, 0.0, 0.0};
  EXPECT_FALSE(bitmap.ExifGravity(gravity));

  // Test fallback to alternative tag name
  const float kAccelerationVector[3] = {1.2f, -2.4f, 8.9f};
  bitmap.SetMetaData("Exif:Acceleration", "point", kAccelerationVector);

  EXPECT_TRUE(bitmap.ExifGravity(gravity));
  EXPECT_NEAR(gravity[0], -1.2, 1e-6);
  EXPECT_NEAR(gravity[1], 2.4, 1e-6);
  EXPECT_NEAR(gravity[2], -8.9, 1e-6);
}

TEST(Bitmap, ReadWriteAsRGB) {
  Bitmap bitmap(2, 3, /*as_rgb=*/true);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 1, BitmapColor<uint8_t>(1, 0, 0));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(2, 0, 0));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(3, 0, 0));
  bitmap.SetPixel(0, 2, BitmapColor<uint8_t>(4, 2, 0));
  bitmap.SetPixel(1, 2, BitmapColor<uint8_t>(5, 2, 1));

  const std::string test_dir = CreateTestDir();
  const std::string filename = test_dir + "/bitmap.png";

  EXPECT_TRUE(bitmap.Write(filename));

  Bitmap read_bitmap;
  EXPECT_TRUE(read_bitmap.Read(filename));
  EXPECT_EQ(read_bitmap.Width(), bitmap.Width());
  EXPECT_EQ(read_bitmap.Height(), bitmap.Height());
  EXPECT_EQ(read_bitmap.Channels(), 3);
  EXPECT_EQ(read_bitmap.BitsPerPixel(), 24);
  EXPECT_EQ(read_bitmap.RowMajorData(), bitmap.RowMajorData());

  EXPECT_TRUE(read_bitmap.Read(filename, /*as_rgb=*/false));
  EXPECT_EQ(read_bitmap.Width(), bitmap.Width());
  EXPECT_EQ(read_bitmap.Height(), bitmap.Height());
  EXPECT_EQ(read_bitmap.Channels(), 1);
  EXPECT_EQ(read_bitmap.BitsPerPixel(), 8);
  EXPECT_EQ(read_bitmap.RowMajorData(), bitmap.CloneAsGrey().RowMajorData());
}

TEST(Bitmap, ReadWriteAsGrey) {
  Bitmap bitmap(2, 3, /*as_rgb=*/false);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0));
  bitmap.SetPixel(0, 1, BitmapColor<uint8_t>(1));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(2));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(3));
  bitmap.SetPixel(0, 2, BitmapColor<uint8_t>(4));
  bitmap.SetPixel(1, 2, BitmapColor<uint8_t>(5));

  const std::string test_dir = CreateTestDir();
  const std::string filename = test_dir + "/bitmap.png";

  EXPECT_TRUE(bitmap.Write(filename));

  Bitmap read_bitmap;
  EXPECT_TRUE(read_bitmap.Read(filename));
  EXPECT_EQ(read_bitmap.Width(), bitmap.Width());
  EXPECT_EQ(read_bitmap.Height(), bitmap.Height());
  EXPECT_EQ(read_bitmap.Channels(), 3);
  EXPECT_EQ(read_bitmap.BitsPerPixel(), 24);
  EXPECT_EQ(read_bitmap.RowMajorData(), bitmap.CloneAsRGB().RowMajorData());

  EXPECT_TRUE(read_bitmap.Read(filename, /*as_rgb=*/false));
  EXPECT_EQ(read_bitmap.Width(), bitmap.Width());
  EXPECT_EQ(read_bitmap.Height(), bitmap.Height());
  EXPECT_EQ(read_bitmap.Channels(), 1);
  EXPECT_EQ(read_bitmap.BitsPerPixel(), 8);
  EXPECT_EQ(read_bitmap.RowMajorData(), bitmap.RowMajorData());
}

TEST(Bitmap, ReadWriteAsGreyNonLinear) {
  Bitmap bitmap(2, 3, /*as_rgb=*/false, /*linear_colorspace=*/false);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0));
  bitmap.SetPixel(0, 1, BitmapColor<uint8_t>(1));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(2));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(3));
  bitmap.SetPixel(0, 2, BitmapColor<uint8_t>(4));
  bitmap.SetPixel(1, 2, BitmapColor<uint8_t>(5));

  const std::string test_dir = CreateTestDir();
  const std::string filename = test_dir + "/bitmap.png";

  EXPECT_TRUE(bitmap.Write(filename, /*delinearize_colorspace=*/false));

  Bitmap read_bitmap;
  EXPECT_TRUE(read_bitmap.Read(
      filename, /*as_rgb=*/false, /*linearize_colorspace=*/false));
  EXPECT_EQ(read_bitmap.Width(), bitmap.Width());
  EXPECT_EQ(read_bitmap.Height(), bitmap.Height());
  EXPECT_EQ(read_bitmap.Channels(), 1);
  EXPECT_EQ(read_bitmap.BitsPerPixel(), 8);
  EXPECT_EQ(read_bitmap.RowMajorData(), bitmap.RowMajorData());
}

}  // namespace
}  // namespace colmap
