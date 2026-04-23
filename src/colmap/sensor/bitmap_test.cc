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

#include <fstream>
#include <tuple>

#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void WriteImageOIIO(const int width,
                    const int height,
                    const int channels,
                    const std::filesystem::path& path,
                    const uint8_t* data) {
  const OIIO::ImageSpec spec(width, height, channels, OIIO::TypeDesc::UINT8);
  auto output = OIIO::ImageOutput::create(path.string());
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(output->open(path.string(), spec));
  ASSERT_TRUE(output->write_image(OIIO::TypeDesc::UINT8, data));
  ASSERT_TRUE(output->close());
}

TEST(BitmapColor, Empty) {
  BitmapColor<uint8_t> color;
  EXPECT_EQ(color.r, 0);
  EXPECT_EQ(color.g, 0);
  EXPECT_EQ(color.b, 0);
  EXPECT_EQ(color, BitmapColor<uint8_t>(0));
  EXPECT_EQ(color, BitmapColor<uint8_t>(0, 0, 0));
}

TEST(BitmapColor, Gray) {
  BitmapColor<uint8_t> color(5);
  EXPECT_EQ(color.r, 5);
  EXPECT_EQ(color.g, 5);
  EXPECT_EQ(color.b, 5);
}

TEST(BitmapColor, RGB) {
  BitmapColor<uint8_t> color(1, 2, 3);
  EXPECT_EQ(color.r, 1);
  EXPECT_EQ(color.g, 2);
  EXPECT_EQ(color.b, 3);
}

TEST(BitmapColor, Cast) {
  BitmapColor<float> color1(1.1f, 2.9f, -3.0f);
  BitmapColor<uint8_t> color2 = color1.Cast<uint8_t>();
  EXPECT_EQ(color2.r, 1);
  EXPECT_EQ(color2.g, 3);
  EXPECT_EQ(color2.b, 0);
}

TEST(BitmapColor, PrintUint8) {
  BitmapColor<uint8_t> color(1, 2, 3);
  std::ostringstream stream;
  stream << color;
  EXPECT_EQ(stream.str(), "RGB(1, 2, 3)");
}

TEST(BitmapColor, PrintFloat) {
  BitmapColor<float> color(1.3f, 2.4f, 3.5f);
  std::ostringstream stream;
  stream << color;
  EXPECT_EQ(stream.str(), "RGB(1.3, 2.4, 3.5)");
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

TEST(Bitmap, MoveConstructEmpty) {
  Bitmap bitmap;
  Bitmap moved_bitmap(std::move(bitmap));
  EXPECT_EQ(moved_bitmap.Width(), 0);
  EXPECT_EQ(moved_bitmap.Height(), 0);
  EXPECT_EQ(moved_bitmap.Channels(), 0);
  EXPECT_TRUE(moved_bitmap.IsEmpty());
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
  EXPECT_TRUE(bitmap.IsEmpty());
  // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
}

TEST(Bitmap, MoveAssignEmpty) {
  Bitmap bitmap;
  Bitmap moved_bitmap = std::move(bitmap);
  EXPECT_EQ(moved_bitmap.Width(), 0);
  EXPECT_EQ(moved_bitmap.Height(), 0);
  EXPECT_EQ(moved_bitmap.Channels(), 0);
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
  EXPECT_TRUE(bitmap.IsEmpty());
  // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
}

TEST(Bitmap, ConstructCopyEmpty) {
  Bitmap bitmap;
  const Bitmap& copied_bitmap(bitmap);
  EXPECT_EQ(copied_bitmap.Width(), 0);
  EXPECT_EQ(copied_bitmap.Height(), 0);
  EXPECT_EQ(copied_bitmap.Channels(), 0);
}

TEST(Bitmap, ConstructCopy) {
  Bitmap bitmap(2, 1, /*as_rgb=*/true);
  const Bitmap& copied_bitmap(bitmap);
  EXPECT_EQ(copied_bitmap.Width(), 2);
  EXPECT_EQ(copied_bitmap.Height(), 1);
  EXPECT_EQ(copied_bitmap.Channels(), 3);
  EXPECT_EQ(bitmap.Width(), 2);
  EXPECT_EQ(bitmap.Height(), 1);
  EXPECT_EQ(bitmap.Channels(), 3);
}

TEST(Bitmap, AssignCopyEmpty) {
  Bitmap bitmap;
  // Warning: Do not simplify the following line to `Bitmap copied_bitmap =
  // bitmap;` otherwise the compiler will not necessarily generate a copy
  // assignment.
  Bitmap copied_bitmap;
  copied_bitmap = bitmap;
  EXPECT_EQ(copied_bitmap.Width(), 0);
  EXPECT_EQ(copied_bitmap.Height(), 0);
  EXPECT_EQ(copied_bitmap.Channels(), 0);
}

TEST(Bitmap, AssignCopy) {
  Bitmap bitmap(2, 1, /*as_rgb=*/true);
  // Warning: Do not simplify the following line to `Bitmap copied_bitmap =
  // bitmap;` otherwise the compiler will not necessarily generate a copy
  // assignment.
  Bitmap copied_bitmap;
  copied_bitmap = bitmap;
  EXPECT_EQ(copied_bitmap.Width(), 2);
  EXPECT_EQ(copied_bitmap.Height(), 1);
  EXPECT_EQ(copied_bitmap.Channels(), 3);
  EXPECT_EQ(bitmap.Width(), 2);
  EXPECT_EQ(bitmap.Height(), 1);
  EXPECT_EQ(bitmap.Channels(), 3);
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

TEST(Bitmap, Rot90) {
  Bitmap bitmap(10, 5, /*as_rgb=*/false);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(255));
  bitmap.SetPixel(9, 0, BitmapColor<uint8_t>(128));
  bitmap.SetPixel(9, 4, BitmapColor<uint8_t>(64));

  Bitmap rotated1 = bitmap.Clone();
  rotated1.Rot90(1);  // 90 CCW
  EXPECT_EQ(rotated1.Width(), 5);
  EXPECT_EQ(rotated1.Height(), 10);
  BitmapColor<uint8_t> color;
  rotated1.GetPixel(0, 9, &color);
  EXPECT_EQ(color.r, 255);  // Top-left (0,0) -> Bottom-left (0,9)
  rotated1.GetPixel(0, 0, &color);
  EXPECT_EQ(color.r, 128);  // Top-right (9,0) -> Top-left (0,0)
  rotated1.GetPixel(4, 0, &color);
  EXPECT_EQ(color.r, 64);  // Bottom-right (9,4) -> Top-right (4,0)

  Bitmap rotated2 = bitmap.Clone();
  rotated2.Rot90(2);  // 180 CCW
  EXPECT_EQ(rotated2.Width(), 10);
  EXPECT_EQ(rotated2.Height(), 5);
  rotated2.GetPixel(9, 4, &color);
  EXPECT_EQ(color.r, 255);

  Bitmap rotated3 = bitmap.Clone();
  rotated3.Rot90(3);  // 270 CCW
  EXPECT_EQ(rotated3.Width(), 5);
  EXPECT_EQ(rotated3.Height(), 10);
  rotated3.GetPixel(4, 0, &color);
  EXPECT_EQ(color.r, 255);  // Top-left (0,0) -> Top-right (4,0)
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
  const auto test_dir = CreateTestDir();
  const auto filename = test_dir / "bitmap.png";
  EXPECT_TRUE(cloned_bitmap.Write(filename));
  Bitmap read_bitmap;
  EXPECT_TRUE(read_bitmap.Read(filename, /*as_rgb=*/true));
  EXPECT_EQ(read_bitmap.RowMajorData(), cloned_bitmap.RowMajorData());
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
  const auto test_dir = CreateTestDir();
  const auto filename = test_dir / "bitmap.png";
  EXPECT_TRUE(cloned_bitmap.Write(filename));
  Bitmap read_bitmap;
  EXPECT_TRUE(read_bitmap.Read(filename, /*as_rgb=*/false));
  EXPECT_EQ(read_bitmap.RowMajorData(), cloned_bitmap.RowMajorData());
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
  bitmap.SetMetaData("foobar_str", "string");
  EXPECT_EQ(bitmap.GetMetaData("foobar_str").value(), "string");
  EXPECT_FALSE(bitmap.GetMetaData("foobar_str", "int8", &value));
  EXPECT_FALSE(bitmap.GetMetaData("foobar_str", "float", &value));
  EXPECT_FALSE(bitmap.GetMetaData("does_not_exist").has_value());
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

TEST(Bitmap, ExifOrientation) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  EXPECT_FALSE(bitmap.ExifOrientation().has_value());

  int orientation = 6;
  bitmap.SetMetaData("Orientation", "int", &orientation);

  const auto exif_orientation = bitmap.ExifOrientation();
  EXPECT_TRUE(exif_orientation.has_value());
  EXPECT_EQ(exif_orientation.value(), 6);
}

TEST(Bitmap, ExifCameraModel) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  EXPECT_FALSE(bitmap.ExifCameraModel().has_value());

  bitmap.SetMetaData("Make", "make");
  bitmap.SetMetaData("Model", "model");
  const float focal_length_in_35mm_film = 50.f;
  bitmap.SetMetaData(
      "Exif:FocalLengthIn35mmFilm", "float", &focal_length_in_35mm_film);

  const auto camera_model = bitmap.ExifCameraModel();
  EXPECT_TRUE(camera_model.has_value());
  EXPECT_EQ(camera_model.value(), "make-model-50.000000-100x80");
}

TEST(Bitmap, ExifFocalLengthIn35mm) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  EXPECT_FALSE(bitmap.ExifFocalLength().has_value());

  const float focal_length_in_35mm_film = 70.f;
  bitmap.SetMetaData(
      "Exif:FocalLengthIn35mmFilm", "float", &focal_length_in_35mm_film);

  const auto focal_length = bitmap.ExifFocalLength();
  EXPECT_TRUE(focal_length.has_value());
  EXPECT_NEAR(focal_length.value(), 207.17, 0.1);
}

TEST(Bitmap, ExifFocalLengthWithPlane) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  EXPECT_FALSE(bitmap.ExifFocalLength().has_value());

  const float kFocalLengthVal = 72.f;
  bitmap.SetMetaData("Exif:FocalLength", "float", &kFocalLengthVal);
  bitmap.SetMetaData("Make", "canon");
  bitmap.SetMetaData("Model", "eos1dsmarkiii");

  const auto focal_length = bitmap.ExifFocalLength();
  EXPECT_TRUE(focal_length.has_value());
  EXPECT_EQ(focal_length.value(), 200);
}

TEST(Bitmap, ExifFocalLengthWithDatabaseLookup) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  EXPECT_FALSE(bitmap.ExifFocalLength().has_value());

  const float kFocalLengthVal = 120.f;
  bitmap.SetMetaData("Exif:FocalLength", "float", &kFocalLengthVal);
  const int kPixelXDim = 100;
  bitmap.SetMetaData("Exif:PixelXDimension", "int", &kPixelXDim);
  const float kPlaneXRes = 1.f;
  bitmap.SetMetaData("Exif:FocalPlaneXResolution", "float", &kPlaneXRes);
  const int kPlanResUnit = 4;
  bitmap.SetMetaData("Exif:FocalPlaneResolutionUnit", "int", &kPlanResUnit);

  const auto focal_length = bitmap.ExifFocalLength();
  EXPECT_TRUE(focal_length.has_value());
  EXPECT_EQ(focal_length.value(), 120);
}

TEST(Bitmap, ExifFocalLengthUnits) {
  // Initialize a dummy bitmap
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  // Set the base focal length and resolution values
  const float focal_length_mm = 50.0f;
  const float focal_x_res = 100.0f;
  bitmap.SetMetaData("Exif:FocalLength", "float", &focal_length_mm);
  bitmap.SetMetaData("Exif:FocalPlaneXResolution", "float", &focal_x_res);

  // Case 2: Inches (25.4 mm per inch)
  int unit = 2;
  bitmap.SetMetaData("Exif:FocalPlaneResolutionUnit", "int", &unit);
  EXPECT_NEAR(bitmap.ExifFocalLength().value(), 50.0 * (100.0 / 25.4), 1e-4);

  // Case 3: Centimeters (10 mm per cm)
  unit = 3;
  bitmap.SetMetaData("Exif:FocalPlaneResolutionUnit", "int", &unit);
  EXPECT_NEAR(bitmap.ExifFocalLength().value(), 50.0 * (100.0 / 10.0), 1e-4);

  // Case 4: Millimeters (1 mm per mm)
  unit = 4;
  bitmap.SetMetaData("Exif:FocalPlaneResolutionUnit", "int", &unit);
  EXPECT_NEAR(bitmap.ExifFocalLength().value(), 50.0 * (100.0 * 1.0), 1e-4);

  // Case 5: Micrometers (1000 um per mm)
  unit = 5;
  bitmap.SetMetaData("Exif:FocalPlaneResolutionUnit", "int", &unit);
  EXPECT_NEAR(bitmap.ExifFocalLength().value(), 50.0 * (100.0 * 1000.0), 1e-4);
}

TEST(Bitmap, ExifLatitude) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  EXPECT_FALSE(bitmap.ExifLatitude().has_value());

  bitmap.SetMetaData("GPS:LatitudeRef", "N");
  const float kDegMinSec[3] = {46, 30, 900};
  bitmap.SetMetaData("GPS:Latitude", "point", kDegMinSec);

  auto latitude = bitmap.ExifLatitude();
  EXPECT_TRUE(latitude.has_value());
  EXPECT_EQ(latitude.value(), 46.75);

  bitmap.SetMetaData("GPS:LatitudeRef", "S");

  latitude = bitmap.ExifLatitude();
  EXPECT_TRUE(latitude.has_value());
  EXPECT_EQ(latitude.value(), -46.75);
}

TEST(Bitmap, ExifLongitude) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  EXPECT_FALSE(bitmap.ExifLongitude().has_value());

  bitmap.SetMetaData("GPS:LongitudeRef", "W");
  const float kDegMinSec[3] = {92, 30, 900};
  bitmap.SetMetaData("GPS:Longitude", "point", kDegMinSec);

  auto longitude = bitmap.ExifLongitude();
  EXPECT_TRUE(longitude.has_value());
  EXPECT_EQ(longitude.value(), -92.75);

  bitmap.SetMetaData("GPS:LongitudeRef", "E");

  longitude = bitmap.ExifLongitude();
  EXPECT_TRUE(longitude.has_value());
  EXPECT_EQ(longitude.value(), 92.75);
}

TEST(Bitmap, ExifAltitude) {
  Bitmap bitmap(100, 80, /*as_rgb=*/true);

  EXPECT_FALSE(bitmap.ExifAltitude().has_value());

  bitmap.SetMetaData("GPS:AltitudeRef", "0");
  const float kAltitudeVal = 123.456;
  bitmap.SetMetaData("GPS:Altitude", "float", &kAltitudeVal);

  auto altitude = bitmap.ExifAltitude();
  EXPECT_TRUE(altitude.has_value());
  EXPECT_EQ(altitude.value(), kAltitudeVal);

  bitmap.SetMetaData("GPS:AltitudeRef", "1");

  altitude = bitmap.ExifAltitude();
  EXPECT_TRUE(altitude.has_value());
  EXPECT_EQ(altitude.value(), -kAltitudeVal);
}

TEST(Bitmap, ReadWriteAsRGB) {
  Bitmap bitmap(2, 3, /*as_rgb=*/true);
  bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(0, 0, 0));
  bitmap.SetPixel(0, 1, BitmapColor<uint8_t>(1, 0, 0));
  bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(2, 0, 0));
  bitmap.SetPixel(1, 1, BitmapColor<uint8_t>(3, 0, 0));
  bitmap.SetPixel(0, 2, BitmapColor<uint8_t>(4, 2, 0));
  bitmap.SetPixel(1, 2, BitmapColor<uint8_t>(5, 2, 1));

  const auto test_dir = CreateTestDir();
  const auto filename = test_dir / "bitmap.png";

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

  const auto test_dir = CreateTestDir();
  const auto filename = test_dir / "bitmap.png";

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

  const auto test_dir = CreateTestDir();
  const auto filename = test_dir / "bitmap.png";

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

TEST(Bitmap, WriteJpegWithQuality) {
  Bitmap bitmap(20, 30, /*as_rgb=*/true);
  for (int y = 0; y < bitmap.Height(); ++y) {
    for (int x = 0; x < bitmap.Width(); ++x) {
      const uint8_t r = static_cast<uint8_t>((x + y * bitmap.Width()) * 20);
      const uint8_t g = static_cast<uint8_t>((x + y * bitmap.Width()) * 15);
      const uint8_t b = static_cast<uint8_t>((x + y * bitmap.Width()) * 10);
      bitmap.SetPixel(x, y, BitmapColor<uint8_t>(r, g, b));
    }
  }

  const auto test_dir = CreateTestDir();
  const auto filename_default = test_dir / "bitmap_default.jpg";
  const auto filename_100 = test_dir / "bitmap_100.jpg";
  const auto filename_10 = test_dir / "bitmap_10.jpg";

  EXPECT_TRUE(bitmap.Write(filename_default));

  bitmap.SetJpegQuality(100);
  EXPECT_TRUE(bitmap.Write(filename_100));

  bitmap.SetJpegQuality(10);
  EXPECT_TRUE(bitmap.Write(filename_10));

  EXPECT_EQ(std::filesystem::file_size(filename_default),
            std::filesystem::file_size(filename_100));
  EXPECT_LT(std::filesystem::file_size(filename_10),
            std::filesystem::file_size(filename_100));
}

class ParameterizedBitmapFormatTests
    : public ::testing::TestWithParam<
          std::tuple</*extension=*/std::string,
                     /*is_lossless=*/bool,
                     /*supports_native_grey=*/bool,
                     /*supports_rgba=*/bool,
                     /*supports_grey_alpha=*/bool>> {};

TEST_P(ParameterizedBitmapFormatTests, ReadWriteRGB) {
  const auto [kExtension,
              kIsLossless,
              kSupportsNativeGrey,
              kSupportsRGBA,
              kSupportsGreyAlpha] = GetParam();

  const int width = 4;
  const int height = 3;
  Bitmap write_bitmap(width, height, /*as_rgb=*/true);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const uint8_t r = static_cast<uint8_t>((x + y * width) * 20);
      const uint8_t g = static_cast<uint8_t>((x + y * width) * 15);
      const uint8_t b = static_cast<uint8_t>((x + y * width) * 10);
      write_bitmap.SetPixel(x, y, BitmapColor<uint8_t>(r, g, b));
    }
  }

  const auto test_dir = CreateTestDir();
  const auto filename = test_dir / ("image" + kExtension);
  EXPECT_TRUE(write_bitmap.Write(filename));

  Bitmap rgb_bitmap;
  EXPECT_TRUE(rgb_bitmap.Read(filename));
  EXPECT_EQ(rgb_bitmap.Width(), width);
  EXPECT_EQ(rgb_bitmap.Height(), height);
  EXPECT_EQ(rgb_bitmap.Channels(), 3);
  EXPECT_EQ(rgb_bitmap.BitsPerPixel(), 24);

  if (kIsLossless) {
    EXPECT_EQ(rgb_bitmap.RowMajorData(), write_bitmap.RowMajorData());
  }

  Bitmap grey_bitmap;
  EXPECT_TRUE(grey_bitmap.Read(filename, /*as_rgb=*/false));
  EXPECT_EQ(grey_bitmap.Width(), width);
  EXPECT_EQ(grey_bitmap.Height(), height);
  EXPECT_EQ(grey_bitmap.Channels(), 1);
  EXPECT_EQ(grey_bitmap.BitsPerPixel(), 8);
}

TEST_P(ParameterizedBitmapFormatTests, ReadWriteGrey) {
  const auto [kExtension,
              kIsLossless,
              kSupportsNativeGrey,
              kSupportsRGBA,
              kSupportsGreyAlpha] = GetParam();

  const int width = 4;
  const int height = 3;
  const std::vector<uint8_t> grey_values = {
      0, 64, 128, 192, 32, 96, 160, 224, 50, 100, 150, 200};

  Bitmap write_bitmap;
  if (kSupportsNativeGrey) {
    write_bitmap = Bitmap(width, height, /*as_rgb=*/false);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        write_bitmap.SetPixel(
            x, y, BitmapColor<uint8_t>(grey_values[y * width + x]));
      }
    }
  } else {
    // Write RGB with uniform values for formats that don't support grey.
    write_bitmap = Bitmap(width, height, /*as_rgb=*/true);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const uint8_t v = grey_values[y * width + x];
        write_bitmap.SetPixel(x, y, BitmapColor<uint8_t>(v, v, v));
      }
    }
  }

  const auto test_dir = CreateTestDir();
  const auto filename = test_dir / ("image" + kExtension);
  EXPECT_TRUE(write_bitmap.Write(filename));

  Bitmap grey_bitmap;
  EXPECT_TRUE(grey_bitmap.Read(filename, /*as_rgb=*/false));
  EXPECT_EQ(grey_bitmap.Width(), width);
  EXPECT_EQ(grey_bitmap.Height(), height);
  EXPECT_EQ(grey_bitmap.Channels(), 1);
  EXPECT_EQ(grey_bitmap.BitsPerPixel(), 8);

  if (kIsLossless) {
    const std::vector<uint8_t> expected_grey = {
        0, 64, 128, 192, 32, 96, 160, 224, 50, 100, 150, 200};
    EXPECT_EQ(grey_bitmap.RowMajorData(), expected_grey);
  }

  Bitmap rgb_bitmap;
  EXPECT_TRUE(rgb_bitmap.Read(filename, /*as_rgb=*/true));
  EXPECT_EQ(rgb_bitmap.Width(), width);
  EXPECT_EQ(rgb_bitmap.Height(), height);
  EXPECT_EQ(rgb_bitmap.Channels(), 3);
  EXPECT_EQ(rgb_bitmap.BitsPerPixel(), 24);
}

TEST_P(ParameterizedBitmapFormatTests, ReadRGBA) {
  const auto [kExtension,
              kIsLossless,
              kSupportsNativeGrey,
              kSupportsRGBA,
              kSupportsGreyAlpha] = GetParam();

  if (!kSupportsRGBA) {
    return;
  }

  const int width = 2;
  const int height = 3;
  const int channels = 4;
  const std::vector<uint8_t> rgba_data = {0,  0, 0, 255, 2,  0,  0,  255,
                                          10, 0, 0, 128, 30, 0,  0,  200,
                                          40, 2, 0, 255, 5,  20, 10, 100};

  const auto test_dir = CreateTestDir();
  const auto filename = test_dir / ("image" + kExtension);
  WriteImageOIIO(width, height, channels, filename, rgba_data.data());

  Bitmap rgb_bitmap;
  EXPECT_TRUE(rgb_bitmap.Read(filename));
  EXPECT_EQ(rgb_bitmap.Width(), width);
  EXPECT_EQ(rgb_bitmap.Height(), height);
  EXPECT_EQ(rgb_bitmap.Channels(), 3);
  EXPECT_EQ(rgb_bitmap.BitsPerPixel(), 24);

  if (kIsLossless) {
    const std::vector<uint8_t> expected_rgb = {
        0, 0, 0, 2, 0, 0, 10, 0, 0, 30, 0, 0, 40, 2, 0, 5, 20, 10};
    for (size_t i = 0; i < expected_rgb.size(); ++i) {
      // Older OIIO versions seem to have a off-by-one error due to rounding.
      EXPECT_NEAR(rgb_bitmap.RowMajorData()[i], expected_rgb[i], 1);
    }
  }

  Bitmap grey_bitmap;
  EXPECT_TRUE(grey_bitmap.Read(filename, /*as_rgb=*/false));
  EXPECT_EQ(grey_bitmap.Width(), width);
  EXPECT_EQ(grey_bitmap.Height(), height);
  EXPECT_EQ(grey_bitmap.Channels(), 1);
}

TEST_P(ParameterizedBitmapFormatTests, ReadGreyAlpha) {
  const auto [kExtension,
              kIsLossless,
              kSupportsNativeGrey,
              kSupportsRGBA,
              kSupportsGreyAlpha] = GetParam();

  if (!kSupportsGreyAlpha) {
    return;
  }

  const int width = 2;
  const int height = 3;
  const int channels = 2;  // Gray + Alpha
  const std::vector<uint8_t> grey_alpha_data = {
      10, 255, 30, 200, 20, 255, 40, 128, 50, 255, 60, 100};

  const auto test_dir = CreateTestDir();
  const auto filename = test_dir / ("image" + kExtension);
  WriteImageOIIO(width, height, channels, filename, grey_alpha_data.data());

  Bitmap grey_bitmap;
  EXPECT_TRUE(grey_bitmap.Read(filename, /*as_rgb=*/false));
  EXPECT_EQ(grey_bitmap.Width(), width);
  EXPECT_EQ(grey_bitmap.Height(), height);
  EXPECT_EQ(grey_bitmap.Channels(), 1);
  EXPECT_EQ(grey_bitmap.BitsPerPixel(), 8);

  if (kIsLossless) {
    const std::vector<uint8_t> expected_grey = {10, 30, 20, 40, 50, 60};
    for (size_t i = 0; i < expected_grey.size(); ++i) {
      // Older OIIO versions seem to have a off-by-one error due to rounding.
      EXPECT_NEAR(grey_bitmap.RowMajorData()[i], expected_grey[i], 1);
    }
  }

  Bitmap rgb_bitmap;
  EXPECT_TRUE(rgb_bitmap.Read(filename, /*as_rgb=*/true));
  EXPECT_EQ(rgb_bitmap.Width(), width);
  EXPECT_EQ(rgb_bitmap.Height(), height);
  EXPECT_EQ(rgb_bitmap.Channels(), 3);
}

TEST(Bitmap, ReadNonImageFile) {
  const auto test_dir = CreateTestDir();
  const auto filename = test_dir / "not_an_image.txt";

  // Create a non-image file
  std::ofstream file(filename);
  file << "This is not an image file";
  file.close();

  Bitmap bitmap;
  EXPECT_FALSE(bitmap.Read(filename));

  // Verify that OIIO error was cleared.
  const std::string pending_error = OIIO::geterror();
  EXPECT_TRUE(pending_error.empty())
      << "OIIO error was not cleared: " << pending_error;
}

TEST(Bitmap, ReadNonExistentFile) {
  const auto test_dir = CreateTestDir();

  Bitmap bitmap;
  EXPECT_FALSE(bitmap.Read(test_dir / "non_existent_file.png"));

  // Verify that OIIO error was cleared.
  const std::string pending_error = OIIO::geterror();
  EXPECT_TRUE(pending_error.empty())
      << "OIIO error was not cleared: " << pending_error;
}

INSTANTIATE_TEST_SUITE_P(
    BitmapFormatTests,
    ParameterizedBitmapFormatTests,
    ::testing::Values(std::make_tuple(/*extension=*/".jpg",
                                      /*is_lossless=*/false,
                                      /*supports_native_grey=*/true,
                                      /*supports_rgba=*/false,
                                      /*supports_grey_alpha=*/false),
                      std::make_tuple(/*extension=*/".png",
                                      /*is_lossless=*/true,
                                      /*supports_native_grey=*/true,
                                      /*supports_rgba=*/true,
                                      /*supports_grey_alpha=*/true),
                      std::make_tuple(/*extension=*/".tif",
                                      /*is_lossless=*/true,
                                      /*supports_native_grey=*/true,
                                      /*supports_rgba=*/true,
                                      /*supports_grey_alpha=*/true),
                      std::make_tuple(/*extension=*/".bmp",
                                      /*is_lossless=*/true,
                                      /*supports_native_grey=*/true,
                                      /*supports_rgba=*/false,
                                      /*supports_grey_alpha=*/false)));

}  // namespace
}  // namespace colmap
