// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "util/bitmap"
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <vector>

#include "util/bitmap.h"

using namespace colmap;

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

BOOST_AUTO_TEST_CASE(TestBitsPerPixel) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  BOOST_CHECK_EQUAL(bitmap.BitsPerPixel(), 24);
  bitmap.Allocate(100, 100, false);
  BOOST_CHECK_EQUAL(bitmap.BitsPerPixel(), 8);
}

BOOST_AUTO_TEST_CASE(TestConvertToRowMajorArrayRGB) {
  Bitmap bitmap;
  bitmap.Allocate(2, 2, true);
  bitmap.SetPixel(0, 0, Eigen::Vector3ub(0, 0, 0));
  bitmap.SetPixel(0, 1, Eigen::Vector3ub(1, 0, 0));
  bitmap.SetPixel(1, 0, Eigen::Vector3ub(2, 0, 0));
  bitmap.SetPixel(1, 1, Eigen::Vector3ub(3, 0, 0));
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
  bitmap.SetPixel(0, 0, Eigen::Vector3ub(0, 0, 0));
  bitmap.SetPixel(0, 1, Eigen::Vector3ub(1, 0, 0));
  bitmap.SetPixel(1, 0, Eigen::Vector3ub(2, 0, 0));
  bitmap.SetPixel(1, 1, Eigen::Vector3ub(3, 0, 0));
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
  bitmap.SetPixel(0, 0, Eigen::Vector3ub(0, 0, 0));
  bitmap.SetPixel(0, 1, Eigen::Vector3ub(1, 0, 0));
  bitmap.SetPixel(1, 0, Eigen::Vector3ub(2, 0, 0));
  bitmap.SetPixel(1, 1, Eigen::Vector3ub(3, 0, 0));
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
  bitmap.SetPixel(0, 0, Eigen::Vector3ub(0, 0, 0));
  bitmap.SetPixel(0, 1, Eigen::Vector3ub(1, 0, 0));
  bitmap.SetPixel(1, 0, Eigen::Vector3ub(2, 0, 0));
  bitmap.SetPixel(1, 1, Eigen::Vector3ub(3, 0, 0));
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
  bitmap.SetPixel(0, 0, Eigen::Vector3ub(1, 2, 3));
  Eigen::Vector3ub color;
  BOOST_CHECK(bitmap.GetPixel(0, 0, &color));
  BOOST_CHECK_EQUAL(color, Eigen::Vector3ub(1, 2, 3));
}

BOOST_AUTO_TEST_CASE(TestGetAndSetPixelGrey) {
  Bitmap bitmap;
  bitmap.Allocate(1, 1, false);
  bitmap.SetPixel(0, 0, Eigen::Vector3ub(0, 2, 3));
  Eigen::Vector3ub color;
  BOOST_CHECK(bitmap.GetPixel(0, 0, &color));
  BOOST_CHECK_EQUAL(color, Eigen::Vector3ub(0, 0, 0));
  bitmap.SetPixel(0, 0, Eigen::Vector3ub(1, 2, 3));
  BOOST_CHECK(bitmap.GetPixel(0, 0, &color));
  BOOST_CHECK_EQUAL(color, Eigen::Vector3ub(1, 1, 1));
}

BOOST_AUTO_TEST_CASE(TestFill) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  bitmap.Fill(Eigen::Vector3ub(1, 2, 3));
  for (int y = 0; y < bitmap.Height(); ++y) {
    for (int x = 0; x < bitmap.Width(); ++x) {
      Eigen::Vector3ub color;
      BOOST_CHECK(bitmap.GetPixel(x, y, &color));
      BOOST_CHECK_EQUAL(color, Eigen::Vector3ub(1, 2, 3));
    }
  }
}

BOOST_AUTO_TEST_CASE(TestInterpolateNearestNeighbor) {
  Bitmap bitmap;
  bitmap.Allocate(11, 11, true);
  bitmap.Fill(Eigen::Vector3ub(0, 0, 0));
  bitmap.SetPixel(5, 5, Eigen::Vector3ub(1, 2, 3));
  Eigen::Vector3ub color;
  BOOST_CHECK(bitmap.InterpolateNearestNeighbor(5, 5, &color));
  BOOST_CHECK_EQUAL(color, Eigen::Vector3ub(1, 2, 3));
  BOOST_CHECK(bitmap.InterpolateNearestNeighbor(5.4999, 5.4999, &color));
  BOOST_CHECK_EQUAL(color, Eigen::Vector3ub(1, 2, 3));
  BOOST_CHECK(bitmap.InterpolateNearestNeighbor(5.5, 5.5, &color));
  BOOST_CHECK_EQUAL(color, Eigen::Vector3ub(0, 0, 0));
  BOOST_CHECK(bitmap.InterpolateNearestNeighbor(4.5, 5.4999, &color));
  BOOST_CHECK_EQUAL(color, Eigen::Vector3ub(1, 2, 3));
}

BOOST_AUTO_TEST_CASE(TestInterpolateBilinear) {
  Bitmap bitmap;
  bitmap.Allocate(11, 11, true);
  bitmap.Fill(Eigen::Vector3ub(0, 0, 0));
  bitmap.SetPixel(5, 5, Eigen::Vector3ub(1, 2, 3));
  Eigen::Vector3d color;
  BOOST_CHECK(bitmap.InterpolateBilinear(5, 5, &color));
  BOOST_CHECK_EQUAL(color, Eigen::Vector3d(1, 2, 3));
  BOOST_CHECK(bitmap.InterpolateBilinear(5.5, 5, &color));
  BOOST_CHECK_EQUAL(color, Eigen::Vector3d(0.5, 1, 1.5));
  BOOST_CHECK(bitmap.InterpolateBilinear(5.5, 5.5, &color));
  BOOST_CHECK_EQUAL(color, Eigen::Vector3d(0.25, 0.5, 0.75));
}

BOOST_AUTO_TEST_CASE(TestRescaleRGB) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, true);
  const Bitmap bitmap1 = bitmap.Rescale(50, 25);
  BOOST_CHECK_EQUAL(bitmap1.Width(), 50);
  BOOST_CHECK_EQUAL(bitmap1.Height(), 25);
  BOOST_CHECK_EQUAL(bitmap1.Channels(), 3);
  const Bitmap bitmap2 = bitmap.Rescale(150, 20);
  BOOST_CHECK_EQUAL(bitmap2.Width(), 150);
  BOOST_CHECK_EQUAL(bitmap2.Height(), 20);
  BOOST_CHECK_EQUAL(bitmap2.Channels(), 3);
}

BOOST_AUTO_TEST_CASE(TestRescaleGrey) {
  Bitmap bitmap;
  bitmap.Allocate(100, 100, false);
  const Bitmap bitmap1 = bitmap.Rescale(50, 25);
  BOOST_CHECK_EQUAL(bitmap1.Width(), 50);
  BOOST_CHECK_EQUAL(bitmap1.Height(), 25);
  BOOST_CHECK_EQUAL(bitmap1.Channels(), 1);
  const Bitmap bitmap2 = bitmap.Rescale(150, 20);
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
