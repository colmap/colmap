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

#define TEST_NAME "mvs/gpu_mat_test"
#include "util/testing.h"

#include "mvs/gpu_mat.h"
#include "mvs/gpu_mat_prng.h"
#include "util/math.h"

using namespace colmap;
using namespace colmap::mvs;

BOOST_AUTO_TEST_CASE(TestFillWithVector) {
  GpuMat<float> array(100, 100, 2);
  const std::vector<float> vector = {0.0f, 1.0f};
  array.FillWithVector(vector.data());
}

template <typename T>
void TestTransposeImage(const size_t width, const size_t height,
                        const size_t depth) {
  GpuMat<T> array(width, height, depth);

  GpuMatPRNG prng_array(width, height);
  array.FillWithRandomNumbers(T(0.0), T(100.0), prng_array);

  GpuMat<T> array_transposed(height, width, depth);
  array.Transpose(&array_transposed);

  std::vector<T> array_host(width * height * depth, T(0.0));
  array.CopyToDevice(array_host.data(), width * sizeof(T));

  std::vector<T> array_transposed_host(width * height * depth, 0);
  array_transposed.CopyToDevice(array_transposed_host.data(),
                                height * sizeof(T));

  for (size_t r = 0; r < height; ++r) {
    for (size_t c = 0; c < width; ++c) {
      for (size_t d = 0; d < depth; ++d) {
        BOOST_CHECK_EQUAL(
            array_host[d * width * height + r * width + c],
            array_transposed_host[d * width * height + c * height + r]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(TestTranspose) {
  for (size_t w = 1; w <= 5; ++w) {
    for (size_t h = 1; h <= 5; ++h) {
      for (size_t d = 1; d <= 3; ++d) {
        const size_t width = 20 * w;
        const size_t height = 20 * h;
        TestTransposeImage<int8_t>(width, height, d);
        TestTransposeImage<int16_t>(width, height, d);
        TestTransposeImage<int32_t>(width, height, d);
        TestTransposeImage<int64_t>(width, height, d);
        TestTransposeImage<float>(width, height, d);
        TestTransposeImage<double>(width, height, d);
      }
    }
  }
}

template <typename T>
void TestFlipHorizontalImage(const size_t width, const size_t height,
                             const size_t depth) {
  GpuMat<T> array(width, height, depth);

  GpuMatPRNG prng_array(width, height);
  array.FillWithRandomNumbers(T(0.0), T(100.0), prng_array);

  GpuMat<T> array_flipped(width, height, depth);
  array.FlipHorizontal(&array_flipped);

  std::vector<T> array_host(width * height * depth, T(0.0));
  array.CopyToDevice(array_host.data(), width * sizeof(T));

  std::vector<T> array_flipped_host(width * height * depth, 0);
  array_flipped.CopyToDevice(array_flipped_host.data(), width * sizeof(T));

  for (size_t r = 0; r < height; ++r) {
    for (size_t c = 0; c < width; ++c) {
      for (size_t d = 0; d < depth; ++d) {
        BOOST_CHECK_EQUAL(
            array_host[d * width * height + r * width + c],
            array_flipped_host[d * width * height + r * width + width - 1 - c]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(TestFlipHorizontal) {
  for (size_t w = 1; w <= 5; ++w) {
    for (size_t h = 1; h <= 5; ++h) {
      for (size_t d = 1; d <= 3; ++d) {
        const size_t width = 20 * w;
        const size_t height = 20 * h;
        TestFlipHorizontalImage<int8_t>(width, height, d);
        TestFlipHorizontalImage<int16_t>(width, height, d);
        TestFlipHorizontalImage<int32_t>(width, height, d);
        TestFlipHorizontalImage<int64_t>(width, height, d);
        TestFlipHorizontalImage<float>(width, height, d);
        TestFlipHorizontalImage<double>(width, height, d);
      }
    }
  }
}

template <typename T>
void TestRotateImage(const size_t width, const size_t height,
                     const size_t depth) {
  GpuMat<T> array(width, height, depth);

  GpuMatPRNG prng_array(width, height);
  array.FillWithRandomNumbers(T(0.0), T(100.0), prng_array);

  GpuMat<T> array_rotated(height, width, depth);
  array.Rotate(&array_rotated);

  std::vector<T> array_host(width * height * depth, T(0.0));
  array.CopyToDevice(array_host.data(), width * sizeof(T));

  std::vector<T> array_rotated_host(width * height * depth, 0);
  array_rotated.CopyToDevice(array_rotated_host.data(), height * sizeof(T));

  const double arrayCenterH = width / 2.0 - 0.5;
  const double arrayCenterV = height / 2.0 - 0.5;
  const double angle = -M_PI / 2;
  for (size_t r = 0; r < height; ++r) {
    for (size_t c = 0; c < width; ++c) {
      for (size_t d = 0; d < depth; ++d) {
        const size_t rotc =
            std::round(std::cos(angle) * (c - arrayCenterH) -
                       std::sin(angle) * (r - arrayCenterV) + arrayCenterV);
        const size_t rotr =
            std::round(std::sin(angle) * (c - arrayCenterH) +
                       std::cos(angle) * (r - arrayCenterV) + arrayCenterH);
        BOOST_CHECK_EQUAL(
            array_host[d * width * height + r * width + c],
            array_rotated_host[d * width * height + rotr * height + rotc]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(TestRotate) {
  for (size_t w = 1; w <= 5; ++w) {
    for (size_t h = 1; h <= 5; ++h) {
      for (size_t d = 1; d <= 3; ++d) {
        const size_t width = 20 * w;
        const size_t height = 20 * h;
        TestRotateImage<int8_t>(width, height, d);
        TestRotateImage<int16_t>(width, height, d);
        TestRotateImage<int32_t>(width, height, d);
        TestRotateImage<int64_t>(width, height, d);
        TestRotateImage<float>(width, height, d);
        TestRotateImage<double>(width, height, d);
      }
    }
  }
}
