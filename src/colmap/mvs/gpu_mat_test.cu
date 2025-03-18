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

#ifdef __CUDACC__
#define BOOST_PP_VARIADICS 0
#endif  // __CUDACC__

#include "colmap/math/math.h"
#include "colmap/mvs/gpu_mat.h"
#include "colmap/mvs/gpu_mat_prng.h"

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {

TEST(GpuMat, FillWithVector) {
  GpuMat<float> array(100, 100, 2);
  const std::vector<float> vector = {1.0f, 2.0f};
  array.FillWithVector(vector.data());

  std::vector<float> array_host(100 * 100 * 2, 0.0f);
  array.CopyToHost(array_host.data(), 100 * sizeof(float));

  for (size_t r = 0; r < 100; ++r) {
    for (size_t c = 0; c < 100; ++c) {
      EXPECT_EQ(array_host[0 * 100 * 100 + r * 100 + c], 1.0f);
      EXPECT_EQ(array_host[1 * 100 * 100 + r * 100 + c], 2.0f);
    }
  }
}

template <typename T>
void TestTransposeImage(const size_t width,
                        const size_t height,
                        const size_t depth) {
  GpuMat<T> array(width, height, depth);

  GpuMatPRNG prng_array(width, height);
  array.FillWithRandomNumbers(T(0.0), T(100.0), prng_array);

  GpuMat<T> array_transposed(height, width, depth);
  array.Transpose(&array_transposed);

  std::vector<T> array_host(width * height * depth, T(0.0));
  array.CopyToHost(array_host.data(), width * sizeof(T));

  std::vector<T> array_transposed_host(width * height * depth, 0);
  array_transposed.CopyToHost(array_transposed_host.data(), height * sizeof(T));

  for (size_t r = 0; r < height; ++r) {
    for (size_t c = 0; c < width; ++c) {
      for (size_t d = 0; d < depth; ++d) {
        EXPECT_EQ(array_host[d * width * height + r * width + c],
                  array_transposed_host[d * width * height + c * height + r]);
      }
    }
  }
}

TEST(GpuMat, Transpose) {
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
void TestFlipHorizontalImage(const size_t width,
                             const size_t height,
                             const size_t depth) {
  GpuMat<T> array(width, height, depth);

  GpuMatPRNG prng_array(width, height);
  array.FillWithRandomNumbers(T(0.0), T(100.0), prng_array);

  GpuMat<T> array_flipped(width, height, depth);
  array.FlipHorizontal(&array_flipped);

  std::vector<T> array_host(width * height * depth, T(0.0));
  array.CopyToHost(array_host.data(), width * sizeof(T));

  std::vector<T> array_flipped_host(width * height * depth, 0);
  array_flipped.CopyToHost(array_flipped_host.data(), width * sizeof(T));

  for (size_t r = 0; r < height; ++r) {
    for (size_t c = 0; c < width; ++c) {
      for (size_t d = 0; d < depth; ++d) {
        EXPECT_EQ(
            array_host[d * width * height + r * width + c],
            array_flipped_host[d * width * height + r * width + width - 1 - c]);
      }
    }
  }
}

TEST(GpuMat, FlipHorizontal) {
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
void TestRotateImage(const size_t width,
                     const size_t height,
                     const size_t depth) {
  GpuMat<T> array(width, height, depth);

  GpuMatPRNG prng_array(width, height);
  array.FillWithRandomNumbers(T(0.0), T(100.0), prng_array);

  GpuMat<T> array_rotated(height, width, depth);
  array.Rotate(&array_rotated);

  std::vector<T> array_host(width * height * depth, T(0.0));
  array.CopyToHost(array_host.data(), width * sizeof(T));

  std::vector<T> array_rotated_host(width * height * depth, 0);
  array_rotated.CopyToHost(array_rotated_host.data(), height * sizeof(T));

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
        EXPECT_EQ(
            array_host[d * width * height + r * width + c],
            array_rotated_host[d * width * height + rotr * height + rotc]);
      }
    }
  }
}

TEST(GpuMat, Rotate) {
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

}  // namespace mvs
}  // namespace colmap
