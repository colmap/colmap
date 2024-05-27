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

#include "colmap/util/endian.h"

#include <random>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ReverseBytes, Nominal) {
  for (size_t i = 0; i < 256; ++i) {
    EXPECT_EQ(ReverseBytes<int8_t>(i), static_cast<int8_t>(i));
    EXPECT_EQ(ReverseBytes<uint8_t>(i), static_cast<uint8_t>(i));
  }

  EXPECT_EQ(ReverseBytes<int16_t>(0), 0);
  EXPECT_EQ(ReverseBytes<int16_t>(1), 256);
  EXPECT_EQ(ReverseBytes<int16_t>(2), 512);
  EXPECT_EQ(ReverseBytes<int16_t>(3), 768);
  EXPECT_EQ(ReverseBytes<int16_t>(256), 1);
  EXPECT_EQ(ReverseBytes<int16_t>(512), 2);
  EXPECT_EQ(ReverseBytes<int16_t>(768), 3);

  EXPECT_EQ(ReverseBytes<uint16_t>(0), 0);
  EXPECT_EQ(ReverseBytes<uint16_t>(1), 256);
  EXPECT_EQ(ReverseBytes<uint16_t>(2), 512);
  EXPECT_EQ(ReverseBytes<uint16_t>(3), 768);
  EXPECT_EQ(ReverseBytes<uint16_t>(256), 1);
  EXPECT_EQ(ReverseBytes<uint16_t>(512), 2);
  EXPECT_EQ(ReverseBytes<uint16_t>(768), 3);

  EXPECT_EQ(ReverseBytes<int32_t>(0), 0);
  EXPECT_EQ(ReverseBytes<int32_t>(1), 16777216);
  EXPECT_EQ(ReverseBytes<int32_t>(2), 33554432);
  EXPECT_EQ(ReverseBytes<int32_t>(3), 50331648);
  EXPECT_EQ(ReverseBytes<int32_t>(16777216), 1);
  EXPECT_EQ(ReverseBytes<int32_t>(33554432), 2);
  EXPECT_EQ(ReverseBytes<int32_t>(50331648), 3);

  EXPECT_EQ(ReverseBytes<uint32_t>(0), 0);
  EXPECT_EQ(ReverseBytes<uint32_t>(1), 16777216);
  EXPECT_EQ(ReverseBytes<uint32_t>(2), 33554432);
  EXPECT_EQ(ReverseBytes<uint32_t>(3), 50331648);
  EXPECT_EQ(ReverseBytes<uint32_t>(16777216), 1);
  EXPECT_EQ(ReverseBytes<uint32_t>(33554432), 2);
  EXPECT_EQ(ReverseBytes<uint32_t>(50331648), 3);

  EXPECT_EQ(ReverseBytes<int64_t>(0), 0);
  EXPECT_EQ(ReverseBytes<int64_t>(1), 72057594037927936);
  EXPECT_EQ(ReverseBytes<int64_t>(2), 144115188075855872);
  EXPECT_EQ(ReverseBytes<int64_t>(3), 216172782113783808);
  EXPECT_EQ(ReverseBytes<int64_t>(72057594037927936), 1);
  EXPECT_EQ(ReverseBytes<int64_t>(144115188075855872), 2);
  EXPECT_EQ(ReverseBytes<int64_t>(216172782113783808), 3);

  EXPECT_EQ(ReverseBytes<uint64_t>(0), 0);
  EXPECT_EQ(ReverseBytes<uint64_t>(1), 72057594037927936);
  EXPECT_EQ(ReverseBytes<uint64_t>(2), 144115188075855872);
  EXPECT_EQ(ReverseBytes<uint64_t>(3), 216172782113783808);
  EXPECT_EQ(ReverseBytes<uint64_t>(72057594037927936), 1);
  EXPECT_EQ(ReverseBytes<uint64_t>(144115188075855872), 2);
  EXPECT_EQ(ReverseBytes<uint64_t>(216172782113783808), 3);
}

TEST(IsLittleBigEndian, Nominal) { EXPECT_NE(IsLittleEndian(), IsBigEndian()); }

template <typename T>
void TestIntNativeToLitteBigEndian() {
  std::default_random_engine prng;
  std::uniform_int_distribution<T> distribution(
      std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
  for (int i = 0; i < 100; ++i) {
    const T x = distribution(prng);
    EXPECT_EQ(LittleEndianToNative<T>(NativeToLittleEndian<T>(x)), x);
    EXPECT_EQ(BigEndianToNative<T>(NativeToBigEndian<T>(x)), x);
  }
}

template <typename T>
void TestRealNativeToLitteBigEndian() {
  std::default_random_engine prng;
  std::uniform_real_distribution<T> distribution(
      std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
  for (int i = 0; i < 100; ++i) {
    const T x = distribution(prng);
    EXPECT_EQ(LittleEndianToNative<T>(NativeToLittleEndian<T>(x)), x);
    EXPECT_EQ(BigEndianToNative<T>(NativeToBigEndian<T>(x)), x);
    EXPECT_EQ(NativeToLittleEndian<T>(LittleEndianToNative<T>(x)), x);
    EXPECT_EQ(NativeToBigEndian<T>(BigEndianToNative<T>(x)), x);
  }
}

TEST(NativeToLitteBigEndian, Nominal) {
#ifndef _MSC_VER  // There is no random number generator in MSVC for char's.
  TestIntNativeToLitteBigEndian<int8_t>();
#endif
  TestIntNativeToLitteBigEndian<int16_t>();
  TestIntNativeToLitteBigEndian<int32_t>();
  TestIntNativeToLitteBigEndian<int64_t>();
#ifndef _MSC_VER  // There is no random number generator in MSVC for char's.
  TestIntNativeToLitteBigEndian<uint8_t>();
#endif
  TestIntNativeToLitteBigEndian<uint16_t>();
  TestIntNativeToLitteBigEndian<uint32_t>();
  TestIntNativeToLitteBigEndian<uint64_t>();
  TestRealNativeToLitteBigEndian<float>();
  TestRealNativeToLitteBigEndian<double>();
}

template <typename T>
void TestIntReadWriteBinaryLittleEndian() {
  std::default_random_engine prng;
  std::uniform_int_distribution<T> distribution(
      std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
  for (int i = 0; i < 100; ++i) {
    std::stringstream file;
    const T orig_value = distribution(prng);
    WriteBinaryLittleEndian<T>(&file, orig_value);
    const T read_value = ReadBinaryLittleEndian<T>(&file);
    EXPECT_EQ(orig_value, read_value);

    std::stringstream file_vector;
    std::vector<T> orig_vector(100);
    std::generate(orig_vector.begin(), orig_vector.end(), [&]() {
      return distribution(prng);
    });
    WriteBinaryLittleEndian<T>(&file_vector, orig_vector);
    std::vector<T> read_vector(orig_vector.size());
    ReadBinaryLittleEndian<T>(&file_vector, &read_vector);
    for (size_t i = 0; i < orig_vector.size(); ++i) {
      EXPECT_EQ(orig_vector[i], read_vector[i]);
    }
  }
}

template <typename T>
void TestFloatReadWriteBinaryLittleEndian() {
  std::default_random_engine prng;
  std::uniform_real_distribution<T> distribution(
      std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
  for (int i = 0; i < 100; ++i) {
    std::stringstream file;
    const T orig_value = distribution(prng);
    WriteBinaryLittleEndian<T>(&file, orig_value);
    const T read_value = ReadBinaryLittleEndian<T>(&file);
    EXPECT_EQ(orig_value, read_value);

    std::stringstream file_vector;
    std::vector<T> orig_vector(100);
    std::generate(orig_vector.begin(), orig_vector.end(), [&]() {
      return distribution(prng);
    });
    WriteBinaryLittleEndian<T>(&file_vector, orig_vector);
    std::vector<T> read_vector(orig_vector.size());
    ReadBinaryLittleEndian<T>(&file_vector, &read_vector);
    for (size_t i = 0; i < orig_vector.size(); ++i) {
      EXPECT_EQ(orig_vector[i], read_vector[i]);
    }
  }
}

TEST(ReadWriteBinaryLittleEndian, Nominal) {
#ifndef _MSC_VER  // There is no random number generator in MSVC for char's.
  TestIntReadWriteBinaryLittleEndian<int8_t>();
#endif
  TestIntReadWriteBinaryLittleEndian<int16_t>();
  TestIntReadWriteBinaryLittleEndian<int32_t>();
  TestIntReadWriteBinaryLittleEndian<int64_t>();
#ifndef _MSC_VER  // There is no random number generator in MSVC for char's.
  TestIntReadWriteBinaryLittleEndian<uint8_t>();
#endif
  TestIntReadWriteBinaryLittleEndian<uint16_t>();
  TestIntReadWriteBinaryLittleEndian<uint32_t>();
  TestIntReadWriteBinaryLittleEndian<uint64_t>();
  TestFloatReadWriteBinaryLittleEndian<float>();
  TestFloatReadWriteBinaryLittleEndian<double>();
}

}  // namespace
}  // namespace colmap
