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

#define TEST_NAME "util/endian"
#include "util/testing.h"

#include "util/endian.h"
#include "util/random.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestReverseBytes) {
  for (size_t i = 0; i < 256; ++i) {
    BOOST_CHECK_EQUAL(ReverseBytes<int8_t>(i), static_cast<int8_t>(i));
    BOOST_CHECK_EQUAL(ReverseBytes<uint8_t>(i), static_cast<uint8_t>(i));
  }

  BOOST_CHECK_EQUAL(ReverseBytes<int16_t>(0), 0);
  BOOST_CHECK_EQUAL(ReverseBytes<int16_t>(1), 256);
  BOOST_CHECK_EQUAL(ReverseBytes<int16_t>(2), 512);
  BOOST_CHECK_EQUAL(ReverseBytes<int16_t>(3), 768);
  BOOST_CHECK_EQUAL(ReverseBytes<int16_t>(256), 1);
  BOOST_CHECK_EQUAL(ReverseBytes<int16_t>(512), 2);
  BOOST_CHECK_EQUAL(ReverseBytes<int16_t>(768), 3);

  BOOST_CHECK_EQUAL(ReverseBytes<uint16_t>(0), 0);
  BOOST_CHECK_EQUAL(ReverseBytes<uint16_t>(1), 256);
  BOOST_CHECK_EQUAL(ReverseBytes<uint16_t>(2), 512);
  BOOST_CHECK_EQUAL(ReverseBytes<uint16_t>(3), 768);
  BOOST_CHECK_EQUAL(ReverseBytes<uint16_t>(256), 1);
  BOOST_CHECK_EQUAL(ReverseBytes<uint16_t>(512), 2);
  BOOST_CHECK_EQUAL(ReverseBytes<uint16_t>(768), 3);

  BOOST_CHECK_EQUAL(ReverseBytes<int32_t>(0), 0);
  BOOST_CHECK_EQUAL(ReverseBytes<int32_t>(1), 16777216);
  BOOST_CHECK_EQUAL(ReverseBytes<int32_t>(2), 33554432);
  BOOST_CHECK_EQUAL(ReverseBytes<int32_t>(3), 50331648);
  BOOST_CHECK_EQUAL(ReverseBytes<int32_t>(16777216), 1);
  BOOST_CHECK_EQUAL(ReverseBytes<int32_t>(33554432), 2);
  BOOST_CHECK_EQUAL(ReverseBytes<int32_t>(50331648), 3);

  BOOST_CHECK_EQUAL(ReverseBytes<uint32_t>(0), 0);
  BOOST_CHECK_EQUAL(ReverseBytes<uint32_t>(1), 16777216);
  BOOST_CHECK_EQUAL(ReverseBytes<uint32_t>(2), 33554432);
  BOOST_CHECK_EQUAL(ReverseBytes<uint32_t>(3), 50331648);
  BOOST_CHECK_EQUAL(ReverseBytes<uint32_t>(16777216), 1);
  BOOST_CHECK_EQUAL(ReverseBytes<uint32_t>(33554432), 2);
  BOOST_CHECK_EQUAL(ReverseBytes<uint32_t>(50331648), 3);

  BOOST_CHECK_EQUAL(ReverseBytes<int64_t>(0), 0);
  BOOST_CHECK_EQUAL(ReverseBytes<int64_t>(1), 72057594037927936);
  BOOST_CHECK_EQUAL(ReverseBytes<int64_t>(2), 144115188075855872);
  BOOST_CHECK_EQUAL(ReverseBytes<int64_t>(3), 216172782113783808);
  BOOST_CHECK_EQUAL(ReverseBytes<int64_t>(72057594037927936), 1);
  BOOST_CHECK_EQUAL(ReverseBytes<int64_t>(144115188075855872), 2);
  BOOST_CHECK_EQUAL(ReverseBytes<int64_t>(216172782113783808), 3);

  BOOST_CHECK_EQUAL(ReverseBytes<uint64_t>(0), 0);
  BOOST_CHECK_EQUAL(ReverseBytes<uint64_t>(1), 72057594037927936);
  BOOST_CHECK_EQUAL(ReverseBytes<uint64_t>(2), 144115188075855872);
  BOOST_CHECK_EQUAL(ReverseBytes<uint64_t>(3), 216172782113783808);
  BOOST_CHECK_EQUAL(ReverseBytes<uint64_t>(72057594037927936), 1);
  BOOST_CHECK_EQUAL(ReverseBytes<uint64_t>(144115188075855872), 2);
  BOOST_CHECK_EQUAL(ReverseBytes<uint64_t>(216172782113783808), 3);
}

BOOST_AUTO_TEST_CASE(TestIsLittleBigEndian) {
  BOOST_CHECK_NE(IsLittleEndian(), IsBigEndian());
}

template <typename T>
void TestIntNativeToLitteBigEndian() {
  const T x = RandomInteger<T>(std::numeric_limits<T>::lowest(),
                               std::numeric_limits<T>::max());
  BOOST_CHECK_EQUAL(LittleEndianToNative<T>(NativeToLittleEndian<T>(x)), x);
  BOOST_CHECK_EQUAL(BigEndianToNative<T>(NativeToBigEndian<T>(x)), x);
}

template <typename T>
void TestRealNativeToLitteBigEndian() {
  const T x = RandomReal<T>(std::numeric_limits<T>::lowest(),
                            std::numeric_limits<T>::max());
  BOOST_CHECK_EQUAL(LittleEndianToNative<T>(NativeToLittleEndian<T>(x)), x);
  BOOST_CHECK_EQUAL(BigEndianToNative<T>(NativeToBigEndian<T>(x)), x);
  BOOST_CHECK_EQUAL(NativeToLittleEndian<T>(LittleEndianToNative<T>(x)), x);
  BOOST_CHECK_EQUAL(NativeToBigEndian<T>(BigEndianToNative<T>(x)), x);
}

BOOST_AUTO_TEST_CASE(TestNativeToLitteBigEndian) {
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
  std::stringstream file;
  const T orig_value = RandomInteger<T>(std::numeric_limits<T>::lowest(),
                                        std::numeric_limits<T>::max());
  WriteBinaryLittleEndian<T>(&file, orig_value);
  const T read_value = ReadBinaryLittleEndian<T>(&file);
  BOOST_CHECK_EQUAL(orig_value, read_value);

  std::stringstream file_vector;
  std::vector<T> orig_vector(100);
  std::generate(orig_vector.begin(), orig_vector.end(), []() {
    return RandomInteger<T>(std::numeric_limits<T>::lowest(),
                            std::numeric_limits<T>::max());
  });
  WriteBinaryLittleEndian<T>(&file_vector, orig_vector);
  std::vector<T> read_vector(orig_vector.size());
  ReadBinaryLittleEndian<T>(&file_vector, &read_vector);
  for (size_t i = 0; i < orig_vector.size(); ++i) {
    BOOST_CHECK_EQUAL(orig_vector[i], read_vector[i]);
  }
}

template <typename T>
void TestFloatReadWriteBinaryLittleEndian() {
  std::stringstream file;
  const T orig_value = RandomReal<T>(std::numeric_limits<T>::lowest(),
                                     std::numeric_limits<T>::max());
  WriteBinaryLittleEndian<T>(&file, orig_value);
  const T read_value = ReadBinaryLittleEndian<T>(&file);
  BOOST_CHECK_EQUAL(orig_value, read_value);

  std::stringstream file_vector;
  std::vector<T> orig_vector(100);
  std::generate(orig_vector.begin(), orig_vector.end(), []() {
    return RandomReal<T>(std::numeric_limits<T>::lowest(),
                         std::numeric_limits<T>::max());
  });
  WriteBinaryLittleEndian<T>(&file_vector, orig_vector);
  std::vector<T> read_vector(orig_vector.size());
  ReadBinaryLittleEndian<T>(&file_vector, &read_vector);
  for (size_t i = 0; i < orig_vector.size(); ++i) {
    BOOST_CHECK_EQUAL(orig_vector[i], read_vector[i]);
  }
}

BOOST_AUTO_TEST_CASE(TestReadWriteBinaryLittleEndian) {
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
