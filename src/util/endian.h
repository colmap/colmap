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

#ifndef COLMAP_SRC_UTIL_ENDIAN_H_
#define COLMAP_SRC_UTIL_ENDIAN_H_

#include <algorithm>

namespace colmap {

// Reverse the order of each byte.
template <typename T>
T ReverseBytes(const T& data);

// Check the order in which bytes are stored in computer memory.
bool IsLittleEndian();
bool IsBigEndian();

// Convert data between endianness and the native format. Note that, for float
// and double types, these functions are only valid if the format is IEEE-754.
// This is the case for pretty much most processors.
template <typename T>
T LittleEndianToNative(const T x);
template <typename T>
T BigEndianToNative(const T x);
template <typename T>
T NativeToLittleEndian(const T x);
template <typename T>
T NativeToBigEndian(const T x);

// Read data in little endian format for cross-platform support.
template <typename T>
T ReadBinaryLittleEndian(std::istream* stream);
template <typename T>
void ReadBinaryLittleEndian(std::istream* stream, std::vector<T>* data);

// Write data in little endian format for cross-platform support.
template <typename T>
void WriteBinaryLittleEndian(std::ostream* stream, const T& data);
template <typename T>
void WriteBinaryLittleEndian(std::ostream* stream, const std::vector<T>& data);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
T ReverseBytes(const T& data) {
  T data_reversed = data;
  std::reverse(reinterpret_cast<char*>(&data_reversed),
               reinterpret_cast<char*>(&data_reversed) + sizeof(T));
  return data_reversed;
}

inline bool IsLittleEndian() {
#ifdef BOOST_BIG_ENDIAN
  return false;
#else
  return true;
#endif
}

inline bool IsBigEndian() {
#ifdef BOOST_BIG_ENDIAN
  return true;
#else
  return false;
#endif
}

template <typename T>
T LittleEndianToNative(const T x) {
  if (IsLittleEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}

template <typename T>
T BigEndianToNative(const T x) {
  if (IsBigEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}

template <typename T>
T NativeToLittleEndian(const T x) {
  if (IsLittleEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}

template <typename T>
T NativeToBigEndian(const T x) {
  if (IsBigEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}

template <typename T>
T ReadBinaryLittleEndian(std::istream* stream) {
  T data_little_endian;
  stream->read(reinterpret_cast<char*>(&data_little_endian), sizeof(T));
  return LittleEndianToNative(data_little_endian);
}

template <typename T>
void ReadBinaryLittleEndian(std::istream* stream, std::vector<T>* data) {
  for (size_t i = 0; i < data->size(); ++i) {
    (*data)[i] = ReadBinaryLittleEndian<T>(stream);
  }
}

template <typename T>
void WriteBinaryLittleEndian(std::ostream* stream, const T& data) {
  const T data_little_endian = NativeToLittleEndian(data);
  stream->write(reinterpret_cast<const char*>(&data_little_endian), sizeof(T));
}

template <typename T>
void WriteBinaryLittleEndian(std::ostream* stream, const std::vector<T>& data) {
  for (const auto& elem : data) {
    WriteBinaryLittleEndian<T>(stream, elem);
  }
}

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_ENDIAN_H_
