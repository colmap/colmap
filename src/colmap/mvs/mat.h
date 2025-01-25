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

#pragma once

#include <string>
#include <vector>

namespace colmap {
namespace mvs {

template <typename T>
class Mat {
 public:
  Mat();
  Mat(size_t width, size_t height, size_t depth);

  size_t GetWidth() const;
  size_t GetHeight() const;
  size_t GetDepth() const;

  size_t GetNumBytes() const;

  T Get(size_t row, size_t col, size_t slice = 0) const;
  void GetSlice(size_t row, size_t col, T* values) const;
  T* GetPtr();
  const T* GetPtr() const;

  const std::vector<T>& GetData() const;

  void Set(size_t row, size_t col, T value);
  void Set(size_t row, size_t col, size_t slice, T value);

  void Fill(T value);

  void Read(const std::string& path);
  void Write(const std::string& path) const;

 protected:
  size_t width_ = 0;
  size_t height_ = 0;
  size_t depth_ = 0;
  std::vector<T> data_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
Mat<T>::Mat() : Mat(0, 0, 0) {}

template <typename T>
Mat<T>::Mat(const size_t width, const size_t height, const size_t depth)
    : width_(width), height_(height), depth_(depth) {
  data_.resize(width_ * height_ * depth_, 0);
}

template <typename T>
size_t Mat<T>::GetWidth() const {
  return width_;
}

template <typename T>
size_t Mat<T>::GetHeight() const {
  return height_;
}

template <typename T>
size_t Mat<T>::GetDepth() const {
  return depth_;
}

template <typename T>
size_t Mat<T>::GetNumBytes() const {
  return data_.size() * sizeof(T);
}

template <typename T>
T Mat<T>::Get(const size_t row, const size_t col, const size_t slice) const {
  return data_.at(slice * width_ * height_ + row * width_ + col);
}

template <typename T>
void Mat<T>::GetSlice(const size_t row, const size_t col, T* values) const {
  for (size_t slice = 0; slice < depth_; ++slice) {
    values[slice] = Get(row, col, slice);
  }
}

template <typename T>
T* Mat<T>::GetPtr() {
  return data_.data();
}

template <typename T>
const T* Mat<T>::GetPtr() const {
  return data_.data();
}

template <typename T>
const std::vector<T>& Mat<T>::GetData() const {
  return data_;
}

template <typename T>
void Mat<T>::Set(const size_t row, const size_t col, const T value) {
  Set(row, col, 0, value);
}

template <typename T>
void Mat<T>::Set(const size_t row,
                 const size_t col,
                 const size_t slice,
                 const T value) {
  data_.at(slice * width_ * height_ + row * width_ + col) = value;
}

template <typename T>
void Mat<T>::Fill(const T value) {
  std::fill(data_.begin(), data_.end(), value);
}

}  // namespace mvs
}  // namespace colmap
