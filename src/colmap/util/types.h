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

#include <functional>
#include <iterator>

#ifdef _MSC_VER
#if _MSC_VER >= 1600
#include <cstdint>
#else
typedef __int8 int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned __int8 uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#endif
#elif __GNUC__ >= 3
#include <cstdint>
#endif

// Define non-copyable or non-movable classes.
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define NON_COPYABLE(class_name)          \
  class_name(class_name const&) = delete; \
  void operator=(class_name const& obj) = delete;
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define NON_MOVABLE(class_name) class_name(class_name&&) = delete;

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/enum_utils.h"

#include <Eigen/Core>

namespace Eigen {

using Matrix3x4f = Matrix<float, 3, 4>;
using Matrix3x4d = Matrix<double, 3, 4>;
using Matrix2x3d = Matrix<double, 2, 3>;
using Matrix6d = Matrix<double, 6, 6>;
using Vector3ub = Matrix<uint8_t, 3, 1>;
using Vector4ub = Matrix<uint8_t, 4, 1>;
using Vector6d = Matrix<double, 6, 1>;
using RowMajorMatrixXf = Matrix<float, Dynamic, Dynamic, RowMajor>;
using RowMajorMatrixXd = Matrix<double, Dynamic, Dynamic, RowMajor>;
using RowMajorMatrixXi = Matrix<int, Dynamic, Dynamic, RowMajor>;

}  // namespace Eigen

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// Index types, determines the maximum number of objects.
////////////////////////////////////////////////////////////////////////////////

// Unique identifier for rigs.
typedef uint32_t rig_t;
constexpr rig_t kInvalidRigId = std::numeric_limits<rig_t>::max();

// Unique identifier for cameras.
typedef uint32_t camera_t;
constexpr camera_t kInvalidCameraId = std::numeric_limits<camera_t>::max();

// Unique identifier for images.
typedef uint32_t image_t;
constexpr image_t kInvalidImageId = std::numeric_limits<image_t>::max();

// Unique identifier for frames.
typedef uint32_t frame_t;
constexpr frame_t kInvalidFrameId = std::numeric_limits<frame_t>::max();

// Each image pair gets a unique ID, see `Database::ImagePairToPairId`.
typedef uint64_t image_pair_t;
constexpr image_pair_t kInvalidImagePairId =
    std::numeric_limits<image_pair_t>::max();

// Index per image, i.e. determines maximum number of 2D points per image.
typedef uint32_t point2D_t;
constexpr point2D_t kInvalidPoint2DIdx = std::numeric_limits<point2D_t>::max();

// Unique identifier per added 3D point. Since we add many 3D points,
// delete them, and possibly re-add them again, the maximum number of allowed
// unique indices should be large.
typedef uint64_t point3D_t;
constexpr point3D_t kInvalidPoint3DId = std::numeric_limits<point3D_t>::max();

// Sensor type.
#ifdef __CUDACC__
enum class SensorType {
  INVALID = -1,
  CAMERA = 0,
  IMU = 1,
};
#else
MAKE_ENUM_CLASS_OVERLOAD_STREAM(SensorType, -1, INVALID, CAMERA, IMU);
#endif

struct sensor_t {
  // Type of the sensor (INVALID / CAMERA / IMU)
  SensorType type;
  // Unique identifier of the sensor.
  // This can be camera_t / imu_t (not supported yet)
  uint32_t id;

  constexpr sensor_t()
      : type(SensorType::INVALID), id(std::numeric_limits<uint32_t>::max()) {}
  constexpr sensor_t(const SensorType& type, uint32_t id)
      : type(type), id(id) {}

  inline bool operator<(const sensor_t& other) const {
    return std::tie(type, id) < std::tie(other.type, other.id);
  }
  inline bool operator==(const sensor_t& other) const {
    return type == other.type && id == other.id;
  }
  inline bool operator!=(const sensor_t& other) const {
    return !(*this == other);
  }
};

constexpr sensor_t kInvalidSensorId =
    sensor_t(SensorType::INVALID, std::numeric_limits<uint32_t>::max());

struct data_t {
  // Unique identifer of the sensor
  sensor_t sensor_id;
  // Unique identifier of the data (measurement)
  // This can be image_t / imu_sample_t (not supported yet)
  uint64_t id;

  constexpr data_t()
      : sensor_id(kInvalidSensorId), id(std::numeric_limits<uint32_t>::max()) {}
  constexpr data_t(const sensor_t& sensor_id, uint32_t id)
      : sensor_id(sensor_id), id(id) {}

  inline bool operator<(const data_t& other) const {
    return std::tie(sensor_id, id) < std::tie(other.sensor_id, other.id);
  }
  inline bool operator==(const data_t& other) const {
    return sensor_id == other.sensor_id && id == other.id;
  }
  inline bool operator!=(const data_t& other) const {
    return !(*this == other);
  }
};

constexpr data_t kInvalidDataId =
    data_t(kInvalidSensorId, std::numeric_limits<uint32_t>::max());

// Simple implementation of C++20's std::span, as Ubuntu 20.04's default GCC
// version does not come with full C++20 and we still want to support it.
template <typename T>
class span {
  T* ptr_;
  const size_t size_;

 public:
  span(T* ptr, size_t len) noexcept : ptr_{ptr}, size_{len} {}

  T& operator[](size_t i) noexcept { return ptr_[i]; }
  T const& operator[](size_t i) const noexcept { return ptr_[i]; }

  size_t size() const noexcept { return size_; }

  T* begin() noexcept { return ptr_; }
  T* end() noexcept { return ptr_ + size_; }
  const T* begin() const noexcept { return ptr_; }
  const T* end() const noexcept { return ptr_ + size_; }
};

// Simple implementation of C++20's std::ranges::filter_view.

template <class Iterator, class Predicate>
struct filter_iterator {
  template <class OtherIterator, class OtherPredicate>
  friend struct filter_iterator;

  typedef
      typename std::iterator_traits<Iterator>::iterator_category base_category;
  typedef typename std::conditional<
      std::is_same<base_category, std::random_access_iterator_tag>::value,
      std::bidirectional_iterator_tag,
      base_category>::type iterator_category;

  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  typedef typename std::iterator_traits<Iterator>::reference reference;
  typedef typename std::iterator_traits<Iterator>::pointer pointer;
  typedef
      typename std::iterator_traits<Iterator>::difference_type difference_type;

  filter_iterator() = default;
  filter_iterator(const Predicate& filter, Iterator it, Iterator end)
      : filter_(filter), it_(std::move(it)), end_(std::move(end)) {
    while (it_ != end_ && !filter_(*it_)) {
      ++it_;
    }
  }

  // Enable conversion from const to non-const iterator and vice versa.
  template <class OtherIterator>
  explicit filter_iterator(
      const filter_iterator<OtherIterator, Predicate>& f,
      typename std::enable_if<
          std::is_convertible<OtherIterator, Iterator>::value>::type* = nullptr)
      : filter_(f.filter_), it_(f.it_), end_(f.end_) {}

  reference operator*() const { return *it_; }
  pointer operator->() { return std::addressof(*it_); }

  filter_iterator& operator++() {
    do {
      ++it_;
    } while (it_ != end_ && !filter_(*it_));
    return *this;
  }

  filter_iterator operator++(int) {
    filter_iterator copy = *this;
    ++it_;
    return copy;
  }

  inline friend bool operator==(const filter_iterator& left,
                                const filter_iterator& right) {
    return left.it_ == right.it_;
  }

  inline friend bool operator!=(const filter_iterator& left,
                                const filter_iterator& right) {
    return left.it_ != right.it_;
  }

 private:
  const Predicate& filter_;
  Iterator it_;
  const Iterator end_;
};

template <class Iterator, class Predicate>
struct filter_view {
 public:
  filter_view(Predicate filter, Iterator beg, Iterator end)
      : filter_(std::move(filter)),
        beg_(filter_, beg, end),
        end_(filter_, end, end) {}

  filter_iterator<Iterator, Predicate> begin() const { return beg_; }
  filter_iterator<Iterator, Predicate> end() const { return end_; }

 private:
  const Predicate filter_;
  const filter_iterator<Iterator, Predicate> beg_;
  const filter_iterator<Iterator, Predicate> end_;
};

}  // namespace colmap

// This file provides specializations of the templated hash function for
// custom types. These are used for comparison in unordered sets/maps.
namespace std {
// Hash function specialization for uint32_t pairs, e.g., image_t or camera_t.
template <>
struct hash<std::pair<uint32_t, uint32_t>> {
  std::size_t operator()(const std::pair<uint32_t, uint32_t>& p) const {
    const uint64_t s = (static_cast<uint64_t>(p.first) << 32) +
                       static_cast<uint64_t>(p.second);
    return std::hash<uint64_t>()(s);
  }
};

template <>
struct hash<colmap::sensor_t> {
  std::size_t operator()(const colmap::sensor_t& s) const noexcept {
    return std::hash<std::pair<uint32_t, uint32_t>>{}(
        std::make_pair(static_cast<uint32_t>(s.type), s.id));
  }
};

template <>
struct hash<colmap::data_t> {
  std::size_t operator()(const colmap::data_t& d) const noexcept {
    const size_t h1 =
        std::hash<uint64_t>{}(std::hash<colmap::sensor_t>{}(d.sensor_id));
    const size_t h2 = std::hash<uint64_t>{}(d.id);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

}  // namespace std
