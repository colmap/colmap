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

#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <list>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace colmap {

// Return 1 if number is positive (including 0), -1 if negative.
// Undefined behavior if the value is NaN.
template <typename T>
int SignOfNumber(T val);

// Clamp the given value to a low and maximum value.
template <typename T>
inline T Clamp(const T& value, const T& low, const T& high);

// Convert angle in degree to radians.
inline float DegToRad(float deg);
inline double DegToRad(double deg);

// Convert angle in radians to degree.
inline float RadToDeg(float rad);
inline double RadToDeg(double rad);

// Compute the n-th percentile. Reorders elements in-place.
// Performs linear interpolation between values.
template <typename T>
double Percentile(std::vector<T>& elems, double p);
template <typename T>
double Percentile(std::vector<T>&& elems, double p);

// Determine median value. Reorderes elements in-place.
// Performs linear interpolation between mid values.
template <typename T>
double Median(std::vector<T>& elems);
template <typename T>
double Median(std::vector<T>&& elems);

// Determine mean value in a vector.
template <typename T>
double Mean(const std::vector<T>& elems);

// Determine sample variance in a vector.
template <typename T>
double Variance(const std::vector<T>& elems);

// Determine sample standard deviation in a vector.
template <typename T>
double StdDev(const std::vector<T>& elems);

// Generate N-choose-K combinations.
//
// Note that elements in range [first, last) must be in sorted order,
// according to `std::less`.
template <class Iterator>
bool NextCombination(Iterator first, Iterator middle, Iterator last);

// Sigmoid function.
template <typename T>
T Sigmoid(T x, T alpha = 1);

// Scale values according to sigmoid transform.
//
//   x \in [0, 1] -> x \in [-x0, x0] -> sigmoid(x, alpha) -> x \in [0, 1]
//
// @param x        Value to be scaled in the range [0, 1].
// @param x0       Spread that determines the range x is scaled to.
// @param alpha    Exponential sigmoid factor.
//
// @return         The scaled value in the range [0, 1].
template <typename T>
T ScaleSigmoid(T x, T alpha = 1, T x0 = 10);

// Binomial coefficient or all combinations, defined as n! / ((n - k)! k!).
uint64_t NChooseK(uint64_t n, uint64_t k);

// Cast value from one type to another and truncate instead of overflow, if the
// input value is out of range of the output data type.
template <typename T1, typename T2>
T2 TruncateCast(T1 value);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

namespace internal {

template <class Iterator>
bool NextCombination(Iterator first1,
                     Iterator last1,
                     Iterator first2,
                     Iterator last2) {
  if ((first1 == last1) || (first2 == last2)) {
    return false;
  }
  Iterator m1 = last1;
  Iterator m2 = last2;
  --m1;
  --m2;
  while (m1 != first1 && *m1 >= *m2) {
    --m1;
  }
  bool result = (m1 == first1) && *first1 >= *m2;
  if (!result) {
    while (first2 != m2 && *m1 >= *first2) {
      ++first2;
    }
    first1 = m1;
    std::iter_swap(first1, first2);
    ++first1;
    ++first2;
  }
  if ((first1 != last1) && (first2 != last2)) {
    m1 = last1;
    m2 = first2;
    while ((m1 != first1) && (m2 != last2)) {
      std::iter_swap(--m1, m2);
      ++m2;
    }
    std::reverse(first1, m1);
    std::reverse(first1, last1);
    std::reverse(m2, last2);
    std::reverse(first2, last2);
  }
  return !result;
}

}  // namespace internal

template <typename T>
int SignOfNumber(const T val) {
  return val >= 0 ? 1 : -1;
}

template <typename T>
T Clamp(const T& value, const T& low, const T& high) {
  return std::max(low, std::min(value, high));
}

float DegToRad(const float deg) {
  return deg * 0.0174532925199432954743716805978692718781530857086181640625f;
}

double DegToRad(const double deg) {
  return deg * 0.0174532925199432954743716805978692718781530857086181640625;
}

// Convert angle in radians to degree.
float RadToDeg(const float rad) {
  return rad * 57.29577951308232286464772187173366546630859375f;
}

double RadToDeg(const double rad) {
  return rad * 57.29577951308232286464772187173366546630859375;
}

template <typename T>
double Percentile(std::vector<T>& elems, const double p) {
  THROW_CHECK(!elems.empty());
  THROW_CHECK_GE(p, 0);
  THROW_CHECK_LE(p, 100);
  const double idx_double = p / 100. * (elems.size() - 1);
  const double left_idx_double = std::floor(idx_double);
  const size_t left_idx = static_cast<size_t>(left_idx_double);
  const double right_idx_double = std::ceil(idx_double);
  const size_t right_idx = static_cast<size_t>(right_idx_double);
  std::nth_element(elems.begin(), elems.begin() + right_idx, elems.end());
  const double right = elems[right_idx];
  if (left_idx == right_idx) {
    return right;
  } else {
    const double left =
        *std::max_element(elems.begin(), elems.begin() + right_idx);
    return (right_idx_double - idx_double) * left +
           (idx_double - left_idx_double) * right;
  }
}

template <typename T>
double Percentile(std::vector<T>&& elems, const double p) {
  return Percentile(elems, p);
}

template <typename T>
double Median(std::vector<T>& elems) {
  return Percentile(elems, 50);
}

template <typename T>
double Median(std::vector<T>&& elems) {
  return Median(elems);
}

template <typename T>
double Mean(const std::vector<T>& elems) {
  THROW_CHECK(!elems.empty());
  double sum = 0;
  for (const auto el : elems) {
    sum += static_cast<double>(el);
  }
  return sum / elems.size();
}

template <typename T>
double Variance(const std::vector<T>& elems) {
  const double mean = Mean(elems);
  double var = 0;
  for (const auto el : elems) {
    const double diff = el - mean;
    var += diff * diff;
  }
  return var / (elems.size() - 1);
}

template <typename T>
double StdDev(const std::vector<T>& elems) {
  return std::sqrt(Variance(elems));
}

template <class Iterator>
bool NextCombination(Iterator first, Iterator middle, Iterator last) {
  return internal::NextCombination(first, middle, middle, last);
}

template <typename T>
T Sigmoid(const T x, const T alpha) {
  return T(1) / (T(1) + std::exp(-x * alpha));
}

template <typename T>
T ScaleSigmoid(T x, const T alpha, const T x0) {
  const T t0 = Sigmoid(-x0, alpha);
  const T t1 = Sigmoid(x0, alpha);
  x = (Sigmoid(2 * x0 * x - x0, alpha) - t0) / (t1 - t0);
  return x;
}

template <typename T1, typename T2>
T2 TruncateCast(const T1 value) {
  return static_cast<T2>(std::min(
      static_cast<T1>(std::numeric_limits<T2>::max()),
      std::max(static_cast<T1>(std::numeric_limits<T2>::min()), value)));
}

}  // namespace colmap
