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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

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

// Return 1 if number is positive, -1 if negative, and 0 if the number is 0.
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

// Determine median value in vector. Returns NaN for empty vectors.
template <typename T>
double Median(const std::vector<T>& elems);

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
size_t NChooseK(size_t n, size_t k);

// Cast value from one type to another and truncate instead of overflow, if the
// input value is out of range of the output data type.
template <typename T1, typename T2>
T2 TruncateCast(T1 value);

// Compute the n-th percentile in the given sequence.
template <typename T>
T Percentile(const std::vector<T>& elems, double p);

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
  --m2;
  while (--m1 != first1 && *m1 >= *m2) {
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
  return (T(0) < val) - (val < T(0));
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
double Median(const std::vector<T>& elems) {
  CHECK(!elems.empty());

  const size_t mid_idx = elems.size() / 2;

  std::vector<T> ordered_elems = elems;
  std::nth_element(ordered_elems.begin(),
                   ordered_elems.begin() + mid_idx,
                   ordered_elems.end());

  if (elems.size() % 2 == 0) {
    const T mid_element1 = ordered_elems[mid_idx];
    const T mid_element2 = *std::max_element(ordered_elems.begin(),
                                             ordered_elems.begin() + mid_idx);
    return 0.5 * mid_element1 + 0.5 * mid_element2;
  } else {
    return ordered_elems[mid_idx];
  }
}

template <typename T>
T Percentile(const std::vector<T>& elems, const double p) {
  CHECK(!elems.empty());
  CHECK_GE(p, 0);
  CHECK_LE(p, 100);

  const int idx = static_cast<int>(std::round(p / 100 * (elems.size() - 1)));
  const size_t percentile_idx =
      std::max(0, std::min(static_cast<int>(elems.size() - 1), idx));

  std::vector<T> ordered_elems = elems;
  std::nth_element(ordered_elems.begin(),
                   ordered_elems.begin() + percentile_idx,
                   ordered_elems.end());

  return ordered_elems.at(percentile_idx);
}

template <typename T>
double Mean(const std::vector<T>& elems) {
  CHECK(!elems.empty());
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
  return std::min(
      static_cast<T1>(std::numeric_limits<T2>::max()),
      std::max(static_cast<T1>(std::numeric_limits<T2>::min()), value));
}

}  // namespace colmap
