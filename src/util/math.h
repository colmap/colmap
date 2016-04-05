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

#ifndef COLMAP_SRC_UTIL_MATH_H_
#define COLMAP_SRC_UTIL_MATH_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <list>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace colmap {

// Return 1 if number is positive, -1 if negative, and 0 if the number is 0.
template <typename T>
int SignOfNumber(const T val);

// Check if the given floating point number is a not-a-number (NaN) value.
inline bool IsNaN(const float x);
inline bool IsNaN(const double x);

// Check if the given floating point array contains a NaN value.
template <typename Derived>
inline bool IsNaN(const Eigen::MatrixBase<Derived>& x);

// Check if the given floating point number is a infinity.
inline bool IsInf(const float x);
inline bool IsInf(const double x);

// Check if the given floating point array contains infinity.
template <typename Derived>
inline bool IsInf(const Eigen::MatrixBase<Derived>& x);

// Clip the given value to a low and maximum value.
template <typename T>
inline T Clip(const T& value, const T& low, const T& high);

// Convert angle in degree to radians.
inline float DegToRad(const float deg);
inline double DegToRad(const double deg);

// Convert angle in radians to degree.
inline float RadToDeg(const float rad);
inline double RadToDeg(const double rad);

// Determine median value in vector. Returns NaN for empty vectors.
template <typename T>
double Median(const std::vector<T>& elems);

// Determine mean value in vector.
template <typename T>
double Mean(const std::vector<T>& elems);

// Check if any of the values in the vector is less than the given threshold.
template <typename T>
bool AnyLessThan(std::vector<T> elems, T threshold);

// Check if any of the values in the vector is greater than the given threshold.
template <typename T>
bool AnyGreaterThan(std::vector<T> elems, T threshold);

// Generate N-choose-K combinations.
//
// Note that elements in range [first, last) must be in sorted order,
// according to `std::less`.
template <class Iterator>
bool NextCombination(Iterator first, Iterator middle, Iterator last);

// Sigmoid function.
template <typename T>
T Sigmoid(const T x, const T alpha = 1);

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
T ScaleSigmoid(T x, const T alpha = 1, const T x0 = 10);

// Binomial coefficient or all combinations, defined as n! / ((n - k)! k!).
size_t NChooseK(const size_t n, const size_t k);

// Evaluate the polynomial with given coefficients at given position.
//
//    y(x) = c0 + c1 * x + c2 * x^2 + ...
//
// @param coeffs      Polynomial coefficients.
// @param x           Position to evaluate.
//
// @return            The function value.
std::complex<double> EvaluatePolynomial(const std::vector<double>& coeffs,
                                        const std::complex<double> x);

// Solve for the real roots of a linear polynomial.
//
// The polynomial coefficients must be given in increasing order:
//
//    y(x) = a + b * x
//
// @param a, b        Polynomial coefficients.
//
// @return            Real root.
std::vector<double> SolvePolynomial1(const double a, const double b);

// Solve for the real roots of a quadratic polynomial.
//
// The polynomial coefficients must be given in increasing order:
//
//    y(x) = a + b * x + c * x^2
//
// @param a, b, c     Polynomial coefficients.
//
// @return            Real roots.
std::vector<double> SolvePolynomial2(const double a, const double b,
                                     const double c);

// Solve for the real roots of a cubic polynomial.
//
// The polynomial coefficients must be given in increasing order:
//
//    y(x) = a + b * x + c * x^2 + d * x^3
//
// Implementation of Cardano's Formula.
//
// @param a, b, c, d  Polynomial coefficients.
//
// @return            Real roots.
std::vector<double> SolvePolynomial3(double a, double b, double c, double d);

// Solve for the real or complex roots of a polynomial.
//
// The polynomial coefficients must be given in increasing order:
//
//    y(x) = a0 + a1 * x + a2 * x^2 + a3 * x^3 + ...
//
// where ``coeffs[0] = a0``, ``coeffs[1] = a1``, ``coeffs[2] = a2`` etc.
//
// Implementation of the Durand-Kerner method, see:
//
//    https://en.wikipedia.org/wiki/Durand%E2%80%93Kerner_method
//
// @param coeffs      Vector of real coefficients.
// @param max_iter    Maximum number of iterations.
// @param eps         Convergence criteria, iteration stops if none of the root
//                    elements changed more than this threshold.
//
// @return            Complex roots of the polynomial of length N-1,
//                    where N is the number of coefficients.
std::vector<std::complex<double>> SolvePolynomialN(
    const std::vector<double>& coeffs, const int max_iter = 100,
    const double eps = 1e-10);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

namespace internal {

template <class Iterator>
bool NextCombination(Iterator first1, Iterator last1, Iterator first2,
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

bool IsNaN(const float x) { return x != x; }
bool IsNaN(const double x) { return x != x; }

template <typename Derived>
bool IsNaN(const Eigen::MatrixBase<Derived>& x) {
  return !(x.array() == x.array()).all();
}

bool IsInf(const float x) { return !IsNaN(x) && IsNaN(x - x); }
bool IsInf(const double x) { return !IsNaN(x) && IsNaN(x - x); }

template <typename Derived>
bool IsInf(const Eigen::MatrixBase<Derived>& x) {
  return !((x - x).array() == (x - x).array()).all();
}

template <typename T>
T Clip(const T& value, const T& low, const T& high) {
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
  size_t size = elems.size();
  size_t mid = size / 2;
  if (size == 0) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  std::vector<T> sorted_elems = elems;
  std::sort(sorted_elems.begin(), sorted_elems.end());

  if (size % 2 == 0) {
    return (sorted_elems[mid] + sorted_elems[mid - 1]) / 2.0;
  } else {
    return sorted_elems[mid];
  }
}

template <typename T>
double Mean(const std::vector<T>& elems) {
  double sum = 0;
  for (const auto el : elems) {
    sum += static_cast<double>(el);
  }
  return sum / elems.size();
}

template <typename T>
bool AnyLessThan(std::vector<T> elems, T threshold) {
  for (const auto& el : elems) {
    if (el < threshold) {
      return true;
    }
  }
  return false;
}

template <typename T>
bool AnyGreaterThan(std::vector<T> elems, T threshold) {
  for (const auto& el : elems) {
    if (el > threshold) {
      return true;
    }
  }
  return false;
}

template <class Iterator>
bool NextCombination(Iterator first, Iterator middle, Iterator last) {
  return internal::NextCombination(first, middle, middle, last);
}

template <typename T>
T Sigmoid(const T x, const T alpha) {
  return T(1) / (T(1) + exp(-x * alpha));
}

template <typename T>
T ScaleSigmoid(T x, const T alpha, const T x0) {
  const T t0 = Sigmoid(-x0, alpha);
  const T t1 = Sigmoid(x0, alpha);
  x = (Sigmoid(2 * x0 * x - x0, alpha) - t0) / (t1 - t0);
  return x;
}

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_MATH_H_
