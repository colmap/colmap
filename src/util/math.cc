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

#include "util/math.h"

namespace colmap {

size_t NChooseK(const size_t n, const size_t k) {
  if (k == 0) {
    return 1;
  }

  return (n * NChooseK(n - 1, k - 1)) / k;
}

std::complex<double> EvaluatePolynomial(const std::vector<double>& coeffs,
                                        const std::complex<double> x) {
  const int n = static_cast<int>(coeffs.size());
  std::complex<double> xn = x;
  std::complex<double> y = 0;
  for (int i = n - 2; i >= 0; --i) {
    y += coeffs[i] * xn;
    xn *= x;
  }
  y += coeffs.back();
  return y;
}

std::vector<double> SolvePolynomial1(const double a, const double b) {
  std::vector<double> roots;
  if (a != 0) {
    roots.resize(1);
    roots[0] = -b / a;
  }
  return roots;
}

std::vector<double> SolvePolynomial2(const double a, const double b,
                                     const double c) {
  std::vector<double> roots;
  if (a == 0) {
    roots = SolvePolynomial1(b, c);
  } else {
    const double d = b * b - 4 * a * c;
    if (d == 0) {
      roots.resize(1);
      roots[0] = -b / (2 * a);
    } else if (d > 0) {
      const double s = std::sqrt(d);
      const double q = -(b + (b > 0 ? s : -s)) / 2;
      roots.resize(2);
      roots[0] = q / a;
      roots[1] = c / q;
    }
  }
  return roots;
}

std::vector<double> SolvePolynomial3(double a, double b, double c, double d) {
  std::vector<double> roots;
  if (a == 0) {
    roots = SolvePolynomial2(b, c, d);
  } else {
    // Solve normalized cubic equation

    // Normalize
    const double inv_a = 1 / a;
    b *= inv_a;
    c *= inv_a;
    d *= inv_a;

    const double p = (3 * c - b * b) / 3;
    const double q = 2 * b * b * b / 27 - b * c / 3 + d;
    const double b3 = b / 3;
    const double p3 = p / 3;
    const double q2 = q / 2;
    const double d = p3 * p3 * p3 + q2 * q2;

    if (d == 0 && p3 == 0) {
      roots.resize(3);
      roots[0] = -b3;
      roots[1] = -b3;
      roots[2] = -b3;
    } else {
      const std::complex<double> u =
          std::pow(-q / 2 + std::sqrt(std::complex<double>(d)), 1 / 3.0);
      const std::complex<double> v = -p / (3.0 * u);
      const std::complex<double> y0 = u + v;

      if (d > 0) {
        roots.resize(1);
        roots[0] = y0.real() - b3;
      } else {
        // sqrt(3), since std::sqrt is not a constexpr (?)
        const double sqrt3 =
            1.7320508075688772935274463415058723669428052538103806;
        const std::complex<double> m = -y0 / 2.0;
        const std::complex<double> n =
            (u - v) / 2.0 * std::complex<double>(0, sqrt3);
        const std::complex<double> y1 = m + n;
        if (d == 0) {
          roots.resize(2);
          roots[0] = y0.real() - b3;
          roots[1] = y1.real() - b3;
        } else {
          roots.resize(3);
          const std::complex<double> y2 = m - n;
          roots[0] = y0.real() - b3;
          roots[1] = y1.real() - b3;
          roots[2] = y2.real() - b3;
        }
      }
    }
  }
  return roots;
}

std::vector<std::complex<double>> SolvePolynomialN(
    const std::vector<double>& coeffs, const int max_iter, const double eps) {
  const size_t cn = coeffs.size();
  const size_t n = cn - 1;

  std::vector<std::complex<double>> roots(n);
  std::complex<double> p(1, 0);
  std::complex<double> r(1, 1);

  // Initial roots
  for (size_t i = 0; i < n; ++i) {
    roots[i] = p;
    p = p * r;
  }

  // Iterative solver
  for (int iter = 0; iter < max_iter; ++iter) {
    double max_diff = 0;
    for (size_t i = 0; i < n; ++i) {
      p = roots[i];
      std::complex<double> nom = coeffs[n];
      std::complex<double> denom = coeffs[n];
      for (size_t j = 0; j < n; ++j) {
        nom = nom * p + coeffs[n - j - 1];
        if (j != i) {
          denom = denom * (p - roots[j]);
        }
      }
      nom /= denom;
      roots[i] = p - nom;
      max_diff = std::max(max_diff, std::abs(nom.real()));
      max_diff = std::max(max_diff, std::abs(nom.imag()));
    }

    // Break, if roots do not change anymore
    if (max_diff <= eps) {
      break;
    }
  }

  return roots;
}

}  // namespace colmap
