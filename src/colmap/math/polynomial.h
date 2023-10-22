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

#pragma once

#include <Eigen/Core>

namespace colmap {

// All polynomials are assumed to be the form:
//
//   sum_{i=0}^N polynomial(i) x^{N-i}.
//
// and are given by a vector of coefficients of size N + 1.
//
// The implementation is based on COLMAP's old polynomial functionality and is
// inspired by Ceres-Solver's/Theia's implementation to support complex
// polynomials. The companion matrix implementation is based on NumPy.

// Evaluate the polynomial for the given coefficients at x using the Horner
// scheme. This function is templated such that the polynomial may be evaluated
// at real and/or imaginary points.
template <typename T>
T EvaluatePolynomial(const Eigen::VectorXd& coeffs, const T& x);

// Find the root of polynomials of the form: a * x + b = 0.
// The real and/or imaginary variable may be NULL if the output is not needed.
bool FindLinearPolynomialRoots(const Eigen::VectorXd& coeffs,
                               Eigen::VectorXd* real,
                               Eigen::VectorXd* imag);

// Find the roots of polynomials of the form: a * x^2 + b * x + c = 0.
// The real and/or imaginary variable may be NULL if the output is not needed.
bool FindQuadraticPolynomialRoots(const Eigen::VectorXd& coeffs,
                                  Eigen::VectorXd* real,
                                  Eigen::VectorXd* imag);

// Find the roots of a polynomial using the Durand-Kerner method, based on:
//
//    https://en.wikipedia.org/wiki/Durand%E2%80%93Kerner_method
//
// The Durand-Kerner is comparatively fast but often unstable/inaccurate.
// The real and/or imaginary variable may be NULL if the output is not needed.
bool FindPolynomialRootsDurandKerner(const Eigen::VectorXd& coeffs,
                                     Eigen::VectorXd* real,
                                     Eigen::VectorXd* imag);

// Find the roots of a polynomial using the companion matrix method, based on:
//
//    R. A. Horn & C. R. Johnson, Matrix Analysis. Cambridge,
//    UK: Cambridge University Press, 1999, pp. 146-7.
//
// Compared to Durand-Kerner, this method is slower but more stable/accurate.
// The real and/or imaginary variable may be NULL if the output is not needed.
bool FindPolynomialRootsCompanionMatrix(const Eigen::VectorXd& coeffs,
                                        Eigen::VectorXd* real,
                                        Eigen::VectorXd* imag);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
T EvaluatePolynomial(const Eigen::VectorXd& coeffs, const T& x) {
  T value = 0.0;
  for (Eigen::VectorXd::Index i = 0; i < coeffs.size(); ++i) {
    value = value * x + coeffs(i);
  }
  return value;
}

}  // namespace colmap
