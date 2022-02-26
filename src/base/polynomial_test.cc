// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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

#define TEST_NAME "base/polynomial"
#include "util/testing.h"

#include "base/polynomial.h"

using namespace colmap;

#define CHECK_EQUAL_RESULT(find_func1, coeffs1, find_func2, coeffs2) \
  {                                                                  \
    Eigen::VectorXd real1;                                           \
    Eigen::VectorXd imag1;                                           \
    const bool success1 = find_func1(coeffs1, &real1, &imag1);       \
    Eigen::VectorXd real2;                                           \
    Eigen::VectorXd imag2;                                           \
    const bool success2 = find_func2(coeffs2, &real2, &imag2);       \
    BOOST_CHECK_EQUAL(success1, success2);                           \
    if (success1) {                                                  \
      BOOST_CHECK_EQUAL(real1, real2);                               \
      BOOST_CHECK_EQUAL(imag1, imag2);                               \
    }                                                                \
  }

BOOST_AUTO_TEST_CASE(TestEvaluatePolynomial) {
  BOOST_CHECK_EQUAL(EvaluatePolynomial(
                        (Eigen::VectorXd(5) << 1, -3, 3, -5, 10).finished(), 1),
                    1 - 3 + 3 - 5 + 10);
  BOOST_CHECK_CLOSE(
      EvaluatePolynomial((Eigen::VectorXd(4) << 1, -3, 3, -5).finished(), 2.0),
      1 * 2 * 2 * 2 - 3 * 2 * 2 + 3 * 2 - 5, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestFindLinearPolynomialRoots) {
  Eigen::VectorXd real;
  Eigen::VectorXd imag;
  BOOST_CHECK(FindLinearPolynomialRoots(Eigen::Vector2d(3, -2), &real, &imag));
  BOOST_CHECK_EQUAL(real(0), 2.0 / 3.0);
  BOOST_CHECK_EQUAL(imag(0), 0);
  BOOST_CHECK_CLOSE(EvaluatePolynomial(Eigen::Vector2d(3, -2),
                                       std::complex<double>(real(0), imag(0)))
                        .real(),
                    0.0, 1e-6);
  BOOST_CHECK_CLOSE(EvaluatePolynomial(Eigen::Vector2d(3, -2),
                                       std::complex<double>(real(0), imag(0)))
                        .imag(),
                    0.0, 1e-6);

  BOOST_CHECK(!FindLinearPolynomialRoots(Eigen::Vector2d(0, 1), &real, &imag));
}

BOOST_AUTO_TEST_CASE(TestFindQuadraticPolynomialRootsReal) {
  Eigen::VectorXd real;
  Eigen::VectorXd imag;
  Eigen::Vector3d coeffs(3, -2, -4);
  BOOST_CHECK(FindQuadraticPolynomialRoots(coeffs, &real, &imag));
  BOOST_CHECK(real.isApprox(Eigen::Vector2d(-0.868517092, 1.535183758), 1e-6));
  BOOST_CHECK_EQUAL(imag, Eigen::Vector2d(0, 0));
  BOOST_CHECK_CLOSE(
      EvaluatePolynomial(coeffs, std::complex<double>(real(0), imag(0))).real(),
      0.0, 1e-6);
  BOOST_CHECK_CLOSE(
      EvaluatePolynomial(coeffs, std::complex<double>(real(1), imag(1))).imag(),
      0.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestFindQuadraticPolynomialRootsComplex) {
  Eigen::VectorXd real;
  Eigen::VectorXd imag;
  const Eigen::Vector3d coeffs(0.276025076998578, 0.679702676853675,
                               0.655098003973841);
  BOOST_CHECK(FindQuadraticPolynomialRoots(coeffs, &real, &imag));
  BOOST_CHECK(real.isApprox(
      Eigen::Vector2d(-1.231233560813707, -1.231233560813707), 1e-6));
  BOOST_CHECK(imag.isApprox(
      Eigen::Vector2d(0.925954520440279, -0.925954520440279), 1e-6));
  BOOST_CHECK_CLOSE(
      EvaluatePolynomial(coeffs, std::complex<double>(real(0), imag(0))).real(),
      0.0, 1e-6);
  BOOST_CHECK_CLOSE(
      EvaluatePolynomial(coeffs, std::complex<double>(real(1), imag(1))).imag(),
      0.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestFindPolynomialRootsDurandKerner) {
  Eigen::VectorXd real;
  Eigen::VectorXd imag;
  Eigen::VectorXd coeffs(5);
  coeffs << 10, -5, 3, -3, 1;
  BOOST_CHECK(FindPolynomialRootsDurandKerner(coeffs, &real, &imag));
  // Reference values generated with OpenCV/Matlab.
  Eigen::VectorXd ref_real(4);
  ref_real << -0.201826, -0.201826, 0.451826, 0.451826;
  BOOST_CHECK(real.isApprox(ref_real, 1e-6));
  Eigen::VectorXd ref_imag(4);
  ref_imag << -0.627696, 0.627696, 0.160867, -0.160867;
  BOOST_CHECK(imag.isApprox(ref_imag, 1e-6));
}

BOOST_AUTO_TEST_CASE(TestFindPolynomialRootsDurandKernerLinearQuadratic) {
  CHECK_EQUAL_RESULT(FindPolynomialRootsDurandKerner, Eigen::Vector2d(1, 2),
                     FindLinearPolynomialRoots, Eigen::Vector2d(1, 2));
  CHECK_EQUAL_RESULT(FindPolynomialRootsDurandKerner,
                     (Eigen::VectorXd(4) << 0, 0, 1, 2).finished(),
                     FindLinearPolynomialRoots, Eigen::Vector2d(1, 2));
  CHECK_EQUAL_RESULT(FindPolynomialRootsDurandKerner, Eigen::Vector3d(1, 2, 3),
                     FindQuadraticPolynomialRoots, Eigen::Vector3d(1, 2, 3));
  CHECK_EQUAL_RESULT(FindPolynomialRootsDurandKerner,
                     (Eigen::VectorXd(5) << 0, 0, 1, 2, 3).finished(),
                     FindQuadraticPolynomialRoots, Eigen::Vector3d(1, 2, 3));
}

BOOST_AUTO_TEST_CASE(TestFindPolynomialRootsCompanionMatrix) {
  Eigen::VectorXd real;
  Eigen::VectorXd imag;
  Eigen::VectorXd coeffs(5);
  coeffs << 10, -5, 3, -3, 1;
  BOOST_CHECK(FindPolynomialRootsCompanionMatrix(coeffs, &real, &imag));
  // Reference values generated with OpenCV/Matlab.
  Eigen::VectorXd ref_real(4);
  ref_real << -0.201826, -0.201826, 0.451826, 0.451826;
  BOOST_CHECK(real.isApprox(ref_real, 1e-6));
  Eigen::VectorXd ref_imag(4);
  ref_imag << 0.627696, -0.627696, 0.160867, -0.160867;
  BOOST_CHECK(imag.isApprox(ref_imag, 1e-6));
}

BOOST_AUTO_TEST_CASE(TestFindPolynomialRootsCompanionMatrixLinearQuadratic) {
  CHECK_EQUAL_RESULT(FindPolynomialRootsCompanionMatrix, Eigen::Vector2d(1, 2),
                     FindLinearPolynomialRoots, Eigen::Vector2d(1, 2));
  CHECK_EQUAL_RESULT(FindPolynomialRootsCompanionMatrix,
                     (Eigen::VectorXd(4) << 0, 0, 1, 2).finished(),
                     FindLinearPolynomialRoots, Eigen::Vector2d(1, 2));
  CHECK_EQUAL_RESULT(FindPolynomialRootsCompanionMatrix,
                     Eigen::Vector3d(1, 2, 3), FindQuadraticPolynomialRoots,
                     Eigen::Vector3d(1, 2, 3));
  CHECK_EQUAL_RESULT(FindPolynomialRootsCompanionMatrix,
                     (Eigen::VectorXd(5) << 0, 0, 1, 2, 3).finished(),
                     FindQuadraticPolynomialRoots, Eigen::Vector3d(1, 2, 3));
}

BOOST_AUTO_TEST_CASE(TestFindPolynomialRootsCompanionMatrixZeroSolution) {
  Eigen::VectorXd real;
  Eigen::VectorXd imag;
  Eigen::VectorXd coeffs(5);
  coeffs << 10, -5, 3, -3, 0;
  BOOST_CHECK(FindPolynomialRootsCompanionMatrix(coeffs, &real, &imag));
  // Reference values generated with Matlab.
  Eigen::VectorXd ref_real(4);
  ref_real << 0.692438, -0.0962191, -0.0962191, 0;
  BOOST_CHECK(real.isApprox(ref_real, 1e-6));
  Eigen::VectorXd ref_imag(4);
  ref_imag << 0, 0.651148, -0.651148, 0;
  BOOST_CHECK(imag.isApprox(ref_imag, 1e-6));
}
