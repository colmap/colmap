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

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "util/math"
#include <boost/test/unit_test.hpp>

#include <vector>

#include "util/math.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestSignOfNumber) {
  BOOST_CHECK_EQUAL(SignOfNumber(0), 0);
  BOOST_CHECK_EQUAL(SignOfNumber(-0.1), -1);
  BOOST_CHECK_EQUAL(SignOfNumber(0.1), 1);
  BOOST_CHECK_EQUAL(SignOfNumber(std::numeric_limits<float>::quiet_NaN()), 0);
  BOOST_CHECK_EQUAL(SignOfNumber(std::numeric_limits<float>::infinity()), 1);
  BOOST_CHECK_EQUAL(SignOfNumber(-std::numeric_limits<float>::infinity()), -1);
}

BOOST_AUTO_TEST_CASE(TestIsNaN) {
  BOOST_CHECK(!IsNaN(0.0f));
  BOOST_CHECK(!IsNaN(0.0));
  BOOST_CHECK(IsNaN(std::numeric_limits<float>::quiet_NaN()));
  BOOST_CHECK(IsNaN(std::numeric_limits<double>::quiet_NaN()));
}

BOOST_AUTO_TEST_CASE(TestIsNaNArray) {
  BOOST_CHECK(!IsNaN(Eigen::Vector3f::Zero()));
  BOOST_CHECK(!IsNaN(Eigen::Vector3d::Zero()));
  BOOST_CHECK(IsNaN(
      Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f)));
  BOOST_CHECK(IsNaN(
      Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0.0f, 0.0f)));
}

BOOST_AUTO_TEST_CASE(TestIsInf) {
  BOOST_CHECK(!IsInf(0.0f));
  BOOST_CHECK(!IsInf(0.0));
  BOOST_CHECK(IsInf(std::numeric_limits<float>::infinity()));
  BOOST_CHECK(IsInf(std::numeric_limits<double>::infinity()));
  BOOST_CHECK(IsInf(-std::numeric_limits<float>::infinity()));
  BOOST_CHECK(IsInf(-std::numeric_limits<double>::infinity()));
}

BOOST_AUTO_TEST_CASE(TestIsInfArray) {
  BOOST_CHECK(!IsInf(Eigen::Vector3f::Zero()));
  BOOST_CHECK(!IsInf(Eigen::Vector3d::Zero()));
  BOOST_CHECK(IsInf(
      Eigen::Vector3f(std::numeric_limits<float>::infinity(), 0.0f, 0.0f)));
  BOOST_CHECK(IsInf(
      Eigen::Vector3d(std::numeric_limits<double>::infinity(), 0.0f, 0.0f)));
}

BOOST_AUTO_TEST_CASE(TestClip) {
  BOOST_CHECK_EQUAL(Clip(0, -1, 1), 0);
  BOOST_CHECK_EQUAL(Clip(0, 0, 1), 0);
  BOOST_CHECK_EQUAL(Clip(0, -1, 0), 0);
  BOOST_CHECK_EQUAL(Clip(0, -1, 1), 0);
  BOOST_CHECK_EQUAL(Clip(0, 1, 2), 1);
  BOOST_CHECK_EQUAL(Clip(0, -2, -1), -1);
  BOOST_CHECK_EQUAL(Clip(0, 0, 0), 0);
}

BOOST_AUTO_TEST_CASE(TestDegToRad) {
  BOOST_CHECK_EQUAL(DegToRad(0.0f), 0.0f);
  BOOST_CHECK_EQUAL(DegToRad(0.0), 0.0);
  BOOST_CHECK_LT(std::abs(DegToRad(180.0f) - M_PI), 1e-6f);
  BOOST_CHECK_LT(std::abs(DegToRad(180.0) - M_PI), 1e-6);
}

BOOST_AUTO_TEST_CASE(TestRadToDeg) {
  BOOST_CHECK_EQUAL(RadToDeg(0.0f), 0.0f);
  BOOST_CHECK_EQUAL(RadToDeg(0.0), 0.0);
  BOOST_CHECK_LT(std::abs(RadToDeg(M_PI) - 180.0f), 1e-6f);
  BOOST_CHECK_LT(std::abs(RadToDeg(M_PI) - 180.0), 1e-6);
}

BOOST_AUTO_TEST_CASE(TestMedian) {
  BOOST_CHECK_EQUAL(Median<int>({1, 2, 3, 4}), 2.5);
  BOOST_CHECK_EQUAL(Median<int>({1, 2, 3, 100}), 2.5);
  BOOST_CHECK_EQUAL(Median<int>({1, 2, 3, 4, 100}), 3);
  BOOST_CHECK_EQUAL(Median<int>({-100, 1, 2, 3, 4}), 2);
  BOOST_CHECK_EQUAL(Median<int>({-1, -2, -3, -4}), -2.5);
  BOOST_CHECK_EQUAL(Median<int>({-1, -2, 3, 4}), 1);
}

BOOST_AUTO_TEST_CASE(TestMean) {
  BOOST_CHECK_EQUAL(Mean<int>({1, 2, 3, 4}), 2.5);
  BOOST_CHECK_EQUAL(Mean<int>({1, 2, 3, 100}), 26.5);
  BOOST_CHECK_EQUAL(Mean<int>({1, 2, 3, 4, 100}), 22);
  BOOST_CHECK_EQUAL(Mean<int>({-100, 1, 2, 3, 4}), -18);
  BOOST_CHECK_EQUAL(Mean<int>({-1, -2, -3, -4}), -2.5);
  BOOST_CHECK_EQUAL(Mean<int>({-1, -2, 3, 4}), 1);
}

BOOST_AUTO_TEST_CASE(TestAnyLessThan) {
  BOOST_CHECK(AnyLessThan<int>({1, 2, 3, 4}, 5));
  BOOST_CHECK(AnyLessThan<int>({1, 2, 3, 4}, 4));
  BOOST_CHECK(AnyLessThan<int>({1, 2, 3, 4}, 3));
  BOOST_CHECK(AnyLessThan<int>({1, 2, 3, 4}, 2));
  BOOST_CHECK(!AnyLessThan<int>({1, 2, 3, 4}, 1));
  BOOST_CHECK(!AnyLessThan<int>({1, 2, 3, 4}, 0));
}

BOOST_AUTO_TEST_CASE(TestAnyGreaterThan) {
  BOOST_CHECK(!AnyGreaterThan<int>({1, 2, 3, 4}, 5));
  BOOST_CHECK(!AnyGreaterThan<int>({1, 2, 3, 4}, 4));
  BOOST_CHECK(AnyGreaterThan<int>({1, 2, 3, 4}, 3));
  BOOST_CHECK(AnyGreaterThan<int>({1, 2, 3, 4}, 2));
  BOOST_CHECK(AnyGreaterThan<int>({1, 2, 3, 4}, 1));
  BOOST_CHECK(AnyGreaterThan<int>({1, 2, 3, 4}, 0));
}

BOOST_AUTO_TEST_CASE(TestNextCombination) {
  std::vector<int> list{0};
  BOOST_CHECK(!NextCombination(list.begin(), list.begin() + 1, list.end()));
  list = {0, 1};
  BOOST_CHECK(!NextCombination(list.begin(), list.begin() + 2, list.end()));
  BOOST_CHECK_EQUAL(list[0], 0);
  BOOST_CHECK(NextCombination(list.begin(), list.begin() + 1, list.end()));
  BOOST_CHECK_EQUAL(list[0], 1);
  BOOST_CHECK(!NextCombination(list.begin(), list.begin() + 1, list.end()));
  BOOST_CHECK_EQUAL(list[0], 0);
  list = {0, 1, 2};
  BOOST_CHECK_EQUAL(list[0], 0);
  BOOST_CHECK_EQUAL(list[1], 1);
  BOOST_CHECK_EQUAL(list[2], 2);
  BOOST_CHECK(NextCombination(list.begin(), list.begin() + 2, list.end()));
  BOOST_CHECK_EQUAL(list[0], 0);
  BOOST_CHECK_EQUAL(list[1], 2);
  BOOST_CHECK_EQUAL(list[2], 1);
  BOOST_CHECK(NextCombination(list.begin(), list.begin() + 2, list.end()));
  BOOST_CHECK_EQUAL(list[0], 1);
  BOOST_CHECK_EQUAL(list[1], 2);
  BOOST_CHECK_EQUAL(list[2], 0);
  BOOST_CHECK(!NextCombination(list.begin(), list.begin() + 2, list.end()));
  BOOST_CHECK_EQUAL(list[0], 0);
  BOOST_CHECK_EQUAL(list[1], 1);
  BOOST_CHECK_EQUAL(list[2], 2);
}

BOOST_AUTO_TEST_CASE(TestSigmoid) {
  BOOST_CHECK_EQUAL(Sigmoid(0.0), 0.5);
  BOOST_CHECK_CLOSE(Sigmoid(100.0), 1.0, 1e-10);
  BOOST_CHECK_SMALL(Sigmoid(-100.0), 1e-10);
}

BOOST_AUTO_TEST_CASE(TestScaleSigmoid) {
  BOOST_CHECK_CLOSE(ScaleSigmoid(0.5), 0.5, 1e-10);
  BOOST_CHECK_CLOSE(ScaleSigmoid(1.0), 1.0, 1e-10);
  BOOST_CHECK_SMALL(ScaleSigmoid(-1.0), 1e-4);
}

BOOST_AUTO_TEST_CASE(TestNChooseK) {
  BOOST_CHECK_EQUAL(NChooseK(1, 0), 1);
  BOOST_CHECK_EQUAL(NChooseK(2, 0), 1);
  BOOST_CHECK_EQUAL(NChooseK(3, 0), 1);

  BOOST_CHECK_EQUAL(NChooseK(1, 1), 1);
  BOOST_CHECK_EQUAL(NChooseK(2, 1), 2);
  BOOST_CHECK_EQUAL(NChooseK(3, 1), 3);

  BOOST_CHECK_EQUAL(NChooseK(2, 2), 1);
  BOOST_CHECK_EQUAL(NChooseK(2, 3), 0);

  BOOST_CHECK_EQUAL(NChooseK(3, 2), 3);
  BOOST_CHECK_EQUAL(NChooseK(4, 2), 6);
  BOOST_CHECK_EQUAL(NChooseK(5, 2), 10);

  BOOST_CHECK_EQUAL(NChooseK(500, 3), 20708500);
}

BOOST_AUTO_TEST_CASE(TestEvaluatePolynomial) {
  BOOST_CHECK_EQUAL(EvaluatePolynomial({1, -3, 3, -5, 10}, 1),
                    std::complex<double>(1 - 3 + 3 - 5 + 10, 0));
  BOOST_CHECK_CLOSE(std::abs(EvaluatePolynomial({1, -3, 3, -5}, 2)),
                    std::abs(1 * 2 * 2 * 2 - 3 * 2 * 2 + 3 * 2 - 5), 1e-6);
}

BOOST_AUTO_TEST_CASE(TestSolvePolynomial) {
  std::vector<double> coeffs;
  coeffs.push_back(1);
  coeffs.push_back(-3);
  coeffs.push_back(3);
  coeffs.push_back(-5);
  coeffs.push_back(10);

  std::vector<std::complex<double>> roots = SolvePolynomialN(coeffs);

  // Generated with OpenCV
  std::vector<std::complex<double>> ref;
  ref.push_back(std::complex<double>(0.451826, 0.160867));
  ref.push_back(std::complex<double>(0.451826, -0.160867));
  ref.push_back(std::complex<double>(-0.201826, 0.627696));
  ref.push_back(std::complex<double>(-0.201826, -0.627696));

  for (size_t i = 0; i < roots.size(); ++i) {
    BOOST_CHECK_CLOSE(std::abs(roots[i]), std::abs(ref[i]), 0.0001);
  }
}
