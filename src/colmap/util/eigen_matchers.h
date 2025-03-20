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

#include <Eigen/Core>
#include <gmock/gmock.h>

namespace colmap {

template <typename T>
bool EigenMatrixMatchAndExplainShape(T lhs,
                                     T rhs,
                                     testing::MatchResultListener* listener) {
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
    *listener << " have different shape (" << lhs.rows() << ", " << lhs.cols()
              << ") vs. (" << rhs.rows() << ", " << rhs.cols() << ")";
    return false;
  }
  return true;
}

template <typename T>
class EigenMatrixEqMatcher : public testing::MatcherInterface<T> {
 public:
  explicit EigenMatrixEqMatcher(T rhs) : rhs_(std::forward<T>(rhs)) {}

  void DescribeTo(std::ostream* os) const override { *os << rhs_; }

  bool MatchAndExplain(T lhs,
                       testing::MatchResultListener* listener) const override {
    if (!EigenMatrixMatchAndExplainShape(lhs, rhs_, listener)) {
      return false;
    }
    return lhs == rhs_;
  }

 private:
  const T rhs_;
};

template <typename T>
testing::PolymorphicMatcher<EigenMatrixEqMatcher<T>> EigenMatrixEq(T rhs) {
  return testing::MakePolymorphicMatcher(
      EigenMatrixEqMatcher<T>(std::forward<T>(rhs)));
}

template <typename T>
class EigenMatrixNearMatcher : public testing::MatcherInterface<T> {
 public:
  EigenMatrixNearMatcher(T rhs, double tol)
      : rhs_(std::forward<T>(rhs)), tol_(tol) {}

  void DescribeTo(std::ostream* os) const override { *os << rhs_; }

  bool MatchAndExplain(T lhs,
                       testing::MatchResultListener* listener) const override {
    if (!EigenMatrixMatchAndExplainShape(lhs, rhs_, listener)) {
      return false;
    }
    return lhs.isApprox(rhs_, tol_);
  }

 private:
  const T rhs_;
  const double tol_;
};

template <typename T>
testing::PolymorphicMatcher<EigenMatrixNearMatcher<T>> EigenMatrixNear(
    T rhs, double tol = Eigen::NumTraits<double>::dummy_precision()) {
  return testing::MakePolymorphicMatcher(
      EigenMatrixNearMatcher<T>(std::forward<T>(rhs), tol));
}

}  // namespace colmap
