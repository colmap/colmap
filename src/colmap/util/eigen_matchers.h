// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <Eigen/Core>
#include <gmock/gmock.h>

namespace colmap {

template <typename T>
bool EigenMatrixMatchAndExplainShape(const T& lhs,
                                     const T& rhs,
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
    if (rhs_.isZero()) {
      // isApprox() is not well-defined for zero matrices.
      return lhs.norm() <= tol_;
    } else {
      return lhs.isApprox(rhs_, tol_);
    }
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
