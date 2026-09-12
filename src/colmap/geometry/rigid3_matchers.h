// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/rigid3.h"

#include <gmock/gmock.h>

namespace colmap {

template <typename T>
class Rigid3dEqMatcher : public testing::MatcherInterface<T> {
 public:
  explicit Rigid3dEqMatcher(T rhs) : rhs_(std::forward<T>(rhs)) {}

  void DescribeTo(std::ostream* os) const override { *os << rhs_; }

  bool MatchAndExplain(T lhs,
                       testing::MatchResultListener* listener) const override {
    // Note that with use !(a == b) to handle NaNs.
    if (!(lhs.rotation().coeffs() == rhs_.rotation().coeffs())) {
      return false;
    }
    if (!(lhs.translation() == rhs_.translation())) {
      return false;
    }
    return true;
  }

 private:
  const Rigid3d rhs_;
};

template <typename T>
testing::PolymorphicMatcher<Rigid3dEqMatcher<T>> Rigid3dEq(T rhs) {
  return testing::MakePolymorphicMatcher(
      Rigid3dEqMatcher<T>(std::forward<T>(rhs)));
}

template <typename T>
class Rigid3dNearMatcher : public testing::MatcherInterface<T> {
 public:
  Rigid3dNearMatcher(T rhs, double rtol, double ttol)
      : rhs_(std::forward<T>(rhs)), rtol_(rtol), ttol_(ttol) {}

  void DescribeTo(std::ostream* os) const override { *os << rhs_; }

  bool MatchAndExplain(T lhs,
                       testing::MatchResultListener* listener) const override {
    // Note that with use !(a <= b) to handle NaNs.
    if (!(lhs.rotation().angularDistance(rhs_.rotation()) <= rtol_)) {
      *listener << " exceed rotation threshold " << rtol_;
      return false;
    }
    if (rhs_.translation().isZero()) {
      // isApprox() is not well-defined for zero matrices.
      return lhs.translation().norm() <= ttol_;
    } else {
      if (!lhs.translation().isApprox(rhs_.translation(), ttol_)) {
        *listener << " exceed translation threshold " << ttol_;
        return false;
      }
    }
    return true;
  }

 private:
  const Rigid3d rhs_;
  const double rtol_;
  const double ttol_;
};

template <typename T>
testing::PolymorphicMatcher<Rigid3dNearMatcher<T>> Rigid3dNear(
    T rhs,
    double rtol = Eigen::NumTraits<double>::dummy_precision(),
    double ttol = Eigen::NumTraits<double>::dummy_precision()) {
  return testing::MakePolymorphicMatcher(
      Rigid3dNearMatcher<T>(std::forward<T>(rhs), rtol, ttol));
}

}  // namespace colmap
