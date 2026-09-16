// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/sim3.h"

#include <gmock/gmock.h>

namespace colmap {

template <typename T>
class Sim3dEqMatcher : public testing::MatcherInterface<T> {
 public:
  explicit Sim3dEqMatcher(T rhs) : rhs_(std::forward<T>(rhs)) {}

  void DescribeTo(std::ostream* os) const override { *os << rhs_; }

  bool MatchAndExplain(T lhs,
                       testing::MatchResultListener* listener) const override {
    // Note that with use !(a == b) to handle NaNs.
    if (!(lhs.scale() == rhs_.scale())) {
      return false;
    }
    if (!(lhs.rotation().coeffs() == rhs_.rotation().coeffs())) {
      return false;
    }
    if (!(Eigen::Vector3d(lhs.translation()) ==
          Eigen::Vector3d(rhs_.translation()))) {
      return false;
    }
    return true;
  }

 private:
  const Sim3d rhs_;
};

template <typename T>
testing::PolymorphicMatcher<Sim3dEqMatcher<T>> Sim3dEq(T rhs) {
  return testing::MakePolymorphicMatcher(
      Sim3dEqMatcher<T>(std::forward<T>(rhs)));
}

template <typename T>
class Sim3dNearMatcher : public testing::MatcherInterface<T> {
 public:
  Sim3dNearMatcher(T rhs, double stol, double rtol, double ttol)
      : rhs_(std::forward<T>(rhs)), stol_(stol), rtol_(rtol), ttol_(ttol) {}

  void DescribeTo(std::ostream* os) const override { *os << rhs_; }

  bool MatchAndExplain(T lhs,
                       testing::MatchResultListener* listener) const override {
    // Note that with use !(a <= b) to handle NaNs.
    if (!(std::abs(lhs.scale() - rhs_.scale()) <= stol_)) {
      *listener << " exceed scale threshold " << stol_;
      return false;
    }
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
  const Sim3d rhs_;
  const double stol_;
  const double rtol_;
  const double ttol_;
};

template <typename T>
testing::PolymorphicMatcher<Sim3dNearMatcher<T>> Sim3dNear(
    T rhs,
    double stol = Eigen::NumTraits<double>::dummy_precision(),
    double rtol = Eigen::NumTraits<double>::dummy_precision(),
    double ttol = Eigen::NumTraits<double>::dummy_precision()) {
  return testing::MakePolymorphicMatcher(
      Sim3dNearMatcher<T>(std::forward<T>(rhs), stol, rtol, ttol));
}

}  // namespace colmap
