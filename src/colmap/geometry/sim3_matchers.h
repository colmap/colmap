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
    if (!(lhs.scale == rhs_.scale)) {
      return false;
    }
    if (!(lhs.rotation.coeffs() == rhs_.rotation.coeffs())) {
      return false;
    }
    if (!(lhs.translation == rhs_.translation)) {
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
      : rhs_(std::forward<T>(rhs)), stol_(rtol), rtol_(rtol), ttol_(ttol) {}

  void DescribeTo(std::ostream* os) const override { *os << rhs_; }

  bool MatchAndExplain(T lhs,
                       testing::MatchResultListener* listener) const override {
    // Note that with use !(a <= b) to handle NaNs.
    if (!(std::abs(lhs.scale - rhs_.scale) <= stol_)) {
      *listener << " exceed scale threshold " << stol_;
      return false;
    }
    if (!(lhs.rotation.angularDistance(rhs_.rotation) <= rtol_)) {
      *listener << " exceed rotation threshold " << rtol_;
      return false;
    }
    if (!lhs.translation.isApprox(rhs_.translation, ttol_)) {
      *listener << " exceed translation threshold " << ttol_;
      return false;
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
