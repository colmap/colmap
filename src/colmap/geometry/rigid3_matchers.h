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
    if (!(lhs.rotation.coeffs() == rhs_.rotation.coeffs())) {
      return false;
    }
    if (!(lhs.translation == rhs_.translation)) {
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
