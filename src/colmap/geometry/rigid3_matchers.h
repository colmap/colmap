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

class Rigid3dEqMatcher : public testing::MatcherInterface<const Rigid3d&> {
 public:
  explicit Rigid3dEqMatcher(const Rigid3d& rhs);

  void DescribeTo(std::ostream* os) const override;

  bool MatchAndExplain(const Rigid3d& lhs,
                       testing::MatchResultListener* listener) const override;

 private:
  const Rigid3d rhs_;
};

testing::PolymorphicMatcher<Rigid3dEqMatcher> Rigid3dEq(const Rigid3d& rhs);

class Rigid3dNearMatcher : public testing::MatcherInterface<const Rigid3d&> {
 public:
  Rigid3dNearMatcher(const Rigid3d& rhs, double rtol, double ttol);

  void DescribeTo(std::ostream* os) const override;

  bool MatchAndExplain(const Rigid3d& lhs,
                       testing::MatchResultListener* listener) const override;

 private:
  const Rigid3d rhs_;
  const double rtol_;
  const double ttol_;
};

testing::PolymorphicMatcher<Rigid3dNearMatcher> Rigid3dNear(
    const Rigid3d& rhs,
    double rtol = Eigen::NumTraits<double>::dummy_precision(),
    double ttol = Eigen::NumTraits<double>::dummy_precision());

}  // namespace colmap
