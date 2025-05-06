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

#include "colmap/geometry/sim3_matchers.h"

namespace colmap {

Sim3dEqMatcher::Sim3dEqMatcher(const Sim3d& rhs) : rhs_(rhs) {}

void Sim3dEqMatcher::DescribeTo(std::ostream* os) const { *os << rhs_; }

bool Sim3dEqMatcher::MatchAndExplain(
    const Sim3d& lhs, testing::MatchResultListener* listener) const {
  // Note that with use !(a == b) to handle NaNs.
  if (!(lhs.scale == rhs_.scale)) {
    return false;
  }
  if (!(lhs.rotation == rhs_.rotation)) {
    return false;
  }
  if (!(lhs.translation == rhs_.translation)) {
    return false;
  }
  return true;
}

testing::PolymorphicMatcher<Sim3dEqMatcher> Sim3dEq(const Sim3d& rhs) {
  return testing::MakePolymorphicMatcher(Sim3dEqMatcher(rhs));
}

Sim3dNearMatcher::Sim3dNearMatcher(const Sim3d& rhs,
                                   double stol,
                                   double rtol,
                                   double ttol)
    : rhs_(rhs), stol_(rtol), rtol_(rtol), ttol_(ttol) {}

void Sim3dNearMatcher::DescribeTo(std::ostream* os) const { *os << rhs_; }

bool Sim3dNearMatcher::MatchAndExplain(
    const Sim3d& lhs, testing::MatchResultListener* listener) const {
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

testing::PolymorphicMatcher<Sim3dNearMatcher> Sim3dNear(const Sim3d& rhs,
                                                        double stol,
                                                        double rtol,
                                                        double ttol) {
  return testing::MakePolymorphicMatcher(
      Sim3dNearMatcher(rhs, stol, rtol, ttol));
}

}  // namespace colmap
