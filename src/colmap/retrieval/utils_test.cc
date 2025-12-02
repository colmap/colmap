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

#include "colmap/retrieval/utils.h"

#include <cmath>

#include <gtest/gtest.h>

namespace colmap::retrieval {
namespace {

TEST(HammingDistWeightFunctor, ZeroDistance) {
  HammingDistWeightFunctor<64> functor;
  EXPECT_EQ(functor(0), 1.0f);
}

TEST(HammingDistWeightFunctor, SmallDistance) {
  HammingDistWeightFunctor<64> functor;
  constexpr float kSigma = 16.0f;
  constexpr float kSigmaSquared = kSigma * kSigma;
  const size_t kDist = 5;
  const float expected =
      std::exp(-static_cast<float>(kDist * kDist) / kSigmaSquared);
  EXPECT_EQ(functor(kDist), expected);
}

TEST(HammingDistWeightFunctor, MaxDistance) {
  HammingDistWeightFunctor<64> functor;
  // At max hamming distance, should have non-zero weight
  EXPECT_GT(functor(functor.kMaxHammingDistance), 0.0f);
  EXPECT_LT(functor(functor.kMaxHammingDistance), 1.0f);
  // Beyond max hamming distance, weight should be 0
  EXPECT_EQ(functor(functor.kMaxHammingDistance + 1), 0.0f);
}

TEST(HammingDistWeightFunctor, Monotonicity) {
  HammingDistWeightFunctor<128, 20> functor;
  // Weight should decrease as distance increases (up to max distance)
  for (size_t i = 0; i <= functor.kMaxHammingDistance; ++i) {
    EXPECT_GT(functor(i), functor(i + 1)) << i;
  }
}

}  // namespace
}  // namespace colmap::retrieval
