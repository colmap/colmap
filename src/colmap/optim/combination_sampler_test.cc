// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/optim/combination_sampler.h"

#include "colmap/math/math.h"

#include <unordered_set>

#include <gtest/gtest.h>

namespace colmap {

TEST(CombinationSampler, LessSamples) {
  CombinationSampler sampler(2);
  sampler.Initialize(5);
  EXPECT_EQ(sampler.MaxNumSamples(), 10);
  std::vector<std::unordered_set<size_t>> sample_sets;
  for (size_t i = 0; i < 10; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_EQ(samples.size(), 2);
    sample_sets.emplace_back(samples.begin(), samples.end());
    EXPECT_EQ(sample_sets.back().size(), 2);
    for (size_t j = 0; j < i; ++j) {
      EXPECT_TRUE(sample_sets[j].count(samples[0]) == 0 ||
                  sample_sets[j].count(samples[1]) == 0);
    }
  }
  std::vector<size_t> samples;
  sampler.Sample(&samples);
  EXPECT_TRUE(sample_sets[0].count(samples[0]) == 1 &&
              sample_sets[0].count(samples[1]) == 1);
}

TEST(CombinationSampler, EqualSamples) {
  CombinationSampler sampler(5);
  sampler.Initialize(5);
  EXPECT_EQ(sampler.MaxNumSamples(), 1);
  for (size_t i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_EQ(samples.size(), 5);
    EXPECT_EQ(std::unordered_set<size_t>(samples.begin(), samples.end()).size(),
              5);
  }
}

}  // namespace colmap
