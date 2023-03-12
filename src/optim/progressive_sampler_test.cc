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

#define TEST_NAME "optim/progressive_sampler"
#include "util/testing.h"

#include <unordered_set>

#include "optim/progressive_sampler.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestLessSamples) {
  ProgressiveSampler sampler(2);
  sampler.Initialize(5);
  BOOST_CHECK_EQUAL(sampler.MaxNumSamples(),
                    std::numeric_limits<size_t>::max());
  for (size_t i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    BOOST_CHECK_EQUAL(samples.size(), 2);
    BOOST_CHECK_EQUAL(
        std::unordered_set<size_t>(samples.begin(), samples.end()).size(), 2);
  }
}

BOOST_AUTO_TEST_CASE(TestEqualSamples) {
  ProgressiveSampler sampler(5);
  sampler.Initialize(5);
  BOOST_CHECK_EQUAL(sampler.MaxNumSamples(),
                    std::numeric_limits<size_t>::max());
  for (size_t i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    BOOST_CHECK_EQUAL(samples.size(), 5);
    BOOST_CHECK_EQUAL(
        std::unordered_set<size_t>(samples.begin(), samples.end()).size(), 5);
  }
}

BOOST_AUTO_TEST_CASE(TestProgressive) {
  const size_t kNumSamples = 5;
  ProgressiveSampler sampler(kNumSamples);
  sampler.Initialize(50);
  size_t prev_last_sample = 5;
  for (size_t i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    for (size_t i = 0; i < samples.size() - 1; ++i) {
      BOOST_CHECK_LT(samples[i], samples.back());
      BOOST_CHECK_GE(samples.back(), prev_last_sample);
      prev_last_sample = samples.back();
    }
  }
}
