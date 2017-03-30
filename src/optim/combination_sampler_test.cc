// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#define TEST_NAME "optim/combination_sampler"
#include "util/testing.h"

#include <unordered_set>

#include "optim/combination_sampler.h"
#include "util/math.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestLessSamples) {
  CombinationSampler sampler(2);
  sampler.Initialize(5);
  BOOST_CHECK_EQUAL(sampler.MaxNumSamples(), 10);
  std::vector<std::unordered_set<size_t>> sample_sets;
  for (size_t i = 0; i < 10; ++i) {
    const auto samples = sampler.Sample();
    BOOST_CHECK_EQUAL(samples.size(), 2);
    sample_sets.emplace_back(samples.begin(), samples.end());
    BOOST_CHECK_EQUAL(sample_sets.back().size(), 2);
    for (size_t j = 0; j < i; ++j) {
      BOOST_CHECK(sample_sets[j].count(samples[0]) == 0 ||
                  sample_sets[j].count(samples[1]) == 0);
    }
  }
  const auto samples = sampler.Sample();
  BOOST_CHECK(sample_sets[0].count(samples[0]) == 1 &&
              sample_sets[0].count(samples[1]) == 1);
}

BOOST_AUTO_TEST_CASE(TestEqualSamples) {
  CombinationSampler sampler(5);
  sampler.Initialize(5);
  BOOST_CHECK_EQUAL(sampler.MaxNumSamples(), 1);
  for (size_t i = 0; i < 100; ++i) {
    const auto samples = sampler.Sample();
    BOOST_CHECK_EQUAL(samples.size(), 5);
    BOOST_CHECK_EQUAL(
        std::unordered_set<size_t>(samples.begin(), samples.end()).size(), 5);
  }
}
