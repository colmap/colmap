// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "optim/random_sampler"
#include <boost/test/unit_test.hpp>

#include <unordered_set>

#include "optim/random_sampler.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestLessSamples) {
  RandomSampler sampler(2);
  sampler.Initialize(5);
  BOOST_CHECK_EQUAL(sampler.MaxNumSamples(),
                    std::numeric_limits<size_t>::max());
  for (size_t i = 0; i < 100; ++i) {
    const auto samples = sampler.Sample();
    BOOST_CHECK_EQUAL(samples.size(), 2);
    BOOST_CHECK_EQUAL(
        std::unordered_set<size_t>(samples.begin(), samples.end()).size(), 2);
  }
}

BOOST_AUTO_TEST_CASE(TestEqualSamples) {
  RandomSampler sampler(5);
  sampler.Initialize(5);
  BOOST_CHECK_EQUAL(sampler.MaxNumSamples(),
                    std::numeric_limits<size_t>::max());
  for (size_t i = 0; i < 100; ++i) {
    const auto samples = sampler.Sample();
    BOOST_CHECK_EQUAL(samples.size(), 5);
    BOOST_CHECK_EQUAL(
        std::unordered_set<size_t>(samples.begin(), samples.end()).size(), 5);
  }
}
