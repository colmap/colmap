// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/combination_sampler.h"

#include "colmap/math/math.h"

#include <numeric>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CombinationSampler, MatchesNextCombinationAcrossCyclesAndReinit) {
  for (size_t num_samples = 0; num_samples <= 6; ++num_samples) {
    // Reuse one sampler to check that Initialize resets mid-cycle progress.
    CombinationSampler sampler(num_samples);
    for (size_t total = num_samples; total <= 6; ++total) {
      sampler.Initialize(total);
      const size_t max_num_samples = NChooseK(total, num_samples);
      EXPECT_EQ(sampler.MaxNumSamples(), max_num_samples);
      std::vector<size_t> population(total);
      std::iota(population.begin(), population.end(), 0);
      std::vector<size_t> sample;
      for (size_t i = 0; i < 2 * max_num_samples + 1; ++i) {
        sampler.Sample(&sample);
        ASSERT_EQ(sample,
                  std::vector<size_t>(population.begin(),
                                      population.begin() + num_samples));
        if (!NextCombination(population.begin(),
                             population.begin() + num_samples,
                             population.end())) {
          std::iota(population.begin(), population.end(), 0);
        }
      }
    }
  }
}

}  // namespace
}  // namespace colmap
