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

#include "colmap/util/random.h"

#include "colmap/util/math.h"

#include <numeric>
#include <thread>

#include <gtest/gtest.h>

namespace colmap {

TEST(PRNGSeed, Nominal) {
  EXPECT_TRUE(PRNG == nullptr);
  SetPRNGSeed();
  EXPECT_TRUE(PRNG != nullptr);
  SetPRNGSeed(0);
  EXPECT_TRUE(PRNG != nullptr);
  std::thread thread([]() {
    // Each thread defines their own PRNG instance.
    EXPECT_TRUE(PRNG == nullptr);
    SetPRNGSeed();
    EXPECT_TRUE(PRNG != nullptr);
    SetPRNGSeed(0);
    EXPECT_TRUE(PRNG != nullptr);
  });
  thread.join();
}

TEST(Repeatability, Nominal) {
  SetPRNGSeed(0);
  std::vector<int> numbers1;
  for (size_t i = 0; i < 100; ++i) {
    numbers1.push_back(RandomInteger(0, 10000));
  }
  SetPRNGSeed(1);
  std::vector<int> numbers2;
  for (size_t i = 0; i < 100; ++i) {
    numbers2.push_back(RandomInteger(0, 10000));
  }
  SetPRNGSeed(0);
  std::vector<int> numbers3;
  for (size_t i = 0; i < 100; ++i) {
    numbers3.push_back(RandomInteger(0, 10000));
  }
  EXPECT_EQ(numbers1, numbers3);
  bool all_equal = true;
  for (size_t i = 0; i < numbers1.size(); ++i) {
    if (numbers1[i] != numbers2[i]) {
      all_equal = false;
    }
  }
  EXPECT_FALSE(all_equal);
}

TEST(RandomInteger, Nominal) {
  SetPRNGSeed();
  for (size_t i = 0; i < 1000; ++i) {
    EXPECT_GE(RandomInteger(-100, 100), -100);
    EXPECT_LE(RandomInteger(-100, 100), 100);
  }
}

TEST(RandomReal, Nominal) {
  SetPRNGSeed();
  for (size_t i = 0; i < 1000; ++i) {
    EXPECT_GE(RandomReal(-100.0, 100.0), -100.0);
    EXPECT_LE(RandomReal(-100.0, 100.0), 100.0);
  }
}

TEST(RandomGaussian, Nominal) {
  SetPRNGSeed(0);
  const double kMean = 1.0;
  const double kSigma = 1.0;
  const size_t kNumValues = 100000;
  std::vector<double> values;
  for (size_t i = 0; i < kNumValues; ++i) {
    values.push_back(RandomGaussian(kMean, kSigma));
  }
  EXPECT_LE(std::abs(Mean(values) - kMean), 1e-2);
  EXPECT_LE(std::abs(StdDev(values) - kSigma), 1e-2);
}

TEST(ShuffleNone, Nominal) {
  SetPRNGSeed();
  std::vector<int> numbers(0);
  Shuffle(0, &numbers);
  numbers = {1, 2, 3, 4, 5};
  std::vector<int> shuffled_numbers = numbers;
  Shuffle(0, &shuffled_numbers);
  EXPECT_EQ(numbers, shuffled_numbers);
}

TEST(ShuffleAll, Nominal) {
  SetPRNGSeed(0);
  std::vector<int> numbers(1000);
  std::iota(numbers.begin(), numbers.end(), 0);
  std::vector<int> shuffled_numbers = numbers;
  Shuffle(1000, &shuffled_numbers);
  size_t num_shuffled = 0;
  for (size_t i = 0; i < numbers.size(); ++i) {
    if (numbers[i] != shuffled_numbers[i]) {
      num_shuffled += 1;
    }
  }
  EXPECT_GT(num_shuffled, 0);
}

}  // namespace colmap
