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

#include "colmap/util/math.h"

#include <gtest/gtest.h>

namespace colmap {

TEST(SignOfNumber, Nominal) {
  EXPECT_EQ(SignOfNumber(0), 0);
  EXPECT_EQ(SignOfNumber(-0.1), -1);
  EXPECT_EQ(SignOfNumber(0.1), 1);
  EXPECT_EQ(SignOfNumber(std::numeric_limits<float>::quiet_NaN()), 0);
  EXPECT_EQ(SignOfNumber(std::numeric_limits<float>::infinity()), 1);
  EXPECT_EQ(SignOfNumber(-std::numeric_limits<float>::infinity()), -1);
}

TEST(Clamp, Nominal) {
  EXPECT_EQ(Clamp(0, -1, 1), 0);
  EXPECT_EQ(Clamp(0, 0, 1), 0);
  EXPECT_EQ(Clamp(0, -1, 0), 0);
  EXPECT_EQ(Clamp(0, -1, 1), 0);
  EXPECT_EQ(Clamp(0, 1, 2), 1);
  EXPECT_EQ(Clamp(0, -2, -1), -1);
  EXPECT_EQ(Clamp(0, 0, 0), 0);
}

TEST(DegToRad, Nominal) {
  EXPECT_EQ(DegToRad(0.0f), 0.0f);
  EXPECT_EQ(DegToRad(0.0), 0.0);
  EXPECT_LT(std::abs(DegToRad(180.0f) - M_PI), 1e-6f);
  EXPECT_LT(std::abs(DegToRad(180.0) - M_PI), 1e-6);
}

TEST(RadToDeg, Nominal) {
  EXPECT_EQ(RadToDeg(0.0f), 0.0f);
  EXPECT_EQ(RadToDeg(0.0), 0.0);
  EXPECT_LT(std::abs(RadToDeg(M_PI) - 180.0f), 1e-6f);
  EXPECT_LT(std::abs(RadToDeg(M_PI) - 180.0), 1e-6);
}

TEST(Median, Nominal) {
  EXPECT_EQ(Median<int>({1, 2, 3, 4}), 2.5);
  EXPECT_EQ(Median<int>({1, 2, 3, 100}), 2.5);
  EXPECT_EQ(Median<int>({1, 2, 3, 4, 100}), 3);
  EXPECT_EQ(Median<int>({-100, 1, 2, 3, 4}), 2);
  EXPECT_EQ(Median<int>({-1, -2, -3, -4}), -2.5);
  EXPECT_EQ(Median<int>({-1, -2, 3, 4}), 1);
  // Test integer overflow scenario.
  EXPECT_EQ(Median<int8_t>({100, 115, 119, 127}), 117);
}

TEST(Percentile, Nominal) {
  EXPECT_EQ(Percentile<int>({0}, 0), 0);
  EXPECT_EQ(Percentile<int>({0}, 50), 0);
  EXPECT_EQ(Percentile<int>({0}, 100), 0);
  EXPECT_EQ(Percentile<int>({0, 1}, 0), 0);
  EXPECT_EQ(Percentile<int>({0, 1}, 50), 1);
  EXPECT_EQ(Percentile<int>({0, 1}, 100), 1);
  EXPECT_EQ(Percentile<int>({0, 1, 2}, 0), 0);
  EXPECT_EQ(Percentile<int>({0, 1, 2}, 50), 1);
  EXPECT_EQ(Percentile<int>({0, 1, 2}, 100), 2);
  EXPECT_EQ(Percentile<int>({0, 1, 1, 2}, 0), 0);
  EXPECT_EQ(Percentile<int>({0, 1, 1, 2}, 33), 1);
  EXPECT_EQ(Percentile<int>({0, 1, 1, 2}, 50), 1);
  EXPECT_EQ(Percentile<int>({0, 1, 1, 2}, 66), 1);
  EXPECT_EQ(Percentile<int>({0, 1, 1, 2}, 100), 2);
}

TEST(Mean, Nominal) {
  EXPECT_EQ(Mean<int>({1, 2, 3, 4}), 2.5);
  EXPECT_EQ(Mean<int>({1, 2, 3, 100}), 26.5);
  EXPECT_EQ(Mean<int>({1, 2, 3, 4, 100}), 22);
  EXPECT_EQ(Mean<int>({-100, 1, 2, 3, 4}), -18);
  EXPECT_EQ(Mean<int>({-1, -2, -3, -4}), -2.5);
  EXPECT_EQ(Mean<int>({-1, -2, 3, 4}), 1);
}

TEST(Variance, Nominal) {
  EXPECT_LE(std::abs(Variance<int>({1, 2, 3, 4}) - 1.66666666), 1e-6);
  EXPECT_LE(std::abs(Variance<int>({1, 2, 3, 100}) - 2401.66666666), 1e-6);
  EXPECT_LE(std::abs(Variance<int>({1, 2, 3, 4, 100}) - 1902.5), 1e-6);
  EXPECT_LE(std::abs(Variance<int>({-100, 1, 2, 3, 4}) - 2102.5), 1e-6);
  EXPECT_LE(std::abs(Variance<int>({-1, -2, -3, -4}) - 1.66666666), 1e-6);
  EXPECT_LE(std::abs(Variance<int>({-1, -2, 3, 4}) - 8.66666666), 1e-6);
}

TEST(StdDev, Nominal) {
  EXPECT_LE(std::abs(std::sqrt(Variance<int>({1, 2, 3, 4})) -
                     StdDev<int>({1, 2, 3, 4})),
            1e-6);
  EXPECT_LE(std::abs(std::sqrt(Variance<int>({1, 2, 3, 100})) -
                     StdDev<int>({1, 2, 3, 100})),
            1e-6);
}

TEST(AnyLessThan, Nominal) {
  EXPECT_TRUE(AnyLessThan<int>({1, 2, 3, 4}, 5));
  EXPECT_TRUE(AnyLessThan<int>({1, 2, 3, 4}, 4));
  EXPECT_TRUE(AnyLessThan<int>({1, 2, 3, 4}, 3));
  EXPECT_TRUE(AnyLessThan<int>({1, 2, 3, 4}, 2));
  EXPECT_FALSE(AnyLessThan<int>({1, 2, 3, 4}, 1));
  EXPECT_FALSE(AnyLessThan<int>({1, 2, 3, 4}, 0));
}

TEST(AnyGreaterThan, Nominal) {
  EXPECT_FALSE(AnyGreaterThan<int>({1, 2, 3, 4}, 5));
  EXPECT_FALSE(AnyGreaterThan<int>({1, 2, 3, 4}, 4));
  EXPECT_TRUE(AnyGreaterThan<int>({1, 2, 3, 4}, 3));
  EXPECT_TRUE(AnyGreaterThan<int>({1, 2, 3, 4}, 2));
  EXPECT_TRUE(AnyGreaterThan<int>({1, 2, 3, 4}, 1));
  EXPECT_TRUE(AnyGreaterThan<int>({1, 2, 3, 4}, 0));
}

TEST(NextCombination, Nominal) {
  std::vector<int> list{0};
  EXPECT_FALSE(NextCombination(list.begin(), list.begin() + 1, list.end()));
  list = {0, 1};
  EXPECT_FALSE(NextCombination(list.begin(), list.begin() + 2, list.end()));
  EXPECT_EQ(list[0], 0);
  EXPECT_TRUE(NextCombination(list.begin(), list.begin() + 1, list.end()));
  EXPECT_EQ(list[0], 1);
  EXPECT_FALSE(NextCombination(list.begin(), list.begin() + 1, list.end()));
  EXPECT_EQ(list[0], 0);
  list = {0, 1, 2};
  EXPECT_EQ(list[0], 0);
  EXPECT_EQ(list[1], 1);
  EXPECT_EQ(list[2], 2);
  EXPECT_TRUE(NextCombination(list.begin(), list.begin() + 2, list.end()));
  EXPECT_EQ(list[0], 0);
  EXPECT_EQ(list[1], 2);
  EXPECT_EQ(list[2], 1);
  EXPECT_TRUE(NextCombination(list.begin(), list.begin() + 2, list.end()));
  EXPECT_EQ(list[0], 1);
  EXPECT_EQ(list[1], 2);
  EXPECT_EQ(list[2], 0);
  EXPECT_FALSE(NextCombination(list.begin(), list.begin() + 2, list.end()));
  EXPECT_EQ(list[0], 0);
  EXPECT_EQ(list[1], 1);
  EXPECT_EQ(list[2], 2);
}

TEST(Sigmoid, Nominal) {
  EXPECT_EQ(Sigmoid(0.0), 0.5);
  EXPECT_NEAR(Sigmoid(100.0), 1.0, 1e-10);
  EXPECT_NEAR(Sigmoid(-100.0), 0, 1e-10);
}

TEST(ScaleSigmoid, Nominal) {
  EXPECT_NEAR(ScaleSigmoid(0.5), 0.5, 1e-10);
  EXPECT_NEAR(ScaleSigmoid(1.0), 1.0, 1e-10);
  EXPECT_NEAR(ScaleSigmoid(-1.0), 0, 1e-4);
}

TEST(NChooseK, Nominal) {
  EXPECT_EQ(NChooseK(1, 0), 1);
  EXPECT_EQ(NChooseK(2, 0), 1);
  EXPECT_EQ(NChooseK(3, 0), 1);

  EXPECT_EQ(NChooseK(1, 1), 1);
  EXPECT_EQ(NChooseK(2, 1), 2);
  EXPECT_EQ(NChooseK(3, 1), 3);

  EXPECT_EQ(NChooseK(2, 2), 1);
  EXPECT_EQ(NChooseK(2, 3), 0);

  EXPECT_EQ(NChooseK(3, 2), 3);
  EXPECT_EQ(NChooseK(4, 2), 6);
  EXPECT_EQ(NChooseK(5, 2), 10);

  EXPECT_EQ(NChooseK(500, 3), 20708500);
}

TEST(TruncateCast, Nominal) {
  EXPECT_EQ((TruncateCast<int, int8_t>(-129)), -128);
  EXPECT_EQ((TruncateCast<int, int8_t>(128)), 127);
  EXPECT_EQ((TruncateCast<int, uint8_t>(-1)), 0);
  EXPECT_EQ((TruncateCast<int, uint8_t>(256)), 255);
  EXPECT_EQ((TruncateCast<int, uint16_t>(-1)), 0);
  EXPECT_EQ((TruncateCast<int, uint16_t>(65536)), 65535);
}

}  // namespace colmap
