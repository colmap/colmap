// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/math/math.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(SignOfNumber, Nominal) {
  EXPECT_EQ(SignOfNumber(0), 1);
  EXPECT_EQ(SignOfNumber(-0.1), -1);
  EXPECT_EQ(SignOfNumber(0.1), 1);
  EXPECT_EQ(SignOfNumber(std::numeric_limits<float>::infinity()), 1);
  EXPECT_EQ(SignOfNumber(-std::numeric_limits<float>::infinity()), -1);
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

TEST(RadToDeg, Roundtrip) {
  for (int i = 0; i < 360; ++i) {
    const auto angle = static_cast<double>(i);
    EXPECT_NEAR(angle, RadToDeg(DegToRad(angle)), 1e-6);
  }
}

TEST(Median, Nominal) {
  EXPECT_EQ(Median<int>({1, 2, 3, 4}), 2.5);
  EXPECT_EQ(Median<int>({4, 1, 3, 2}), 2.5);
  EXPECT_EQ(Median<int>({1, 2, 3, 100}), 2.5);
  EXPECT_EQ(Median<int>({1, 2, 3, 4, 100}), 3);
  EXPECT_EQ(Median<int>({4, 100, 1, 3, 2}), 3);
  EXPECT_EQ(Median<int>({-100, 1, 2, 3, 4}), 2);
  EXPECT_EQ(Median<int>({-1, -2, -3, -4}), -2.5);
  EXPECT_EQ(Median<int>({-3, -1, -4, -2}), -2.5);
  EXPECT_EQ(Median<int>({-1, -2, 3, 4}), 1);
  // Test integer overflow scenario.
  EXPECT_EQ(Median<int8_t>({100, 115, 119, 127}), 117);
}

TEST(MedianAbsoluteDeviation, Nominal) {
  // {1, 2, 3, 4, 5} -> median=3, deviations={2, 1, 0, 1, 2}, MAD=1
  auto [median1, mad1] = MedianAbsoluteDeviation<int>({1, 2, 3, 4, 5});
  EXPECT_EQ(median1, 3);
  EXPECT_EQ(mad1, 1);

  // {1, 2, 3, 4} -> median=2.5, deviations={1.5, 0.5, 0.5, 1.5}, MAD=1
  auto [median2, mad2] = MedianAbsoluteDeviation<int>({1, 2, 3, 4});
  EXPECT_EQ(median2, 2.5);
  EXPECT_EQ(mad2, 1);

  // Unsorted input: {5, 1, 3, 2, 4} -> same as {1, 2, 3, 4, 5}
  auto [median3, mad3] = MedianAbsoluteDeviation<int>({5, 1, 3, 2, 4});
  EXPECT_EQ(median3, 3);
  EXPECT_EQ(mad3, 1);

  // Single element: {42} -> median=42, MAD=0
  auto [median4, mad4] = MedianAbsoluteDeviation<int>({42});
  EXPECT_EQ(median4, 42);
  EXPECT_EQ(mad4, 0);

  // With outlier: {1, 2, 3, 4, 100} -> median=3, deviations={2, 1, 0, 1, 97}
  auto [median5, mad5] = MedianAbsoluteDeviation<int>({1, 2, 3, 4, 100});
  EXPECT_EQ(median5, 3);
  EXPECT_EQ(mad5, 1);
}

TEST(Percentile, Nominal) {
  EXPECT_EQ((Percentile(std::vector<int>{0}, 0)), 0);
  EXPECT_EQ((Percentile(std::vector<int>{0}, 50)), 0);
  EXPECT_EQ((Percentile(std::vector<int>{0}, 100)), 0);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1}, 0)), 0);
  EXPECT_EQ((Percentile(std::vector<int>{1, 0}, 0)), 0);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1}, 50)), 0.5);
  EXPECT_EQ((Percentile(std::vector<int>{1, 0}, 50)), 0.5);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1}, 100)), 1);
  EXPECT_EQ((Percentile(std::vector<int>{1, 0}, 100)), 1);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1, 2}, 0)), 0);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1, 2}, 50)), 1);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1, 2}, 100)), 2);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1, 1, 2}, 0)), 0);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1, 1, 2}, 100. / 3.)), 1);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1, 1, 2}, 50)), 1);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1, 1, 2}, 100 / 3. * 2.)), 1);
  EXPECT_EQ((Percentile(std::vector<int>{1, 2, 0, 1}, 100 / 3. * 2.)), 1);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1, 1, 2}, 100)), 2);
  EXPECT_EQ((Percentile(std::vector<int>{1, 2, 0, 1}, 100)), 2);
  EXPECT_EQ((Percentile(std::vector<int>{0, 100}, 1)), 1);
  EXPECT_EQ((Percentile(std::vector<int>{0, 100}, 50)), 50);
  EXPECT_EQ((Percentile(std::vector<int>{0, 100}, 50.1)), 50.1);
  EXPECT_EQ((Percentile(std::vector<int>{0, 100}, 99)), 99);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1, 2, 3}, 1)), 0.03);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1, 2, 3}, 2)), 0.06);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1, 2, 3}, 33)), 0.99);
  EXPECT_EQ((Percentile(std::vector<int>{0, 1, 2, 3}, 34)), 1.02);
  EXPECT_EQ((Percentile(std::vector<int>{3, 0, 1, 2}, 34)), 1.02);
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
  EXPECT_EQ(NChooseK(0, 0), 0);

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
  EXPECT_EQ(NChooseK(500, 7), 1486071034734000);
  EXPECT_EQ(NChooseK(10000, 5), 832500291625002000);
}

TEST(NChooseK, AvoidIntermediateOverflow) {
  EXPECT_EQ(NChooseK(66, 33), 7219428434016265740ULL);
  EXPECT_EQ(NChooseK(67, 33), 14226520737620288370ULL);
  EXPECT_EQ(NChooseK(100, 99), 100);
  EXPECT_EQ(NChooseK(100, 90), 17310309456440ULL);
  EXPECT_EQ(NChooseK(1000, 993), 194280608456793000ULL);
}

TEST(TruncateCast, Nominal) {
  EXPECT_EQ((TruncateCast<int, int8_t>(-129)), -128);
  EXPECT_EQ((TruncateCast<int, int8_t>(128)), 127);
  EXPECT_EQ((TruncateCast<int, uint8_t>(-1)), 0);
  EXPECT_EQ((TruncateCast<int, uint8_t>(256)), 255);
  EXPECT_EQ((TruncateCast<int, uint16_t>(-1)), 0);
  EXPECT_EQ((TruncateCast<int, uint16_t>(65536)), 65535);
  EXPECT_EQ((TruncateCast<float, uint8_t>(-1.0f)), 0);
  EXPECT_EQ((TruncateCast<float, uint8_t>(42.75f)), 42);
  EXPECT_EQ((TruncateCast<float, uint8_t>(256.0f)), 255);
}

TEST(TruncateCast, Signedness) {
  EXPECT_EQ((TruncateCast<uint16_t, int16_t>(1234)), 1234);
  EXPECT_EQ((TruncateCast<uint16_t, int16_t>(65535)), 32767);
  EXPECT_EQ((TruncateCast<int16_t, uint16_t>(1234)), 1234);
  EXPECT_EQ((TruncateCast<int16_t, uint16_t>(-1)), 0);
  EXPECT_EQ((TruncateCast<int64_t, uint64_t>(42)), 42u);
  EXPECT_EQ((TruncateCast<uint64_t, int64_t>(42)), 42);
  EXPECT_EQ(
      (TruncateCast<uint64_t, int64_t>(std::numeric_limits<uint64_t>::max())),
      std::numeric_limits<int64_t>::max());
}

TEST(TruncateCast, Widening) {
  EXPECT_EQ((TruncateCast<int8_t, int16_t>(-5)), -5);
  EXPECT_EQ((TruncateCast<int16_t, int32_t>(-1234)), -1234);
  EXPECT_EQ((TruncateCast<int16_t, int32_t>(1234)), 1234);
  EXPECT_EQ((TruncateCast<uint8_t, uint16_t>(42)), 42);
  EXPECT_EQ(
      (TruncateCast<uint32_t, int64_t>(std::numeric_limits<uint32_t>::max())),
      std::numeric_limits<uint32_t>::max());
}

TEST(TruncateCast, FloatingDestination) {
  EXPECT_EQ((TruncateCast<double, float>(-1.0)), -1.0f);
  EXPECT_EQ((TruncateCast<double, float>(0.0)), 0.0f);
  EXPECT_EQ((TruncateCast<double, float>(-1e300)),
            std::numeric_limits<float>::lowest());
  EXPECT_EQ((TruncateCast<double, float>(1e300)),
            std::numeric_limits<float>::max());
  EXPECT_EQ((TruncateCast<float, double>(-1.0f)), -1.0);
  EXPECT_EQ((TruncateCast<float, double>(0.0f)), 0.0);
  EXPECT_EQ((TruncateCast<int, float>(-42)), -42.0f);
  EXPECT_EQ((TruncateCast<int, double>(42)), 42.0);
  const float subnormal = std::numeric_limits<float>::denorm_min();
  EXPECT_EQ((TruncateCast<double, float>(subnormal)), subnormal);
}

TEST(TruncateCast, FloatingIntegerEndpoints) {
  EXPECT_EQ((TruncateCast<float, int32_t>(2147483648.0f)),
            std::numeric_limits<int32_t>::max());
  EXPECT_EQ((TruncateCast<double, int64_t>(9223372036854775808.0)),
            std::numeric_limits<int64_t>::max());
  EXPECT_EQ((TruncateCast<double, uint64_t>(18446744073709551616.0)),
            std::numeric_limits<uint64_t>::max());
  EXPECT_EQ((TruncateCast<double, int64_t>(
                std::nextafter(9223372036854775808.0, 0.0))),
            9223372036854774784LL);
}

TEST(TruncateCast, PreserveIntegerPrecision) {
  const int64_t signed_value = std::numeric_limits<int64_t>::max() - 1;
  const uint64_t unsigned_value = std::numeric_limits<uint64_t>::max() - 1;
  EXPECT_EQ((TruncateCast<int64_t, int64_t>(signed_value)), signed_value);
  EXPECT_EQ((TruncateCast<int64_t, uint64_t>(signed_value)),
            static_cast<uint64_t>(signed_value));
  EXPECT_EQ((TruncateCast<uint64_t, uint64_t>(unsigned_value)), unsigned_value);
}

}  // namespace
}  // namespace colmap
