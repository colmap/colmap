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

#include "colmap/util/timestamp.h"

#include <map>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Timestamp, TimestampToSeconds) {
  EXPECT_DOUBLE_EQ(TimestampToSeconds(0), 0.0);
  EXPECT_DOUBLE_EQ(TimestampToSeconds(1000000000), 1.0);
  EXPECT_DOUBLE_EQ(TimestampToSeconds(500000000), 0.5);
  EXPECT_DOUBLE_EQ(TimestampToSeconds(-1000000000), -1.0);
}

TEST(Timestamp, SecondsToTimestamp) {
  EXPECT_EQ(SecondsToTimestamp(0.0), 0);
  EXPECT_EQ(SecondsToTimestamp(1.0), 1000000000);
  EXPECT_EQ(SecondsToTimestamp(0.5), 500000000);
  EXPECT_EQ(SecondsToTimestamp(0.005), 5000000);  // 5ms.
}

TEST(Timestamp, TimestampDiffSeconds) {
  timestamp_t t0 = 1000000000;  // 1s.
  timestamp_t t1 = 1005000000;  // 1.005s.
  EXPECT_DOUBLE_EQ(TimestampDiffSeconds(t1, t0), 0.005);
  EXPECT_DOUBLE_EQ(TimestampDiffSeconds(t0, t1), -0.005);
}

TEST(Timestamp, SecondsToTimestampPrecision) {
  // SecondsToTimestamp is only used for config durations (small values),
  // never for large absolute timestamps (which are parsed as int64 directly).
  // Verify nanosecond-level precision for realistic config values.
  EXPECT_EQ(SecondsToTimestamp(0.005), 5000000);
  EXPECT_EQ(SecondsToTimestamp(0.25), 250000000);
  EXPECT_EQ(SecondsToTimestamp(9.81), 9810000000LL);

  // Verify that the conversion truncates (not rounds) sub-nanosecond values.
  EXPECT_EQ(SecondsToTimestamp(0.1), 100000000);

  // Differences of int64 timestamps converted back to seconds preserve
  // nanosecond precision, unlike subtracting two large doubles.
  timestamp_t t0 = 1403636579763555584LL;
  timestamp_t t1 = t0 + SecondsToTimestamp(0.005);
  EXPECT_DOUBLE_EQ(TimestampDiffSeconds(t1, t0), 0.005);
}

TEST(Timestamp, LargeNanosecondValues) {
  // Typical EuRoC timestamp: 1403636579763555584 ns.
  timestamp_t t = 1403636579763555584LL;
  double s = TimestampToSeconds(t);
  EXPECT_NEAR(s, 1403636579.763555584, 1e-3);

  // Difference between two timestamps (5ms apart).
  timestamp_t t2 = t + 5000000;  // +5ms.
  EXPECT_DOUBLE_EQ(TimestampDiffSeconds(t2, t), 0.005);
}

TEST(Timestamp, MapKeyExactEquality) {
  // int64 map keys support exact lookup, unlike double keys.
  std::map<timestamp_t, int> m;
  timestamp_t key = 1403636579763555584LL;
  m[key] = 42;
  EXPECT_NE(m.find(key), m.end());
  EXPECT_EQ(m.at(key), 42);
  m.erase(key);
  EXPECT_EQ(m.find(key), m.end());
}

TEST(Timestamp, InvalidTimestamp) {
  EXPECT_EQ(kInvalidTimestamp, -1);
  EXPECT_LT(kInvalidTimestamp, 0);
}

}  // namespace
}  // namespace colmap
