// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <cmath>

namespace colmap {

// Timestamps whose magnitude exceeds this bound (2^53 ns) can no longer be
// represented exactly as double, so converting them to seconds loses
// sub-nanosecond precision; differences within the bound stay exact.
constexpr timestamp_t kMaxStableTimestamp = timestamp_t{1} << 53;

// Convert a nanosecond timestamp to seconds. Note that converting a large
// absolute timestamp (magnitude > 2^53 ns) to double loses sub-nanosecond
// precision; to difference absolute timestamps use TimestampDiffSeconds, which
// subtracts in int64 first.
inline double SecondsFromTimestamp(timestamp_t t) {
  VLOG_IF(2, t > kMaxStableTimestamp || t < -kMaxStableTimestamp)
      << "Converting timestamp " << t
      << " ns to seconds loses sub-nanosecond precision (magnitude > 2^53); "
         "use TimestampDiffSeconds for precise differences.";
  return t * 1e-9;
}

// Convert seconds to a nanosecond timestamp, rounding to the nearest
// nanosecond. Intended for small durations (e.g., config values), not large
// absolute timestamps which should be parsed as int64 directly.
inline timestamp_t TimestampFromSeconds(double s) {
  return static_cast<timestamp_t>(std::round(s * 1e9));
}

// Compute the time difference (t1 - t0) in seconds with nanosecond precision.
// Unlike subtracting two large doubles, differencing int64 timestamps and then
// converting preserves full precision. The result may be negative.
inline double TimestampDiffSeconds(timestamp_t t1, timestamp_t t0) {
  return SecondsFromTimestamp(t1 - t0);
}

}  // namespace colmap
