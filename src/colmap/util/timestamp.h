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

#pragma once

#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <cmath>

namespace colmap {

// Convert a (non-negative) nanosecond timestamp to seconds. Note that
// converting a large absolute timestamp (magnitude > 2^53 ns) to double loses
// sub-nanosecond precision; to difference absolute timestamps use
// TimestampDiffSeconds, which subtracts in int64 first.
inline double SecondsFromTimestamp(timestamp_t t) {
  THROW_CHECK_GE(t, 0);
  return t * 1e-9;
}

// Convert (non-negative) seconds to a nanosecond timestamp, rounding to the
// nearest nanosecond. Intended for small durations (e.g., config values), not
// large absolute timestamps which should be parsed as int64 directly.
inline timestamp_t TimestampFromSeconds(double s) {
  THROW_CHECK_GE(s, 0.0);
  return static_cast<timestamp_t>(std::round(s * 1e9));
}

// Compute the time difference (t1 - t0) in seconds with nanosecond precision.
// Unlike subtracting two large doubles, differencing int64 timestamps and then
// converting preserves full precision. The result may be negative, but both
// timestamps must be valid (non-negative).
inline double TimestampDiffSeconds(timestamp_t t1, timestamp_t t0) {
  THROW_CHECK_GE(t1, 0);
  THROW_CHECK_GE(t0, 0);
  return (t1 - t0) * 1e-9;
}

}  // namespace colmap
