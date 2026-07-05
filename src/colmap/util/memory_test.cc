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

#include "colmap/util/memory.h"

#include <cstring>
#include <memory>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// On supported platforms the peak RSS should be a plausible non-zero value.
// On unsupported platforms the helper returns 0, which we tolerate.
TEST(Memory, PeakRSSIsPlausibleOrZero) {
  const size_t peak = GetPeakRSSBytes();
#if defined(_WIN32) || defined(__APPLE__) || defined(__linux__)
  EXPECT_GT(peak, 0u);
  // A running test process uses more than 1 MB but far less than 1 TB.
  EXPECT_LT(peak, size_t{1} << 40);
#else
  EXPECT_EQ(peak, 0u);
#endif
}

TEST(Memory, PeakRSSDoesNotDecreaseAfterLargeAllocation) {
#if defined(_WIN32) || defined(__APPLE__) || defined(__linux__)
  const size_t before = GetPeakRSSBytes();
  ASSERT_GT(before, 0u);

  // Allocate and touch ~256 MB so the pages become resident.
  constexpr size_t kNumBytes = size_t{256} << 20;
  auto buffer = std::make_unique<char[]>(kNumBytes);
  std::memset(buffer.get(), 1, kNumBytes);
  // Prevent the compiler from optimizing the allocation away.
  volatile char sink = buffer[kNumBytes - 1];
  (void)sink;

  const size_t after = GetPeakRSSBytes();
  // Peak RSS is a high-water mark, so it must never decrease.
  EXPECT_GE(after, before);
#else
  GTEST_SKIP() << "GetPeakRSSBytes not supported on this platform.";
#endif
}

TEST(Memory, CurrentRSSIsPlausibleOrZero) {
  const size_t current = GetCurrentRSSBytes();
#if defined(_WIN32) || defined(__APPLE__) || defined(__linux__)
  EXPECT_GT(current, 0u);
  EXPECT_LE(current, GetPeakRSSBytes());
#else
  EXPECT_EQ(current, 0u);
#endif
}

}  // namespace
}  // namespace colmap
