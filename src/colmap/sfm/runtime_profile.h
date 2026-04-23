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

// TEMPORARY profiling instrumentation for incremental_pipeline.cc and
// incremental_mapper.cc. Provides a single global accumulator of per-step
// elapsed wall-clock; the pipeline's `Run()` logs the totals at the end.
// Remove before merging.

#pragma once

#include <chrono>
#include <cstdint>

namespace colmap {

struct RuntimeProfileStat {
  int64_t calls = 0;
  int64_t total_us = 0;
};

struct RuntimeProfile {
  RuntimeProfileStat init;
  RuntimeProfileStat find_next_images;
  RuntimeProfileStat register_next_image;
  RuntimeProfileStat triangulate_image;
  RuntimeProfileStat local_refinement;
  RuntimeProfileStat local_bundle_adjustment;
  RuntimeProfileStat local_merge_tracks;
  RuntimeProfileStat local_complete_tracks;
  RuntimeProfileStat local_complete_image;
  RuntimeProfileStat local_filter;
  RuntimeProfileStat global_refinement;
  RuntimeProfileStat global_bundle_adjustment;
  RuntimeProfileStat global_complete_all_tracks;
  RuntimeProfileStat global_merge_all_tracks;
  RuntimeProfileStat global_retriangulate;
  RuntimeProfileStat global_filter;
  RuntimeProfileStat extract_colors;
  RuntimeProfileStat snapshot;

  void Reset() { *this = RuntimeProfile(); }
};

inline RuntimeProfile& GlobalRuntimeProfile() {
  static RuntimeProfile profile;
  return profile;
}

class ScopedAccumulator {
 public:
  explicit ScopedAccumulator(RuntimeProfileStat& stat)
      : stat_(stat), start_(std::chrono::steady_clock::now()) {}
  ~ScopedAccumulator() {
    stat_.calls += 1;
    stat_.total_us += std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - start_)
                          .count();
  }

 private:
  RuntimeProfileStat& stat_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace colmap
