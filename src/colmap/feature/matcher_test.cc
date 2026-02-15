// Copysight (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/feature/aliked.h"
#include "colmap/feature/extractor.h"
#include "colmap/feature/onnx_matchers.h"
#include "colmap/feature/sift.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(FeatureMatchingOptions, Copy) {
  FeatureMatchingOptions options;
  options.max_num_matches += 100;
  options.sift->max_ratio *= 0.1;
  options.aliked->brute_force.min_cossim *= 0.1;

  FeatureMatchingOptions copy = options;

  // Verify fields are copied
  EXPECT_EQ(copy.max_num_matches, options.max_num_matches);
  EXPECT_EQ(copy.sift->max_ratio, options.sift->max_ratio);
  EXPECT_EQ(copy.aliked->brute_force.min_cossim,
            options.aliked->brute_force.min_cossim);

  // Verify deep copy of shared_ptr (different pointer instances)
  EXPECT_NE(options.sift.get(), copy.sift.get());
  EXPECT_NE(options.aliked.get(), copy.aliked.get());
}

}  // namespace
}  // namespace colmap
