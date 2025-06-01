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

#include "colmap/feature/torch_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(GetDeviceName, Nominal) {
  EXPECT_THAT(GetDeviceName(/*use_gpu=*/true, /*gpu_index=*/""),
              testing::AnyOf("cpu", "cuda"));
  EXPECT_THAT(GetDeviceName(/*use_gpu=*/true, /*gpu_index=*/"-1"),
              testing::AnyOf("cpu", "cuda"));
  EXPECT_THAT(GetDeviceName(/*use_gpu=*/true, /*gpu_index=*/"2"),
              testing::AnyOf("cpu", "cuda:2"));
  EXPECT_EQ(GetDeviceName(/*use_gpu=*/false, /*gpu_index=*/""), "cpu");
  EXPECT_EQ(GetDeviceName(/*use_gpu=*/false, /*gpu_index=*/"-1"), "cpu");
  EXPECT_EQ(GetDeviceName(/*use_gpu=*/false, /*gpu_index=*/"0,1,2"), "cpu");
}

}  // namespace
}  // namespace colmap
