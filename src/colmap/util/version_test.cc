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

#include "colmap/util/version.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(MakeDatabaseVersionNumber, Nominal) {
  EXPECT_EQ(MakeDatabaseVersionNumber(3, 14, 0, 0), 3140000);
  EXPECT_EQ(MakeDatabaseVersionNumber(3, 14, 0, 1), 3140001);
  EXPECT_EQ(MakeDatabaseVersionNumber(3, 14, 1, 0), 3140100);
  EXPECT_EQ(MakeDatabaseVersionNumber(3, 15, 0, 0), 3150000);
  EXPECT_EQ(MakeDatabaseVersionNumber(4, 0, 0, 0), 4000000);
  EXPECT_GT(GetDatabaseVersionNumber(), 0);
  EXPECT_ANY_THROW(MakeDatabaseVersionNumber(3, 100, 0, 0));
  EXPECT_ANY_THROW(MakeDatabaseVersionNumber(3, 14, 100, 0));
  EXPECT_ANY_THROW(MakeDatabaseVersionNumber(3, 14, 0, 100));
}

TEST(GetVersionInfo, Nominal) {
  const std::string version = GetVersionInfo();
  EXPECT_FALSE(version.empty());
  EXPECT_THAT(version, testing::HasSubstr("COLMAP"));
}

TEST(GetBuildInfo, Nominal) {
  const std::string build_info = GetBuildInfo();
  EXPECT_FALSE(build_info.empty());
  EXPECT_THAT(build_info, testing::HasSubstr("Commit"));
}

}  // namespace
}  // namespace colmap
