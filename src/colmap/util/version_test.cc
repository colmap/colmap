// SPDX-License-Identifier: BSD-3-Clause

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
