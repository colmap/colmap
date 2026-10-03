// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/math/random.h"

#include <glog/logging.h>
#include <gmock/gmock.h>

namespace {

constexpr unsigned kDefaultTestPRNGSeed = 0;

class PRNGTestEventListener : public testing::EmptyTestEventListener {
 public:
  void OnTestStart(const testing::TestInfo&) override {
    colmap::SetPRNGSeed(kDefaultTestPRNGSeed);
  }
};

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleMock(&argc, argv);
  testing::UnitTest::GetInstance()->listeners().Append(
      new PRNGTestEventListener);
  return RUN_ALL_TESTS();
}
