// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/timer.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Timer, Default) {
  Timer timer;
  EXPECT_EQ(timer.ElapsedMicroSeconds(), 0);
  EXPECT_EQ(timer.ElapsedSeconds(), 0);
  EXPECT_EQ(timer.ElapsedMinutes(), 0);
  EXPECT_EQ(timer.ElapsedHours(), 0);
}

TEST(Timer, Start) {
  Timer timer;
  timer.Start();
  EXPECT_GE(timer.ElapsedMicroSeconds(), 0);
  EXPECT_GE(timer.ElapsedSeconds(), 0);
  EXPECT_GE(timer.ElapsedMinutes(), 0);
  EXPECT_GE(timer.ElapsedHours(), 0);
}

TEST(Timer, Pause) {
  Timer timer;
  timer.Start();
  timer.Pause();
  double prev_time = timer.ElapsedMicroSeconds();
  for (size_t i = 0; i < 1000; ++i) {
    EXPECT_EQ(timer.ElapsedMicroSeconds(), prev_time);
    prev_time = timer.ElapsedMicroSeconds();
  }
  timer.Resume();
  for (size_t i = 0; i < 1000; ++i) {
    EXPECT_GE(timer.ElapsedMicroSeconds(), prev_time);
  }
  timer.Reset();
  EXPECT_EQ(timer.ElapsedMicroSeconds(), 0);
}

}  // namespace
}  // namespace colmap
