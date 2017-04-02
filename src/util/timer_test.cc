// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#define TEST_NAME "util/timer"
#include "util/testing.h"

#include "util/timer.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestDefault) {
  Timer timer;
  BOOST_CHECK_EQUAL(timer.ElapsedMicroSeconds(), 0);
  BOOST_CHECK_EQUAL(timer.ElapsedSeconds(), 0);
  BOOST_CHECK_EQUAL(timer.ElapsedMinutes(), 0);
  BOOST_CHECK_EQUAL(timer.ElapsedHours(), 0);
}

BOOST_AUTO_TEST_CASE(TestStart) {
  Timer timer;
  timer.Start();
  BOOST_CHECK_GE(timer.ElapsedMicroSeconds(), 0);
  BOOST_CHECK_GE(timer.ElapsedSeconds(), 0);
  BOOST_CHECK_GE(timer.ElapsedMinutes(), 0);
  BOOST_CHECK_GE(timer.ElapsedHours(), 0);
}

BOOST_AUTO_TEST_CASE(TestPause) {
  Timer timer;
  timer.Start();
  timer.Pause();
  double prev_time = timer.ElapsedMicroSeconds();
  for (size_t i = 0; i < 1000; ++i) {
    BOOST_CHECK_EQUAL(timer.ElapsedMicroSeconds(), prev_time);
    prev_time = timer.ElapsedMicroSeconds();
  }
  timer.Resume();
  for (size_t i = 0; i < 1000; ++i) {
    BOOST_CHECK_GE(timer.ElapsedMicroSeconds(), prev_time);
  }
  timer.Reset();
  BOOST_CHECK_EQUAL(timer.ElapsedMicroSeconds(), 0);
}
