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

#include "util/timer.h"

#include "util/logging.h"
#include "util/misc.h"

using namespace std::chrono;

namespace colmap {

Timer::Timer() : started_(false), paused_(false) {}

void Timer::Start() {
  started_ = true;
  paused_ = false;
  start_time_ = high_resolution_clock::now();
}

void Timer::Restart() {
  started_ = false;
  Start();
}

void Timer::Pause() {
  paused_ = true;
  pause_time_ = high_resolution_clock::now();
}

void Timer::Resume() {
  paused_ = false;
  start_time_ += high_resolution_clock::now() - pause_time_;
}

void Timer::Reset() {
  started_ = false;
  paused_ = false;
}

double Timer::ElapsedMicroSeconds() const {
  if (!started_) {
    return 0.0;
  }
  if (paused_) {
    return duration_cast<microseconds>(pause_time_ - start_time_).count();
  } else {
    return duration_cast<microseconds>(high_resolution_clock::now() -
                                       start_time_)
        .count();
  }
}

double Timer::ElapsedSeconds() const { return ElapsedMicroSeconds() / 1e6; }

double Timer::ElapsedMinutes() const { return ElapsedSeconds() / 60; }

double Timer::ElapsedHours() const { return ElapsedMinutes() / 60; }

void Timer::PrintSeconds() const {
  std::cout << StringPrintf("Elapsed time: %.5f [seconds]", ElapsedSeconds())
            << std::endl;
}

void Timer::PrintMinutes() const {
  std::cout << StringPrintf("Elapsed time: %.3f [minutes]", ElapsedMinutes())
            << std::endl;
}

void Timer::PrintHours() const {
  std::cout << StringPrintf("Elapsed time: %.3f [hours]", ElapsedHours())
            << std::endl;
}

}  // namespace colmap
