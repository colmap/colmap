// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

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
