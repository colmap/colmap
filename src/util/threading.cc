// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "util/threading.h"

#include "util/logging.h"

namespace colmap {

Thread::Thread()
    : started_(false),
      stopped_(false),
      paused_(false),
      pausing_(false),
      finished_(false) {}

void Thread::Start() {
  std::unique_lock<std::mutex> lock(mutex_);
  CHECK(!started_ || finished_);
  timer_.Restart();
  thread_ = std::thread(&Thread::RunFunc, this);
  started_ = true;
  stopped_ = false;
  paused_ = false;
  pausing_ = false;
  finished_ = false;
}

void Thread::Stop() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    stopped_ = true;
  }
  Resume();
}

void Thread::Pause() {
  std::unique_lock<std::mutex> lock(mutex_);
  paused_ = true;
}

void Thread::Resume() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (paused_) {
    paused_ = false;
    pause_condition_.notify_all();
  }
}

void Thread::Wait() { thread_.join(); }

bool Thread::IsStarted() {
  std::unique_lock<std::mutex> lock(mutex_);
  return started_;
}

bool Thread::IsStopped() {
  std::unique_lock<std::mutex> lock(mutex_);
  return stopped_;
}

bool Thread::IsPaused() {
  std::unique_lock<std::mutex> lock(mutex_);
  return paused_;
}

bool Thread::IsRunning() {
  std::unique_lock<std::mutex> lock(mutex_);
  return started_ && !pausing_ && !finished_;
}

bool Thread::IsFinished() {
  std::unique_lock<std::mutex> lock(mutex_);
  return finished_;
}

const class Timer& Thread::Timer() const { return timer_; }

void Thread::WaitIfPaused() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (paused_) {
    pausing_ = true;
    timer_.Pause();
    pause_condition_.wait(lock);
    pausing_ = false;
    timer_.Resume();
  }
}

void Thread::RunFunc() {
  Run();
  {
    std::unique_lock<std::mutex> lock(mutex_);
    finished_ = true;
    timer_.Pause();
  }
}

ThreadPool::ThreadPool(const int num_threads)
    : stopped_(false), num_active_workers_(0) {
  int num_effective_threads = num_threads;
  if (num_threads == kMaxNumThreads) {
    num_effective_threads = std::thread::hardware_concurrency();
  }

  if (num_effective_threads <= 0) {
    num_effective_threads = 1;
  }

  for (int i = 0; i < num_effective_threads; ++i) {
    std::function<void(void)> worker = std::bind(&ThreadPool::WorkerFunc, this);
    workers_.emplace_back(worker);
  }
}

ThreadPool::~ThreadPool() { Stop(); }

void ThreadPool::Stop() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopped_) {
      return;
    }
    stopped_ = true;
  }

  {
    std::queue<std::function<void()>> empty_tasks;
    std::swap(tasks_, empty_tasks);
  }

  task_condition_.notify_all();
  for (auto& worker : workers_) {
    worker.join();
  }
}

void ThreadPool::Wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  finished_condition_.wait(
      lock, [this]() { return tasks_.empty() && num_active_workers_ == 0; });
}

void ThreadPool::WorkerFunc() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      task_condition_.wait(lock,
                           [this] { return stopped_ || !tasks_.empty(); });
      if (stopped_ && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
      num_active_workers_ += 1;
    }

    task();

    num_active_workers_ -= 1;
    finished_condition_.notify_one();
  }
}

}  // namespace colmap
