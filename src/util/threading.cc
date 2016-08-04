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

#include <iostream>

namespace colmap {

ThreadPool::ThreadPool(const int num_threads)
    : stop_(false), num_active_workers_(0) {
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

ThreadPool::~ThreadPool() {
  Stop();
}

void ThreadPool::Stop() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stop_) {
      return;
    }
    stop_ = true;
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
      task_condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty()) {
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
