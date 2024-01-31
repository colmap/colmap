// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/util/threading.h"

#include "colmap/util/logging.h"

namespace colmap {

Thread::Thread()
    : started_(false),
      stopped_(false),
      paused_(false),
      pausing_(false),
      finished_(false),
      setup_(false),
      setup_valid_(false) {
  RegisterCallback(STARTED_CALLBACK);
  RegisterCallback(FINISHED_CALLBACK);
}

void Thread::Start() {
  std::unique_lock<std::mutex> lock(mutex_);
  THROW_CHECK(!started_ || finished_);
  Wait();
  timer_.Restart();
  thread_ = std::thread(&Thread::RunFunc, this);
  started_ = true;
  stopped_ = false;
  paused_ = false;
  pausing_ = false;
  finished_ = false;
  setup_ = false;
  setup_valid_ = false;
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

void Thread::Wait() {
  if (thread_.joinable()) {
    thread_.join();
  }
}

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

void Thread::AddCallback(const int id, const std::function<void()>& func) {
  THROW_CHECK(func);
  THROW_CHECK_GT(callbacks_.count(id), 0) << "Callback not registered";
  callbacks_.at(id).push_back(func);
}

void Thread::RegisterCallback(const int id) {
  callbacks_.emplace(id, std::list<std::function<void()>>());
}

void Thread::Callback(const int id) const {
  THROW_CHECK_GT(callbacks_.count(id), 0) << "Callback not registered";
  for (const auto& callback : callbacks_.at(id)) {
    callback();
  }
}

std::thread::id Thread::GetThreadId() const {
  return std::this_thread::get_id();
}

void Thread::SignalValidSetup() {
  std::unique_lock<std::mutex> lock(mutex_);
  THROW_CHECK(!setup_);
  setup_ = true;
  setup_valid_ = true;
  setup_condition_.notify_all();
}

void Thread::SignalInvalidSetup() {
  std::unique_lock<std::mutex> lock(mutex_);
  THROW_CHECK(!setup_);
  setup_ = true;
  setup_valid_ = false;
  setup_condition_.notify_all();
}

const class Timer& Thread::GetTimer() const { return timer_; }

void Thread::BlockIfPaused() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (paused_) {
    pausing_ = true;
    timer_.Pause();
    pause_condition_.wait(lock);
    pausing_ = false;
    timer_.Resume();
  }
}

bool Thread::CheckValidSetup() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!setup_) {
    setup_condition_.wait(lock);
  }
  return setup_valid_;
}

void Thread::RunFunc() {
  Callback(STARTED_CALLBACK);
  Run();
  {
    std::unique_lock<std::mutex> lock(mutex_);
    finished_ = true;
    timer_.Pause();
  }
  Callback(FINISHED_CALLBACK);
}

ThreadPool::ThreadPool(const int num_threads)
    : stopped_(false), num_active_workers_(0) {
  const int num_effective_threads = GetEffectiveNumThreads(num_threads);
  for (int index = 0; index < num_effective_threads; ++index) {
    std::function<void(void)> worker =
        std::bind(&ThreadPool::WorkerFunc, this, index);
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

    std::queue<std::function<void()>> empty_tasks;
    std::swap(tasks_, empty_tasks);
  }

  task_condition_.notify_all();

  for (auto& worker : workers_) {
    worker.join();
  }

  finished_condition_.notify_all();
}

void ThreadPool::Wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!tasks_.empty() || num_active_workers_ > 0) {
    finished_condition_.wait(
        lock, [this]() { return tasks_.empty() && num_active_workers_ == 0; });
  }
}

void ThreadPool::WorkerFunc(const int index) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    thread_id_to_index_.emplace(GetThreadId(), index);
  }

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

    {
      std::unique_lock<std::mutex> lock(mutex_);
      num_active_workers_ -= 1;
    }

    finished_condition_.notify_all();
  }
}

std::thread::id ThreadPool::GetThreadId() const {
  return std::this_thread::get_id();
}

int ThreadPool::GetThreadIndex() {
  std::unique_lock<std::mutex> lock(mutex_);
  return thread_id_to_index_.at(GetThreadId());
}

int GetEffectiveNumThreads(const int num_threads) {
  int num_effective_threads = num_threads;
  if (num_threads <= 0) {
    num_effective_threads = std::thread::hardware_concurrency();
  }

  if (num_effective_threads <= 0) {
    num_effective_threads = 1;
  }

  return num_effective_threads;
}

}  // namespace colmap
