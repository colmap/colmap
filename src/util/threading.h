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

#ifndef COLMAP_SRC_UTIL_THREADING_
#define COLMAP_SRC_UTIL_THREADING_

#include <atomic>
#include <functional>
#include <future>
#include <queue>

namespace colmap {

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif

// Define `thread_local` cross-platform
#ifndef thread_local
#if __STDC_VERSION__ >= 201112 && !defined __STDC_NO_THREADS__
#define thread_local _Thread_local
#elif defined _WIN32 && (defined _MSC_VER || defined __ICL || \
                         defined __DMC__ || defined __BORLANDC__)
#define thread_local __declspec(thread)
#elif defined __GNUC__ || defined __SUNPRO_C || defined __xlC__
#define thread_local __thread
#else
#error "Cannot define thread_local"
#endif
#endif

#ifdef __clang__
#pragma clang diagnostic pop  // -Wkeyword-macro
#endif

class ThreadPool {
 public:
  static const int kMaxNumThreads = -1;

  ThreadPool(const int num_threads = kMaxNumThreads);
  ~ThreadPool();

  inline int NumThreads() const;

  // Add new task to thread pool.
  template <class func_t, class... args_t>
  auto AddTask(func_t&& f, args_t&&... args)
      -> std::future<typename std::result_of<func_t(args_t...)>::type>;

  // Stop the execution of all workers.
  void Stop();

 private:
  void WorkerFunc();

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;

  std::mutex mutex_;
  std::condition_variable condition_;
  bool stop_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

int ThreadPool::NumThreads() const { return workers_.size(); }

template <class func_t, class... args_t>
auto ThreadPool::AddTask(func_t&& f, args_t&&... args)
    -> std::future<typename std::result_of<func_t(args_t...)>::type> {
  typedef typename std::result_of<func_t(args_t...)>::type return_t;

  auto task = std::make_shared<std::packaged_task<return_t()>>(
      std::bind(std::forward<func_t>(f), std::forward<args_t>(args)...));

  std::future<return_t> result = task->get_future();

  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stop_) {
      throw std::runtime_error("Cannot add task to stopped thread pool.");
    }
    tasks_.emplace([task]() { (*task)(); });
  }

  condition_.notify_one();

  return result;
}

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_THREADING_
