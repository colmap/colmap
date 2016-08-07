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
#include <unordered_map>

#include "util/timer.h"

namespace colmap {

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif

// Define `thread_local` cross-platform.
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

// Helper class to create single threads with simple controls and timing, e.g.:
//
//      class MyThread : public Thread {
//        void Run() {
//          // Some pre-processing...
//          for (const auto& item : items) {
//            WaitIfPaused();
//            if (IsStopped()) {
//              // Tear down...
//              break;
//            }
//            // Process item...
//          }
//          Callback("Finished");
//        }
//      };
//
//      MyThread thread;
//      thread.SetCallback("Finished", []() { std::cout << "Finished"; })
//      thread.Start();
//      // Pause, resume, stop, ...
//      thread.Wait();
//      thread.Timer().PrintElapsedSeconds();
//
class Thread {
 public:
  Thread();
  virtual ~Thread() = default;

  // Control the state of the thread.
  void Start();
  void Stop();
  void Pause();
  void Resume();
  void Wait();

  // Check the state of the thread.
  bool IsStarted();
  bool IsStopped();
  bool IsPaused();
  bool IsRunning();
  bool IsFinished();

  // Set callbacks that can be triggered within the main run function.
  void SetCallback(const std::string& name, const std::function<void()>& func);
  void ResetCallback(const std::string& name);

  // Get timing information of the thread, properly accounting for pause times.
  const Timer& GetTimer() const;

 protected:
  // This is the main run function to be implemented by the child class. If you
  // are looping over data and want to support the pause operation, call
  // `WaitIfPaused` at appropriate places in the loop. To support the stop
  // operation, check the `IsStopped` state and early return from this method.
  virtual void Run() = 0;

  // To be called from inside the main run function. This blocks the main
  // caller, if the thread is paused, until the thread is resumed.
  void WaitIfPaused();

  // Call back to the function with the specified name, if it exists.
  void Callback(const std::string& name) const;

 private:
  // Wrapper around the main run function to set the finished flag.
  void RunFunc();

  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable pause_condition_;

  Timer timer_;

  bool started_;
  bool stopped_;
  bool paused_;
  bool pausing_;
  bool finished_;

  std::unordered_map<std::string, std::function<void()>> callbacks_;
};

class ThreadPool {
 public:
  static const int kMaxNumThreads = -1;

  ThreadPool(const int num_threads = kMaxNumThreads);
  ~ThreadPool();

  inline int NumThreads() const;

  // Add new task to the thread pool.
  template <class func_t, class... args_t>
  auto AddTask(func_t&& f, args_t&&... args)
      -> std::future<typename std::result_of<func_t(args_t...)>::type>;

  // Stop the execution of all workers.
  void Stop();

  // Wait until tasks are finished.
  void Wait();

 private:
  void WorkerFunc();

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;

  std::mutex mutex_;
  std::condition_variable task_condition_;
  std::condition_variable finished_condition_;

  bool stopped_;
  int num_active_workers_;
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
    if (stopped_) {
      throw std::runtime_error("Cannot add task to stopped thread pool.");
    }
    tasks_.emplace([task]() { (*task)(); });
  }

  task_condition_.notify_one();

  return result;
}

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_THREADING_
