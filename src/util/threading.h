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

#ifndef COLMAP_SRC_UTIL_THREADING_
#define COLMAP_SRC_UTIL_THREADING_

#include <atomic>
#include <climits>
#include <functional>
#include <future>
#include <list>
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
//        enum {
//          PROCESSED_CALLBACK,
//        };
//
//        MyThread() { RegisterCallback(PROCESSED_CALLBACK); }
//        void Run() {
//          // Some pre-processing...
//          for (const auto& item : items) {
//            BlockIfPaused();
//            if (IsStopped()) {
//              // Tear down...
//              break;
//            }
//            // Process item...
//            Callback(PROCESSED_CALLBACK);
//          }
//        }
//      };
//
//      MyThread thread;
//      thread.AddCallback(MyThread::PROCESSED_CALLBACK, []() {
//        std::cout << "Processed item"; })
//      thread.AddCallback(MyThread::STARTED_CALLBACK, []() {
//        std::cout << "Start"; })
//      thread.AddCallback(MyThread::FINISHED_CALLBACK, []() {
//        std::cout << "Finished"; })
//      thread.Start();
//      // Pause, resume, stop, ...
//      thread.Wait();
//      thread.Timer().PrintElapsedSeconds();
//
class Thread {
 public:
  enum {
    STARTED_CALLBACK = INT_MIN,
    FINISHED_CALLBACK,
  };

  Thread();
  virtual ~Thread() = default;

  // Control the state of the thread.
  virtual void Start();
  virtual void Stop();
  virtual void Pause();
  virtual void Resume();
  virtual void Wait();

  // Check the state of the thread.
  bool IsStarted();
  bool IsStopped();
  bool IsPaused();
  bool IsRunning();
  bool IsFinished();

  // To be called from inside the main run function. This blocks the main
  // caller, if the thread is paused, until the thread is resumed.
  void BlockIfPaused();

  // Set callbacks that can be triggered within the main run function.
  void AddCallback(const int id, const std::function<void()>& func);

  // Get timing information of the thread, properly accounting for pause times.
  const Timer& GetTimer() const;

 protected:
  // This is the main run function to be implemented by the child class. If you
  // are looping over data and want to support the pause operation, call
  // `BlockIfPaused` at appropriate places in the loop. To support the stop
  // operation, check the `IsStopped` state and early return from this method.
  virtual void Run() = 0;

  // Register a new callback. Note that only registered callbacks can be
  // set/reset and called from within the thread. Hence, this method should be
  // called from the derived thread constructor.
  void RegisterCallback(const int id);

  // Call back to the function with the specified name, if it exists.
  void Callback(const int id) const;

  // Get the unique identifier of the current thread.
  std::thread::id GetThreadId() const;

 private:
  // Wrapper around the main run function to set the finished flag.
  void RunFunc();

  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable pause_condition_;

  Timer timer_;

  std::atomic<bool> started_;
  std::atomic<bool> stopped_;
  std::atomic<bool> paused_;
  std::atomic<bool> pausing_;
  std::atomic<bool> finished_;

  std::unordered_map<int, std::list<std::function<void()>>> callbacks_;
};

// A thread pool class to submit generic tasks (functors) to a pool of workers:
//
//    ThreadPool thread_pool;
//    thread_pool.AddTask([]() { /* Do some work */ });
//    auto future = thread_pool.AddTask([]() { /* Do some work */ return 1; });
//    const auto result = future.get();
//    for (int i = 0; i < 10; ++i) {
//      thread_pool.AddTask([](const int i) { /* Do some work */ });
//    }
//    thread_pool.Wait();
//
class ThreadPool {
 public:
  static const int kMaxNumThreads = -1;

  explicit ThreadPool(const int num_threads = kMaxNumThreads);
  ~ThreadPool();

  inline size_t NumThreads() const;

  // Add new task to the thread pool.
  template <class func_t, class... args_t>
  auto AddTask(func_t&& f, args_t&&... args)
      -> std::future<typename std::result_of<func_t(args_t...)>::type>;

  // Stop the execution of all workers.
  void Stop();

  // Wait until tasks are finished.
  void Wait();

  // Get the unique identifier of the current thread.
  std::thread::id GetThreadId() const;

  // Get the index of the current thread. In a thread pool of size N,
  // the thread index defines the 0-based index of the thread in the pool.
  // In other words, there are the thread indices 0, ..., N-1.
  int GetThreadIndex();

 private:
  void WorkerFunc(const int index);

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;

  std::mutex mutex_;
  std::condition_variable task_condition_;
  std::condition_variable finished_condition_;

  bool stopped_;
  std::atomic<int> num_active_workers_;

  std::unordered_map<std::thread::id, int> thread_id_to_index_;
};

// A job queue class for the producer-consumer paradigm.
//
//    JobQueue<int> job_queue;
//
//    std::thread producer_thread([&job_queue]() {
//      for (int i = 0; i < 10; ++i) {
//        job_queue.Push(i);
//      }
//    });
//
//    std::thread consumer_thread([&job_queue]() {
//      for (int i = 0; i < 10; ++i) {
//        const auto job = job_queue.Pop();
//        if (job.IsValid()) { /* Do some work */ }
//        else { break; }
//      }
//    });
//
//    producer_thread.join();
//    consumer_thread.join();
//
template <typename T>
class JobQueue {
 public:
  class Job {
   public:
    Job() : valid_(false) {}
    explicit Job(const T& data) : data_(data), valid_(true) {}

    // Check whether the data is valid.
    bool IsValid() const { return valid_; }

    // Get reference to the data.
    T& Data() { return data_; }
    const T& Data() const { return data_; }

   private:
    T data_;
    bool valid_;
  };

  JobQueue();
  explicit JobQueue(const size_t max_num_jobs);
  ~JobQueue();

  // The number of pushed and not popped jobs in the queue.
  size_t Size();

  // Push a new job to the queue. Waits if the number of jobs is exceeded.
  bool Push(const T& data);

  // Pop a job from the queue. Waits if there is no job in the queue.
  Job Pop();

  // Wait for all jobs to be popped and then stop the queue.
  void Wait();

  // Stop the queue and return from all push/pop calls with false.
  void Stop();

  // Clear all pushed and not popped jobs from the queue.
  void Clear();

 private:
  size_t max_num_jobs_;
  std::atomic<bool> stop_;
  std::queue<T> jobs_;
  std::mutex mutex_;
  std::condition_variable push_condition_;
  std::condition_variable pop_condition_;
  std::condition_variable empty_condition_;
};

// Return the number of logical CPU cores if num_threads <= 0,
// otherwise return the input value of num_threads.
int GetEffectiveNumThreads(const int num_threads);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t ThreadPool::NumThreads() const { return workers_.size(); }

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

template <typename T>
JobQueue<T>::JobQueue() : JobQueue(std::numeric_limits<size_t>::max()) {}

template <typename T>
JobQueue<T>::JobQueue(const size_t max_num_jobs)
    : max_num_jobs_(max_num_jobs), stop_(false) {}

template <typename T>
JobQueue<T>::~JobQueue() {
  Stop();
}

template <typename T>
size_t JobQueue<T>::Size() {
  std::unique_lock<std::mutex> lock(mutex_);
  return jobs_.size();
}

template <typename T>
bool JobQueue<T>::Push(const T& data) {
  std::unique_lock<std::mutex> lock(mutex_);
  while (jobs_.size() >= max_num_jobs_ && !stop_) {
    pop_condition_.wait(lock);
  }
  if (stop_) {
    return false;
  } else {
    jobs_.push(data);
    push_condition_.notify_one();
    return true;
  }
}

template <typename T>
typename JobQueue<T>::Job JobQueue<T>::Pop() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (jobs_.empty() && !stop_) {
    push_condition_.wait(lock);
  }
  if (stop_) {
    return Job();
  } else {
    const T data = jobs_.front();
    jobs_.pop();
    pop_condition_.notify_one();
    if (jobs_.empty()) {
      empty_condition_.notify_all();
    }
    return Job(data);
  }
}

template <typename T>
void JobQueue<T>::Wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!jobs_.empty()) {
    empty_condition_.wait(lock);
  }
}

template <typename T>
void JobQueue<T>::Stop() {
  stop_ = true;
  push_condition_.notify_all();
  pop_condition_.notify_all();
}

template <typename T>
void JobQueue<T>::Clear() {
  std::unique_lock<std::mutex> lock(mutex_);
  std::queue<T> empty_jobs;
  std::swap(jobs_, empty_jobs);
}

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_THREADING_
