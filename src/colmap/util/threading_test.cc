// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include <gtest/gtest.h>

namespace colmap {
namespace {

namespace {

// Custom implementation of std::barrier that allows us to execute the below
// tests deterministically.
class Barrier {
 public:
  Barrier() : Barrier(2) {}

  explicit Barrier(const size_t count)
      : threshold_(count), count_(count), generation_(0) {}

  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    auto current_generation = generation_;
    if (!--count_) {
      ++generation_;
      count_ = threshold_;
      condition_.notify_all();
    } else {
      condition_.wait(lock, [this, current_generation] {
        return current_generation != generation_;
      });
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  const size_t threshold_;
  size_t count_;
  size_t generation_;
};

}  // namespace

// IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
//            so we use glog's CHECK macros inside threads.

TEST(Thread, Wait) {
  class TestThread : public Thread {
   public:
    Barrier startBarrier;
    Barrier endBarrier;

    void Run() {
      startBarrier.Wait();
      endBarrier.Wait();
    }
  };

  TestThread thread;
  EXPECT_FALSE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.endBarrier.Wait();
  thread.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_TRUE(thread.IsFinished());
}

TEST(Thread, Pause) {
  class TestThread : public Thread {
   public:
    Barrier startBarrier;
    Barrier pauseBarrier;
    Barrier pausedBarrier;
    Barrier resumedBarrier;
    Barrier endBarrier;

    void Run() {
      startBarrier.Wait();
      pauseBarrier.Wait();
      pausedBarrier.Wait();
      BlockIfPaused();
      resumedBarrier.Wait();
      endBarrier.Wait();
    }
  };

  TestThread thread;
  EXPECT_FALSE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.pauseBarrier.Wait();
  thread.Pause();
  thread.pausedBarrier.Wait();
  while (!thread.IsPaused() || thread.IsRunning()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_TRUE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Resume();
  thread.resumedBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.endBarrier.Wait();
  thread.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_TRUE(thread.IsFinished());
}

TEST(Thread, PauseStop) {
  class TestThread : public Thread {
   public:
    Barrier startBarrier;
    Barrier pauseBarrier;
    Barrier pausedBarrier;
    Barrier resumedBarrier;
    Barrier stopBarrier;
    Barrier stoppingBarrier;
    Barrier stoppedBarrier;
    Barrier endBarrier;

    void Run() {
      startBarrier.Wait();
      pauseBarrier.Wait();
      pausedBarrier.Wait();
      BlockIfPaused();
      resumedBarrier.Wait();
      stopBarrier.Wait();
      stoppingBarrier.Wait();

      if (IsStopped()) {
        stoppedBarrier.Wait();
        endBarrier.Wait();
        return;
      }
    }
  };

  TestThread thread;
  EXPECT_FALSE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.pauseBarrier.Wait();
  thread.Pause();
  thread.pausedBarrier.Wait();
  while (!thread.IsPaused() || thread.IsRunning()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_TRUE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Resume();
  thread.resumedBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.stopBarrier.Wait();
  thread.Stop();
  thread.stoppingBarrier.Wait();
  thread.stoppedBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_TRUE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.endBarrier.Wait();
  thread.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_TRUE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_TRUE(thread.IsFinished());
}

TEST(Thread, Restart) {
  class TestThread : public Thread {
   public:
    Barrier startBarrier;
    Barrier endBarrier;

    void Run() {
      startBarrier.Wait();
      endBarrier.Wait();
    }
  };

  TestThread thread;
  EXPECT_FALSE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  for (size_t i = 0; i < 2; ++i) {
    thread.Start();

    thread.startBarrier.Wait();
    EXPECT_TRUE(thread.IsStarted());
    EXPECT_FALSE(thread.IsStopped());
    EXPECT_FALSE(thread.IsPaused());
    EXPECT_TRUE(thread.IsRunning());
    EXPECT_FALSE(thread.IsFinished());

    thread.endBarrier.Wait();
    thread.Wait();
    EXPECT_TRUE(thread.IsStarted());
    EXPECT_FALSE(thread.IsStopped());
    EXPECT_FALSE(thread.IsPaused());
    EXPECT_FALSE(thread.IsRunning());
    EXPECT_TRUE(thread.IsFinished());
  }
}

TEST(Thread, ValidSetup) {
  class TestThread : public Thread {
   public:
    Barrier startBarrier;
    Barrier signalBarrier;
    Barrier endBarrier;

    void Run() {
      startBarrier.Wait();
      SignalValidSetup();
      signalBarrier.Wait();
      endBarrier.Wait();
    }
  };

  TestThread thread;
  EXPECT_FALSE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.signalBarrier.Wait();
  EXPECT_TRUE(thread.CheckValidSetup());

  thread.endBarrier.Wait();
  thread.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_TRUE(thread.IsFinished());
}

TEST(Thread, InvalidSetup) {
  class TestThread : public Thread {
   public:
    Barrier startBarrier;
    Barrier signalBarrier;
    Barrier endBarrier;

    void Run() {
      startBarrier.Wait();
      SignalInvalidSetup();
      signalBarrier.Wait();
      endBarrier.Wait();
    }
  };

  TestThread thread;
  EXPECT_FALSE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.signalBarrier.Wait();
  EXPECT_FALSE(thread.CheckValidSetup());

  thread.endBarrier.Wait();
  thread.Wait();
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsStopped());
  EXPECT_FALSE(thread.IsPaused());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_TRUE(thread.IsFinished());
}

TEST(Thread, Callback) {
  class TestThread : public Thread {
   public:
    enum Callbacks {
      CALLBACK1,
      CALLBACK2,
    };

    TestThread() {
      RegisterCallback(CALLBACK1);
      RegisterCallback(CALLBACK2);
    }

   private:
    void Run() {
      Callback(CALLBACK1);
      Callback(CALLBACK2);
    }
  };

  bool called_back1 = false;
  std::function<void()> CallbackFunc1 = [&called_back1]() {
    called_back1 = true;
  };

  bool called_back2 = false;
  std::function<void()> CallbackFunc2 = [&called_back2]() {
    called_back2 = true;
  };

  bool called_back3 = false;
  std::function<void()> CallbackFunc3 = [&called_back3]() {
    called_back3 = true;
  };

  TestThread thread;
  thread.AddCallback(TestThread::CALLBACK1, CallbackFunc1);
  thread.Start();
  thread.Wait();
  EXPECT_TRUE(called_back1);
  EXPECT_FALSE(called_back2);
  EXPECT_FALSE(called_back3);

  called_back1 = false;
  called_back2 = false;
  thread.AddCallback(TestThread::CALLBACK2, CallbackFunc2);
  thread.Start();
  thread.Wait();
  EXPECT_TRUE(called_back1);
  EXPECT_TRUE(called_back2);
  EXPECT_FALSE(called_back3);

  called_back1 = false;
  called_back2 = false;
  called_back3 = false;
  thread.AddCallback(TestThread::CALLBACK1, CallbackFunc3);
  thread.Start();
  thread.Wait();
  EXPECT_TRUE(called_back1);
  EXPECT_TRUE(called_back2);
  EXPECT_TRUE(called_back3);
}

TEST(Thread, DefaultCallback) {
  class TestThread : public Thread {
   public:
    Barrier startBarrier;
    Barrier signalBarrier;
    Barrier endBarrier;

    void Run() {
      startBarrier.Wait();
      endBarrier.Wait();
    }
  };

  bool called_back1 = false;
  std::function<void()> CallbackFunc1 = [&called_back1]() {
    called_back1 = true;
  };

  bool called_back2 = false;
  std::function<void()> CallbackFunc2 = [&called_back2]() {
    called_back2 = true;
  };

  TestThread thread;
  thread.AddCallback(TestThread::STARTED_CALLBACK, CallbackFunc1);
  thread.AddCallback(TestThread::FINISHED_CALLBACK, CallbackFunc2);
  thread.Start();
  thread.startBarrier.Wait();
  EXPECT_TRUE(called_back1);
  EXPECT_FALSE(called_back2);
  thread.endBarrier.Wait();
  thread.Wait();
  EXPECT_TRUE(called_back1);
  EXPECT_TRUE(called_back2);
}

TEST(ThreadPool, NoArgNoReturn) {
  std::function<void(void)> Func = []() {
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  };

  ThreadPool pool(4);
  std::vector<std::future<void>> futures;
  futures.reserve(100);
  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func));
  }

  for (auto& future : futures) {
    future.get();
  }
}

TEST(ThreadPool, ArgNoReturn) {
  std::function<void(int)> Func = [](int num) {
    for (int i = 0; i < 100; ++i) {
      num += i;
    }
  };

  ThreadPool pool(4);
  std::vector<std::future<void>> futures;
  futures.reserve(100);
  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func, i));
  }

  for (auto& future : futures) {
    future.get();
  }
}

TEST(ThreadPool, NoArgReturn) {
  std::function<int(void)> Func = []() { return 0; };

  ThreadPool pool(4);
  std::vector<std::future<int>> futures;
  futures.reserve(100);
  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func));
  }

  for (auto& future : futures) {
    future.get();
  }
}

TEST(ThreadPool, ArgReturn) {
  std::function<int(int)> Func = [](int num) {
    for (int i = 0; i < 100; ++i) {
      num += i;
    }
    return num;
  };

  ThreadPool pool(4);
  std::vector<std::future<int>> futures;
  futures.reserve(100);
  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func, i));
  }

  for (auto& future : futures) {
    future.get();
  }
}

TEST(ThreadPool, Stop) {
  std::function<int(int)> Func = [](int num) {
    for (int i = 0; i < 100; ++i) {
      num += i;
    }
    return num;
  };

  ThreadPool pool(4);
  std::vector<std::future<int>> futures;
  futures.reserve(100);
  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func, i));
  }

  pool.Stop();

  EXPECT_THROW(pool.AddTask(Func, 100), std::runtime_error);

  pool.Stop();
}

TEST(ThreadPool, Wait) {
  std::vector<uint8_t> results(100, 0);
  std::function<void(int)> Func = [&results](const int num) {
    results[num] = 1;
  };

  ThreadPool pool(4);
  pool.Wait();

  for (size_t i = 0; i < results.size(); ++i) {
    pool.AddTask(Func, i);
  }

  pool.Wait();

  for (const auto result : results) {
    EXPECT_EQ(result, 1);
  }
}

TEST(ThreadPool, WaitEverytime) {
  std::vector<uint8_t> results(4, 0);
  std::function<void(int)> Func = [&results](const int num) {
    results[num] = 1;
  };

  ThreadPool pool(4);

  for (size_t i = 0; i < results.size(); ++i) {
    pool.AddTask(Func, i);
    pool.Wait();

    for (size_t j = 0; j < results.size(); ++j) {
      if (j <= i) {
        EXPECT_EQ(results[j], 1);
      } else {
        EXPECT_EQ(results[j], 0);
      }
    }
  }

  pool.Wait();
}

TEST(ThreadPool, GetThreadIndex) {
  ThreadPool pool(4);

  std::vector<int> results(100, -1);
  std::function<void(int)> Func = [&](const int num) {
    results[num] = pool.GetThreadIndex();
  };

  for (size_t i = 0; i < results.size(); ++i) {
    pool.AddTask(Func, i);
  }

  pool.Wait();

  for (const auto result : results) {
    EXPECT_GE(result, 0);
    EXPECT_LE(result, 3);
  }
}

TEST(JobQueue, SingleProducerSingleConsumer) {
  JobQueue<int> job_queue;

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  std::thread producer_thread([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread consumer_thread([&job_queue]() {
    CHECK_LE(job_queue.Size(), 10);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 10);
    }
  });

  producer_thread.join();
  consumer_thread.join();
}

TEST(JobQueue, SingleProducerSingleConsumerMaxNumJobs) {
  JobQueue<int> job_queue(2);

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  std::thread producer_thread([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread consumer_thread([&job_queue]() {
    CHECK_LE(job_queue.Size(), 2);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 10);
    }
  });

  producer_thread.join();
  consumer_thread.join();
}

TEST(JobQueue, MultipleProducerSingleConsumer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  std::thread producer_thread1([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread producer_thread2([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread consumer_thread([&job_queue]() {
    CHECK_LE(job_queue.Size(), 1);
    for (int i = 0; i < 20; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 10);
    }
  });

  producer_thread1.join();
  producer_thread2.join();
  consumer_thread.join();
}

TEST(JobQueue, SingleProducerMultipleConsumer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  std::thread producer_thread([&job_queue]() {
    for (int i = 0; i < 20; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread consumer_thread1([&job_queue]() {
    CHECK_LE(job_queue.Size(), 1);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 20);
    }
  });

  std::thread consumer_thread2([&job_queue]() {
    CHECK_LE(job_queue.Size(), 1);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 20);
    }
  });

  producer_thread.join();
  consumer_thread1.join();
  consumer_thread2.join();
}

TEST(JobQueue, MultipleProducerMultipleConsumer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  std::thread producer_thread1([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread producer_thread2([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread consumer_thread1([&job_queue]() {
    CHECK_LE(job_queue.Size(), 1);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 10);
    }
  });

  std::thread consumer_thread2([&job_queue]() {
    CHECK_LE(job_queue.Size(), 1);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 10);
    }
  });

  producer_thread1.join();
  producer_thread2.join();
  consumer_thread1.join();
  consumer_thread2.join();
}

TEST(JobQueue, Wait) {
  JobQueue<int> job_queue;

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  for (int i = 0; i < 10; ++i) {
    CHECK(job_queue.Push(i));
  }

  std::thread consumer_thread([&job_queue]() {
    CHECK_EQ(job_queue.Size(), 10);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_EQ(job.Data(), i);
    }
  });

  job_queue.Wait();

  EXPECT_EQ(job_queue.Size(), 0);
  EXPECT_TRUE(job_queue.Push(0));
  EXPECT_TRUE(job_queue.Pop().IsValid());

  consumer_thread.join();
}

TEST(JobQueue, StopProducer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  Barrier stopBarrier;
  std::thread producer_thread([&job_queue, &stopBarrier]() {
    CHECK(job_queue.Push(0));
    stopBarrier.Wait();
    CHECK(!job_queue.Push(0));
  });

  stopBarrier.Wait();
  EXPECT_EQ(job_queue.Size(), 1);

  job_queue.Stop();
  producer_thread.join();

  EXPECT_FALSE(job_queue.Push(0));
  EXPECT_FALSE(job_queue.Pop().IsValid());
}

TEST(JobQueue, StopConsumer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: EXPECT_TRUE_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  EXPECT_TRUE(job_queue.Push(0));

  Barrier pop_barrier;
  std::thread consumer_thread([&job_queue, &pop_barrier]() {
    const auto job = job_queue.Pop();
    CHECK(job.IsValid());
    CHECK_EQ(job.Data(), 0);
    pop_barrier.Wait();
    CHECK(!job_queue.Pop().IsValid());
  });

  pop_barrier.Wait();
  EXPECT_EQ(job_queue.Size(), 0);

  job_queue.Stop();
  consumer_thread.join();

  EXPECT_FALSE(job_queue.Push(0));
  EXPECT_FALSE(job_queue.Pop().IsValid());
}

TEST(JobQueue, Clear) {
  JobQueue<int> job_queue(1);

  EXPECT_TRUE(job_queue.Push(0));
  EXPECT_EQ(job_queue.Size(), 1);

  job_queue.Clear();
  EXPECT_EQ(job_queue.Size(), 0);
}

TEST(GetEffectiveNumThreads, Nominal) {
  EXPECT_GT(GetEffectiveNumThreads(-2), 0);
  EXPECT_GT(GetEffectiveNumThreads(-1), 0);
  EXPECT_GT(GetEffectiveNumThreads(0), 0);
  EXPECT_EQ(GetEffectiveNumThreads(1), 1);
  EXPECT_EQ(GetEffectiveNumThreads(2), 2);
  EXPECT_EQ(GetEffectiveNumThreads(3), 3);
}

}  // namespace
}  // namespace colmap
