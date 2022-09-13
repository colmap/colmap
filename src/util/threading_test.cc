// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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

#define TEST_NAME "util/threading"
#include "util/testing.h"

#include "util/logging.h"
#include "util/threading.h"

using namespace colmap;

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

// IMPORTANT: BOOST_CHECK_* macros are not thread-safe,
//            so we use glog's CHECK macros inside threads.

BOOST_AUTO_TEST_CASE(TestThreadWait) {
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
  BOOST_CHECK(!thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.endBarrier.Wait();
  thread.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(thread.IsFinished());
}

BOOST_AUTO_TEST_CASE(TestThreadPause) {
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
  BOOST_CHECK(!thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.pauseBarrier.Wait();
  thread.Pause();
  thread.pausedBarrier.Wait();
  while (!thread.IsPaused() || thread.IsRunning()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Resume();
  thread.resumedBarrier.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.endBarrier.Wait();
  thread.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(thread.IsFinished());
}

BOOST_AUTO_TEST_CASE(TestThreadPauseStop) {
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
  BOOST_CHECK(!thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.pauseBarrier.Wait();
  thread.Pause();
  thread.pausedBarrier.Wait();
  while (!thread.IsPaused() || thread.IsRunning()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Resume();
  thread.resumedBarrier.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.stopBarrier.Wait();
  thread.Stop();
  thread.stoppingBarrier.Wait();
  thread.stoppedBarrier.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.endBarrier.Wait();
  thread.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(thread.IsFinished());
}

BOOST_AUTO_TEST_CASE(TestThreadRestart) {
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
  BOOST_CHECK(!thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  for (size_t i = 0; i < 2; ++i) {
    thread.Start();

    thread.startBarrier.Wait();
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.endBarrier.Wait();
    thread.Wait();
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(thread.IsFinished());
  }
}

BOOST_AUTO_TEST_CASE(TestThreadValidSetup) {
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
  BOOST_CHECK(!thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.signalBarrier.Wait();
  BOOST_CHECK(thread.CheckValidSetup());

  thread.endBarrier.Wait();
  thread.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(thread.IsFinished());
}

BOOST_AUTO_TEST_CASE(TestThreadInvalidSetup) {
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
  BOOST_CHECK(!thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Start();

  thread.startBarrier.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.signalBarrier.Wait();
  BOOST_CHECK(!thread.CheckValidSetup());

  thread.endBarrier.Wait();
  thread.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(thread.IsFinished());
}

BOOST_AUTO_TEST_CASE(TestCallback) {
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
  BOOST_CHECK(called_back1);
  BOOST_CHECK(!called_back2);
  BOOST_CHECK(!called_back3);

  called_back1 = false;
  called_back2 = false;
  thread.AddCallback(TestThread::CALLBACK2, CallbackFunc2);
  thread.Start();
  thread.Wait();
  BOOST_CHECK(called_back1);
  BOOST_CHECK(called_back2);
  BOOST_CHECK(!called_back3);

  called_back1 = false;
  called_back2 = false;
  called_back3 = false;
  thread.AddCallback(TestThread::CALLBACK1, CallbackFunc3);
  thread.Start();
  thread.Wait();
  BOOST_CHECK(called_back1);
  BOOST_CHECK(called_back2);
  BOOST_CHECK(called_back3);
}

BOOST_AUTO_TEST_CASE(TestDefaultCallback) {
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
  BOOST_CHECK(called_back1);
  BOOST_CHECK(!called_back2);
  thread.endBarrier.Wait();
  thread.Wait();
  BOOST_CHECK(called_back1);
  BOOST_CHECK(called_back2);
}

BOOST_AUTO_TEST_CASE(TestThreadPoolNoArgNoReturn) {
  std::function<void(void)> Func = []() {
    int num = 0;
    for (int i = 0; i < 100; ++i) {
      num += i;
    }
  };

  ThreadPool pool(4);
  std::vector<std::future<void>> futures;

  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func));
  }

  for (auto& future : futures) {
    future.get();
  }
}

BOOST_AUTO_TEST_CASE(TestThreadPoolArgNoReturn) {
  std::function<void(int)> Func = [](int num) {
    for (int i = 0; i < 100; ++i) {
      num += i;
    }
  };

  ThreadPool pool(4);
  std::vector<std::future<void>> futures;

  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func, i));
  }

  for (auto& future : futures) {
    future.get();
  }
}

BOOST_AUTO_TEST_CASE(TestThreadPoolNoArgReturn) {
  std::function<int(void)> Func = []() { return 0; };

  ThreadPool pool(4);
  std::vector<std::future<int>> futures;

  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func));
  }

  for (auto& future : futures) {
    future.get();
  }
}

BOOST_AUTO_TEST_CASE(TestThreadPoolArgReturn) {
  std::function<int(int)> Func = [](int num) {
    for (int i = 0; i < 100; ++i) {
      num += i;
    }
    return num;
  };

  ThreadPool pool(4);
  std::vector<std::future<int>> futures;

  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func, i));
  }

  for (auto& future : futures) {
    future.get();
  }
}

BOOST_AUTO_TEST_CASE(TestThreadPoolStop) {
  std::function<int(int)> Func = [](int num) {
    for (int i = 0; i < 100; ++i) {
      num += i;
    }
    return num;
  };

  ThreadPool pool(4);
  std::vector<std::future<int>> futures;

  for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.AddTask(Func, i));
  }

  pool.Stop();

  BOOST_CHECK_THROW(pool.AddTask(Func, 100), std::runtime_error);

  pool.Stop();
}

BOOST_AUTO_TEST_CASE(TestThreadPoolWait) {
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
    BOOST_CHECK_EQUAL(result, 1);
  }
}

BOOST_AUTO_TEST_CASE(TestThreadPoolWaitEverytime) {
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
        BOOST_CHECK_EQUAL(results[j], 1);
      } else {
        BOOST_CHECK_EQUAL(results[j], 0);
      }
    }
  }

  pool.Wait();
}

BOOST_AUTO_TEST_CASE(TestThreadPoolGetThreadIndex) {
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
    BOOST_CHECK_GE(result, 0);
    BOOST_CHECK_LE(result, 3);
  }
}

BOOST_AUTO_TEST_CASE(TestJobQueueSingleProducerSingleConsumer) {
  JobQueue<int> job_queue;

  // IMPORTANT: BOOST_CHECK_* macros are not thread-safe,
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

BOOST_AUTO_TEST_CASE(TestJobQueueSingleProducerSingleConsumerMaxNumJobs) {
  JobQueue<int> job_queue(2);

  // IMPORTANT: BOOST_CHECK_* macros are not thread-safe,
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

BOOST_AUTO_TEST_CASE(TestJobQueueMultipleProducerSingleConsumer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: BOOST_CHECK_* macros are not thread-safe,
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

BOOST_AUTO_TEST_CASE(TestJobQueueSingleProducerMultipleConsumer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: BOOST_CHECK_* macros are not thread-safe,
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

BOOST_AUTO_TEST_CASE(TestJobQueueMultipleProducerMultipleConsumer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: BOOST_CHECK_* macros are not thread-safe,
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

BOOST_AUTO_TEST_CASE(TestJobQueueWait) {
  JobQueue<int> job_queue;

  // IMPORTANT: BOOST_CHECK_* macros are not thread-safe,
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

  BOOST_CHECK_EQUAL(job_queue.Size(), 0);
  BOOST_CHECK(job_queue.Push(0));
  BOOST_CHECK(job_queue.Pop().IsValid());

  consumer_thread.join();
}

BOOST_AUTO_TEST_CASE(TestJobQueueStopProducer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: BOOST_CHECK_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  Barrier stopBarrier;
  std::thread producer_thread([&job_queue, &stopBarrier]() {
    CHECK(job_queue.Push(0));
    stopBarrier.Wait();
    CHECK(!job_queue.Push(0));
  });

  stopBarrier.Wait();
  BOOST_CHECK_EQUAL(job_queue.Size(), 1);

  job_queue.Stop();
  producer_thread.join();

  BOOST_CHECK(!job_queue.Push(0));
  BOOST_CHECK(!job_queue.Pop().IsValid());
}

BOOST_AUTO_TEST_CASE(TestJobQueueStopConsumer) {
  JobQueue<int> job_queue(1);

  // IMPORTANT: BOOST_CHECK_* macros are not thread-safe,
  //            so we use glog's CHECK macros inside threads.

  BOOST_CHECK(job_queue.Push(0));

  Barrier popBarrier;
  std::thread consumer_thread([&job_queue, &popBarrier]() {
    const auto job = job_queue.Pop();
    CHECK(job.IsValid());
    CHECK_EQ(job.Data(), 0);
    popBarrier.Wait();
    CHECK(!job_queue.Pop().IsValid());
  });

  popBarrier.Wait();
  BOOST_CHECK_EQUAL(job_queue.Size(), 0);

  job_queue.Stop();
  consumer_thread.join();

  BOOST_CHECK(!job_queue.Push(0));
  BOOST_CHECK(!job_queue.Pop().IsValid());
}

BOOST_AUTO_TEST_CASE(TestJobQueueClear) {
  JobQueue<int> job_queue(1);

  BOOST_CHECK(job_queue.Push(0));
  BOOST_CHECK_EQUAL(job_queue.Size(), 1);

  job_queue.Clear();
  BOOST_CHECK_EQUAL(job_queue.Size(), 0);
}

BOOST_AUTO_TEST_CASE(TestGetEffectiveNumThreads) {
  BOOST_CHECK_GT(GetEffectiveNumThreads(-2), 0);
  BOOST_CHECK_GT(GetEffectiveNumThreads(-1), 0);
  BOOST_CHECK_GT(GetEffectiveNumThreads(0), 0);
  BOOST_CHECK_EQUAL(GetEffectiveNumThreads(1), 1);
  BOOST_CHECK_EQUAL(GetEffectiveNumThreads(2), 2);
  BOOST_CHECK_EQUAL(GetEffectiveNumThreads(3), 3);
}
