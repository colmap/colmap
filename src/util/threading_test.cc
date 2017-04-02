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

#define TEST_NAME "util/threading"
#include "util/testing.h"

#include "util/logging.h"
#include "util/threading.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestThreadWait) {
  class TestThread : public Thread {
    void Run() { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }
  };

  TestThread thread;
  BOOST_CHECK(!thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Start();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(thread.IsFinished());
}

BOOST_AUTO_TEST_CASE(TestThreadPause) {
  class TestThread : public Thread {
    void Run() {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      BlockIfPaused();
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  };

  TestThread thread;
  BOOST_CHECK(!thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Start();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Pause();
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Resume();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(thread.IsFinished());
}

BOOST_AUTO_TEST_CASE(TestThreadStop) {
  class TestThread : public Thread {
    void Run() {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      if (IsStopped()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(thread.IsFinished());
}

BOOST_AUTO_TEST_CASE(TestThreadPauseStop) {
  class TestThread : public Thread {
    void Run() {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      BlockIfPaused();
      if (IsStopped()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Pause();
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  thread.Wait();
  BOOST_CHECK(thread.IsStarted());
  BOOST_CHECK(thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(thread.IsFinished());
}

BOOST_AUTO_TEST_CASE(TestThreadRestart) {
  class TestThread : public Thread {
    void Run() { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }
  };

  TestThread thread;
  BOOST_CHECK(!thread.IsStarted());
  BOOST_CHECK(!thread.IsStopped());
  BOOST_CHECK(!thread.IsPaused());
  BOOST_CHECK(!thread.IsRunning());
  BOOST_CHECK(!thread.IsFinished());

  for (size_t i = 0; i < 2; ++i) {
    thread.Start();
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Wait();
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(thread.IsFinished());
  }
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
   private:
    void Run() { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
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
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  BOOST_CHECK(called_back1);
  BOOST_CHECK(!called_back2);
  thread.Wait();
  BOOST_CHECK(called_back1);
  BOOST_CHECK(called_back2);
}

BOOST_AUTO_TEST_CASE(TestThreadTimer) {
  class TestThread : public Thread {
    void Run() {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      BlockIfPaused();
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  };

  TestThread thread;
  thread.Start();
  thread.Wait();
  const auto elapsed_seconds1 = thread.GetTimer().ElapsedSeconds();
  BOOST_CHECK_GT(elapsed_seconds1, 0.35);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  BOOST_CHECK_EQUAL(thread.GetTimer().ElapsedSeconds(), elapsed_seconds1);

  thread.Start();
  BOOST_CHECK_LT(thread.GetTimer().ElapsedSeconds(), elapsed_seconds1);

  thread.Pause();
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  const auto elapsed_seconds2 = thread.GetTimer().ElapsedSeconds();
  BOOST_CHECK_LT(elapsed_seconds2, 0.225);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  BOOST_CHECK_EQUAL(thread.GetTimer().ElapsedSeconds(), elapsed_seconds2);

  thread.Resume();
  thread.Wait();
  BOOST_CHECK_GT(thread.GetTimer().ElapsedSeconds(), elapsed_seconds2);
  BOOST_CHECK_GT(thread.GetTimer().ElapsedSeconds(), 0.35);
}

BOOST_AUTO_TEST_CASE(TestThreadPoolNoArgNoReturn) {
  std::function<void(void)> Func = []() {
    int num = 0;
    for (int i = 0; i < 1000; ++i) {
      num += i;
    }
  };

  ThreadPool pool(4);
  std::vector<std::future<void>> futures;

  for (int i = 0; i < 1000; ++i) {
    futures.push_back(pool.AddTask(Func));
  }

  for (auto& future : futures) {
    future.get();
  }
}

BOOST_AUTO_TEST_CASE(TestThreadPoolArgNoReturn) {
  std::function<void(int)> Func = [](int num) {
    for (int i = 0; i < 1000; ++i) {
      num += i;
    }
  };

  ThreadPool pool(4);
  std::vector<std::future<void>> futures;

  for (int i = 0; i < 1000; ++i) {
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

  for (int i = 0; i < 1000; ++i) {
    futures.push_back(pool.AddTask(Func));
  }

  for (auto& future : futures) {
    future.get();
  }
}

BOOST_AUTO_TEST_CASE(TestThreadPoolArgReturn) {
  std::function<int(int)> Func = [](int num) {
    for (int i = 0; i < 1000; ++i) {
      num += i;
    }
    return num;
  };

  ThreadPool pool(4);
  std::vector<std::future<int>> futures;

  for (int i = 0; i < 1000; ++i) {
    futures.push_back(pool.AddTask(Func, i));
  }

  for (auto& future : futures) {
    future.get();
  }
}

BOOST_AUTO_TEST_CASE(TestThreadPoolDestructor) {
  std::vector<uint8_t> results(1000, 0);
  std::function<void(int)> Func = [&results](const int num) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    results[num] = 1;
  };

  {
    ThreadPool pool(4);
    for (size_t i = 0; i < results.size(); ++i) {
      pool.AddTask(Func, i);
    }
  }

  bool missing_result = false;
  for (const auto result : results) {
    if (result == 0) {
      missing_result = true;
      break;
    }
  }

  BOOST_CHECK(missing_result);
}

BOOST_AUTO_TEST_CASE(TestThreadPoolStop) {
  std::function<int(int)> Func = [](int num) {
    for (int i = 0; i < 1000; ++i) {
      num += i;
    }
    return num;
  };

  ThreadPool pool(4);
  std::vector<std::future<int>> futures;

  for (int i = 0; i < 1000; ++i) {
    futures.push_back(pool.AddTask(Func, i));
  }

  pool.Stop();

  BOOST_CHECK_THROW(pool.AddTask(Func, 1000), std::runtime_error);

  pool.Stop();
}

BOOST_AUTO_TEST_CASE(TestThreadPoolWait) {
  std::vector<uint8_t> results(1000, 0);
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

BOOST_AUTO_TEST_CASE(TestThreadPoolGetThreadIndex) {
  ThreadPool pool(4);

  std::vector<int> results(1000, -1);
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

  std::thread producer_thread([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread consumer_thread([&job_queue]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK_EQ(job_queue.Size(), 10);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_EQ(job.Data(), i);
    }
  });

  producer_thread.join();
  consumer_thread.join();
}

BOOST_AUTO_TEST_CASE(TestJobQueueSingleProducerSingleConsumerMaxNumJobs) {
  JobQueue<int> job_queue(2);

  std::thread producer_thread([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread consumer_thread([&job_queue]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK_EQ(job_queue.Size(), 2);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_EQ(job.Data(), i);
    }
  });

  producer_thread.join();
  consumer_thread.join();
}

BOOST_AUTO_TEST_CASE(TestJobQueueMultipleProducerSingleConsumer) {
  JobQueue<int> job_queue(1);

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
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK_EQ(job_queue.Size(), 1);
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

  std::thread producer_thread([&job_queue]() {
    for (int i = 0; i < 20; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread consumer_thread1([&job_queue]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK_LE(job_queue.Size(), 1);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 20);
    }
  });

  std::thread consumer_thread2([&job_queue]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK_LE(job_queue.Size(), 1);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_LT(job.Data(), 10);
    }
  });

  std::thread consumer_thread2([&job_queue]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
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

  std::thread producer_thread([&job_queue]() {
    for (int i = 0; i < 10; ++i) {
      CHECK(job_queue.Push(i));
    }
  });

  std::thread consumer_thread([&job_queue]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK_EQ(job_queue.Size(), 10);
    for (int i = 0; i < 10; ++i) {
      const auto job = job_queue.Pop();
      CHECK(job.IsValid());
      CHECK_EQ(job.Data(), i);
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  job_queue.Wait();

  BOOST_CHECK_EQUAL(job_queue.Size(), 0);
  BOOST_CHECK(job_queue.Push(0));
  BOOST_CHECK(job_queue.Pop().IsValid());

  producer_thread.join();
  consumer_thread.join();
}

BOOST_AUTO_TEST_CASE(TestJobQueueStopProducer) {
  JobQueue<int> job_queue(1);

  std::thread producer_thread([&job_queue]() {
    CHECK(job_queue.Push(0));
    CHECK(!job_queue.Push(0));
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  BOOST_CHECK_EQUAL(job_queue.Size(), 1);

  job_queue.Stop();
  producer_thread.join();

  BOOST_CHECK(!job_queue.Push(0));
  BOOST_CHECK(!job_queue.Pop().IsValid());
}

BOOST_AUTO_TEST_CASE(TestJobQueueStopConsumer) {
  JobQueue<int> job_queue(1);

  BOOST_CHECK(job_queue.Push(0));

  std::thread consumer_thread([&job_queue]() {
    const auto job = job_queue.Pop();
    CHECK(job.IsValid());
    CHECK_EQ(job.Data(), 0);
    CHECK(!job_queue.Pop().IsValid());
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
