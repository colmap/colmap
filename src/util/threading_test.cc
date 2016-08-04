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

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "util/threading"
#include <boost/test/unit_test.hpp>

#include <chrono>
#include <vector>

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
      WaitIfPaused();
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
      WaitIfPaused();
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

BOOST_AUTO_TEST_CASE(TestThreadTimer) {
  class TestThread : public Thread {
    void Run() {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      WaitIfPaused();
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  };

  TestThread thread;
  thread.Start();
  thread.Wait();
  const auto elapsed_seconds1 = thread.Timer().ElapsedSeconds();
  BOOST_CHECK_GT(elapsed_seconds1, 0.35);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  BOOST_CHECK_EQUAL(thread.Timer().ElapsedSeconds(), elapsed_seconds1);

  thread.Start();
  BOOST_CHECK_LT(thread.Timer().ElapsedSeconds(), elapsed_seconds1);

  thread.Pause();
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  const auto elapsed_seconds2 = thread.Timer().ElapsedSeconds();
  BOOST_CHECK_LT(elapsed_seconds2, 0.225);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  BOOST_CHECK_EQUAL(thread.Timer().ElapsedSeconds(), elapsed_seconds2);

  thread.Resume();
  thread.Wait();
  BOOST_CHECK_GT(thread.Timer().ElapsedSeconds(), elapsed_seconds2);
  BOOST_CHECK_GT(thread.Timer().ElapsedSeconds(), 0.35);
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
  std::vector<bool> results(1000, false);
  std::function<void(int)> Func = [&results](int num) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    results[num] = true;
  };

  {
    ThreadPool pool(4);
    for (int i = 0; i < results.size(); ++i) {
      pool.AddTask(Func, i);
    }
  }

  bool missing_result = false;
  for (const auto result : results) {
    if (!result) {
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
  std::vector<bool> results(1000, false);
  std::function<void(int)> Func = [&results](int num) { results[num] = true; };

  ThreadPool pool(4);
  std::vector<std::future<void>> futures;
  for (int i = 0; i < results.size(); ++i) {
    futures.push_back(pool.AddTask(Func, i));
  }

  pool.Wait();

  for (const auto result : results) {
    BOOST_CHECK_EQUAL(result, true);
  }
}
