// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_tasks.h"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace colmap::mvs::internal {
namespace {

struct FailingTask {
  explicit FailingTask(std::promise<void>& copied) : copied(copied) {}

  FailingTask(const FailingTask& other) : copied(other.copied) {
    copied.set_value();
    throw std::runtime_error("task could not be copied");
  }

  void operator()() const {}

  std::promise<void>& copied;
};

TEST(RunPatchMatchPhase, JoinsBorrowingWorkerOnSubmissionFailure) {
  using namespace std::chrono_literals;
  ThreadPool pool(2);
  std::promise<void> copied;
  std::promise<void> release;
  auto attempted = copied.get_future();
  const auto can_finish = release.get_future().share();
  std::atomic<bool> state_alive = false;

  auto caller = std::async(std::launch::async, [&]() {
    const auto state = std::make_shared<int>(42);
    const std::weak_ptr<int> borrowed = state;
    try {
      RunPatchMatchPhase(pool, [&]() {
        pool.AddTask([&, borrowed]() {
          can_finish.wait();
          const auto owned = borrowed.lock();
          state_alive = owned && *owned == 42;
        });
        pool.AddTask([]() { throw std::runtime_error("worker failed"); });
        const FailingTask bad_task(copied);
        pool.AddTask(bad_task);
      });
    } catch (const std::runtime_error& error) {
      return std::string(error.what());
    }
    return std::string();
  });

  EXPECT_EQ(attempted.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(caller.wait_for(30ms), std::future_status::timeout);
  release.set_value();
  EXPECT_EQ(caller.get(), "task could not be copied");
  // A broken drain may leave a completed worker exception in the pool.
  EXPECT_NO_THROW(pool.Wait());
  EXPECT_TRUE(state_alive);
}

TEST(RunPatchMatchPhase, JoinsRemainingReadsAfterFutureThrows) {
  using namespace std::chrono_literals;
  ThreadPool pool(2);
  std::promise<void> failed;
  std::promise<void> second_started;
  std::promise<void> release;
  auto first_failure = failed.get_future();
  auto started = second_started.get_future();
  const auto can_finish = release.get_future().share();
  std::atomic<bool> second_saw_live_state = false;

  auto caller = std::async(std::launch::async, [&]() {
    const auto state = std::make_shared<int>(42);
    const std::weak_ptr<int> borrowed = state;
    try {
      RunPatchMatchPhase(pool, [&]() {
        const auto first = pool.AddTask(
            []() { throw std::runtime_error("first read failed"); });
        pool.AddTask([&, borrowed]() {
          second_started.set_value();
          can_finish.wait();
          second_saw_live_state = !borrowed.expired();
        });
        try {
          first.get();
        } catch (...) {
          failed.set_value();
          throw;
        }
      });
    } catch (const std::runtime_error& error) {
      return std::string(error.what());
    }
    return std::string();
  });

  EXPECT_EQ(first_failure.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(started.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(caller.wait_for(30ms), std::future_status::timeout);
  release.set_value();
  EXPECT_EQ(caller.get(), "first read failed");
  EXPECT_TRUE(second_saw_live_state);
  EXPECT_NO_THROW(pool.Wait());
}

TEST(RunPatchMatchPhase, PropagatesWorkerFailureAndAllowsReuse) {
  ThreadPool pool(1);
  EXPECT_THROW(RunPatchMatchPhase(pool,
                                  [&]() {
                                    pool.AddTask([]() {
                                      throw std::runtime_error("worker failed");
                                    });
                                  }),
               AggregateException);
  bool executed = false;
  RunPatchMatchPhase(pool, [&]() { pool.AddTask([&]() { executed = true; }); });
  EXPECT_TRUE(executed);
}

}  // namespace
}  // namespace colmap::mvs::internal
