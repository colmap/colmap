// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/threading.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>

namespace colmap::mvs::internal {

struct DelaunayWorkerDrain {
  void operator()(ThreadPool& pool) const { pool.Wait(); }
};

// Bound the pool to the image count, draining all accepted tasks before
// returning or propagating an exception. Submission is injectable so callers
// can exercise failure after work has already entered the pool.
template <typename ImageOperation,
          typename Submit,
          typename Drain = DelaunayWorkerDrain>
void RunDelaunayImageWorkers(const size_t num_images,
                             const int num_threads,
                             ImageOperation&& integrate_image,
                             Submit&& submit,
                             Drain drain = {}) {
  if (num_images == 0) return;

  std::atomic<size_t> next_image_idx{0};
  auto integrate_images = [&]() {
    try {
      while (true) {
        const size_t image_idx =
            next_image_idx.fetch_add(1, std::memory_order_relaxed);
        if (image_idx >= num_images) return;
        integrate_image(image_idx);
      }
    } catch (...) {
      next_image_idx.store(num_images, std::memory_order_relaxed);
      throw;
    }
  };

  // The pool dies before the state and lambdas referenced by its tasks.
  const size_t num_workers = std::min(
      num_images, static_cast<size_t>(GetEffectiveNumThreads(num_threads)));
  ThreadPool thread_pool(static_cast<int>(num_workers));
  try {
    for (size_t i = 0; i < num_workers; ++i) {
      submit(thread_pool, integrate_images);
    }
    drain(thread_pool);
  } catch (...) {
    const auto error = std::current_exception();
    next_image_idx.store(num_images, std::memory_order_relaxed);
    try {
      drain(thread_pool);
    } catch (...) {
      // Preserve the original failure; Wait has consumed task exceptions.
      std::rethrow_exception(error);
    }
    std::rethrow_exception(error);
  }
}

}  // namespace colmap::mvs::internal
