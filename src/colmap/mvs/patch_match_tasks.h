// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/threading.h"

#include <utility>

namespace colmap::mvs::internal {

// Workers may borrow controller state. Join accepted tasks even when
// constructing or enqueueing a later task throws.
template <typename Submit>
void RunPatchMatchPhase(ThreadPool& pool, Submit&& submit) {
  try {
    std::forward<Submit>(submit)();
  } catch (...) {
    try {
      pool.Wait();
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // Preserve the initiating submission failure.
    }
    throw;
  }
  pool.Wait();
}

}  // namespace colmap::mvs::internal
