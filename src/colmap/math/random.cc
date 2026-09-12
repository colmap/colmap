// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/math/random.h"

#include <mutex>

namespace colmap {

thread_local std::unique_ptr<std::mt19937> PRNG;

int kDefaultPRNGSeed = 0;

void SetPRNGSeed(unsigned seed) {
  PRNG = std::make_unique<std::mt19937>(seed);
  // srand is not thread-safe.
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);
  srand(seed);
}

}  // namespace colmap
