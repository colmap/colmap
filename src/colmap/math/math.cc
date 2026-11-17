// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/math/math.h"

#include <numeric>

namespace colmap {

// Implementation based on: https://blog.plover.com/math/choose.html
uint64_t NChooseK(uint64_t n, uint64_t k) {
  if (n == 0 || n < k) {
    return 0;
  }

  k = std::min(k, n - k);
  uint64_t r = 1;
  for (uint64_t d = 1; d <= k; ++d) {
    const uint64_t divisor = std::gcd(n, d);
    // Cancel the denominator before multiplying to avoid intermediate overflow.
    r /= d / divisor;
    r *= n-- / divisor;
  }
  return r;
}

}  // namespace colmap
