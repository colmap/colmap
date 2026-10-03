// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/math/math.h"

namespace colmap {

// Implementation based on: https://blog.plover.com/math/choose.html
uint64_t NChooseK(uint64_t n, uint64_t k) {
  if (n == 0 || n < k) {
    return 0;
  }

  uint64_t r = 1;
  for (uint64_t d = 1; d <= k; ++d) {
    r *= n--;
    r /= d;
  }
  return r;
}

}  // namespace colmap
