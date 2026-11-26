// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/gpu_mat.h"

namespace colmap {
namespace mvs {

inline constexpr size_t kConsistencyFlagsPerByte = 8;

// Overflow-safe ceil(sources / 8), including zero and all size_t counts.
inline constexpr size_t PackedConsistencyMaskDepth(const size_t sources) {
  return sources / kConsistencyFlagsPerByte +
         (sources % kConsistencyFlagsPerByte != 0);
}

__host__ __device__ inline constexpr uint8_t PackedConsistencyMaskBit(
    const size_t source) {
  return static_cast<uint8_t>(1u << (source % kConsistencyFlagsPerByte));
}

// Non-owning, trivially-copyable kernel view. Exactly one sweep owner thread
// writes each pixel. That owner serially visits all source slots, so byte
// read/modify/write needs no atomic. Never use from parallel source helpers.
struct PackedConsistencyMaskView {
  GpuMatView<uint8_t> bytes;

  __device__ void Set(const size_t row,
                      const size_t col,
                      const size_t source,
                      const uint8_t value) {
    const size_t plane = source / kConsistencyFlagsPerByte;
    const uint8_t bit = PackedConsistencyMaskBit(source);
    uint8_t packed = bytes.Get(row, col, plane);
    packed = value ? static_cast<uint8_t>(packed | bit)
                   : static_cast<uint8_t>(packed & static_cast<uint8_t>(~bit));
    bytes.Set(row, col, plane, packed);
  }
};

}  // namespace mvs
}  // namespace colmap
