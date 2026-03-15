/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace caspar {

/**
 * Struct used in shared index operations in a cuda block. See memops.cuh for usage.
 *
 * There is one shared index per thread in the block.
 * @param unique: The unique indices accessed by the thread block in sorted order.
 *    Does not belong to a particular thread.
 * @param target: The target index of the thread. The following acces is valid (but inefficient):
 *    loat var = data[shared_indices_out[shared_indices_out[threadIdx.x].target].unique];
 * @param argsort: Indices making the access pattern sorted. i.e. the following is sorted:
 *    sorted_data[threadIdx.x] = shared_indices_out[shared_indices_out[threadIdx.x].argsort].unique
 *    This is used to for better warp coalescing in reduction operations.
 */
struct SharedIndex {
  uint32_t unique;
  uint16_t target;
  uint16_t argsort;
};

void shared_indices(const unsigned int* const indices, SharedIndex* const shared_indices_out,
                    const unsigned int size);

};  // namespace caspar
