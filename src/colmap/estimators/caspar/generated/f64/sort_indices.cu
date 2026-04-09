/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <cub/cub.cuh>

#include "sort_indices.h"

namespace {

__global__ void arange(uint* values, size_t problem_size) {
  const uint global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread_idx >= problem_size) {
    return;
  }
  values[global_thread_idx] = global_thread_idx;
}

__global__ void target_from_argsort(uint* argsort, uint* target, size_t problem_size) {
  const uint global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread_idx >= problem_size) {
    return;
  }
  target[argsort[global_thread_idx]] = global_thread_idx;
}

}  // namespace

namespace caspar {

size_t sort_indices_get_tmp_nbytes(size_t problem_size) {
  size_t tmp_storage_bytes = 0;
  auto err =
      cub::DeviceRadixSort::SortPairs(nullptr, tmp_storage_bytes, (uint*)nullptr, (uint*)nullptr,
                                      (uint*)nullptr, (uint*)nullptr, (uint)problem_size);
  return tmp_storage_bytes;
}

void sort_indices(void* tmp_storage, size_t tmp_storage_bytes, const uint* keys_in,
                  uint* sorted_out, uint* target_out, uint* argsort_out, size_t problem_size) {
  const size_t grid_size = (problem_size + 1023) / 1024;
  arange<<<grid_size, 1024>>>(target_out, problem_size);
  cub::DeviceRadixSort::SortPairs(tmp_storage, tmp_storage_bytes, keys_in, sorted_out, target_out,
                                  argsort_out, (uint)problem_size);
  target_from_argsort<<<grid_size, 1024>>>(argsort_out, target_out, problem_size);
}

size_t sort_keys_get_tmp_nbytes(size_t problem_size) {
  size_t tmp_storage_bytes = 0;
  auto err = cub::DeviceRadixSort::SortKeys(nullptr, tmp_storage_bytes, (uint*)nullptr,
                                            (uint*)nullptr, (uint)problem_size);
  return tmp_storage_bytes;
}

void sort_keys(void* tmp_storage, size_t tmp_storage_bytes, const uint* keys_in, uint* sorted_out,
               size_t problem_size) {
  cub::DeviceRadixSort::SortKeys(tmp_storage, tmp_storage_bytes, keys_in, sorted_out,
                                 (uint)problem_size);
}

__global__ void select_index_kernel(const uint* const input, const uint* const selections,
                                    uint* const output, size_t problem_size) {
  const uint global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread_idx >= problem_size) {
    return;
  }
  output[global_thread_idx] = input[selections[global_thread_idx]];
}

void select_index(const uint* const input, const uint* const selections, uint* const output,
                  size_t problem_size) {
  const size_t grid_size = (problem_size + 1023) / 1024;
  select_index_kernel<<<grid_size, 1024>>>(input, selections, output, problem_size);
}

};  // namespace caspar
