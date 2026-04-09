/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <stdio.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "shared_indices.h"

namespace cg = cooperative_groups;

// READ INDEXED
namespace caspar {

template <uint block_size, typename KernelT, typename StorageT, typename VecT>
__forceinline__ __device__ void read_idx_1(const StorageT* const input, const uint offset,
                                           const uint idx, KernelT& x) {
  x = input[offset + idx];
}

template <uint block_size, typename KernelT, typename StorageT, typename VecT>
__forceinline__ __device__ void read_idx_2(const StorageT* const input, const uint offset,
                                           const uint idx, KernelT& x, KernelT& y) {
  const VecT tmp = *reinterpret_cast<const VecT*>(&input[offset + idx * 2]);
  x = tmp.x;
  y = tmp.y;
}

template <uint block_size, typename KernelT, typename StorageT, typename VecT>
__forceinline__ __device__ void read_idx_3(const StorageT* const input, const uint offset,
                                           const uint idx, KernelT& x, KernelT& y, KernelT& z) {
  const VecT tmp = *reinterpret_cast<const VecT*>(&input[offset + idx * 4]);
  x = tmp.x;
  y = tmp.y;
  z = tmp.z;
}

template <uint block_size, typename KernelT, typename StorageT, typename VecT>
__forceinline__ __device__ void read_idx_4(const StorageT* const input, const uint offset,
                                           const uint idx, KernelT& x, KernelT& y, KernelT& z,
                                           KernelT& w) {
  const VecT tmp = *reinterpret_cast<const VecT*>(&input[offset + idx * 4]);
  x = tmp.x;
  y = tmp.y;
  z = tmp.z;
  w = tmp.w;
}
// READ INDEXED WITH DEFAULT
template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void read_idx_with_default_1(const StorageT* const input,
                                                        const uint offset, const uint num_alloc,
                                                        const int idx, KernelT& x,
                                                        const KernelT x_default) {
  if (idx >= 0 && idx < num_alloc) {
    x = input[offset + idx];
  } else {
    x = x_default;
  }
}

template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void read_idx_with_default_2(const StorageT* const input,
                                                        const uint offset, const uint num_alloc,
                                                        const int idx, KernelT& x, KernelT& y,
                                                        KernelT x_default, KernelT y_default) {
  VecT tmp;
  if (idx >= 0 && idx < num_alloc) {
    tmp = *reinterpret_cast<const VecT*>(&input[offset + idx * 2]);
    x = tmp.x;
    y = tmp.y;
  } else {
    x = x_default;
    y = y_default;
  }
}

template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void read_idx_with_default_3(const StorageT* const input,
                                                        const uint offset, const uint num_alloc,
                                                        const int idx, KernelT& x, KernelT& y,
                                                        KernelT& z, KernelT x_default,
                                                        KernelT y_default, KernelT z_default) {
  VecT tmp;
  if (idx >= 0 && idx < num_alloc) {
    tmp = *reinterpret_cast<const VecT*>(&input[offset + idx * 4]);
    x = tmp.x;
    y = tmp.y;
    z = tmp.z;
  } else {
    x = x_default;
    y = y_default;
    z = z_default;
  }
}

template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void read_idx_with_default_4(const StorageT* const input,
                                                        const uint offset, const uint num_alloc,
                                                        const int idx, KernelT& x, KernelT& y,
                                                        KernelT& z, KernelT& w, KernelT x_default,
                                                        KernelT y_default, KernelT z_default,
                                                        KernelT w_default) {
  VecT tmp;
  if (idx >= 0 && idx < num_alloc) {
    tmp = *reinterpret_cast<const VecT*>(&input[offset + idx * 4]);
    x = tmp.x;
    y = tmp.y;
    z = tmp.z;
    w = tmp.w;
  } else {
    x = x_default;
    y = y_default;
    z = z_default;
    w = w_default;
  }
}

// WRITE INDEXED

template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void write_idx_1(StorageT* const output, const uint offset,
                                            const uint idx, const KernelT x) {
  output[offset + idx] = x;
}

template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void write_idx_2(StorageT* const output, const uint offset,
                                            const uint idx, const KernelT x, const KernelT y) {
  VecT tmp;
  tmp.x = x;
  tmp.y = y;
  *reinterpret_cast<VecT*>(&output[offset + idx * 2]) = tmp;
}

template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void write_idx_3(StorageT* const output, const uint offset,
                                            const uint idx, const KernelT x, const KernelT y,
                                            const KernelT z) {
  VecT tmp;
  tmp.x = x;
  tmp.y = y;
  tmp.z = z;
  *reinterpret_cast<VecT*>(&output[offset + idx * 4]) = tmp;
}

template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void write_idx_4(StorageT* const output, const uint offset,
                                            const uint idx, const KernelT x, const KernelT y,
                                            const KernelT z, const KernelT w) {
  VecT tmp;
  tmp.x = x;
  tmp.y = y;
  tmp.z = z;
  tmp.w = w;
  *reinterpret_cast<VecT*>(&output[offset + idx * 4]) = tmp;
}

// WRITE ADD
template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void add_idx_1(StorageT* const output, const uint offset, const uint idx,
                                          const KernelT x) {
  output[offset + idx] += x;
}

template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void add_idx_2(StorageT* const output, const uint offset, const uint idx,
                                          const KernelT x, const KernelT y) {
  const VecT existing = *reinterpret_cast<const VecT*>(&output[offset + idx * 2]);
  VecT tmp;
  tmp.x = x + existing.x;
  tmp.y = y + existing.y;
  *reinterpret_cast<VecT*>(&output[offset + idx * 2]) = tmp;
}

template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void add_idx_3(StorageT* const output, const uint offset, const uint idx,
                                          const KernelT x, const KernelT y, const KernelT z) {
  const VecT existing = *reinterpret_cast<const VecT*>(&output[offset + idx * 4]);
  VecT tmp;
  tmp.x = x + existing.x;
  tmp.y = y + existing.y;
  tmp.z = z + existing.z;
  *reinterpret_cast<VecT*>(&output[offset + idx * 4]) = tmp;
}

template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void add_idx_4(StorageT* const output, const uint offset, const uint idx,
                                          const KernelT x, const KernelT y, const KernelT z,
                                          const KernelT w) {
  const VecT existing = *reinterpret_cast<const VecT*>(&output[offset + idx * 4]);
  VecT tmp;
  tmp.x = x + existing.x;
  tmp.y = y + existing.y;
  tmp.z = z + existing.z;
  tmp.w = w + existing.w;
  *reinterpret_cast<VecT*>(&output[offset + idx * 4]) = tmp;
}

// WRITE SUM
/**
 * The write_sum_x writes data to shared local data.
 * flush_sum or flush_sum_block should be called after all write_sum_x calls to perform reduction
 * and write to the output.
 */
template <typename KernelT, typename StorageT>
__forceinline__ __device__ void write_sum_1(StorageT* const inout_shared, const KernelT x) {
  inout_shared[threadIdx.x] = x;
}

template <typename KernelT, typename StorageT>
__forceinline__ __device__ void write_sum_2(StorageT* const inout_shared, const KernelT x,
                                            const KernelT y) {
  inout_shared[threadIdx.x * 2 + 0] = x;
  inout_shared[threadIdx.x * 2 + 1] = y;
}

template <typename KernelT, typename StorageT>
__forceinline__ __device__ void write_sum_3(StorageT* const inout_shared, const KernelT x,
                                            const KernelT y, const KernelT z) {
  inout_shared[threadIdx.x * 3 + 0] = x;
  inout_shared[threadIdx.x * 3 + 1] = y;
  inout_shared[threadIdx.x * 3 + 2] = z;
}

template <typename KernelT, typename StorageT>
__forceinline__ __device__ void write_sum_4(StorageT* const inout_shared, const KernelT x,
                                            const KernelT y, const KernelT z, const KernelT w) {
  inout_shared[threadIdx.x * 4 + 0] = x;
  inout_shared[threadIdx.x * 4 + 1] = y;
  inout_shared[threadIdx.x * 4 + 2] = z;
  inout_shared[threadIdx.x * 4 + 3] = w;
}

/**
 * Function used to perform collaborative reductions. Read more on caspar.argtypes.accessors.Sum.
 */
template <uint dim_target, typename StorageT>
__forceinline__ __device__ void flush_sum_shared(StorageT* const output, const uint offset,
                                                 const SharedIndex* const indices,
                                                 StorageT* const inout_shared) {
  __syncthreads();

  const SharedIndex idx = indices[threadIdx.x];
  uint unique = 0xffffffff;
  if (idx.argsort != 0xffff) {  // 0xffff indicates the thread is not used.
    unique = indices[indices[idx.argsort].target].unique;
  }
  const cg::coalesced_group group = cg::labeled_partition(cg::coalesced_threads(), unique);

#pragma unroll
  for (int i = 0; i < dim_target; i++) {
    const SharedIndex idx = indices[threadIdx.x];
    StorageT tot;
    if (idx.argsort != 0xffff) {  // 0xffff indicates the thread is not used.
      tot = cg::reduce(group, inout_shared[idx.argsort * dim_target + i], cg::plus<StorageT>());
    }
    __syncthreads();
    inout_shared[threadIdx.x * dim_target + i] = 0.0f;
    __syncthreads();

    // 0xffff indicates the thread is not used.
    if (idx.argsort != 0xffff && group.thread_rank() == 0) {
      atomicAdd_block(&inout_shared[indices[idx.argsort].target * dim_target + i], tot);
    }
    __syncthreads();
  }

  constexpr uint dim_aligned = dim_target == 3 ? 4 : dim_target;
  for (int i = 0; i < dim_target; i++) {
    const uint val_idx = threadIdx.x + blockDim.x * i;
    const uint obj_idx = indices[val_idx / dim_target].unique;
    const uint elem = val_idx % dim_target;
    // 0xffffffff indicates this thread has nothing to write.
    if (obj_idx == 0xffffffff) {
      break;
    }
    atomicAdd(&output[offset + obj_idx * dim_aligned + elem], inout_shared[val_idx]);
  }
  __syncthreads();
}

/**
 * Function used to perform a reduction over the block.
 *
 * Read more on caspar.argtypes.accessors.BlockSum.
 */
template <uint dim_target, typename StorageT>
__forceinline__ __device__ void flush_sum_block(StorageT* const output,
                                                StorageT* const inout_shared, const bool valid) {
  __syncthreads();
  const cg::coalesced_group group = cg::binary_partition(cg::coalesced_threads(), valid);
  constexpr uint dim_aligned = dim_target == 3 ? 4 : dim_target;

#pragma unroll
  for (int i = 0; i < dim_target; i++) {
    StorageT tot;

    if (valid) {
      tot = cg::reduce(group, inout_shared[threadIdx.x * dim_target + i], cg::plus<StorageT>());
    }
    __syncthreads();
    inout_shared[threadIdx.x * dim_target + i] = 0.0f;
    __syncthreads();

    if (valid && group.thread_rank() == 0) {
      // TODO(Emil Martes): use first warp to reduce instead of atomicAdd_block
      atomicAdd_block(&inout_shared[i], tot);
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < dim_target; i += blockDim.x) {
    output[blockIdx.x * dim_aligned + i] = inout_shared[i];
  }
}
template <typename StorageT>
__forceinline__ __device__ void sum_store(StorageT* const shared_tmp, StorageT* const inout_shared,
                                          const uint offset, const bool valid, StorageT data) {
  auto group = cg::tiled_partition<32>(cg::this_thread_block());

  __syncthreads();
  StorageT tot = cg::reduce(group, valid ? data : 0.0f, cg::plus<StorageT>());
  if (group.thread_rank() == 0) {
    inout_shared[group.meta_group_rank()] = tot;
  }
  __syncthreads();
  if (group.meta_group_rank() == 0) {
    tot = cg::reduce(group, inout_shared[group.thread_rank()], cg::plus<StorageT>());
    if (group.thread_rank() == 0) {
      shared_tmp[offset] = tot;
    }
  }
}

template <typename StorageT>
__forceinline__ __device__ void sum_flush_final(const StorageT* const shared_tmp,
                                                StorageT* const output, const uint dim) {
  __syncthreads();
  if (threadIdx.x < dim) {
    atomicAdd(&output[threadIdx.x], shared_tmp[threadIdx.x]);
  }
}

/**
 * Function used to perform a reduction over a block and add to the output.
 *
 * Read more on caspar.argtypes.accessors.BlockSumAdd.
 */
template <uint dim_target, typename StorageT>
__forceinline__ __device__ void flush_sum_block_add(StorageT* const output,
                                                    StorageT* const inout_shared,
                                                    const bool valid) {
  __syncthreads();
  const cg::coalesced_group group = cg::binary_partition(cg::coalesced_threads(), valid);
  constexpr uint dim_aligned = dim_target == 3 ? 4 : dim_target;

#pragma unroll
  for (int i = 0; i < dim_target; i++) {
    StorageT tot;

    if (valid) {
      tot = cg::reduce(group, inout_shared[threadIdx.x * dim_target + i], cg::plus<StorageT>());
    }
    __syncthreads();
    inout_shared[threadIdx.x * dim_target + i] = 0.0f;
    __syncthreads();

    if (valid && group.thread_rank() == 0) {
      // TODO(Emil Martes): use first warp to reduce instead of atomicAdd_block
      atomicAdd_block(&inout_shared[i], tot);
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < dim_target; i += blockDim.x) {
    output[blockIdx.x * dim_aligned + i] += inout_shared[i];
  }
}

// READ SHARED

/**
 * Function used to load data from global memory into shared memory.
 *
 * Read more on caspar.argtypes.accessors.Shared.
 * Should be followed by read_shared_x.
 */
template <uint dim_target, typename KernelT, typename StorageT>
__forceinline__ __device__ void load_shared(const StorageT* const input, const uint offset,
                                            const SharedIndex* const indices,
                                            KernelT* const inout_shared) {
  __syncthreads();
  constexpr uint dim_aligned = dim_target == 3 ? 4 : dim_target;

#pragma unroll
  for (int i = 0; i < dim_target; i++) {
    const uint val_idx = blockDim.x * i + threadIdx.x;
    const uint obj_idx = indices[val_idx / dim_target].unique;
    const uint elem = val_idx % dim_target;

    // 0xffffffff indicates this thread has nothing to read.
    if (obj_idx == 0xffffffff) {
      break;
    }
    inout_shared[val_idx] = input[offset + obj_idx * dim_aligned + elem];
  }
  __syncthreads();
}

/**
 * Function used to load data from a unique global element into shared memory.
 *
 * Read more on caspar.argtypes.accessors.Unique.
 * Should be followed by read_shared_x.
 */
template <uint dim_target, typename KernelT, typename StorageT>
__forceinline__ __device__ void load_unique(const StorageT* const input, const uint offset,
                                            KernelT* const inout_shared) {
  __syncthreads();
  if (threadIdx.x < dim_target) {
    inout_shared[threadIdx.x] = input[offset + threadIdx.x];
  }
  __syncthreads();
}

// READ SHARED
template <typename KernelT>
__forceinline__ __device__ void read_shared_1(const KernelT* const inout_shared, const uint target,
                                              KernelT& x) {
  x = inout_shared[target];
}

template <typename KernelT>
__forceinline__ __device__ void read_shared_2(const KernelT* const inout_shared, const uint target,
                                              KernelT& x, KernelT& y) {
  x = inout_shared[target * 2 + 0];
  y = inout_shared[target * 2 + 1];
}

template <typename KernelT>
__forceinline__ __device__ void read_shared_3(const KernelT* const inout_shared, const uint target,
                                              KernelT& x, KernelT& y, KernelT& z) {
  x = inout_shared[target * 3 + 0];
  y = inout_shared[target * 3 + 1];
  z = inout_shared[target * 3 + 2];
}

template <typename KernelT>
__forceinline__ __device__ void read_shared_4(const KernelT* const inout_shared, const uint target,
                                              KernelT& x, KernelT& y, KernelT& z, KernelT& w) {
  x = inout_shared[target * 4 + 0];
  y = inout_shared[target * 4 + 1];
  z = inout_shared[target * 4 + 2];
  w = inout_shared[target * 4 + 3];
}

// READ OVERLAPPED
template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void read_and_shuffle_1(KernelT* const inout_shared,
                                                   const StorageT* const input, const uint offset,
                                                   const uint idx, bool is_last, bool should_read,
                                                   KernelT& x0, KernelT& x1) {
  if (should_read) {
    x0 = input[offset + idx];
    inout_shared[threadIdx.x] = x0;
  }
  __syncthreads();
  if (should_read && !is_last) {
    x1 = inout_shared[threadIdx.x + 1];
  }
  __syncthreads();
  if (should_read && is_last) {
    x1 = input[offset + idx + 1];
  }
}
template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void read_and_shuffle_2(KernelT* const inout_shared,
                                                   const StorageT* const input, const uint offset,
                                                   const uint idx, bool is_last, bool should_read,
                                                   KernelT& x0, KernelT& y0, KernelT& x1,
                                                   KernelT& y1) {
  VecT tmp;
  if (should_read) {
    tmp = reinterpret_cast<const VecT*>(&input[offset])[idx];
    x0 = tmp.x;
    y0 = tmp.y;
    reinterpret_cast<VecT*>(inout_shared)[threadIdx.x] = tmp;
  }
  __syncthreads();
  if (should_read && !is_last) {
    tmp = reinterpret_cast<VecT*>(inout_shared)[threadIdx.x + 1];
  }
  __syncthreads();
  if (should_read && is_last) {
    tmp = reinterpret_cast<const VecT*>(&input[offset])[idx + 1];
  }
  x1 = tmp.x;
  y1 = tmp.y;
}

template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void read_and_shuffle_3(KernelT* const inout_shared,
                                                   const StorageT* const input, const uint offset,
                                                   const uint idx, bool is_last, bool should_read,
                                                   KernelT& x0, KernelT& y0, KernelT& z0,
                                                   KernelT& x1, KernelT& y1, KernelT& z1) {
  VecT tmp;
  if (should_read) {
    tmp = reinterpret_cast<const VecT*>(&input[offset])[idx];
    x0 = tmp.x;
    y0 = tmp.y;
    z0 = tmp.z;
    reinterpret_cast<VecT*>(inout_shared)[threadIdx.x] = tmp;
  }
  __syncthreads();
  if (should_read && !is_last) {
    tmp = reinterpret_cast<VecT*>(inout_shared)[threadIdx.x + 1];
  }
  __syncthreads();
  if (should_read && is_last) {
    tmp = reinterpret_cast<const VecT*>(&input[offset])[idx + 1];
  }
  x1 = tmp.x;
  y1 = tmp.y;
  z1 = tmp.z;
}

template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void read_and_shuffle_4(KernelT* const inout_shared,
                                                   const StorageT* const input, const uint offset,
                                                   const uint idx, bool is_last, bool should_read,
                                                   KernelT& x0, KernelT& y0, KernelT& z0,
                                                   KernelT& w0, KernelT& x1, KernelT& y1,
                                                   KernelT& z1, KernelT& w1) {
  VecT tmp;
  if (should_read) {
    tmp = reinterpret_cast<const VecT*>(&input[offset])[idx];
    x0 = tmp.x;
    y0 = tmp.y;
    z0 = tmp.z;
    w0 = tmp.w;
    reinterpret_cast<VecT*>(inout_shared)[threadIdx.x] = tmp;
  }
  __syncthreads();
  if (should_read && !is_last) {
    tmp = reinterpret_cast<VecT*>(inout_shared)[threadIdx.x + 1];
  }
  __syncthreads();
  if (should_read && is_last) {
    tmp = reinterpret_cast<const VecT*>(&input[offset])[idx + 1];
  }
  x1 = tmp.x;
  y1 = tmp.y;
  z1 = tmp.z;
  w1 = tmp.w;
}
// READ WITH DEFAULT
template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void read_and_shuffle_with_default_1(
    KernelT* const inout_shared, const StorageT* const input, const uint offset, const int idx,
    bool is_last, bool should_read, const uint data_size, int step, KernelT& x0, KernelT& x1,
    KernelT x_default) {
  if (should_read && idx >= 0 && idx < data_size) {
    x0 = input[offset + idx];
    inout_shared[threadIdx.x] = x0;
  } else {
    x0 = x_default;
    inout_shared[threadIdx.x] = x_default;
  }
  __syncthreads();

  if (!is_last) {
    x1 = inout_shared[threadIdx.x + 1];
  } else if (idx + step >= 0 && idx + step < data_size) {
    x1 = input[offset + idx + step];
  } else {
    x1 = x_default;
  }
  __syncthreads();
}
template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void read_and_shuffle_with_default_2(
    KernelT* const inout_shared, const StorageT* const input, const uint offset, const int idx,
    bool is_last, bool should_read, const uint data_size, int step, KernelT& x0, KernelT& y0,
    KernelT& x1, KernelT& y1, KernelT x_default, KernelT y_default) {
  VecT tmp;
  if (should_read && idx >= 0 && idx < data_size) {
    tmp = reinterpret_cast<const VecT*>(&input[offset])[idx];
  } else {
    tmp.x = x_default;
    tmp.y = y_default;
  }
  x0 = tmp.x;
  y0 = tmp.y;
  reinterpret_cast<VecT*>(inout_shared)[threadIdx.x] = tmp;
  __syncthreads();
  if (!is_last) {
    tmp = reinterpret_cast<VecT*>(inout_shared)[threadIdx.x + 1];
  } else if (idx + step >= 0 && idx + step < data_size) {
    tmp = reinterpret_cast<const VecT*>(&input[offset])[idx + step];
  } else {
    tmp.x = x_default;
    tmp.y = y_default;
  }
  x1 = tmp.x;
  y1 = tmp.y;
  __syncthreads();
}
template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void read_and_shuffle_with_default_3(
    KernelT* const inout_shared, const StorageT* const input, const uint offset, const int idx,
    bool is_last, bool should_read, const uint data_size, int step, KernelT& x0, KernelT& y0,
    KernelT& z0, KernelT& x1, KernelT& y1, KernelT& z1, KernelT x_default, KernelT y_default,
    KernelT z_default) {
  VecT tmp;
  if (should_read && idx >= 0 && idx < data_size) {
    tmp = reinterpret_cast<const VecT*>(&input[offset])[idx];
  } else {
    tmp.x = x_default;
    tmp.y = y_default;
    tmp.z = z_default;
  }
  x0 = tmp.x;
  y0 = tmp.y;
  z0 = tmp.z;
  reinterpret_cast<VecT*>(inout_shared)[threadIdx.x] = tmp;
  __syncthreads();

  if (!is_last) {
    tmp = reinterpret_cast<VecT*>(inout_shared)[threadIdx.x + 1];
  } else if (idx + step >= 0 && idx + step < data_size) {
    tmp = reinterpret_cast<const VecT*>(&input[offset])[idx + step];
  } else {
    tmp.x = x_default;
    tmp.y = y_default;
    tmp.z = z_default;
  }
  x1 = tmp.x;
  y1 = tmp.y;
  z1 = tmp.z;
  __syncthreads();
}
template <uint block_size, typename KernelT, typename StorageT, typename VecT>

__forceinline__ __device__ void read_and_shuffle_with_default_4(
    KernelT* const inout_shared, const StorageT* const input, const uint offset, const int idx,
    bool is_last, bool should_read, const uint data_size, int step, KernelT& x0, KernelT& y0,
    KernelT& z0, KernelT& w0, KernelT& x1, KernelT& y1, KernelT& z1, KernelT& w1, KernelT x_default,
    KernelT y_default, KernelT z_default, KernelT w_default) {
  VecT tmp;
  if (should_read && idx >= 0 && idx < data_size) {
    tmp = reinterpret_cast<const VecT*>(&input[offset])[idx];
  } else {
    tmp.x = x_default;
    tmp.y = y_default;
    tmp.z = z_default;
    tmp.w = w_default;
  }
  x0 = tmp.x;
  y0 = tmp.y;
  z0 = tmp.z;
  w0 = tmp.w;
  reinterpret_cast<VecT*>(inout_shared)[threadIdx.x] = tmp;
  __syncthreads();

  if (!is_last) {
    tmp = reinterpret_cast<VecT*>(inout_shared)[threadIdx.x + 1];
  } else if (idx + step >= 0 && idx + step < data_size) {
    tmp = reinterpret_cast<const VecT*>(&input[offset])[idx + step];
  } else {
    tmp.x = x_default;
    tmp.y = y_default;
    tmp.z = z_default;
    tmp.w = w_default;
  }
  x1 = tmp.x;
  y1 = tmp.y;
  z1 = tmp.z;
  w1 = tmp.w;
  __syncthreads();
}

template <bool do_add, uint block_size, typename KernelT, typename StorageT, typename VecT>
__forceinline__ __device__ void shuffle_and_write_1(StorageT* const inout_shared,
                                                    StorageT* const output, const uint offset,
                                                    const uint gtidx, uint problem_size,
                                                    const KernelT x0, const KernelT x1) {
  if (gtidx < problem_size) {
    inout_shared[threadIdx.x + 1] = x1;
  }
  if (threadIdx.x == 0) {
    inout_shared[0] = x0;
  }
  __syncthreads();
  VecT from_prev;
  if (gtidx <= problem_size) {
    from_prev = inout_shared[threadIdx.x];
  }
  if (threadIdx.x == 0) {
    from_prev = 0.0f;
  }

  const uint block_start = gtidx & (-1u << 10);
  {
    bool help_intrablock_down = block_start != 0 && threadIdx.x < 1;
    bool help_intrablock_up =
        block_start + 1024 <= problem_size && threadIdx.x >= 1 && threadIdx.x < 2;

    StorageT data;
    if (help_intrablock_down) {
      data = inout_shared[threadIdx.x * 1025];
    }
    if (help_intrablock_up) {
      data = inout_shared[(threadIdx.x) * 1025 + 1024 - 1025];
    }
    __syncthreads();
    if (help_intrablock_down || help_intrablock_up) {
      if (block_start != 0 && block_start + 1024 <= problem_size) {
        uint idx = threadIdx.x < 1 ? threadIdx.x : threadIdx.x - 1u + 1024u;
        atomicAdd(&output[offset + block_start + idx], data);
      } else if (block_start == 0) {
        atomicAdd(&output[offset + block_start + threadIdx.x - 1u + 1024u], data);
      } else {
        atomicAdd(&output[offset + block_start + threadIdx.x], data);
      }
    }
  }
  if ((threadIdx.x > 0 || block_start == 0) && gtidx <= problem_size) {
    VecT tmp = from_prev;
    if (gtidx < problem_size) {
      tmp += x0;
    }
    if (do_add) {
      tmp += output[offset + gtidx];
    }
    output[offset + gtidx] = tmp;
  }
}

template <bool do_add, uint block_size, typename KernelT, typename StorageT, typename VecT>
__forceinline__ __device__ void shuffle_and_write_2(StorageT* const inout_shared,
                                                    StorageT* const output, const uint offset,
                                                    const uint gtidx, uint problem_size,
                                                    const KernelT x0, const KernelT y0,
                                                    const KernelT x1, const KernelT y1) {
  if (gtidx < problem_size) {
    inout_shared[threadIdx.x + 1] = x1;
    inout_shared[threadIdx.x + 1 + 1025] = y1;
  }
  if (threadIdx.x == 0) {
    inout_shared[0] = x0;
    inout_shared[1025] = y0;
  }
  __syncthreads();
  VecT from_prev;
  if (gtidx <= problem_size) {
    from_prev.x = inout_shared[threadIdx.x];
    from_prev.y = inout_shared[threadIdx.x + 1025];
  }
  if (threadIdx.x == 0) {
    from_prev.x = 0.0f;
    from_prev.y = 0.0f;
  }

  const uint block_start = gtidx & (-1u << 10);
  {
    bool help_intrablock_down = block_start != 0 && threadIdx.x < 2;
    bool help_intrablock_up =
        block_start + 1024 <= problem_size && threadIdx.x >= 2 && threadIdx.x < 4;

    StorageT data;
    if (help_intrablock_down) {
      data = inout_shared[threadIdx.x * 1025];
    }
    if (help_intrablock_up) {
      data = inout_shared[(threadIdx.x) * 1025 + 1024 - 2 * 1025];
    }
    __syncthreads();
    if (help_intrablock_down || help_intrablock_up) {
      if (block_start != 0 && block_start + 1024 <= problem_size) {
        uint idx = threadIdx.x < 2 ? threadIdx.x : threadIdx.x - 2u + 2u * 1024u;
        atomicAdd(&output[offset + 2 * block_start + idx], data);
      } else if (block_start == 0) {
        atomicAdd(&output[offset + 2 * block_start + threadIdx.x - 2u + 2u * 1024u], data);
      } else {
        atomicAdd(&output[offset + 2 * block_start + threadIdx.x], data);
      }
    }
  }
  if ((threadIdx.x > 0 || block_start == 0) && gtidx <= problem_size) {
    VecT tmp2 = from_prev;
    if (gtidx < problem_size) {
      tmp2.x += x0;
      tmp2.y += y0;
    }
    if (do_add) {
      VecT current = reinterpret_cast<VecT*>(&output[offset])[gtidx];
      tmp2.x += current.x;
      tmp2.y += current.y;
    }
    reinterpret_cast<VecT*>(&output[offset])[gtidx] = tmp2;
  }
}

template <bool do_add, uint block_size, typename KernelT, typename StorageT, typename VecT>
__forceinline__ __device__ void shuffle_and_write_3(StorageT* const inout_shared,
                                                    StorageT* const output, const uint offset,
                                                    const uint gtidx, uint problem_size,
                                                    const KernelT x0, const KernelT y0,
                                                    const KernelT z0, const KernelT x1,
                                                    const KernelT y1, const KernelT z1) {
  if (gtidx < problem_size) {
    inout_shared[threadIdx.x + 1] = x1;
    inout_shared[threadIdx.x + 1 + 1025] = y1;
    inout_shared[threadIdx.x + 1 + 2 * 1025] = z1;
  }
  if (threadIdx.x == 0) {
    inout_shared[0] = x0;
    inout_shared[1025] = y0;
    inout_shared[2 * 1025] = z0;
  }
  __syncthreads();
  VecT from_prev;
  if (gtidx <= problem_size) {
    from_prev.x = inout_shared[threadIdx.x];
    from_prev.y = inout_shared[threadIdx.x + 1025];
    from_prev.z = inout_shared[threadIdx.x + 2 * 1025];
  }
  if (threadIdx.x == 0) {
    from_prev.x = 0.0f;
    from_prev.y = 0.0f;
    from_prev.z = 0.0f;
  }

  const uint block_start = gtidx & (-1u << 10);
  {
    bool help_intrablock_down = block_start != 0 && threadIdx.x < 3;
    bool help_intrablock_up =
        block_start + 1024 <= problem_size && threadIdx.x >= 3 && threadIdx.x < 6;

    StorageT data;
    if (help_intrablock_down) {
      data = inout_shared[threadIdx.x * 1025];
    }
    if (help_intrablock_up) {
      data = inout_shared[(threadIdx.x) * 1025 + 1024 - 3 * 1025];
    }
    __syncthreads();
    if (help_intrablock_down || help_intrablock_up) {
      if (block_start != 0 && block_start + 1024 <= problem_size) {
        uint idx = threadIdx.x < 3 ? threadIdx.x : threadIdx.x - 3u + 4u * 1024u;
        atomicAdd(&output[offset + 4 * block_start + idx], data);
      } else if (block_start == 0) {
        atomicAdd(&output[offset + 4 * block_start + threadIdx.x - 3u + 4u * 1024u], data);
      } else {
        atomicAdd(&output[offset + 4 * block_start + threadIdx.x], data);
      }
    }
  }
  if ((threadIdx.x > 0 || block_start == 0) && gtidx <= problem_size) {
    VecT tmp3 = from_prev;
    if (gtidx < problem_size) {
      tmp3.x += x0;
      tmp3.y += y0;
      tmp3.z += z0;
    }
    if (do_add) {
      VecT current = reinterpret_cast<VecT*>(&output[offset])[gtidx];
      tmp3.x += current.x;
      tmp3.y += current.y;
      tmp3.z += current.z;
    }
    reinterpret_cast<VecT*>(&output[offset])[gtidx] = tmp3;
  }
}

template <bool do_add, uint block_size, typename KernelT, typename StorageT, typename VecT>
__forceinline__ __device__ void shuffle_and_write_4(
    StorageT* const inout_shared, StorageT* const output, const uint offset, const uint gtidx,
    uint problem_size, const KernelT x0, const KernelT y0, const KernelT z0, const KernelT w0,
    const KernelT x1, const KernelT y1, const KernelT z1, const KernelT w1) {
  if (gtidx < problem_size) {
    inout_shared[threadIdx.x + 1] = x1;
    inout_shared[threadIdx.x + 1 + 1025] = y1;
    inout_shared[threadIdx.x + 1 + 2 * 1025] = z1;
    inout_shared[threadIdx.x + 1 + 3 * 1025] = w1;
  }
  if (threadIdx.x == 0) {
    inout_shared[0] = x0;
    inout_shared[1025] = y0;
    inout_shared[2 * 1025] = z0;
    inout_shared[3 * 1025] = w0;
  }
  __syncthreads();
  VecT from_prev;
  if (gtidx <= problem_size) {
    from_prev.x = inout_shared[threadIdx.x];
    from_prev.y = inout_shared[threadIdx.x + 1025];
    from_prev.z = inout_shared[threadIdx.x + 2 * 1025];
    from_prev.w = inout_shared[threadIdx.x + 3 * 1025];
  }
  if (threadIdx.x == 0) {
    from_prev.x = 0.0f;
    from_prev.y = 0.0f;
    from_prev.z = 0.0f;
    from_prev.w = 0.0f;
  }

  const uint block_start = gtidx & (-1u << 10);
  {
    bool help_intrablock_down = block_start != 0 && threadIdx.x < 4;
    bool help_intrablock_up =
        block_start + 1024 <= problem_size && threadIdx.x >= 4 && threadIdx.x < 8;

    StorageT data;
    if (help_intrablock_down) {
      data = inout_shared[threadIdx.x * 1025];
    }
    if (help_intrablock_up) {
      data = inout_shared[(threadIdx.x) * 1025 + 1024 - 4 * 1025];
    }
    __syncthreads();
    if (help_intrablock_down || help_intrablock_up) {
      if (block_start != 0 && block_start + 1024 <= problem_size) {
        uint idx = threadIdx.x < 4 ? threadIdx.x : threadIdx.x - 4u + 4u * 1024u;
        atomicAdd(&output[offset + 4 * block_start + idx], data);
      } else if (block_start == 0) {
        atomicAdd(&output[offset + 4 * block_start + threadIdx.x - 4u + 4u * 1024u], data);
      } else {
        atomicAdd(&output[offset + 4 * block_start + threadIdx.x], data);
      }
    }
  }
  if ((threadIdx.x > 0 || block_start == 0) && gtidx <= problem_size) {
    VecT tmp4 = from_prev;
    if (gtidx < problem_size) {
      tmp4.x += x0;
      tmp4.y += y0;
      tmp4.z += z0;
      tmp4.w += w0;
    }
    if (do_add) {
      VecT current = reinterpret_cast<VecT*>(&output[offset])[gtidx];
      tmp4.x += current.x;
      tmp4.y += current.y;
      tmp4.z += current.z;
      tmp4.w += current.w;
    }
    reinterpret_cast<VecT*>(&output[offset])[gtidx] = tmp4;
  }
}

}  // namespace caspar
