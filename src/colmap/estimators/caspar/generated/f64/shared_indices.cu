/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include "shared_indices.h"

namespace cg = cooperative_groups;

namespace caspar {

template <typename T>
__device__ void swap(T& a, T& b) {
  T tmp = a;
  a = b;
  b = tmp;
}

// Batcher odd–even mergesort, sorts at most 1024 elements
template <typename T_val, uint arrayLength, typename T_idx = uint>
__forceinline__ __device__ void sort(T_val* const values, T_idx* const indices = nullptr) {
  __shared__ bool is_sorted;
  if (threadIdx.x == 0)
    is_sorted = true;

  __syncthreads();
  if (threadIdx.x + 1 < arrayLength)
    if (values[threadIdx.x] > values[threadIdx.x + 1])
      is_sorted = false;
  __syncthreads();
  if (is_sorted)
    return;

#pragma unroll
  for (uint size = 2; size <= arrayLength; size <<= 1) {
    uint stride = size / 2;
    uint offset = threadIdx.x & (stride - 1);
    {
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      if (pos + stride < arrayLength)
        if (values[pos] > values[pos + stride]) {
          swap<T_val>(values[pos], values[pos + stride]);
          if (indices)
            swap<T_idx>(indices[pos], indices[pos + stride]);
        }
      stride >>= 1;
    }
#pragma unroll
    for (; stride > 0; stride >>= 1) {
      if (size <= 32)
        __syncwarp();
      else
        __syncthreads();
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      if (offset >= stride && pos < arrayLength)
        if (values[pos - stride] > values[pos]) {
          swap<T_val>(values[pos - stride], values[pos]);
          if (indices)
            swap<T_idx>(indices[pos - stride], indices[pos]);
        }
    }
    if (size <= 16)
      __syncwarp();
    else
      __syncthreads();
  }
}

template <int arrayLength>
__forceinline__ __device__ void make_unique(uint* const values, uint* const indices) {
  sort<uint, arrayLength>(values, indices);
  bool pred = false;
  if (threadIdx.x < arrayLength - 1)
    pred = values[threadIdx.x] == values[threadIdx.x + 1];
  __syncthreads();
  if (pred)
    values[threadIdx.x + 1] = 0xFFFFFFFF;
  sort<uint, arrayLength>(values);
}

template <int arrayLength>
__forceinline__ __device__ uint get_n_unique(const uint* values) {
  __shared__ uint nsorted;  // last block might have less than 1024 elements
  if (threadIdx.x == 0)
    nsorted = arrayLength;
  __syncthreads();
  if (threadIdx.x < arrayLength - 1)
    if (values[threadIdx.x + 1] == 0xFFFFFFFF && values[threadIdx.x] != 0xFFFFFFFF)
      nsorted = threadIdx.x + 1;
  __syncthreads();
  return nsorted;
}

__forceinline__ __device__ uint get_ord(const uint* values, const uint length, const uint val) {
  const uint* lo = values;
  const uint* hi = values + length;
  while (lo < hi) {
    const uint* mid = lo + (hi - lo) / 2;
    if (*mid < val)
      lo = mid + 1;
    else
      hi = mid;
  }
  return static_cast<uint>(lo - values);
}

__global__ void shared_indices_kernel(const uint* const __restrict__ indices,
                                      SharedIndex* const __restrict__ shared_indices_out,
                                      const uint size) {
  const auto block = cg::this_thread_block();

  const auto gtrank = cg::this_grid().thread_rank();
  const auto btrank = block.thread_rank();
  const int start = block.group_index().x * block.group_dim().x;
  const int num = min(1024, size - start) * sizeof(uint);

  __shared__ uint indices_loc[1024];
  cg::memcpy_async(block, indices_loc, indices + start, num);
  cg::wait(block);

  const uint val = gtrank < size ? indices_loc[btrank] : 0xFFFFFFFF;

  __shared__ uint values_sh[1024];
  __shared__ uint indices_sh[1024];
  values_sh[threadIdx.x] = val;
  indices_sh[threadIdx.x] = threadIdx.x;
  __syncthreads();

  make_unique<1024>(values_sh, indices_sh);
  SharedIndex shared_index;
  if (gtrank < size) {
    const uint n_unique = get_n_unique<1024>(values_sh);
    shared_index.unique = values_sh[btrank];
    shared_index.argsort = indices_sh[btrank];
    shared_index.target = get_ord(values_sh, n_unique, val);
  } else {
    shared_index.unique = 0xffffffff;
    shared_index.argsort = 0xffff;
    shared_index.target = 0xffff;
  };
  if (gtrank < size)
    shared_indices_out[gtrank] = shared_index;
}

void shared_indices(const uint* const indices, SharedIndex* const shared_indices_out,
                    const uint problem_size) {
  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  shared_indices_kernel<<<n_blocks, 1024>>>(indices, shared_indices_out, problem_size);
}

}  // namespace caspar
