/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 *
 * Author: Jeff Daily <jeff.daily@amd.com>
 * ---------------------------------------------------------------------------- */

#pragma once

// CUDA-to-HIP compatibility header for ROCm/HIP builds.
// Provides the CUDA API spellings the project uses via aliases to HIP equivalents.

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <hipcub/hipcub.hpp>

// CUDA runtime API -> HIP runtime API
#define cudaMalloc              hipMalloc
#define cudaFree                hipFree
#define cudaMemset              hipMemset
#define cudaMemcpy              hipMemcpy
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
#define cudaError_t             hipError_t
#define cudaSuccess             hipSuccess
#define cudaStream_t            hipStream_t
#define cudaDeviceSynchronize   hipDeviceSynchronize
#define cudaGetLastError        hipGetLastError
#define cudaGetErrorString      hipGetErrorString
#define cudaSetDevice           hipSetDevice
#define cudaGetDevice           hipGetDevice
#define cudaGetDeviceCount      hipGetDeviceCount
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaPointerAttributes   hipPointerAttribute_t

// HIP shared-memory atomics are block-scoped by definition (no inter-block visibility),
// so atomicAdd_block is equivalent to atomicAdd on shared memory.
#define atomicAdd_block atomicAdd

// CUB -> hipCUB namespace
namespace cub = hipcub;

// Cooperative groups: HIP has cg basics but lacks cg::reduce, cg::labeled_partition, memcpy_async.
// Provide manual implementations where needed.

namespace caspar_hip {

// Butterfly reduction within a cooperative group (thread_block_tile or coalesced_group).
// Replaces cg::reduce(group, val, cg::plus<T>()).
template <typename GroupT, typename T>
__device__ __forceinline__ T reduce_sum(GroupT group, T val) {
  for (unsigned int offset = group.size() / 2; offset > 0; offset >>= 1) {
    val += group.shfl_xor(val, offset);
  }
  return val;
}

// Butterfly reduction for max (cg::greater<T>).
template <typename GroupT, typename T>
__device__ __forceinline__ T reduce_max(GroupT group, T val) {
  for (unsigned int offset = group.size() / 2; offset > 0; offset >>= 1) {
    T other = group.shfl_xor(val, offset);
    val = (val > other) ? val : other;
  }
  return val;
}

// Match_any: return a mask of lanes in the tile that have the same label.
// HIP CG has match_any() for coalesced groups.
template <typename GroupT, typename LabelT>
__device__ __forceinline__ unsigned long long match_any_mask(GroupT group, LabelT label) {
  // HIP cooperative_groups::coalesced_group has match_any
  return group.match_any(label);
}

// Labeled partition emulation: reduce values within lanes sharing the same label,
// and have exactly one lane per unique label perform the atomic.
// Returns the reduced value and sets is_leader=true for exactly one lane per label.
template <typename GroupT, typename T, typename LabelT>
__device__ __forceinline__ T labeled_reduce_sum(GroupT group, T val, LabelT label, bool& is_leader) {
  // Get mask of lanes with same label
  unsigned long long same_label_mask = group.match_any(label);

  // Find my position within the matching lanes
  unsigned int my_lane = group.thread_rank();
  unsigned long long lower_mask = (1ULL << my_lane) - 1;
  unsigned int rank_in_label = __popcll(same_label_mask & lower_mask);

  // Leader is the lowest-numbered lane in the group
  is_leader = (rank_in_label == 0);

  // Count total lanes with this label
  unsigned int label_size = __popcll(same_label_mask);

  // Butterfly reduction over the masked lanes
  // For each reduction step, exchange with lane at offset if both are in same_label_mask
  T result = val;
  for (unsigned int offset = 1; offset < group.size(); offset <<= 1) {
    unsigned int partner_lane = my_lane ^ offset;
    bool partner_has_same_label = (same_label_mask >> partner_lane) & 1ULL;
    T partner_val = group.shfl_xor(result, offset);
    if (partner_has_same_label && partner_lane < group.size()) {
      result += partner_val;
    }
  }

  return result;
}

}  // namespace caspar_hip

// Macros to replace cg::reduce calls (used in device code)
#define CG_REDUCE_SUM(group, val) caspar_hip::reduce_sum(group, val)
#define CG_LABELED_REDUCE_SUM(group, val, label, is_leader) \
    caspar_hip::labeled_reduce_sum(group, val, label, is_leader)

#else  // CUDA path

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

// On CUDA, cg::reduce is available
#define CG_REDUCE_SUM(group, val) cg::reduce(group, val, cg::plus<decltype(val)>())

#endif
