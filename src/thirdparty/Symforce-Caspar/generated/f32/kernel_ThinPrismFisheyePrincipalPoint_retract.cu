#include "kernel_ThinPrismFisheyePrincipalPoint_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyePrincipalPointRetractKernel(
        float* ThinPrismFisheyePrincipalPoint,
        unsigned int ThinPrismFisheyePrincipalPoint_num_alloc,
        float* delta,
        unsigned int delta_num_alloc,
        float* out_ThinPrismFisheyePrincipalPoint_retracted,
        unsigned int out_ThinPrismFisheyePrincipalPoint_retracted_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  float r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        ThinPrismFisheyePrincipalPoint,
        0 * ThinPrismFisheyePrincipalPoint_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, float, float, float2>(
        delta, 0 * delta_num_alloc, global_thread_idx, r2, r3);
    r2 = r0 + r2;
    r3 = r1 + r3;
    WriteIdx2<1024, float, float, float2>(
        out_ThinPrismFisheyePrincipalPoint_retracted,
        0 * out_ThinPrismFisheyePrincipalPoint_retracted_num_alloc,
        global_thread_idx,
        r2,
        r3);
  };
}

void ThinPrismFisheyePrincipalPointRetract(
    float* ThinPrismFisheyePrincipalPoint,
    unsigned int ThinPrismFisheyePrincipalPoint_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_ThinPrismFisheyePrincipalPoint_retracted,
    unsigned int out_ThinPrismFisheyePrincipalPoint_retracted_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyePrincipalPointRetractKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyePrincipalPoint,
      ThinPrismFisheyePrincipalPoint_num_alloc,
      delta,
      delta_num_alloc,
      out_ThinPrismFisheyePrincipalPoint_retracted,
      out_ThinPrismFisheyePrincipalPoint_retracted_num_alloc,
      problem_size);
}

}  // namespace caspar