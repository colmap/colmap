#include "kernel_ThinPrismFisheyePrincipalPoint_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyePrincipalPointAlphaNumeratorDenominatorKernel(
        float* ThinPrismFisheyePrincipalPoint_p_kp1,
        unsigned int ThinPrismFisheyePrincipalPoint_p_kp1_num_alloc,
        float* ThinPrismFisheyePrincipalPoint_r_k,
        unsigned int ThinPrismFisheyePrincipalPoint_r_k_num_alloc,
        float* ThinPrismFisheyePrincipalPoint_w,
        unsigned int ThinPrismFisheyePrincipalPoint_w_num_alloc,
        float* const ThinPrismFisheyePrincipalPoint_total_ag,
        float* const ThinPrismFisheyePrincipalPoint_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float ThinPrismFisheyePrincipalPoint_total_ag_local[1];

  __shared__ float ThinPrismFisheyePrincipalPoint_total_ac_local[1];

  float r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        ThinPrismFisheyePrincipalPoint_p_kp1,
        0 * ThinPrismFisheyePrincipalPoint_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, float, float, float2>(
        ThinPrismFisheyePrincipalPoint_r_k,
        0 * ThinPrismFisheyePrincipalPoint_r_k_num_alloc,
        global_thread_idx,
        r2,
        r3);
    r2 = fmaf(r0, r2, r1 * r3);
  };
  SumStore<float>(ThinPrismFisheyePrincipalPoint_total_ag_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r2);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        ThinPrismFisheyePrincipalPoint_w,
        0 * ThinPrismFisheyePrincipalPoint_w_num_alloc,
        global_thread_idx,
        r2,
        r3);
    r3 = fmaf(r1, r3, r0 * r2);
  };
  SumStore<float>(ThinPrismFisheyePrincipalPoint_total_ac_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r3);
  SumFlushFinal<float>(ThinPrismFisheyePrincipalPoint_total_ag_local,
                       ThinPrismFisheyePrincipalPoint_total_ag,
                       1);
  SumFlushFinal<float>(ThinPrismFisheyePrincipalPoint_total_ac_local,
                       ThinPrismFisheyePrincipalPoint_total_ac,
                       1);
}

void ThinPrismFisheyePrincipalPointAlphaNumeratorDenominator(
    float* ThinPrismFisheyePrincipalPoint_p_kp1,
    unsigned int ThinPrismFisheyePrincipalPoint_p_kp1_num_alloc,
    float* ThinPrismFisheyePrincipalPoint_r_k,
    unsigned int ThinPrismFisheyePrincipalPoint_r_k_num_alloc,
    float* ThinPrismFisheyePrincipalPoint_w,
    unsigned int ThinPrismFisheyePrincipalPoint_w_num_alloc,
    float* const ThinPrismFisheyePrincipalPoint_total_ag,
    float* const ThinPrismFisheyePrincipalPoint_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyePrincipalPointAlphaNumeratorDenominatorKernel<<<n_blocks,
                                                                  1024>>>(
      ThinPrismFisheyePrincipalPoint_p_kp1,
      ThinPrismFisheyePrincipalPoint_p_kp1_num_alloc,
      ThinPrismFisheyePrincipalPoint_r_k,
      ThinPrismFisheyePrincipalPoint_r_k_num_alloc,
      ThinPrismFisheyePrincipalPoint_w,
      ThinPrismFisheyePrincipalPoint_w_num_alloc,
      ThinPrismFisheyePrincipalPoint_total_ag,
      ThinPrismFisheyePrincipalPoint_total_ac,
      problem_size);
}

}  // namespace caspar