#include "kernel_SphericalPose_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SphericalPoseAlphaNumeratorDenominatorKernel(
        float* SphericalPose_p_kp1,
        unsigned int SphericalPose_p_kp1_num_alloc,
        float* SphericalPose_r_k,
        unsigned int SphericalPose_r_k_num_alloc,
        float* SphericalPose_w,
        unsigned int SphericalPose_w_num_alloc,
        float* const SphericalPose_total_ag,
        float* const SphericalPose_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float SphericalPose_total_ag_local[1];

  __shared__ float SphericalPose_total_ac_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(SphericalPose_p_kp1,
                                         4 * SphericalPose_p_kp1_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1);
    ReadIdx2<1024, float, float, float2>(SphericalPose_r_k,
                                         4 * SphericalPose_r_k_num_alloc,
                                         global_thread_idx,
                                         r2,
                                         r3);
    r2 = fmaf(r0, r2, r1 * r3);
    ReadIdx4<1024, float, float, float4>(SphericalPose_p_kp1,
                                         0 * SphericalPose_p_kp1_num_alloc,
                                         global_thread_idx,
                                         r3,
                                         r4,
                                         r5,
                                         r6);
    ReadIdx4<1024, float, float, float4>(SphericalPose_r_k,
                                         0 * SphericalPose_r_k_num_alloc,
                                         global_thread_idx,
                                         r7,
                                         r8,
                                         r9,
                                         r10);
    r2 = fmaf(r3, r7, r2);
    r2 = fmaf(r4, r8, r2);
    r2 = fmaf(r5, r9, r2);
    r2 = fmaf(r6, r10, r2);
  };
  SumStore<float>(SphericalPose_total_ag_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r2);
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(SphericalPose_w,
                                         0 * SphericalPose_w_num_alloc,
                                         global_thread_idx,
                                         r2,
                                         r10,
                                         r9,
                                         r8);
    ReadIdx2<1024, float, float, float2>(SphericalPose_w,
                                         4 * SphericalPose_w_num_alloc,
                                         global_thread_idx,
                                         r7,
                                         r11);
    r7 = fmaf(r0, r7, r6 * r8);
    r7 = fmaf(r1, r11, r7);
    r7 = fmaf(r3, r2, r7);
    r7 = fmaf(r5, r9, r7);
    r7 = fmaf(r4, r10, r7);
  };
  SumStore<float>(SphericalPose_total_ac_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r7);
  SumFlushFinal<float>(SphericalPose_total_ag_local, SphericalPose_total_ag, 1);
  SumFlushFinal<float>(SphericalPose_total_ac_local, SphericalPose_total_ac, 1);
}

void SphericalPoseAlphaNumeratorDenominator(
    float* SphericalPose_p_kp1,
    unsigned int SphericalPose_p_kp1_num_alloc,
    float* SphericalPose_r_k,
    unsigned int SphericalPose_r_k_num_alloc,
    float* SphericalPose_w,
    unsigned int SphericalPose_w_num_alloc,
    float* const SphericalPose_total_ag,
    float* const SphericalPose_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalPoseAlphaNumeratorDenominatorKernel<<<n_blocks, 1024>>>(
      SphericalPose_p_kp1,
      SphericalPose_p_kp1_num_alloc,
      SphericalPose_r_k,
      SphericalPose_r_k_num_alloc,
      SphericalPose_w,
      SphericalPose_w_num_alloc,
      SphericalPose_total_ag,
      SphericalPose_total_ac,
      problem_size);
}

}  // namespace caspar