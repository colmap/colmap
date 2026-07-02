#include "kernel_SphericalPose_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SphericalPoseAlphaDenominatorOrBetaNumeratorKernel(
        float* SphericalPose_p_kp1,
        unsigned int SphericalPose_p_kp1_num_alloc,
        float* SphericalPose_w,
        unsigned int SphericalPose_w_num_alloc,
        float* const SphericalPose_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float SphericalPose_out_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(SphericalPose_p_kp1,
                                         0 * SphericalPose_p_kp1_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
    ReadIdx4<1024, float, float, float4>(SphericalPose_w,
                                         0 * SphericalPose_w_num_alloc,
                                         global_thread_idx,
                                         r4,
                                         r5,
                                         r6,
                                         r7);
    ReadIdx2<1024, float, float, float2>(SphericalPose_p_kp1,
                                         4 * SphericalPose_p_kp1_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r9);
    ReadIdx2<1024, float, float, float2>(SphericalPose_w,
                                         4 * SphericalPose_w_num_alloc,
                                         global_thread_idx,
                                         r10,
                                         r11);
    r10 = fmaf(r8, r10, r3 * r7);
    r10 = fmaf(r9, r11, r10);
    r10 = fmaf(r0, r4, r10);
    r10 = fmaf(r2, r6, r10);
    r10 = fmaf(r1, r5, r10);
  };
  SumStore<float>(SphericalPose_out_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r10);
  SumFlushFinal<float>(SphericalPose_out_local, SphericalPose_out, 1);
}

void SphericalPoseAlphaDenominatorOrBetaNumerator(
    float* SphericalPose_p_kp1,
    unsigned int SphericalPose_p_kp1_num_alloc,
    float* SphericalPose_w,
    unsigned int SphericalPose_w_num_alloc,
    float* const SphericalPose_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalPoseAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks, 1024>>>(
      SphericalPose_p_kp1,
      SphericalPose_p_kp1_num_alloc,
      SphericalPose_w,
      SphericalPose_w_num_alloc,
      SphericalPose_out,
      problem_size);
}

}  // namespace caspar