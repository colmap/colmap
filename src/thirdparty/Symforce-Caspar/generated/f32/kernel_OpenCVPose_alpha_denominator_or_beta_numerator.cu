#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_OpenCVPose_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpenCVPoseAlphaDenominatorOrBetaNumeratorKernel(
        float *OpenCVPose_p_kp1, unsigned int OpenCVPose_p_kp1_num_alloc,
        float *OpenCVPose_w, unsigned int OpenCVPose_w_num_alloc,
        float *const OpenCVPose_out, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float OpenCVPose_out_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(OpenCVPose_p_kp1,
                                         0 * OpenCVPose_p_kp1_num_alloc,
                                         global_thread_idx, r0, r1, r2, r3);
    ReadIdx4<1024, float, float, float4>(OpenCVPose_w,
                                         0 * OpenCVPose_w_num_alloc,
                                         global_thread_idx, r4, r5, r6, r7);
    r7 = fmaf(r3, r7, r1 * r5);
    ReadIdx2<1024, float, float, float2>(OpenCVPose_p_kp1,
                                         4 * OpenCVPose_p_kp1_num_alloc,
                                         global_thread_idx, r3, r5);
    ReadIdx2<1024, float, float, float2>(
        OpenCVPose_w, 4 * OpenCVPose_w_num_alloc, global_thread_idx, r1, r8);
    r7 = fmaf(r0, r4, r7);
    r7 = fmaf(r2, r6, r7);
    r7 = fmaf(r3, r1, r7);
    r7 = fmaf(r5, r8, r7);
  };
  SumStore<float>(OpenCVPose_out_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r7);
  SumFlushFinal<float>(OpenCVPose_out_local, OpenCVPose_out, 1);
}

void OpenCVPoseAlphaDenominatorOrBetaNumerator(
    float *OpenCVPose_p_kp1, unsigned int OpenCVPose_p_kp1_num_alloc,
    float *OpenCVPose_w, unsigned int OpenCVPose_w_num_alloc,
    float *const OpenCVPose_out, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpenCVPoseAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks, 1024>>>(
      OpenCVPose_p_kp1, OpenCVPose_p_kp1_num_alloc, OpenCVPose_w,
      OpenCVPose_w_num_alloc, OpenCVPose_out, problem_size);
}

} // namespace caspar