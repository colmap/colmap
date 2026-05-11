#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_pinhole_pose_prior_core_score.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) PinholePosePriorCoreScoreKernel(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *prior_position, unsigned int prior_position_num_alloc,
    float *sqrt_info, unsigned int sqrt_info_num_alloc, float *const out_rTr,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sqrt_info, 0 * sqrt_info_num_alloc,
                                         global_thread_idx, r0, r1, r2, r3);
    ReadIdx3<1024, float, float, float4>(prior_position,
                                         0 * prior_position_num_alloc,
                                         global_thread_idx, r4, r5, r6);
    r7 = -1.00000000000000000e+00;
  };
  LoadShared<3, float, float>(pose, 4 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r8, r9, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r11 = -2.00000000000000000e+00;
  };
  LoadShared<4, float, float>(pose, 0 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r12, r13, r14,
                       r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = r14 * r14;
    r16 = r11 * r16;
    r17 = 1.00000000000000000e+00;
    r18 = r12 * r12;
    r18 = fmaf(r11, r18, r17);
    r19 = r16 + r18;
    r20 = r12 * r15;
    r21 = 2.00000000000000000e+00;
    r22 = r13 * r21;
    r23 = r14 * r22;
    r20 = fmaf(r21, r20, r23);
    r20 = fmaf(r10, r20, r9 * r19);
    r19 = r15 * r11;
    r24 = r12 * r22;
    r25 = fmaf(r14, r19, r24);
    r20 = fmaf(r8, r25, r20);
    r20 = fmaf(r7, r20, r5 * r7);
    r5 = r13 * r13;
    r5 = r5 * r11;
    r18 = r5 + r18;
    r23 = fmaf(r12, r19, r23);
    r23 = fmaf(r9, r23, r10 * r18);
    r18 = r12 * r14;
    r18 = r18 * r21;
    r22 = fmaf(r15, r22, r18);
    r23 = fmaf(r8, r22, r23);
    r23 = fmaf(r7, r23, r6 * r7);
    r2 = fmaf(r2, r23, r1 * r20);
    r5 = r17 + r5;
    r5 = r5 + r16;
    r16 = r14 * r15;
    r16 = fmaf(r21, r16, r24);
    r16 = fmaf(r9, r16, r8 * r5);
    r19 = fmaf(r13, r19, r18);
    r16 = fmaf(r10, r19, r16);
    r16 = fmaf(r7, r16, r4 * r7);
    r2 = fmaf(r0, r16, r2);
    ReadIdx2<1024, float, float, float2>(sqrt_info, 4 * sqrt_info_num_alloc,
                                         global_thread_idx, r16, r0);
    r16 = fmaf(r16, r23, r3 * r20);
    r16 = fmaf(r16, r16, r2 * r2);
    r23 = r23 * r23;
    r0 = r0 * r0;
    r16 = fmaf(r23, r0, r16);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r16);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholePosePriorCoreScore(float *pose, unsigned int pose_num_alloc,
                               SharedIndex *pose_indices, float *prior_position,
                               unsigned int prior_position_num_alloc,
                               float *sqrt_info,
                               unsigned int sqrt_info_num_alloc,
                               float *const out_rTr, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholePosePriorCoreScoreKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, prior_position,
      prior_position_num_alloc, sqrt_info, sqrt_info_num_alloc, out_rTr,
      problem_size);
}

} // namespace caspar