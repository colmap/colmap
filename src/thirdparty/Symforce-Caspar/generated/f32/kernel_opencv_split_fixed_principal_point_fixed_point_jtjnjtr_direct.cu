#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_split_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedPrincipalPointFixedPointJtjnjtrDirectKernel(
        float *pose_njtr, unsigned int pose_njtr_num_alloc,
        SharedIndex *pose_njtr_indices, float *pose_jac,
        unsigned int pose_jac_num_alloc, float *focal_and_extra_njtr,
        unsigned int focal_and_extra_njtr_num_alloc,
        SharedIndex *focal_and_extra_njtr_indices, float *focal_and_extra_jac,
        unsigned int focal_and_extra_jac_num_alloc, float *const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float *const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex focal_and_extra_njtr_indices_loc[1024];
  focal_and_extra_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(pose_jac, 0 * pose_jac_num_alloc,
                                         global_thread_idx, r0, r1, r2, r3);
  };
  LoadShared<4, float, float>(
      focal_and_extra_njtr, 0 * focal_and_extra_njtr_num_alloc,
      focal_and_extra_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       focal_and_extra_njtr_indices_loc[threadIdx.x].target, r4,
                       r5, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(focal_and_extra_jac,
                                         4 * focal_and_extra_jac_num_alloc,
                                         global_thread_idx, r8, r9, r10, r11);
  };
  LoadShared<2, float, float>(
      focal_and_extra_njtr, 4 * focal_and_extra_njtr_num_alloc,
      focal_and_extra_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                       r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = fmaf(r12, r10, r7 * r8);
    ReadIdx4<1024, float, float, float4>(focal_and_extra_jac,
                                         0 * focal_and_extra_jac_num_alloc,
                                         global_thread_idx, r15, r16, r17, r18);
    ReadIdx2<1024, float, float, float2>(focal_and_extra_jac,
                                         8 * focal_and_extra_jac_num_alloc,
                                         global_thread_idx, r19, r20);
    r14 = fmaf(r4, r15, r14);
    r14 = fmaf(r6, r17, r14);
    r14 = fmaf(r13, r19, r14);
    r6 = fmaf(r6, r18, r7 * r9);
    r6 = fmaf(r12, r11, r6);
    r6 = fmaf(r5, r16, r6);
    r6 = fmaf(r13, r20, r6);
    r13 = fmaf(r1, r6, r0 * r14);
    r5 = fmaf(r3, r6, r2 * r14);
    ReadIdx4<1024, float, float, float4>(pose_jac, 4 * pose_jac_num_alloc,
                                         global_thread_idx, r12, r7, r4, r21);
    r22 = fmaf(r7, r6, r12 * r14);
    r23 = fmaf(r21, r6, r4 * r14);
    WriteSum4<float, float>((float *)inout_shared, r13, r5, r22, r23);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(pose_jac, 8 * pose_jac_num_alloc,
                                         global_thread_idx, r23, r22, r5, r13);
    r24 = fmaf(r22, r6, r23 * r14);
    r6 = fmaf(r13, r6, r5 * r14);
    WriteSum2<float, float>((float *)inout_shared, r24, r6);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc, (float *)inout_shared);
  LoadShared<2, float, float>(pose_njtr, 4 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target, r6, r24);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r23 = fmaf(r6, r23, r24 * r5);
  };
  LoadShared<4, float, float>(pose_njtr, 0 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target, r5, r14, r25,
                       r26);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r23 = fmaf(r25, r12, r23);
    r23 = fmaf(r26, r4, r23);
    r23 = fmaf(r5, r0, r23);
    r23 = fmaf(r14, r2, r23);
    r15 = r15 * r23;
    r21 = fmaf(r26, r21, r24 * r13);
    r21 = fmaf(r25, r7, r21);
    r21 = fmaf(r5, r1, r21);
    r21 = fmaf(r14, r3, r21);
    r21 = fmaf(r6, r22, r21);
    r16 = r16 * r21;
    r18 = fmaf(r18, r21, r17 * r23);
    r9 = fmaf(r9, r21, r8 * r23);
    WriteSum4<float, float>((float *)inout_shared, r15, r16, r18, r9);
  };
  FlushSumShared<4, float>(
      out_focal_and_extra_njtr, 0 * out_focal_and_extra_njtr_num_alloc,
      focal_and_extra_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r11, r21, r10 * r23);
    r21 = fmaf(r20, r21, r19 * r23);
    WriteSum2<float, float>((float *)inout_shared, r11, r21);
  };
  FlushSumShared<2, float>(
      out_focal_and_extra_njtr, 4 * out_focal_and_extra_njtr_num_alloc,
      focal_and_extra_njtr_indices_loc, (float *)inout_shared);
}

void OpencvSplitFixedPrincipalPointFixedPointJtjnjtrDirect(
    float *pose_njtr, unsigned int pose_njtr_num_alloc,
    SharedIndex *pose_njtr_indices, float *pose_jac,
    unsigned int pose_jac_num_alloc, float *focal_and_extra_njtr,
    unsigned int focal_and_extra_njtr_num_alloc,
    SharedIndex *focal_and_extra_njtr_indices, float *focal_and_extra_jac,
    unsigned int focal_and_extra_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedPrincipalPointFixedPointJtjnjtrDirectKernel<<<n_blocks,
                                                                1024>>>(
      pose_njtr, pose_njtr_num_alloc, pose_njtr_indices, pose_jac,
      pose_jac_num_alloc, focal_and_extra_njtr, focal_and_extra_njtr_num_alloc,
      focal_and_extra_njtr_indices, focal_and_extra_jac,
      focal_and_extra_jac_num_alloc, out_pose_njtr, out_pose_njtr_num_alloc,
      out_focal_and_extra_njtr, out_focal_and_extra_njtr_num_alloc,
      problem_size);
}

} // namespace caspar