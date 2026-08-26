#include "kernel_thin_prism_fisheye_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFixedPointJtjnjtrDirectKernel(
        float* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        float* pose_jac,
        unsigned int pose_jac_num_alloc,
        float* calib_njtr,
        unsigned int calib_njtr_num_alloc,
        SharedIndex* calib_njtr_indices,
        float* calib_jac,
        unsigned int calib_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_njtr_indices_loc[1024];
  calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1, r2, r3);
  };
  LoadShared<4, float, float>(calib_njtr,
                              0 * calib_njtr_num_alloc,
                              calib_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_njtr_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  LoadShared<4, float, float>(calib_njtr,
                              8 * calib_njtr_num_alloc,
                              calib_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_njtr_indices_loc[threadIdx.x].target,
                       r8,
                       r9,
                       r10,
                       r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(calib_jac,
                                         12 * calib_jac_num_alloc,
                                         global_thread_idx,
                                         r12,
                                         r13,
                                         r14,
                                         r15);
    r11 = fmaf(r11, r15, r7);
    ReadIdx4<1024, float, float, float4>(calib_jac,
                                         8 * calib_jac_num_alloc,
                                         global_thread_idx,
                                         r7,
                                         r16,
                                         r17,
                                         r18);
  };
  LoadShared<4, float, float>(calib_njtr,
                              4 * calib_njtr_num_alloc,
                              calib_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_njtr_indices_loc[threadIdx.x].target,
                       r19,
                       r20,
                       r21,
                       r22);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(calib_jac,
                                         4 * calib_jac_num_alloc,
                                         global_thread_idx,
                                         r23,
                                         r24,
                                         r25,
                                         r26);
    ReadIdx4<1024, float, float, float4>(calib_jac,
                                         0 * calib_jac_num_alloc,
                                         global_thread_idx,
                                         r27,
                                         r28,
                                         r29,
                                         r30);
    r11 = fmaf(r8, r18, r11);
    r11 = fmaf(r21, r26, r11);
    r11 = fmaf(r22, r16, r11);
    r11 = fmaf(r9, r13, r11);
    r11 = fmaf(r20, r24, r11);
    r11 = fmaf(r19, r30, r11);
    r11 = fmaf(r5, r28, r11);
    r10 = fmaf(r10, r14, r6);
    r10 = fmaf(r9, r12, r10);
    r10 = fmaf(r8, r17, r10);
    r10 = fmaf(r20, r23, r10);
    r10 = fmaf(r22, r7, r10);
    r10 = fmaf(r21, r25, r10);
    r10 = fmaf(r19, r29, r10);
    r10 = fmaf(r4, r27, r10);
    r4 = fmaf(r0, r10, r1 * r11);
    r19 = fmaf(r2, r10, r3 * r11);
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r21, r22, r20, r8);
    r9 = fmaf(r21, r10, r22 * r11);
    r6 = fmaf(r20, r10, r8 * r11);
    WriteSum4<float, float>((float*)inout_shared, r4, r19, r9, r6);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r6, r9, r19, r4);
    r5 = fmaf(r6, r10, r9 * r11);
    r10 = fmaf(r19, r10, r4 * r11);
    WriteSum2<float, float>((float*)inout_shared, r5, r10);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc,
                           (float*)inout_shared);
  LoadShared<2, float, float>(pose_njtr,
                              4 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target,
                       r10,
                       r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = fmaf(r10, r6, r5 * r19);
  };
  LoadShared<4, float, float>(pose_njtr,
                              0 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target,
                       r19,
                       r11,
                       r31,
                       r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = fmaf(r31, r21, r6);
    r6 = fmaf(r32, r20, r6);
    r6 = fmaf(r19, r0, r6);
    r6 = fmaf(r11, r2, r6);
    r27 = r27 * r6;
    r8 = fmaf(r32, r8, r5 * r4);
    r8 = fmaf(r31, r22, r8);
    r8 = fmaf(r19, r1, r8);
    r8 = fmaf(r11, r3, r8);
    r8 = fmaf(r10, r9, r8);
    r28 = r28 * r8;
    WriteSum4<float, float>((float*)inout_shared, r27, r28, r6, r8);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fmaf(r30, r8, r29 * r6);
    r23 = fmaf(r23, r6, r24 * r8);
    r25 = fmaf(r25, r6, r26 * r8);
    r7 = fmaf(r7, r6, r16 * r8);
    WriteSum4<float, float>((float*)inout_shared, r30, r23, r25, r7);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           4 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = fmaf(r17, r6, r18 * r8);
    r12 = fmaf(r12, r6, r13 * r8);
    r6 = r14 * r6;
    r8 = r15 * r8;
    WriteSum4<float, float>((float*)inout_shared, r17, r12, r6, r8);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           8 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeFixedPointJtjnjtrDirect(
    float* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    float* pose_jac,
    unsigned int pose_jac_num_alloc,
    float* calib_njtr,
    unsigned int calib_njtr_num_alloc,
    SharedIndex* calib_njtr_indices,
    float* calib_jac,
    unsigned int calib_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFixedPointJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      calib_njtr,
      calib_njtr_num_alloc,
      calib_njtr_indices,
      calib_jac,
      calib_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar