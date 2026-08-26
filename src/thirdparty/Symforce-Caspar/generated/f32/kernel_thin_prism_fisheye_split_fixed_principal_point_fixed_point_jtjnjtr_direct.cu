#include "kernel_thin_prism_fisheye_split_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPrincipalPointFixedPointJtjnjtrDirectKernel(
        float* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        float* pose_jac,
        unsigned int pose_jac_num_alloc,
        float* focal_and_extra_njtr,
        unsigned int focal_and_extra_njtr_num_alloc,
        SharedIndex* focal_and_extra_njtr_indices,
        float* focal_and_extra_jac,
        unsigned int focal_and_extra_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        size_t problem_size) {
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
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1, r2, r3);
  };
  LoadShared<4, float, float>(focal_and_extra_njtr,
                              0 * focal_and_extra_njtr_num_alloc,
                              focal_and_extra_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(focal_and_extra_jac,
                                         4 * focal_and_extra_jac_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r9,
                                         r10,
                                         r11);
    ReadIdx4<1024, float, float, float4>(focal_and_extra_jac,
                                         0 * focal_and_extra_jac_num_alloc,
                                         global_thread_idx,
                                         r12,
                                         r13,
                                         r14,
                                         r15);
    r16 = fmaf(r6, r15, r7 * r9);
  };
  LoadShared<4, float, float>(focal_and_extra_njtr,
                              4 * focal_and_extra_njtr_num_alloc,
                              focal_and_extra_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                       r17,
                       r18,
                       r19,
                       r20);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(focal_and_extra_jac,
                                         12 * focal_and_extra_jac_num_alloc,
                                         global_thread_idx,
                                         r21,
                                         r22,
                                         r23,
                                         r24);
    ReadIdx4<1024, float, float, float4>(focal_and_extra_jac,
                                         8 * focal_and_extra_jac_num_alloc,
                                         global_thread_idx,
                                         r25,
                                         r26,
                                         r27,
                                         r28);
  };
  LoadShared<2, float, float>(focal_and_extra_njtr,
                              8 * focal_and_extra_njtr_num_alloc,
                              focal_and_extra_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                       r29,
                       r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = fmaf(r17, r11, r16);
    r16 = fmaf(r5, r13, r16);
    r16 = fmaf(r20, r22, r16);
    r16 = fmaf(r19, r28, r16);
    r16 = fmaf(r30, r24, r16);
    r16 = fmaf(r18, r26, r16);
    r17 = fmaf(r17, r10, r7 * r8);
    r17 = fmaf(r4, r12, r17);
    r17 = fmaf(r6, r14, r17);
    r17 = fmaf(r29, r23, r17);
    r17 = fmaf(r18, r25, r17);
    r17 = fmaf(r20, r21, r17);
    r17 = fmaf(r19, r27, r17);
    r19 = fmaf(r0, r17, r1 * r16);
    r20 = fmaf(r2, r17, r3 * r16);
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r18, r29, r6, r4);
    r7 = fmaf(r18, r17, r29 * r16);
    r30 = fmaf(r6, r17, r4 * r16);
    WriteSum4<float, float>((float*)inout_shared, r19, r20, r7, r30);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r30, r7, r20, r19);
    r5 = fmaf(r30, r17, r7 * r16);
    r17 = fmaf(r20, r17, r19 * r16);
    WriteSum2<float, float>((float*)inout_shared, r5, r17);
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
                       r17,
                       r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r30 = fmaf(r17, r30, r5 * r20);
  };
  LoadShared<4, float, float>(pose_njtr,
                              0 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target,
                       r20,
                       r16,
                       r31,
                       r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r30 = fmaf(r31, r18, r30);
    r30 = fmaf(r32, r6, r30);
    r30 = fmaf(r20, r0, r30);
    r30 = fmaf(r16, r2, r30);
    r12 = r12 * r30;
    r4 = fmaf(r32, r4, r5 * r19);
    r4 = fmaf(r31, r29, r4);
    r4 = fmaf(r20, r1, r4);
    r4 = fmaf(r16, r3, r4);
    r4 = fmaf(r17, r7, r4);
    r13 = r13 * r4;
    r15 = fmaf(r15, r4, r14 * r30);
    r9 = fmaf(r9, r4, r8 * r30);
    WriteSum4<float, float>((float*)inout_shared, r12, r13, r15, r9);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r11, r4, r10 * r30);
    r26 = fmaf(r26, r4, r25 * r30);
    r28 = fmaf(r28, r4, r27 * r30);
    r22 = fmaf(r22, r4, r21 * r30);
    WriteSum4<float, float>((float*)inout_shared, r11, r26, r28, r22);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           4 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = r23 * r30;
    r4 = r24 * r4;
    WriteSum2<float, float>((float*)inout_shared, r30, r4);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           8 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_njtr_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeSplitFixedPrincipalPointFixedPointJtjnjtrDirect(
    float* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    float* pose_jac,
    unsigned int pose_jac_num_alloc,
    float* focal_and_extra_njtr,
    unsigned int focal_and_extra_njtr_num_alloc,
    SharedIndex* focal_and_extra_njtr_indices,
    float* focal_and_extra_jac,
    unsigned int focal_and_extra_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedPrincipalPointFixedPointJtjnjtrDirectKernel<<<
      n_blocks,
      1024>>>(pose_njtr,
              pose_njtr_num_alloc,
              pose_njtr_indices,
              pose_jac,
              pose_jac_num_alloc,
              focal_and_extra_njtr,
              focal_and_extra_njtr_num_alloc,
              focal_and_extra_njtr_indices,
              focal_and_extra_jac,
              focal_and_extra_jac_num_alloc,
              out_pose_njtr,
              out_pose_njtr_num_alloc,
              out_focal_and_extra_njtr,
              out_focal_and_extra_njtr_num_alloc,
              problem_size);
}

}  // namespace caspar