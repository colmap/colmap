#include "kernel_thin_prism_fisheye_split_fixed_principal_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPrincipalPointJtjnjtrDirectKernel(
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
        float* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        float* point_jac,
        unsigned int point_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
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

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42;

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
  };
  LoadShared<3, float, float>(point_njtr,
                              0 * point_njtr_num_alloc,
                              point_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_njtr_indices_loc[threadIdx.x].target,
                       r30,
                       r5,
                       r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r32, r33);
    ReadIdx4<1024, float, float, float4>(point_jac,
                                         0 * point_jac_num_alloc,
                                         global_thread_idx,
                                         r34,
                                         r35,
                                         r36,
                                         r37);
    r38 = fmaf(r5, r37, r31 * r33);
    r38 = fmaf(r30, r35, r38);
    r39 = r16 + r38;
    r17 = fmaf(r17, r10, r7 * r8);
    r17 = fmaf(r4, r12, r17);
    r17 = fmaf(r6, r14, r17);
    r17 = fmaf(r29, r23, r17);
    r17 = fmaf(r18, r25, r17);
    r17 = fmaf(r20, r21, r17);
    r17 = fmaf(r19, r27, r17);
    r5 = fmaf(r5, r36, r31 * r32);
    r5 = fmaf(r30, r34, r5);
    r30 = r17 + r5;
    r31 = fmaf(r0, r30, r1 * r39);
    r19 = fmaf(r2, r30, r3 * r39);
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r20, r18, r29, r6);
    r4 = fmaf(r20, r30, r18 * r39);
    r7 = fmaf(r29, r30, r6 * r39);
    WriteSum4<float, float>((float*)inout_shared, r31, r19, r4, r7);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r7, r4, r19, r31);
    r40 = fmaf(r7, r30, r4 * r39);
    r30 = fmaf(r19, r30, r31 * r39);
    WriteSum2<float, float>((float*)inout_shared, r40, r30);
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
                       r30,
                       r40);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r7 = fmaf(r30, r7, r40 * r19);
  };
  LoadShared<4, float, float>(pose_njtr,
                              0 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target,
                       r19,
                       r39,
                       r41,
                       r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r7 = fmaf(r41, r20, r7);
    r7 = fmaf(r42, r29, r7);
    r7 = fmaf(r19, r0, r7);
    r7 = fmaf(r39, r2, r7);
    r5 = r7 + r5;
    r12 = r12 * r5;
    r6 = fmaf(r42, r6, r40 * r31);
    r6 = fmaf(r41, r18, r6);
    r6 = fmaf(r19, r1, r6);
    r6 = fmaf(r39, r3, r6);
    r6 = fmaf(r30, r4, r6);
    r38 = r6 + r38;
    r13 = r13 * r38;
    r15 = fmaf(r15, r38, r14 * r5);
    r9 = fmaf(r9, r38, r8 * r5);
    WriteSum4<float, float>((float*)inout_shared, r12, r13, r15, r9);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r11, r38, r10 * r5);
    r26 = fmaf(r26, r38, r25 * r5);
    r28 = fmaf(r28, r38, r27 * r5);
    r22 = fmaf(r22, r38, r21 * r5);
    WriteSum4<float, float>((float*)inout_shared, r11, r26, r28, r22);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           4 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = r23 * r5;
    r38 = r24 * r38;
    WriteSum2<float, float>((float*)inout_shared, r5, r38);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           8 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r16 + r6;
    r7 = r17 + r7;
    r34 = fmaf(r34, r7, r35 * r6);
    r36 = fmaf(r36, r7, r37 * r6);
    r7 = fmaf(r32, r7, r33 * r6);
    WriteSum3<float, float>((float*)inout_shared, r34, r36, r7);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_njtr_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeSplitFixedPrincipalPointJtjnjtrDirect(
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
    float* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    float* point_jac,
    unsigned int point_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedPrincipalPointJtjnjtrDirectKernel<<<n_blocks,
                                                                1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      focal_and_extra_njtr,
      focal_and_extra_njtr_num_alloc,
      focal_and_extra_njtr_indices,
      focal_and_extra_jac,
      focal_and_extra_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar