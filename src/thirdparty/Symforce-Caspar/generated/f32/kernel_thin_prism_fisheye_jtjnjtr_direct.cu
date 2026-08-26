#include "kernel_thin_prism_fisheye_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeJtjnjtrDirectKernel(float* pose_njtr,
                                        unsigned int pose_njtr_num_alloc,
                                        SharedIndex* pose_njtr_indices,
                                        float* pose_jac,
                                        unsigned int pose_jac_num_alloc,
                                        float* calib_njtr,
                                        unsigned int calib_njtr_num_alloc,
                                        SharedIndex* calib_njtr_indices,
                                        float* calib_jac,
                                        unsigned int calib_jac_num_alloc,
                                        float* point_njtr,
                                        unsigned int point_njtr_num_alloc,
                                        SharedIndex* point_njtr_indices,
                                        float* point_jac,
                                        unsigned int point_jac_num_alloc,
                                        float* const out_pose_njtr,
                                        unsigned int out_pose_njtr_num_alloc,
                                        float* const out_calib_njtr,
                                        unsigned int out_calib_njtr_num_alloc,
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

  __shared__ SharedIndex calib_njtr_indices_loc[1024];
  calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_njtr_indices[global_thread_idx]
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
    r10 = fmaf(r10, r14, r6);
    ReadIdx4<1024, float, float, float4>(calib_jac,
                                         8 * calib_jac_num_alloc,
                                         global_thread_idx,
                                         r6,
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
    r10 = fmaf(r9, r12, r10);
    r10 = fmaf(r8, r17, r10);
    r10 = fmaf(r20, r23, r10);
    r10 = fmaf(r22, r6, r10);
    r10 = fmaf(r21, r25, r10);
    r10 = fmaf(r19, r29, r10);
    r10 = fmaf(r4, r27, r10);
  };
  LoadShared<3, float, float>(point_njtr,
                              0 * point_njtr_num_alloc,
                              point_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_njtr_indices_loc[threadIdx.x].target,
                       r4,
                       r31,
                       r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r33, r34);
    ReadIdx4<1024, float, float, float4>(point_jac,
                                         0 * point_jac_num_alloc,
                                         global_thread_idx,
                                         r35,
                                         r36,
                                         r37,
                                         r38);
    r39 = fmaf(r31, r37, r32 * r33);
    r39 = fmaf(r4, r35, r39);
    r40 = r10 + r39;
    r11 = fmaf(r11, r15, r7);
    r11 = fmaf(r8, r18, r11);
    r11 = fmaf(r21, r26, r11);
    r11 = fmaf(r22, r16, r11);
    r11 = fmaf(r9, r13, r11);
    r11 = fmaf(r20, r24, r11);
    r11 = fmaf(r19, r30, r11);
    r11 = fmaf(r5, r28, r11);
    r31 = fmaf(r31, r38, r32 * r34);
    r31 = fmaf(r4, r36, r31);
    r4 = r11 + r31;
    r32 = fmaf(r1, r4, r0 * r40);
    r5 = fmaf(r3, r4, r2 * r40);
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r19, r20, r9, r22);
    r21 = fmaf(r20, r4, r19 * r40);
    r8 = fmaf(r22, r4, r9 * r40);
    WriteSum4<float, float>((float*)inout_shared, r32, r5, r21, r8);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r8, r21, r5, r32);
    r7 = fmaf(r21, r4, r8 * r40);
    r4 = fmaf(r32, r4, r5 * r40);
    WriteSum2<float, float>((float*)inout_shared, r7, r4);
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
                       r4,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r4, r8, r7 * r5);
  };
  LoadShared<4, float, float>(pose_njtr,
                              0 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target,
                       r5,
                       r40,
                       r41,
                       r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r41, r19, r8);
    r8 = fmaf(r42, r9, r8);
    r8 = fmaf(r5, r0, r8);
    r8 = fmaf(r40, r2, r8);
    r39 = r8 + r39;
    r27 = r27 * r39;
    r22 = fmaf(r42, r22, r7 * r32);
    r22 = fmaf(r41, r20, r22);
    r22 = fmaf(r5, r1, r22);
    r22 = fmaf(r40, r3, r22);
    r22 = fmaf(r4, r21, r22);
    r31 = r22 + r31;
    r28 = r28 * r31;
    WriteSum4<float, float>((float*)inout_shared, r27, r28, r39, r31);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fmaf(r30, r31, r29 * r39);
    r23 = fmaf(r23, r39, r24 * r31);
    r25 = fmaf(r25, r39, r26 * r31);
    r6 = fmaf(r6, r39, r16 * r31);
    WriteSum4<float, float>((float*)inout_shared, r30, r23, r25, r6);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           4 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = fmaf(r17, r39, r18 * r31);
    r12 = fmaf(r12, r39, r13 * r31);
    r39 = r14 * r39;
    r31 = r15 * r31;
    WriteSum4<float, float>((float*)inout_shared, r17, r12, r39, r31);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           8 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = r10 + r8;
    r22 = r11 + r22;
    r36 = fmaf(r36, r22, r35 * r8);
    r38 = fmaf(r38, r22, r37 * r8);
    r22 = fmaf(r34, r22, r33 * r8);
    WriteSum3<float, float>((float*)inout_shared, r36, r38, r22);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_njtr_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeJtjnjtrDirect(float* pose_njtr,
                                   unsigned int pose_njtr_num_alloc,
                                   SharedIndex* pose_njtr_indices,
                                   float* pose_jac,
                                   unsigned int pose_jac_num_alloc,
                                   float* calib_njtr,
                                   unsigned int calib_njtr_num_alloc,
                                   SharedIndex* calib_njtr_indices,
                                   float* calib_jac,
                                   unsigned int calib_jac_num_alloc,
                                   float* point_njtr,
                                   unsigned int point_njtr_num_alloc,
                                   SharedIndex* point_njtr_indices,
                                   float* point_jac,
                                   unsigned int point_jac_num_alloc,
                                   float* const out_pose_njtr,
                                   unsigned int out_pose_njtr_num_alloc,
                                   float* const out_calib_njtr,
                                   unsigned int out_calib_njtr_num_alloc,
                                   float* const out_point_njtr,
                                   unsigned int out_point_njtr_num_alloc,
                                   size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
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
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar