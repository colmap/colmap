#include "kernel_thin_prism_fisheye_fixed_pose_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFixedPoseJtjnjtrDirectKernel(
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
        float* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

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
      r31;
  LoadShared<3, float, float>(point_njtr,
                              0 * point_njtr_num_alloc,
                              point_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_njtr_indices_loc[threadIdx.x].target,
                       r0,
                       r1,
                       r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r3, r4);
    ReadIdx4<1024, float, float, float4>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r5, r6, r7, r8);
    r9 = fmaf(r1, r7, r2 * r3);
    r9 = fmaf(r0, r5, r9);
    ReadIdx4<1024, float, float, float4>(calib_jac,
                                         0 * calib_jac_num_alloc,
                                         global_thread_idx,
                                         r10,
                                         r11,
                                         r12,
                                         r13);
    r14 = r10 * r9;
    r1 = fmaf(r1, r8, r2 * r4);
    r1 = fmaf(r0, r6, r1);
    r0 = r11 * r1;
    WriteSum4<float, float>((float*)inout_shared, r14, r0, r9, r1);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fmaf(r12, r9, r13 * r1);
    ReadIdx4<1024, float, float, float4>(calib_jac,
                                         4 * calib_jac_num_alloc,
                                         global_thread_idx,
                                         r14,
                                         r2,
                                         r15,
                                         r16);
    r17 = fmaf(r2, r1, r14 * r9);
    r18 = fmaf(r16, r1, r15 * r9);
    ReadIdx4<1024, float, float, float4>(calib_jac,
                                         8 * calib_jac_num_alloc,
                                         global_thread_idx,
                                         r19,
                                         r20,
                                         r21,
                                         r22);
    r23 = fmaf(r20, r1, r19 * r9);
    WriteSum4<float, float>((float*)inout_shared, r0, r17, r18, r23);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           4 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fmaf(r22, r1, r21 * r9);
    ReadIdx4<1024, float, float, float4>(calib_jac,
                                         12 * calib_jac_num_alloc,
                                         global_thread_idx,
                                         r18,
                                         r17,
                                         r0,
                                         r24);
    r25 = fmaf(r17, r1, r18 * r9);
    r9 = r0 * r9;
    r1 = r24 * r1;
    WriteSum4<float, float>((float*)inout_shared, r23, r25, r9, r1);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           8 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc,
                           (float*)inout_shared);
  LoadShared<4, float, float>(calib_njtr,
                              0 * calib_njtr_num_alloc,
                              calib_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_njtr_indices_loc[threadIdx.x].target,
                       r1,
                       r9,
                       r25,
                       r23);
  };
  __syncthreads();
  LoadShared<4, float, float>(calib_njtr,
                              8 * calib_njtr_num_alloc,
                              calib_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_njtr_indices_loc[threadIdx.x].target,
                       r26,
                       r27,
                       r28,
                       r29);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = fmaf(r28, r0, r25);
  };
  LoadShared<4, float, float>(calib_njtr,
                              4 * calib_njtr_num_alloc,
                              calib_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_njtr_indices_loc[threadIdx.x].target,
                       r28,
                       r25,
                       r30,
                       r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = fmaf(r27, r18, r0);
    r0 = fmaf(r26, r21, r0);
    r0 = fmaf(r25, r14, r0);
    r0 = fmaf(r31, r19, r0);
    r0 = fmaf(r30, r15, r0);
    r0 = fmaf(r28, r12, r0);
    r0 = fmaf(r1, r10, r0);
    r24 = fmaf(r29, r24, r23);
    r24 = fmaf(r26, r22, r24);
    r24 = fmaf(r30, r16, r24);
    r24 = fmaf(r31, r20, r24);
    r24 = fmaf(r27, r17, r24);
    r24 = fmaf(r25, r2, r24);
    r24 = fmaf(r28, r13, r24);
    r24 = fmaf(r9, r11, r24);
    r6 = fmaf(r6, r24, r5 * r0);
    r8 = fmaf(r8, r24, r7 * r0);
    r24 = fmaf(r4, r24, r3 * r0);
    WriteSum3<float, float>((float*)inout_shared, r6, r8, r24);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_njtr_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeFixedPoseJtjnjtrDirect(
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
    float* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFixedPoseJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
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
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar