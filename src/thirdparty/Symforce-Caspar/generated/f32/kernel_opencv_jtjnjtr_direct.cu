#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) OpencvJtjnjtrDirectKernel(
    float *pose_njtr, unsigned int pose_njtr_num_alloc,
    SharedIndex *pose_njtr_indices, float *pose_jac,
    unsigned int pose_jac_num_alloc, float *calib_njtr,
    unsigned int calib_njtr_num_alloc, SharedIndex *calib_njtr_indices,
    float *calib_jac, unsigned int calib_jac_num_alloc, float *point_njtr,
    unsigned int point_njtr_num_alloc, SharedIndex *point_njtr_indices,
    float *point_jac, unsigned int point_jac_num_alloc,
    float *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
    float *const out_calib_njtr, unsigned int out_calib_njtr_num_alloc,
    float *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
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
      r31, r32, r33, r34, r35, r36;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(pose_jac, 0 * pose_jac_num_alloc,
                                         global_thread_idx, r0, r1, r2, r3);
  };
  LoadShared<4, float, float>(calib_njtr, 4 * calib_njtr_num_alloc,
                              calib_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_njtr_indices_loc[threadIdx.x].target, r4, r5, r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(calib_jac, 8 * calib_jac_num_alloc,
                                         global_thread_idx, r8, r9);
    r7 = fmaf(r5, r9, r7);
    ReadIdx4<1024, float, float, float4>(calib_jac, 4 * calib_jac_num_alloc,
                                         global_thread_idx, r10, r11, r12, r13);
  };
  LoadShared<4, float, float>(calib_njtr, 0 * calib_njtr_num_alloc,
                              calib_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_njtr_indices_loc[threadIdx.x].target, r14, r15,
                       r16, r17);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(calib_jac, 0 * calib_jac_num_alloc,
                                         global_thread_idx, r18, r19, r20, r21);
    r7 = fmaf(r4, r13, r7);
    r7 = fmaf(r17, r11, r7);
    r7 = fmaf(r16, r21, r7);
    r7 = fmaf(r15, r19, r7);
  };
  LoadShared<3, float, float>(point_njtr, 0 * point_njtr_num_alloc,
                              point_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       point_njtr_indices_loc[threadIdx.x].target, r15, r22,
                       r23);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(point_jac, 4 * point_jac_num_alloc,
                                         global_thread_idx, r24, r25);
    ReadIdx4<1024, float, float, float4>(point_jac, 0 * point_jac_num_alloc,
                                         global_thread_idx, r26, r27, r28, r29);
    r30 = fmaf(r22, r29, r23 * r25);
    r30 = fmaf(r15, r27, r30);
    r31 = r7 + r30;
    r5 = fmaf(r5, r8, r6);
    r5 = fmaf(r4, r12, r5);
    r5 = fmaf(r16, r20, r5);
    r5 = fmaf(r17, r10, r5);
    r5 = fmaf(r14, r18, r5);
    r22 = fmaf(r22, r28, r23 * r24);
    r22 = fmaf(r15, r26, r22);
    r15 = r5 + r22;
    r23 = fmaf(r0, r15, r1 * r31);
    r14 = fmaf(r2, r15, r3 * r31);
    ReadIdx4<1024, float, float, float4>(pose_jac, 4 * pose_jac_num_alloc,
                                         global_thread_idx, r17, r16, r4, r6);
    r32 = fmaf(r17, r15, r16 * r31);
    r33 = fmaf(r4, r15, r6 * r31);
    WriteSum4<float, float>((float *)inout_shared, r23, r14, r32, r33);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(pose_jac, 8 * pose_jac_num_alloc,
                                         global_thread_idx, r33, r32, r14, r23);
    r34 = fmaf(r33, r15, r32 * r31);
    r31 = fmaf(r23, r31, r14 * r15);
    WriteSum2<float, float>((float *)inout_shared, r34, r31);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc, (float *)inout_shared);
  LoadShared<2, float, float>(pose_njtr, 4 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target, r31, r34);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r31, r33, r34 * r14);
  };
  LoadShared<4, float, float>(pose_njtr, 0 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target, r14, r15, r35,
                       r36);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r35, r17, r33);
    r33 = fmaf(r36, r4, r33);
    r33 = fmaf(r14, r0, r33);
    r33 = fmaf(r15, r2, r33);
    r22 = r33 + r22;
    r18 = r18 * r22;
    r6 = fmaf(r36, r6, r34 * r23);
    r6 = fmaf(r35, r16, r6);
    r6 = fmaf(r14, r1, r6);
    r6 = fmaf(r15, r3, r6);
    r6 = fmaf(r31, r32, r6);
    r30 = r6 + r30;
    r19 = r19 * r30;
    r21 = fmaf(r21, r30, r20 * r22);
    r11 = fmaf(r11, r30, r10 * r22);
    WriteSum4<float, float>((float *)inout_shared, r18, r19, r21, r11);
  };
  FlushSumShared<4, float>(out_calib_njtr, 0 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fmaf(r13, r30, r12 * r22);
    r8 = fmaf(r8, r22, r9 * r30);
    WriteSum4<float, float>((float *)inout_shared, r13, r8, r22, r30);
  };
  FlushSumShared<4, float>(out_calib_njtr, 4 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r5 + r33;
    r6 = r7 + r6;
    r27 = fmaf(r27, r6, r26 * r33);
    r29 = fmaf(r29, r6, r28 * r33);
    r6 = fmaf(r25, r6, r24 * r33);
    WriteSum3<float, float>((float *)inout_shared, r27, r29, r6);
  };
  FlushSumShared<3, float>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                           point_njtr_indices_loc, (float *)inout_shared);
}

void OpencvJtjnjtrDirect(
    float *pose_njtr, unsigned int pose_njtr_num_alloc,
    SharedIndex *pose_njtr_indices, float *pose_jac,
    unsigned int pose_jac_num_alloc, float *calib_njtr,
    unsigned int calib_njtr_num_alloc, SharedIndex *calib_njtr_indices,
    float *calib_jac, unsigned int calib_jac_num_alloc, float *point_njtr,
    unsigned int point_njtr_num_alloc, SharedIndex *point_njtr_indices,
    float *point_jac, unsigned int point_jac_num_alloc,
    float *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
    float *const out_calib_njtr, unsigned int out_calib_njtr_num_alloc,
    float *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      pose_njtr, pose_njtr_num_alloc, pose_njtr_indices, pose_jac,
      pose_jac_num_alloc, calib_njtr, calib_njtr_num_alloc, calib_njtr_indices,
      calib_jac, calib_jac_num_alloc, point_njtr, point_njtr_num_alloc,
      point_njtr_indices, point_jac, point_jac_num_alloc, out_pose_njtr,
      out_pose_njtr_num_alloc, out_calib_njtr, out_calib_njtr_num_alloc,
      out_point_njtr, out_point_njtr_num_alloc, problem_size);
}

} // namespace caspar