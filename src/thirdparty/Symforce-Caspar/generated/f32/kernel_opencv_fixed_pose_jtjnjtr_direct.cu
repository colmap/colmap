#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_fixed_pose_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) OpencvFixedPoseJtjnjtrDirectKernel(
    float *calib_njtr, unsigned int calib_njtr_num_alloc,
    SharedIndex *calib_njtr_indices, float *calib_jac,
    unsigned int calib_jac_num_alloc, float *point_njtr,
    unsigned int point_njtr_num_alloc, SharedIndex *point_njtr_indices,
    float *point_jac, unsigned int point_jac_num_alloc,
    float *const out_calib_njtr, unsigned int out_calib_njtr_num_alloc,
    float *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
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
      r16, r17, r18, r19, r20, r21, r22;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(calib_jac, 0 * calib_jac_num_alloc,
                                         global_thread_idx, r0, r1, r2, r3);
  };
  LoadShared<3, float, float>(point_njtr, 0 * point_njtr_num_alloc,
                              point_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       point_njtr_indices_loc[threadIdx.x].target, r4, r5, r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(point_jac, 4 * point_jac_num_alloc,
                                         global_thread_idx, r7, r8);
    ReadIdx4<1024, float, float, float4>(point_jac, 0 * point_jac_num_alloc,
                                         global_thread_idx, r9, r10, r11, r12);
    r13 = fmaf(r5, r11, r6 * r7);
    r13 = fmaf(r4, r9, r13);
    r14 = r0 * r13;
    r5 = fmaf(r5, r12, r6 * r8);
    r5 = fmaf(r4, r10, r5);
    r4 = r1 * r5;
    r6 = fmaf(r2, r13, r3 * r5);
    ReadIdx4<1024, float, float, float4>(calib_jac, 4 * calib_jac_num_alloc,
                                         global_thread_idx, r15, r16, r17, r18);
    r19 = fmaf(r15, r13, r16 * r5);
    WriteSum4<float, float>((float *)inout_shared, r14, r4, r6, r19);
  };
  FlushSumShared<4, float>(out_calib_njtr, 0 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fmaf(r17, r13, r18 * r5);
    ReadIdx2<1024, float, float, float2>(calib_jac, 8 * calib_jac_num_alloc,
                                         global_thread_idx, r6, r4);
    r14 = fmaf(r4, r5, r6 * r13);
    WriteSum4<float, float>((float *)inout_shared, r19, r14, r13, r5);
  };
  FlushSumShared<4, float>(out_calib_njtr, 4 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc, (float *)inout_shared);
  LoadShared<4, float, float>(calib_njtr, 4 * calib_njtr_num_alloc,
                              calib_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_njtr_indices_loc[threadIdx.x].target, r5, r13, r14,
                       r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = fmaf(r13, r6, r14);
  };
  LoadShared<4, float, float>(calib_njtr, 0 * calib_njtr_num_alloc,
                              calib_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_njtr_indices_loc[threadIdx.x].target, r14, r20,
                       r21, r22);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = fmaf(r5, r17, r6);
    r6 = fmaf(r21, r2, r6);
    r6 = fmaf(r22, r15, r6);
    r6 = fmaf(r14, r0, r6);
    r4 = fmaf(r13, r4, r19);
    r4 = fmaf(r5, r18, r4);
    r4 = fmaf(r22, r16, r4);
    r4 = fmaf(r21, r3, r4);
    r4 = fmaf(r20, r1, r4);
    r10 = fmaf(r10, r4, r9 * r6);
    r12 = fmaf(r12, r4, r11 * r6);
    r4 = fmaf(r8, r4, r7 * r6);
    WriteSum3<float, float>((float *)inout_shared, r10, r12, r4);
  };
  FlushSumShared<3, float>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                           point_njtr_indices_loc, (float *)inout_shared);
}

void OpencvFixedPoseJtjnjtrDirect(
    float *calib_njtr, unsigned int calib_njtr_num_alloc,
    SharedIndex *calib_njtr_indices, float *calib_jac,
    unsigned int calib_jac_num_alloc, float *point_njtr,
    unsigned int point_njtr_num_alloc, SharedIndex *point_njtr_indices,
    float *point_jac, unsigned int point_jac_num_alloc,
    float *const out_calib_njtr, unsigned int out_calib_njtr_num_alloc,
    float *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvFixedPoseJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      calib_njtr, calib_njtr_num_alloc, calib_njtr_indices, calib_jac,
      calib_jac_num_alloc, point_njtr, point_njtr_num_alloc, point_njtr_indices,
      point_jac, point_jac_num_alloc, out_calib_njtr, out_calib_njtr_num_alloc,
      out_point_njtr, out_point_njtr_num_alloc, problem_size);
}

} // namespace caspar