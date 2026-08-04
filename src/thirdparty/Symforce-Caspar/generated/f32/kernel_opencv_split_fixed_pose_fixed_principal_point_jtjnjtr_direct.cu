#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_split_fixed_pose_fixed_principal_point_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedPoseFixedPrincipalPointJtjnjtrDirectKernel(
        float *focal_and_extra_njtr,
        unsigned int focal_and_extra_njtr_num_alloc,
        SharedIndex *focal_and_extra_njtr_indices, float *focal_and_extra_jac,
        unsigned int focal_and_extra_jac_num_alloc, float *point_njtr,
        unsigned int point_njtr_num_alloc, SharedIndex *point_njtr_indices,
        float *point_jac, unsigned int point_jac_num_alloc,
        float *const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

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
      r16, r17, r18, r19, r20;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(focal_and_extra_jac,
                                         0 * focal_and_extra_jac_num_alloc,
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
    r6 = fmaf(r3, r5, r2 * r13);
    ReadIdx4<1024, float, float, float4>(focal_and_extra_jac,
                                         4 * focal_and_extra_jac_num_alloc,
                                         global_thread_idx, r15, r16, r17, r18);
    r19 = fmaf(r16, r5, r15 * r13);
    WriteSum4<float, float>((float *)inout_shared, r14, r4, r6, r19);
  };
  FlushSumShared<4, float>(
      out_focal_and_extra_njtr, 0 * out_focal_and_extra_njtr_num_alloc,
      focal_and_extra_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fmaf(r18, r5, r17 * r13);
    ReadIdx2<1024, float, float, float2>(focal_and_extra_jac,
                                         8 * focal_and_extra_jac_num_alloc,
                                         global_thread_idx, r6, r4);
    r5 = fmaf(r4, r5, r6 * r13);
    WriteSum2<float, float>((float *)inout_shared, r19, r5);
  };
  FlushSumShared<2, float>(
      out_focal_and_extra_njtr, 4 * out_focal_and_extra_njtr_num_alloc,
      focal_and_extra_njtr_indices_loc, (float *)inout_shared);
  LoadShared<4, float, float>(
      focal_and_extra_njtr, 0 * focal_and_extra_njtr_num_alloc,
      focal_and_extra_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       focal_and_extra_njtr_indices_loc[threadIdx.x].target, r5,
                       r19, r13, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = fmaf(r13, r3, r14 * r16);
  };
  LoadShared<2, float, float>(
      focal_and_extra_njtr, 4 * focal_and_extra_njtr_num_alloc,
      focal_and_extra_njtr_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                       r16, r20);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = fmaf(r16, r18, r3);
    r3 = fmaf(r19, r1, r3);
    r3 = fmaf(r20, r4, r3);
    r17 = fmaf(r16, r17, r14 * r15);
    r17 = fmaf(r5, r0, r17);
    r17 = fmaf(r13, r2, r17);
    r17 = fmaf(r20, r6, r17);
    r9 = fmaf(r9, r17, r10 * r3);
    r11 = fmaf(r11, r17, r12 * r3);
    r17 = fmaf(r7, r17, r8 * r3);
    WriteSum3<float, float>((float *)inout_shared, r9, r11, r17);
  };
  FlushSumShared<3, float>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                           point_njtr_indices_loc, (float *)inout_shared);
}

void OpencvSplitFixedPoseFixedPrincipalPointJtjnjtrDirect(
    float *focal_and_extra_njtr, unsigned int focal_and_extra_njtr_num_alloc,
    SharedIndex *focal_and_extra_njtr_indices, float *focal_and_extra_jac,
    unsigned int focal_and_extra_jac_num_alloc, float *point_njtr,
    unsigned int point_njtr_num_alloc, SharedIndex *point_njtr_indices,
    float *point_jac, unsigned int point_jac_num_alloc,
    float *const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    float *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedPoseFixedPrincipalPointJtjnjtrDirectKernel<<<n_blocks,
                                                               1024>>>(
      focal_and_extra_njtr, focal_and_extra_njtr_num_alloc,
      focal_and_extra_njtr_indices, focal_and_extra_jac,
      focal_and_extra_jac_num_alloc, point_njtr, point_njtr_num_alloc,
      point_njtr_indices, point_jac, point_jac_num_alloc,
      out_focal_and_extra_njtr, out_focal_and_extra_njtr_num_alloc,
      out_point_njtr, out_point_njtr_num_alloc, problem_size);
}

} // namespace caspar