#include "kernel_pinhole_split_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPrincipalPointFixedPointJtjnjtrDirectKernel(
        float* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        float* pose_jac,
        unsigned int pose_jac_num_alloc,
        float* focal_njtr,
        unsigned int focal_njtr_num_alloc,
        SharedIndex* focal_njtr_indices,
        float* focal_jac,
        unsigned int focal_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex focal_njtr_indices_loc[1024];
  focal_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1, r2, r3);
  };
  LoadShared<2, float, float>(focal_njtr,
                              0 * focal_njtr_num_alloc,
                              focal_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_njtr_indices_loc[threadIdx.x].target,
                       r4,
                       r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        focal_jac, 0 * focal_jac_num_alloc, global_thread_idx, r6, r7);
    r4 = r4 * r6;
    r5 = r5 * r7;
    r8 = fmaf(r1, r5, r0 * r4);
    r9 = fmaf(r3, r5, r2 * r4);
    ReadIdx4<1024, float, float, float4>(pose_jac,
                                         4 * pose_jac_num_alloc,
                                         global_thread_idx,
                                         r10,
                                         r11,
                                         r12,
                                         r13);
    r14 = fmaf(r10, r4, r11 * r5);
    r15 = fmaf(r13, r5, r12 * r4);
    WriteSum4<float, float>((float*)inout_shared, r8, r9, r14, r15);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r15, r14, r9, r8);
    r16 = fmaf(r15, r4, r14 * r5);
    r5 = fmaf(r8, r5, r9 * r4);
    WriteSum2<float, float>((float*)inout_shared, r16, r5);
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
                       r5,
                       r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r5, r15, r16 * r9);
  };
  LoadShared<4, float, float>(pose_njtr,
                              0 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target,
                       r9,
                       r4,
                       r17,
                       r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r17, r10, r15);
    r15 = fmaf(r18, r12, r15);
    r15 = fmaf(r9, r0, r15);
    r15 = fmaf(r4, r2, r15);
    r15 = r6 * r15;
    r13 = fmaf(r18, r13, r16 * r8);
    r13 = fmaf(r17, r11, r13);
    r13 = fmaf(r9, r1, r13);
    r13 = fmaf(r4, r3, r13);
    r13 = fmaf(r5, r14, r13);
    r13 = r7 * r13;
    WriteSum2<float, float>((float*)inout_shared, r15, r13);
  };
  FlushSumShared<2, float>(out_focal_njtr,
                           0 * out_focal_njtr_num_alloc,
                           focal_njtr_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitFixedPrincipalPointFixedPointJtjnjtrDirect(
    float* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    float* pose_jac,
    unsigned int pose_jac_num_alloc,
    float* focal_njtr,
    unsigned int focal_njtr_num_alloc,
    SharedIndex* focal_njtr_indices,
    float* focal_jac,
    unsigned int focal_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPrincipalPointFixedPointJtjnjtrDirectKernel<<<n_blocks,
                                                                 1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      focal_njtr,
      focal_njtr_num_alloc,
      focal_njtr_indices,
      focal_jac,
      focal_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar