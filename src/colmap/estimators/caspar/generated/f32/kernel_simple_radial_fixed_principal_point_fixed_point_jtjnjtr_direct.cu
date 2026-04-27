#include "kernel_simple_radial_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_principal_point_fixed_point_jtjnjtr_direct_kernel(
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
      r16, r17, r18, r19, r20;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1, r2, r3);
  };
  load_shared<2, float, float>(focal_and_extra_njtr,
                               0 * focal_and_extra_njtr_num_alloc,
                               focal_and_extra_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                         r4,
                         r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(focal_and_extra_jac,
                                           0 * focal_and_extra_jac_num_alloc,
                                           global_thread_idx,
                                           r6,
                                           r7,
                                           r8,
                                           r9);
    r10 = fmaf(r5, r9, r4 * r7);
    r5 = fmaf(r5, r8, r4 * r6);
    r4 = fmaf(r0, r5, r1 * r10);
    r11 = fmaf(r2, r5, r3 * r10);
    read_idx_4<1024, float, float, float4>(pose_jac,
                                           4 * pose_jac_num_alloc,
                                           global_thread_idx,
                                           r12,
                                           r13,
                                           r14,
                                           r15);
    r16 = fmaf(r12, r5, r13 * r10);
    r17 = fmaf(r14, r5, r15 * r10);
    write_sum_4<float, float>((float*)inout_shared, r4, r11, r16, r17);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r17, r16, r11, r4);
    r18 = fmaf(r17, r5, r16 * r10);
    r5 = fmaf(r11, r5, r4 * r10);
    write_sum_2<float, float>((float*)inout_shared, r18, r5);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_njtr_indices_loc,
                             (float*)inout_shared);
  load_shared<2, float, float>(pose_njtr,
                               4 * pose_njtr_num_alloc,
                               pose_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         pose_njtr_indices_loc[threadIdx.x].target,
                         r5,
                         r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r17 = fmaf(r5, r17, r18 * r11);
  };
  load_shared<4, float, float>(pose_njtr,
                               0 * pose_njtr_num_alloc,
                               pose_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_njtr_indices_loc[threadIdx.x].target,
                         r11,
                         r10,
                         r19,
                         r20);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r17 = fmaf(r19, r12, r17);
    r17 = fmaf(r20, r14, r17);
    r17 = fmaf(r11, r0, r17);
    r17 = fmaf(r10, r2, r17);
    r15 = fmaf(r20, r15, r18 * r4);
    r15 = fmaf(r19, r13, r15);
    r15 = fmaf(r11, r1, r15);
    r15 = fmaf(r10, r3, r15);
    r15 = fmaf(r5, r16, r15);
    r7 = fmaf(r7, r15, r6 * r17);
    r15 = fmaf(r9, r15, r8 * r17);
    write_sum_2<float, float>((float*)inout_shared, r7, r15);
  };
  flush_sum_shared<2, float>(out_focal_and_extra_njtr,
                             0 * out_focal_and_extra_njtr_num_alloc,
                             focal_and_extra_njtr_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_principal_point_fixed_point_jtjnjtr_direct(
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
  simple_radial_fixed_principal_point_fixed_point_jtjnjtr_direct_kernel<<<
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