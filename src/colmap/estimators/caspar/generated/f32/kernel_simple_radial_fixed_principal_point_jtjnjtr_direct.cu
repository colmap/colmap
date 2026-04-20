#include "kernel_simple_radial_fixed_principal_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_principal_point_jtjnjtr_direct_kernel(
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
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1, r2, r3);
  };
  load_shared<3, float, float>(point_njtr,
                               0 * point_njtr_num_alloc,
                               point_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         point_njtr_indices_loc[threadIdx.x].target,
                         r4,
                         r5,
                         r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r7, r8);
    read_idx_4<1024, float, float, float4>(point_jac,
                                           0 * point_jac_num_alloc,
                                           global_thread_idx,
                                           r9,
                                           r10,
                                           r11,
                                           r12);
    r13 = fmaf(r5, r11, r6 * r7);
    r13 = fmaf(r4, r9, r13);
  };
  load_shared<2, float, float>(focal_and_extra_njtr,
                               0 * focal_and_extra_njtr_num_alloc,
                               focal_and_extra_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                         r14,
                         r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(focal_and_extra_jac,
                                           0 * focal_and_extra_jac_num_alloc,
                                           global_thread_idx,
                                           r16,
                                           r17,
                                           r18,
                                           r19);
    r20 = fmaf(r15, r18, r14 * r16);
    r21 = r13 + r20;
    r5 = fmaf(r5, r12, r6 * r8);
    r5 = fmaf(r4, r10, r5);
    r15 = fmaf(r15, r19, r14 * r17);
    r14 = r5 + r15;
    r4 = fmaf(r1, r14, r0 * r21);
    r6 = fmaf(r3, r14, r2 * r21);
    read_idx_4<1024, float, float, float4>(pose_jac,
                                           4 * pose_jac_num_alloc,
                                           global_thread_idx,
                                           r22,
                                           r23,
                                           r24,
                                           r25);
    r26 = fmaf(r23, r14, r22 * r21);
    r27 = fmaf(r25, r14, r24 * r21);
    write_sum_4<float, float>((float*)inout_shared, r4, r6, r26, r27);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r27, r26, r6, r4);
    r28 = fmaf(r26, r14, r27 * r21);
    r21 = fmaf(r6, r21, r4 * r14);
    write_sum_2<float, float>((float*)inout_shared, r28, r21);
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
                         r21,
                         r28);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r27 = fmaf(r21, r27, r28 * r6);
  };
  load_shared<4, float, float>(pose_njtr,
                               0 * pose_njtr_num_alloc,
                               pose_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_njtr_indices_loc[threadIdx.x].target,
                         r6,
                         r14,
                         r29,
                         r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r27 = fmaf(r29, r22, r27);
    r27 = fmaf(r30, r24, r27);
    r27 = fmaf(r6, r0, r27);
    r27 = fmaf(r14, r2, r27);
    r13 = r27 + r13;
    r25 = fmaf(r30, r25, r28 * r4);
    r25 = fmaf(r29, r23, r25);
    r25 = fmaf(r6, r1, r25);
    r25 = fmaf(r14, r3, r25);
    r25 = fmaf(r21, r26, r25);
    r5 = r25 + r5;
    r17 = fmaf(r17, r5, r16 * r13);
    r5 = fmaf(r19, r5, r18 * r13);
    write_sum_2<float, float>((float*)inout_shared, r17, r5);
  };
  flush_sum_shared<2, float>(out_focal_and_extra_njtr,
                             0 * out_focal_and_extra_njtr_num_alloc,
                             focal_and_extra_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r27 + r20;
    r15 = r25 + r15;
    r10 = fmaf(r10, r15, r9 * r20);
    r12 = fmaf(r12, r15, r11 * r20);
    r15 = fmaf(r8, r15, r7 * r20);
    write_sum_3<float, float>((float*)inout_shared, r10, r12, r15);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_njtr_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_principal_point_jtjnjtr_direct(
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
  simple_radial_fixed_principal_point_jtjnjtr_direct_kernel<<<n_blocks, 1024>>>(
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