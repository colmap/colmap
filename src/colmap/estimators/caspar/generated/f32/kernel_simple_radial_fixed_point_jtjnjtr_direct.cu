#include "kernel_simple_radial_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_point_jtjnjtr_direct_kernel(
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
        float* extra_calib_njtr,
        unsigned int extra_calib_njtr_num_alloc,
        SharedIndex* extra_calib_njtr_indices,
        float* extra_calib_jac,
        unsigned int extra_calib_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        float* const out_extra_calib_njtr,
        unsigned int out_extra_calib_njtr_num_alloc,
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

  __shared__ SharedIndex extra_calib_njtr_indices_loc[1024];
  extra_calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? extra_calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1, r2, r3);
  };
  load_shared<1, float, float>(focal_njtr,
                               0 * focal_njtr_num_alloc,
                               focal_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>(
        (float*)inout_shared, focal_njtr_indices_loc[threadIdx.x].target, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        focal_jac, 0 * focal_jac_num_alloc, global_thread_idx, r5, r6);
    r7 = r4 * r6;
  };
  load_shared<3, float, float>(extra_calib_njtr,
                               0 * extra_calib_njtr_num_alloc,
                               extra_calib_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         extra_calib_njtr_indices_loc[threadIdx.x].target,
                         r8,
                         r9,
                         r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(extra_calib_jac,
                                           0 * extra_calib_jac_num_alloc,
                                           global_thread_idx,
                                           r11,
                                           r12);
    r9 = fmaf(r10, r12, r9);
    r13 = r7 + r9;
    r4 = r4 * r5;
    r10 = fmaf(r10, r11, r8);
    r8 = r4 + r10;
    r14 = fmaf(r0, r8, r1 * r13);
    r15 = fmaf(r2, r8, r3 * r13);
    read_idx_4<1024, float, float, float4>(pose_jac,
                                           4 * pose_jac_num_alloc,
                                           global_thread_idx,
                                           r16,
                                           r17,
                                           r18,
                                           r19);
    r20 = fmaf(r16, r8, r17 * r13);
    r21 = fmaf(r18, r8, r19 * r13);
    write_sum_4<float, float>((float*)inout_shared, r14, r15, r20, r21);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(pose_jac,
                                           8 * pose_jac_num_alloc,
                                           global_thread_idx,
                                           r21,
                                           r20,
                                           r15,
                                           r14);
    r22 = fmaf(r21, r8, r20 * r13);
    r8 = fmaf(r15, r8, r14 * r13);
    write_sum_2<float, float>((float*)inout_shared, r22, r8);
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
                         r8,
                         r22);
  };
  __syncthreads();
  load_shared<4, float, float>(pose_njtr,
                               0 * pose_njtr_num_alloc,
                               pose_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_njtr_indices_loc[threadIdx.x].target,
                         r13,
                         r23,
                         r24,
                         r25);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r19 = fmaf(r25, r19, r22 * r14);
    r19 = fmaf(r24, r17, r19);
    r19 = fmaf(r13, r1, r19);
    r19 = fmaf(r23, r3, r19);
    r19 = fmaf(r8, r20, r19);
    r9 = r19 + r9;
    r21 = fmaf(r8, r21, r22 * r15);
    r21 = fmaf(r24, r16, r21);
    r21 = fmaf(r25, r18, r21);
    r21 = fmaf(r13, r0, r21);
    r21 = fmaf(r23, r2, r21);
    r10 = r21 + r10;
    r10 = fmaf(r5, r10, r6 * r9);
    write_sum_1<float, float>((float*)inout_shared, r10);
  };
  flush_sum_shared<1, float>(out_focal_njtr,
                             0 * out_focal_njtr_num_alloc,
                             focal_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r7 + r19;
    r21 = r4 + r21;
    r11 = fmaf(r11, r21, r12 * r19);
    write_sum_3<float, float>((float*)inout_shared, r21, r19, r11);
  };
  flush_sum_shared<3, float>(out_extra_calib_njtr,
                             0 * out_extra_calib_njtr_num_alloc,
                             extra_calib_njtr_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_point_jtjnjtr_direct(
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
    float* extra_calib_njtr,
    unsigned int extra_calib_njtr_num_alloc,
    SharedIndex* extra_calib_njtr_indices,
    float* extra_calib_jac,
    unsigned int extra_calib_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_extra_calib_njtr,
    unsigned int out_extra_calib_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_point_jtjnjtr_direct_kernel<<<n_blocks, 1024>>>(
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
      extra_calib_njtr,
      extra_calib_njtr_num_alloc,
      extra_calib_njtr_indices,
      extra_calib_jac,
      extra_calib_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_extra_calib_njtr,
      out_extra_calib_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar