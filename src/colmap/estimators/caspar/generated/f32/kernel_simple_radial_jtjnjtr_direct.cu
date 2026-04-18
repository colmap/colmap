#include "kernel_simple_radial_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) simple_radial_jtjnjtr_direct_kernel(
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
    float* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    float* point_jac,
    unsigned int point_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_extra_calib_njtr,
    unsigned int out_extra_calib_njtr_num_alloc,
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

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34;

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
    r7 = r4 * r5;
  };
  load_shared<3, float, float>(point_njtr,
                               0 * point_njtr_num_alloc,
                               point_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         point_njtr_indices_loc[threadIdx.x].target,
                         r8,
                         r9,
                         r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r11, r12);
    read_idx_4<1024, float, float, float4>(point_jac,
                                           0 * point_jac_num_alloc,
                                           global_thread_idx,
                                           r13,
                                           r14,
                                           r15,
                                           r16);
    r17 = fmaf(r9, r15, r10 * r11);
    r17 = fmaf(r8, r13, r17);
  };
  load_shared<3, float, float>(extra_calib_njtr,
                               0 * extra_calib_njtr_num_alloc,
                               extra_calib_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         extra_calib_njtr_indices_loc[threadIdx.x].target,
                         r18,
                         r19,
                         r20);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(extra_calib_jac,
                                           0 * extra_calib_jac_num_alloc,
                                           global_thread_idx,
                                           r21,
                                           r22);
    r18 = fmaf(r20, r21, r18);
    r23 = r17 + r18;
    r24 = r7 + r23;
    r4 = r4 * r6;
    r9 = fmaf(r9, r16, r10 * r12);
    r9 = fmaf(r8, r14, r9);
    r20 = fmaf(r20, r22, r19);
    r19 = r9 + r20;
    r8 = r4 + r19;
    r10 = fmaf(r1, r8, r0 * r24);
    r25 = fmaf(r3, r8, r2 * r24);
    read_idx_4<1024, float, float, float4>(pose_jac,
                                           4 * pose_jac_num_alloc,
                                           global_thread_idx,
                                           r26,
                                           r27,
                                           r28,
                                           r29);
    r30 = fmaf(r27, r8, r26 * r24);
    r31 = fmaf(r29, r8, r28 * r24);
    write_sum_4<float, float>((float*)inout_shared, r10, r25, r30, r31);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(pose_jac,
                                           8 * pose_jac_num_alloc,
                                           global_thread_idx,
                                           r31,
                                           r30,
                                           r25,
                                           r10);
    r32 = fmaf(r30, r8, r31 * r24);
    r24 = fmaf(r25, r24, r10 * r8);
    write_sum_2<float, float>((float*)inout_shared, r32, r24);
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
                         r24,
                         r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r24, r31, r32 * r25);
  };
  load_shared<4, float, float>(pose_njtr,
                               0 * pose_njtr_num_alloc,
                               pose_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_njtr_indices_loc[threadIdx.x].target,
                         r25,
                         r8,
                         r33,
                         r34);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r33, r26, r31);
    r31 = fmaf(r34, r28, r31);
    r31 = fmaf(r25, r0, r31);
    r31 = fmaf(r8, r2, r31);
    r23 = r31 + r23;
    r29 = fmaf(r34, r29, r32 * r10);
    r29 = fmaf(r33, r27, r29);
    r29 = fmaf(r25, r1, r29);
    r29 = fmaf(r8, r3, r29);
    r29 = fmaf(r24, r30, r29);
    r19 = r29 + r19;
    r19 = fmaf(r6, r19, r5 * r23);
    write_sum_1<float, float>((float*)inout_shared, r19);
  };
  flush_sum_shared<1, float>(out_focal_njtr,
                             0 * out_focal_njtr_num_alloc,
                             focal_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r7 + r31;
    r17 = r17 + r31;
    r29 = r4 + r29;
    r9 = r9 + r29;
    r22 = fmaf(r22, r9, r21 * r17);
    write_sum_3<float, float>((float*)inout_shared, r17, r9, r22);
  };
  flush_sum_shared<3, float>(out_extra_calib_njtr,
                             0 * out_extra_calib_njtr_num_alloc,
                             extra_calib_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r18 + r31;
    r29 = r20 + r29;
    r14 = fmaf(r14, r29, r13 * r31);
    r16 = fmaf(r16, r29, r15 * r31);
    r29 = fmaf(r12, r29, r11 * r31);
    write_sum_3<float, float>((float*)inout_shared, r14, r16, r29);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_njtr_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_jtjnjtr_direct(float* pose_njtr,
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
                                  float* point_njtr,
                                  unsigned int point_njtr_num_alloc,
                                  SharedIndex* point_njtr_indices,
                                  float* point_jac,
                                  unsigned int point_jac_num_alloc,
                                  float* const out_pose_njtr,
                                  unsigned int out_pose_njtr_num_alloc,
                                  float* const out_focal_njtr,
                                  unsigned int out_focal_njtr_num_alloc,
                                  float* const out_extra_calib_njtr,
                                  unsigned int out_extra_calib_njtr_num_alloc,
                                  float* const out_point_njtr,
                                  unsigned int out_point_njtr_num_alloc,
                                  size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_jtjnjtr_direct_kernel<<<n_blocks, 1024>>>(
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
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_extra_calib_njtr,
      out_extra_calib_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar