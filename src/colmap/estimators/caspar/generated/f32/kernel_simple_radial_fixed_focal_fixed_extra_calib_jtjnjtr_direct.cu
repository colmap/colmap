#include "kernel_simple_radial_fixed_focal_fixed_extra_calib_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_fixed_extra_calib_jtjnjtr_direct_kernel(
        float* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        float* pose_jac,
        unsigned int pose_jac_num_alloc,
        float* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        float* point_jac,
        unsigned int point_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
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

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23;

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
    r5 = fmaf(r5, r12, r6 * r8);
    r5 = fmaf(r4, r10, r5);
    r4 = fmaf(r1, r5, r0 * r13);
    r6 = fmaf(r3, r5, r2 * r13);
    read_idx_4<1024, float, float, float4>(pose_jac,
                                           4 * pose_jac_num_alloc,
                                           global_thread_idx,
                                           r14,
                                           r15,
                                           r16,
                                           r17);
    r18 = fmaf(r15, r5, r14 * r13);
    r19 = fmaf(r17, r5, r16 * r13);
    write_sum_4<float, float>((float*)inout_shared, r4, r6, r18, r19);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r19, r18, r6, r4);
    r20 = fmaf(r18, r5, r19 * r13);
    r13 = fmaf(r6, r13, r4 * r5);
    write_sum_2<float, float>((float*)inout_shared, r20, r13);
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
                         r13,
                         r20);
  };
  __syncthreads();
  load_shared<4, float, float>(pose_njtr,
                               0 * pose_njtr_num_alloc,
                               pose_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_njtr_indices_loc[threadIdx.x].target,
                         r5,
                         r21,
                         r22,
                         r23);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r17 = fmaf(r23, r17, r20 * r4);
    r17 = fmaf(r22, r15, r17);
    r17 = fmaf(r5, r1, r17);
    r17 = fmaf(r21, r3, r17);
    r17 = fmaf(r13, r18, r17);
    r19 = fmaf(r13, r19, r20 * r6);
    r19 = fmaf(r22, r14, r19);
    r19 = fmaf(r23, r16, r19);
    r19 = fmaf(r5, r0, r19);
    r19 = fmaf(r21, r2, r19);
    r9 = fmaf(r9, r19, r10 * r17);
    r11 = fmaf(r11, r19, r12 * r17);
    r19 = fmaf(r7, r19, r8 * r17);
    write_sum_3<float, float>((float*)inout_shared, r9, r11, r19);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_njtr_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_focal_fixed_extra_calib_jtjnjtr_direct(
    float* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    float* pose_jac,
    unsigned int pose_jac_num_alloc,
    float* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    float* point_jac,
    unsigned int point_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_focal_fixed_extra_calib_jtjnjtr_direct_kernel<<<n_blocks,
                                                                      1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar