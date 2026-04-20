#include "kernel_pinhole_fixed_extra_calib_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_extra_calib_fixed_point_jtjnjtr_direct_kernel(
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
      r16, r17;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r0, r1, r2, r3);
  };
  load_shared<2, float, float>(focal_njtr,
                               0 * focal_njtr_num_alloc,
                               focal_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         focal_njtr_indices_loc[threadIdx.x].target,
                         r4,
                         r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        focal_jac, 0 * focal_jac_num_alloc, global_thread_idx, r6, r7);
    r4 = r4 * r6;
    r8 = r2 * r4;
    read_idx_4<1024, float, float, float4>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r9, r10, r11, r12);
    r5 = r5 * r7;
    r13 = fmaf(r10, r5, r9 * r4);
    r14 = fmaf(r12, r5, r11 * r4);
    r15 = fmaf(r0, r4, r1 * r5);
    write_sum_4<float, float>((float*)inout_shared, r13, r14, r15, r8);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = r3 * r5;
    read_idx_2<1024, float, float, float2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r15, r14);
    r5 = fmaf(r14, r5, r15 * r4);
    write_sum_2<float, float>((float*)inout_shared, r8, r5);
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
                         r8);
  };
  __syncthreads();
  load_shared<4, float, float>(pose_njtr,
                               0 * pose_njtr_num_alloc,
                               pose_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_njtr_indices_loc[threadIdx.x].target,
                         r4,
                         r13,
                         r16,
                         r17);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = fmaf(r16, r0, r8 * r15);
    r0 = fmaf(r17, r2, r0);
    r0 = fmaf(r4, r9, r0);
    r0 = fmaf(r13, r11, r0);
    r0 = r6 * r0;
    r1 = fmaf(r16, r1, r8 * r14);
    r1 = fmaf(r4, r10, r1);
    r1 = fmaf(r13, r12, r1);
    r1 = fmaf(r5, r3, r1);
    r1 = r7 * r1;
    write_sum_2<float, float>((float*)inout_shared, r0, r1);
  };
  flush_sum_shared<2, float>(out_focal_njtr,
                             0 * out_focal_njtr_num_alloc,
                             focal_njtr_indices_loc,
                             (float*)inout_shared);
}

void pinhole_fixed_extra_calib_fixed_point_jtjnjtr_direct(
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
  pinhole_fixed_extra_calib_fixed_point_jtjnjtr_direct_kernel<<<n_blocks,
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