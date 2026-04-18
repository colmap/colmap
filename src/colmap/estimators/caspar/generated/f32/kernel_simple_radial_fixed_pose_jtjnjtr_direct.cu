#include "kernel_simple_radial_fixed_pose_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_jtjnjtr_direct_kernel(
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
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        float* const out_extra_calib_njtr,
        unsigned int out_extra_calib_njtr_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

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
      r16, r17;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        focal_jac, 0 * focal_jac_num_alloc, global_thread_idx, r0, r1);
  };
  load_shared<3, float, float>(point_njtr,
                               0 * point_njtr_num_alloc,
                               point_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         point_njtr_indices_loc[threadIdx.x].target,
                         r2,
                         r3,
                         r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r5, r6);
    read_idx_4<1024, float, float, float4>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r7, r8, r9, r10);
    r11 = fmaf(r3, r10, r4 * r6);
    r11 = fmaf(r2, r8, r11);
  };
  load_shared<3, float, float>(extra_calib_njtr,
                               0 * extra_calib_njtr_num_alloc,
                               extra_calib_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         extra_calib_njtr_indices_loc[threadIdx.x].target,
                         r12,
                         r13,
                         r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(extra_calib_jac,
                                           0 * extra_calib_jac_num_alloc,
                                           global_thread_idx,
                                           r15,
                                           r16);
    r13 = fmaf(r14, r16, r13);
    r17 = r11 + r13;
    r3 = fmaf(r3, r9, r4 * r5);
    r3 = fmaf(r2, r7, r3);
    r14 = fmaf(r14, r15, r12);
    r12 = r3 + r14;
    r12 = fmaf(r0, r12, r1 * r17);
    write_sum_1<float, float>((float*)inout_shared, r12);
  };
  flush_sum_shared<1, float>(out_focal_njtr,
                             0 * out_focal_njtr_num_alloc,
                             focal_njtr_indices_loc,
                             (float*)inout_shared);
  load_shared<1, float, float>(focal_njtr,
                               0 * focal_njtr_num_alloc,
                               focal_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>(
        (float*)inout_shared, focal_njtr_indices_loc[threadIdx.x].target, r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r12 * r0;
    r3 = r0 + r3;
    r1 = r12 * r1;
    r11 = r1 + r11;
    r16 = fmaf(r16, r11, r15 * r3);
    write_sum_3<float, float>((float*)inout_shared, r3, r11, r16);
  };
  flush_sum_shared<3, float>(out_extra_calib_njtr,
                             0 * out_extra_calib_njtr_num_alloc,
                             extra_calib_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = r0 + r14;
    r13 = r1 + r13;
    r8 = fmaf(r8, r13, r7 * r14);
    r10 = fmaf(r10, r13, r9 * r14);
    r13 = fmaf(r6, r13, r5 * r14);
    write_sum_3<float, float>((float*)inout_shared, r8, r10, r13);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_njtr_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_pose_jtjnjtr_direct(
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
  simple_radial_fixed_pose_jtjnjtr_direct_kernel<<<n_blocks, 1024>>>(
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
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_extra_calib_njtr,
      out_extra_calib_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar