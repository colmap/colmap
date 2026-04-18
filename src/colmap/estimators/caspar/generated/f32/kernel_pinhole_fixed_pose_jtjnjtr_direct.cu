#include "kernel_pinhole_fixed_pose_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_pose_jtjnjtr_direct_kernel(
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        focal_jac, 0 * focal_jac_num_alloc, global_thread_idx, r0, r1);
  };
  load_shared<2, float, float>(extra_calib_njtr,
                               0 * extra_calib_njtr_num_alloc,
                               extra_calib_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         extra_calib_njtr_indices_loc[threadIdx.x].target,
                         r2,
                         r3);
  };
  __syncthreads();
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
    r14 = r2 + r13;
    r14 = r0 * r14;
    r5 = fmaf(r5, r12, r6 * r8);
    r5 = fmaf(r4, r10, r5);
    r4 = r3 + r5;
    r4 = r1 * r4;
    write_sum_2<float, float>((float*)inout_shared, r14, r4);
  };
  flush_sum_shared<2, float>(out_focal_njtr,
                             0 * out_focal_njtr_num_alloc,
                             focal_njtr_indices_loc,
                             (float*)inout_shared);
  load_shared<2, float, float>(focal_njtr,
                               0 * focal_njtr_num_alloc,
                               focal_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         focal_njtr_indices_loc[threadIdx.x].target,
                         r4,
                         r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r4 * r0;
    r13 = r0 + r13;
    r1 = r14 * r1;
    r5 = r1 + r5;
    write_sum_2<float, float>((float*)inout_shared, r13, r5);
  };
  flush_sum_shared<2, float>(out_extra_calib_njtr,
                             0 * out_extra_calib_njtr_num_alloc,
                             extra_calib_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r3 + r1;
    r0 = r2 + r0;
    r9 = fmaf(r9, r0, r10 * r1);
    r11 = fmaf(r11, r0, r12 * r1);
    r0 = fmaf(r7, r0, r8 * r1);
    write_sum_3<float, float>((float*)inout_shared, r9, r11, r0);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_njtr_indices_loc,
                             (float*)inout_shared);
}

void pinhole_fixed_pose_jtjnjtr_direct(
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
  pinhole_fixed_pose_jtjnjtr_direct_kernel<<<n_blocks, 1024>>>(
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