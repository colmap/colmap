#include "kernel_pinhole_fixed_pose_fixed_extra_calib_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_pose_fixed_extra_calib_jtjnjtr_direct_kernel(
        float* focal_njtr,
        unsigned int focal_njtr_num_alloc,
        SharedIndex* focal_njtr_indices,
        float* focal_jac,
        unsigned int focal_jac_num_alloc,
        float* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        float* point_jac,
        unsigned int point_jac_num_alloc,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
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

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;

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
    r11 = fmaf(r3, r9, r4 * r5);
    r11 = fmaf(r2, r7, r11);
    r11 = r0 * r11;
    r3 = fmaf(r3, r10, r4 * r6);
    r3 = fmaf(r2, r8, r3);
    r3 = r1 * r3;
    write_sum_2<float, float>((float*)inout_shared, r11, r3);
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
                         r3,
                         r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = r11 * r1;
    r0 = r3 * r0;
    r7 = fmaf(r7, r0, r8 * r1);
    r9 = fmaf(r9, r0, r10 * r1);
    r0 = fmaf(r5, r0, r6 * r1);
    write_sum_3<float, float>((float*)inout_shared, r7, r9, r0);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_njtr_indices_loc,
                             (float*)inout_shared);
}

void pinhole_fixed_pose_fixed_extra_calib_jtjnjtr_direct(
    float* focal_njtr,
    unsigned int focal_njtr_num_alloc,
    SharedIndex* focal_njtr_indices,
    float* focal_jac,
    unsigned int focal_jac_num_alloc,
    float* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    float* point_jac,
    unsigned int point_jac_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_pose_fixed_extra_calib_jtjnjtr_direct_kernel<<<n_blocks,
                                                               1024>>>(
      focal_njtr,
      focal_njtr_num_alloc,
      focal_njtr_indices,
      focal_jac,
      focal_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar