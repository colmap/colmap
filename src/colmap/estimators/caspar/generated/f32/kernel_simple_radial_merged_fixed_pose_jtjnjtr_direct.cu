#include "kernel_simple_radial_merged_fixed_pose_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_merged_fixed_pose_jtjnjtr_direct_kernel(
        float* calib_njtr,
        unsigned int calib_njtr_num_alloc,
        SharedIndex* calib_njtr_indices,
        float* calib_jac,
        unsigned int calib_jac_num_alloc,
        float* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        float* point_jac,
        unsigned int point_jac_num_alloc,
        float* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_njtr_indices_loc[1024];
  calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13;
  load_shared<3, float, float>(point_njtr,
                               0 * point_njtr_num_alloc,
                               point_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         point_njtr_indices_loc[threadIdx.x].target,
                         r0,
                         r1,
                         r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r3, r4);
    read_idx_4<1024, float, float, float4>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r5, r6, r7, r8);
    r9 = fmaf(r1, r8, r2 * r4);
    r9 = fmaf(r0, r6, r9);
    r1 = fmaf(r1, r7, r2 * r3);
    r1 = fmaf(r0, r5, r1);
    read_idx_4<1024, float, float, float4>(calib_jac,
                                           0 * calib_jac_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r2,
                                           r10,
                                           r11);
    r12 = fmaf(r0, r1, r2 * r9);
    r13 = fmaf(r10, r1, r11 * r9);
    write_sum_4<float, float>((float*)inout_shared, r12, r13, r1, r9);
  };
  flush_sum_shared<4, float>(out_calib_njtr,
                             0 * out_calib_njtr_num_alloc,
                             calib_njtr_indices_loc,
                             (float*)inout_shared);
  load_shared<4, float, float>(calib_njtr,
                               0 * calib_njtr_num_alloc,
                               calib_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         calib_njtr_indices_loc[threadIdx.x].target,
                         r9,
                         r1,
                         r13,
                         r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fmaf(r9, r2, r12);
    r2 = fmaf(r1, r11, r2);
    r0 = fmaf(r9, r0, r13);
    r0 = fmaf(r1, r10, r0);
    r5 = fmaf(r5, r0, r6 * r2);
    r7 = fmaf(r7, r0, r8 * r2);
    r0 = fmaf(r3, r0, r4 * r2);
    write_sum_3<float, float>((float*)inout_shared, r5, r7, r0);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_njtr_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_merged_fixed_pose_jtjnjtr_direct(
    float* calib_njtr,
    unsigned int calib_njtr_num_alloc,
    SharedIndex* calib_njtr_indices,
    float* calib_jac,
    unsigned int calib_jac_num_alloc,
    float* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    float* point_jac,
    unsigned int point_jac_num_alloc,
    float* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_merged_fixed_pose_jtjnjtr_direct_kernel<<<n_blocks, 1024>>>(
      calib_njtr,
      calib_njtr_num_alloc,
      calib_njtr_indices,
      calib_jac,
      calib_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar