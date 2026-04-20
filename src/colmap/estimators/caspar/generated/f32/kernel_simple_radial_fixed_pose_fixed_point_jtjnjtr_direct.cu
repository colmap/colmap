#include "kernel_simple_radial_fixed_pose_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_fixed_point_jtjnjtr_direct_kernel(
        float* focal_and_extra_njtr,
        unsigned int focal_and_extra_njtr_num_alloc,
        SharedIndex* focal_and_extra_njtr_indices,
        float* focal_and_extra_jac,
        unsigned int focal_and_extra_jac_num_alloc,
        float* principal_point_njtr,
        unsigned int principal_point_njtr_num_alloc,
        SharedIndex* principal_point_njtr_indices,
        float* principal_point_jac,
        unsigned int principal_point_jac_num_alloc,
        float* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex focal_and_extra_njtr_indices_loc[1024];
  focal_and_extra_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex principal_point_njtr_indices_loc[1024];
  principal_point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(focal_and_extra_jac,
                                           0 * focal_and_extra_jac_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1,
                                           r2,
                                           r3);
  };
  load_shared<2, float, float>(principal_point_njtr,
                               0 * principal_point_njtr_num_alloc,
                               principal_point_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         principal_point_njtr_indices_loc[threadIdx.x].target,
                         r4,
                         r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = fmaf(r0, r4, r1 * r5);
    r4 = fmaf(r2, r4, r3 * r5);
    write_sum_2<float, float>((float*)inout_shared, r6, r4);
  };
  flush_sum_shared<2, float>(out_focal_and_extra_njtr,
                             0 * out_focal_and_extra_njtr_num_alloc,
                             focal_and_extra_njtr_indices_loc,
                             (float*)inout_shared);
  load_shared<2, float, float>(focal_and_extra_njtr,
                               0 * focal_and_extra_njtr_num_alloc,
                               focal_and_extra_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                         r4,
                         r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fmaf(r6, r2, r4 * r0);
    r3 = fmaf(r6, r3, r4 * r1);
    write_sum_2<float, float>((float*)inout_shared, r2, r3);
  };
  flush_sum_shared<2, float>(out_principal_point_njtr,
                             0 * out_principal_point_njtr_num_alloc,
                             principal_point_njtr_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_pose_fixed_point_jtjnjtr_direct(
    float* focal_and_extra_njtr,
    unsigned int focal_and_extra_njtr_num_alloc,
    SharedIndex* focal_and_extra_njtr_indices,
    float* focal_and_extra_jac,
    unsigned int focal_and_extra_jac_num_alloc,
    float* principal_point_njtr,
    unsigned int principal_point_njtr_num_alloc,
    SharedIndex* principal_point_njtr_indices,
    float* principal_point_jac,
    unsigned int principal_point_jac_num_alloc,
    float* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_pose_fixed_point_jtjnjtr_direct_kernel<<<n_blocks,
                                                               1024>>>(
      focal_and_extra_njtr,
      focal_and_extra_njtr_num_alloc,
      focal_and_extra_njtr_indices,
      focal_and_extra_jac,
      focal_and_extra_jac_num_alloc,
      principal_point_njtr,
      principal_point_njtr_num_alloc,
      principal_point_njtr_indices,
      principal_point_jac,
      principal_point_jac_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar