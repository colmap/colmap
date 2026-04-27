#include "kernel_simple_radial_fixed_pose_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_fixed_principal_point_fixed_point_jtjnjtr_direct_kernel(
        float* focal_and_extra_njtr,
        unsigned int focal_and_extra_njtr_num_alloc,
        SharedIndex* focal_and_extra_njtr_indices,
        float* focal_and_extra_jac,
        unsigned int focal_and_extra_jac_num_alloc,
        float* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex focal_and_extra_njtr_indices_loc[1024];
  focal_and_extra_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
}

void simple_radial_fixed_pose_fixed_principal_point_fixed_point_jtjnjtr_direct(
    float* focal_and_extra_njtr,
    unsigned int focal_and_extra_njtr_num_alloc,
    SharedIndex* focal_and_extra_njtr_indices,
    float* focal_and_extra_jac,
    unsigned int focal_and_extra_jac_num_alloc,
    float* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_pose_fixed_principal_point_fixed_point_jtjnjtr_direct_kernel<<<
      n_blocks,
      1024>>>(focal_and_extra_njtr,
              focal_and_extra_njtr_num_alloc,
              focal_and_extra_njtr_indices,
              focal_and_extra_jac,
              focal_and_extra_jac_num_alloc,
              out_focal_and_extra_njtr,
              out_focal_and_extra_njtr_num_alloc,
              problem_size);
}

}  // namespace caspar