#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedFocalFixedPrincipalPointJtjnjtrDirectKernel(
        float* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        float* point_jac,
        unsigned int point_jac_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        size_t problem_size) {}

void PinholeSplitFixedPoseFixedFocalFixedPrincipalPointJtjnjtrDirect(
    float* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    float* point_jac,
    unsigned int point_jac_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPoseFixedFocalFixedPrincipalPointJtjnjtrDirectKernel<<<
      n_blocks,
      1024>>>(point_njtr,
              point_njtr_num_alloc,
              point_njtr_indices,
              point_jac,
              point_jac_num_alloc,
              out_point_njtr,
              out_point_njtr_num_alloc,
              problem_size);
}

}  // namespace caspar