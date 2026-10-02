#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_pinhole_fixed_pose_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedPoseFixedPointJtjnjtrDirectKernel(
        double *calib_njtr, unsigned int calib_njtr_num_alloc,
        SharedIndex *calib_njtr_indices, double *calib_jac,
        unsigned int calib_jac_num_alloc, double *const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc, size_t problem_size) {}

void PinholeFixedPoseFixedPointJtjnjtrDirect(
    double *calib_njtr, unsigned int calib_njtr_num_alloc,
    SharedIndex *calib_njtr_indices, double *calib_jac,
    unsigned int calib_jac_num_alloc, double *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedPoseFixedPointJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      calib_njtr, calib_njtr_num_alloc, calib_njtr_indices, calib_jac,
      calib_jac_num_alloc, out_calib_njtr, out_calib_njtr_num_alloc,
      problem_size);
}

} // namespace caspar