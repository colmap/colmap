#include "kernel_PinholeExtraCalib_update_step_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeExtraCalib_update_step_first_kernel(
        double* PinholeExtraCalib_p_kp1,
        unsigned int PinholeExtraCalib_p_kp1_num_alloc,
        const double* const alpha,
        double* out_PinholeExtraCalib_step_kp1,
        unsigned int out_PinholeExtraCalib_step_kp1_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        PinholeExtraCalib_p_kp1,
        0 * PinholeExtraCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
  };
  load_unique<1, double, double>(alpha, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r0 * r2;
    r2 = r1 * r2;
    write_idx_2<1024, double, double, double2>(
        out_PinholeExtraCalib_step_kp1,
        0 * out_PinholeExtraCalib_step_kp1_num_alloc,
        global_thread_idx,
        r0,
        r2);
  };
}

void PinholeExtraCalib_update_step_first(
    double* PinholeExtraCalib_p_kp1,
    unsigned int PinholeExtraCalib_p_kp1_num_alloc,
    const double* const alpha,
    double* out_PinholeExtraCalib_step_kp1,
    unsigned int out_PinholeExtraCalib_step_kp1_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeExtraCalib_update_step_first_kernel<<<n_blocks, 1024>>>(
      PinholeExtraCalib_p_kp1,
      PinholeExtraCalib_p_kp1_num_alloc,
      alpha,
      out_PinholeExtraCalib_step_kp1,
      out_PinholeExtraCalib_step_kp1_num_alloc,
      problem_size);
}

}  // namespace caspar