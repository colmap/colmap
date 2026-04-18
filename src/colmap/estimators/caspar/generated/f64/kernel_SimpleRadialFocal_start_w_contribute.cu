#include "kernel_SimpleRadialFocal_start_w_contribute.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialFocal_start_w_contribute_kernel(
        double* SimpleRadialFocal_precond_diag,
        unsigned int SimpleRadialFocal_precond_diag_num_alloc,
        const double* const diag,
        double* SimpleRadialFocal_p,
        unsigned int SimpleRadialFocal_p_num_alloc,
        double* out_SimpleRadialFocal_w,
        unsigned int out_SimpleRadialFocal_w_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1;

  if (global_thread_idx < problem_size) {
    read_idx_1<1024, double, double, double>(
        SimpleRadialFocal_precond_diag,
        0 * SimpleRadialFocal_precond_diag_num_alloc,
        global_thread_idx,
        r0);
  };
  load_unique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = r0 * r1;
    read_idx_1<1024, double, double, double>(SimpleRadialFocal_p,
                                             0 * SimpleRadialFocal_p_num_alloc,
                                             global_thread_idx,
                                             r0);
    r1 = r1 * r0;
    add_idx_1<1024, double, double, double>(
        out_SimpleRadialFocal_w,
        0 * out_SimpleRadialFocal_w_num_alloc,
        global_thread_idx,
        r1);
  };
}

void SimpleRadialFocal_start_w_contribute(
    double* SimpleRadialFocal_precond_diag,
    unsigned int SimpleRadialFocal_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialFocal_p,
    unsigned int SimpleRadialFocal_p_num_alloc,
    double* out_SimpleRadialFocal_w,
    unsigned int out_SimpleRadialFocal_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFocal_start_w_contribute_kernel<<<n_blocks, 1024>>>(
      SimpleRadialFocal_precond_diag,
      SimpleRadialFocal_precond_diag_num_alloc,
      diag,
      SimpleRadialFocal_p,
      SimpleRadialFocal_p_num_alloc,
      out_SimpleRadialFocal_w,
      out_SimpleRadialFocal_w_num_alloc,
      problem_size);
}

}  // namespace caspar