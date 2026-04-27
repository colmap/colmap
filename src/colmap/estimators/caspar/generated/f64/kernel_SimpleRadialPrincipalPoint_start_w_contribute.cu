#include "kernel_SimpleRadialPrincipalPoint_start_w_contribute.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialPrincipalPoint_start_w_contribute_kernel(
        double* SimpleRadialPrincipalPoint_precond_diag,
        unsigned int SimpleRadialPrincipalPoint_precond_diag_num_alloc,
        const double* const diag,
        double* SimpleRadialPrincipalPoint_p,
        unsigned int SimpleRadialPrincipalPoint_p_num_alloc,
        double* out_SimpleRadialPrincipalPoint_w,
        unsigned int out_SimpleRadialPrincipalPoint_w_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        SimpleRadialPrincipalPoint_precond_diag,
        0 * SimpleRadialPrincipalPoint_precond_diag_num_alloc,
        global_thread_idx,
        r0,
        r1);
  };
  load_unique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r0 * r2;
    read_idx_2<1024, double, double, double2>(
        SimpleRadialPrincipalPoint_p,
        0 * SimpleRadialPrincipalPoint_p_num_alloc,
        global_thread_idx,
        r3,
        r4);
    r0 = r0 * r3;
    r2 = r1 * r2;
    r2 = r2 * r4;
    add_idx_2<1024, double, double, double2>(
        out_SimpleRadialPrincipalPoint_w,
        0 * out_SimpleRadialPrincipalPoint_w_num_alloc,
        global_thread_idx,
        r0,
        r2);
  };
}

void SimpleRadialPrincipalPoint_start_w_contribute(
    double* SimpleRadialPrincipalPoint_precond_diag,
    unsigned int SimpleRadialPrincipalPoint_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialPrincipalPoint_p,
    unsigned int SimpleRadialPrincipalPoint_p_num_alloc,
    double* out_SimpleRadialPrincipalPoint_w,
    unsigned int out_SimpleRadialPrincipalPoint_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialPrincipalPoint_start_w_contribute_kernel<<<n_blocks, 1024>>>(
      SimpleRadialPrincipalPoint_precond_diag,
      SimpleRadialPrincipalPoint_precond_diag_num_alloc,
      diag,
      SimpleRadialPrincipalPoint_p,
      SimpleRadialPrincipalPoint_p_num_alloc,
      out_SimpleRadialPrincipalPoint_w,
      out_SimpleRadialPrincipalPoint_w_num_alloc,
      problem_size);
}

}  // namespace caspar