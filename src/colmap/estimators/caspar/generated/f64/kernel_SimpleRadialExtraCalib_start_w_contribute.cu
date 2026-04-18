#include "kernel_SimpleRadialExtraCalib_start_w_contribute.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialExtraCalib_start_w_contribute_kernel(
        double* SimpleRadialExtraCalib_precond_diag,
        unsigned int SimpleRadialExtraCalib_precond_diag_num_alloc,
        const double* const diag,
        double* SimpleRadialExtraCalib_p,
        unsigned int SimpleRadialExtraCalib_p_num_alloc,
        double* out_SimpleRadialExtraCalib_w,
        unsigned int out_SimpleRadialExtraCalib_w_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        SimpleRadialExtraCalib_precond_diag,
        0 * SimpleRadialExtraCalib_precond_diag_num_alloc,
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
        SimpleRadialExtraCalib_p,
        0 * SimpleRadialExtraCalib_p_num_alloc,
        global_thread_idx,
        r3,
        r4);
    r0 = r0 * r3;
    r1 = r1 * r2;
    r1 = r1 * r4;
    add_idx_2<1024, double, double, double2>(
        out_SimpleRadialExtraCalib_w,
        0 * out_SimpleRadialExtraCalib_w_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_1<1024, double, double, double>(
        SimpleRadialExtraCalib_precond_diag,
        2 * SimpleRadialExtraCalib_precond_diag_num_alloc,
        global_thread_idx,
        r1);
    r2 = r1 * r2;
    read_idx_1<1024, double, double, double>(
        SimpleRadialExtraCalib_p,
        2 * SimpleRadialExtraCalib_p_num_alloc,
        global_thread_idx,
        r1);
    r2 = r2 * r1;
    add_idx_1<1024, double, double, double>(
        out_SimpleRadialExtraCalib_w,
        2 * out_SimpleRadialExtraCalib_w_num_alloc,
        global_thread_idx,
        r2);
  };
}

void SimpleRadialExtraCalib_start_w_contribute(
    double* SimpleRadialExtraCalib_precond_diag,
    unsigned int SimpleRadialExtraCalib_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialExtraCalib_p,
    unsigned int SimpleRadialExtraCalib_p_num_alloc,
    double* out_SimpleRadialExtraCalib_w,
    unsigned int out_SimpleRadialExtraCalib_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialExtraCalib_start_w_contribute_kernel<<<n_blocks, 1024>>>(
      SimpleRadialExtraCalib_precond_diag,
      SimpleRadialExtraCalib_precond_diag_num_alloc,
      diag,
      SimpleRadialExtraCalib_p,
      SimpleRadialExtraCalib_p_num_alloc,
      out_SimpleRadialExtraCalib_w,
      out_SimpleRadialExtraCalib_w_num_alloc,
      problem_size);
}

}  // namespace caspar