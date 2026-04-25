#include "kernel_PinholePose_start_w.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholePose_start_w_kernel(double* PinholePose_precond_diag,
                               unsigned int PinholePose_precond_diag_num_alloc,
                               const double* const diag,
                               double* PinholePose_p,
                               unsigned int PinholePose_p_num_alloc,
                               double* out_PinholePose_w,
                               unsigned int out_PinholePose_w_num_alloc,
                               size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        PinholePose_precond_diag,
        0 * PinholePose_precond_diag_num_alloc,
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
        PinholePose_p, 0 * PinholePose_p_num_alloc, global_thread_idx, r3, r4);
    r0 = r0 * r3;
    r1 = r1 * r2;
    r1 = r1 * r4;
    write_idx_2<1024, double, double, double2>(out_PinholePose_w,
                                               0 * out_PinholePose_w_num_alloc,
                                               global_thread_idx,
                                               r0,
                                               r1);
    read_idx_2<1024, double, double, double2>(
        PinholePose_precond_diag,
        2 * PinholePose_precond_diag_num_alloc,
        global_thread_idx,
        r1,
        r0);
    r1 = r1 * r2;
    read_idx_2<1024, double, double, double2>(
        PinholePose_p, 2 * PinholePose_p_num_alloc, global_thread_idx, r4, r3);
    r1 = r1 * r4;
    r0 = r0 * r2;
    r0 = r0 * r3;
    write_idx_2<1024, double, double, double2>(out_PinholePose_w,
                                               2 * out_PinholePose_w_num_alloc,
                                               global_thread_idx,
                                               r1,
                                               r0);
    read_idx_2<1024, double, double, double2>(
        PinholePose_precond_diag,
        4 * PinholePose_precond_diag_num_alloc,
        global_thread_idx,
        r0,
        r1);
    r0 = r0 * r2;
    read_idx_2<1024, double, double, double2>(
        PinholePose_p, 4 * PinholePose_p_num_alloc, global_thread_idx, r3, r4);
    r0 = r0 * r3;
    r2 = r1 * r2;
    r2 = r2 * r4;
    write_idx_2<1024, double, double, double2>(out_PinholePose_w,
                                               4 * out_PinholePose_w_num_alloc,
                                               global_thread_idx,
                                               r0,
                                               r2);
  };
}

void PinholePose_start_w(double* PinholePose_precond_diag,
                         unsigned int PinholePose_precond_diag_num_alloc,
                         const double* const diag,
                         double* PinholePose_p,
                         unsigned int PinholePose_p_num_alloc,
                         double* out_PinholePose_w,
                         unsigned int out_PinholePose_w_num_alloc,
                         size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholePose_start_w_kernel<<<n_blocks, 1024>>>(
      PinholePose_precond_diag,
      PinholePose_precond_diag_num_alloc,
      diag,
      PinholePose_p,
      PinholePose_p_num_alloc,
      out_PinholePose_w,
      out_PinholePose_w_num_alloc,
      problem_size);
}

}  // namespace caspar