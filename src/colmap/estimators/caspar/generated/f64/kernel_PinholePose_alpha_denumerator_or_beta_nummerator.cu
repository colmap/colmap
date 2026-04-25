#include "kernel_PinholePose_alpha_denumerator_or_beta_nummerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholePose_alpha_denumerator_or_beta_nummerator_kernel(
        double* PinholePose_p_kp1,
        unsigned int PinholePose_p_kp1_num_alloc,
        double* PinholePose_w,
        unsigned int PinholePose_w_num_alloc,
        double* const PinholePose_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double PinholePose_out_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(PinholePose_p_kp1,
                                              2 * PinholePose_p_kp1_num_alloc,
                                              global_thread_idx,
                                              r0,
                                              r1);
    read_idx_2<1024, double, double, double2>(
        PinholePose_w, 2 * PinholePose_w_num_alloc, global_thread_idx, r2, r3);
    read_idx_2<1024, double, double, double2>(PinholePose_p_kp1,
                                              4 * PinholePose_p_kp1_num_alloc,
                                              global_thread_idx,
                                              r4,
                                              r5);
    read_idx_2<1024, double, double, double2>(
        PinholePose_w, 4 * PinholePose_w_num_alloc, global_thread_idx, r6, r7);
    r6 = fma(r4, r6, r1 * r3);
    read_idx_2<1024, double, double, double2>(PinholePose_p_kp1,
                                              0 * PinholePose_p_kp1_num_alloc,
                                              global_thread_idx,
                                              r4,
                                              r3);
    read_idx_2<1024, double, double, double2>(
        PinholePose_w, 0 * PinholePose_w_num_alloc, global_thread_idx, r1, r8);
    r6 = fma(r0, r2, r6);
    r6 = fma(r5, r7, r6);
    r6 = fma(r4, r1, r6);
    r6 = fma(r3, r8, r6);
  };
  sum_store<double>(PinholePose_out_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r6);
  sum_flush_final<double>(PinholePose_out_local, PinholePose_out, 1);
}

void PinholePose_alpha_denumerator_or_beta_nummerator(
    double* PinholePose_p_kp1,
    unsigned int PinholePose_p_kp1_num_alloc,
    double* PinholePose_w,
    unsigned int PinholePose_w_num_alloc,
    double* const PinholePose_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholePose_alpha_denumerator_or_beta_nummerator_kernel<<<n_blocks, 1024>>>(
      PinholePose_p_kp1,
      PinholePose_p_kp1_num_alloc,
      PinholePose_w,
      PinholePose_w_num_alloc,
      PinholePose_out,
      problem_size);
}

}  // namespace caspar