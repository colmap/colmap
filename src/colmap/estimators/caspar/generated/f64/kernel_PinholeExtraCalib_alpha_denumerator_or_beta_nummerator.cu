#include "kernel_PinholeExtraCalib_alpha_denumerator_or_beta_nummerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeExtraCalib_alpha_denumerator_or_beta_nummerator_kernel(
        double* PinholeExtraCalib_p_kp1,
        unsigned int PinholeExtraCalib_p_kp1_num_alloc,
        double* PinholeExtraCalib_w,
        unsigned int PinholeExtraCalib_w_num_alloc,
        double* const PinholeExtraCalib_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double PinholeExtraCalib_out_local[1];

  double r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        PinholeExtraCalib_p_kp1,
        0 * PinholeExtraCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(PinholeExtraCalib_w,
                                              0 * PinholeExtraCalib_w_num_alloc,
                                              global_thread_idx,
                                              r2,
                                              r3);
    r2 = fma(r0, r2, r1 * r3);
  };
  sum_store<double>(PinholeExtraCalib_out_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r2);
  sum_flush_final<double>(
      PinholeExtraCalib_out_local, PinholeExtraCalib_out, 1);
}

void PinholeExtraCalib_alpha_denumerator_or_beta_nummerator(
    double* PinholeExtraCalib_p_kp1,
    unsigned int PinholeExtraCalib_p_kp1_num_alloc,
    double* PinholeExtraCalib_w,
    unsigned int PinholeExtraCalib_w_num_alloc,
    double* const PinholeExtraCalib_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeExtraCalib_alpha_denumerator_or_beta_nummerator_kernel<<<n_blocks,
                                                                  1024>>>(
      PinholeExtraCalib_p_kp1,
      PinholeExtraCalib_p_kp1_num_alloc,
      PinholeExtraCalib_w,
      PinholeExtraCalib_w_num_alloc,
      PinholeExtraCalib_out,
      problem_size);
}

}  // namespace caspar