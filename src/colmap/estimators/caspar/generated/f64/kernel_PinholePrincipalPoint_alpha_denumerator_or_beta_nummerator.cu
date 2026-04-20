#include "kernel_PinholePrincipalPoint_alpha_denumerator_or_beta_nummerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholePrincipalPoint_alpha_denumerator_or_beta_nummerator_kernel(
        double* PinholePrincipalPoint_p_kp1,
        unsigned int PinholePrincipalPoint_p_kp1_num_alloc,
        double* PinholePrincipalPoint_w,
        unsigned int PinholePrincipalPoint_w_num_alloc,
        double* const PinholePrincipalPoint_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double PinholePrincipalPoint_out_local[1];

  double r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        PinholePrincipalPoint_p_kp1,
        0 * PinholePrincipalPoint_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(
        PinholePrincipalPoint_w,
        0 * PinholePrincipalPoint_w_num_alloc,
        global_thread_idx,
        r2,
        r3);
    r3 = fma(r1, r3, r0 * r2);
  };
  sum_store<double>(PinholePrincipalPoint_out_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r3);
  sum_flush_final<double>(
      PinholePrincipalPoint_out_local, PinholePrincipalPoint_out, 1);
}

void PinholePrincipalPoint_alpha_denumerator_or_beta_nummerator(
    double* PinholePrincipalPoint_p_kp1,
    unsigned int PinholePrincipalPoint_p_kp1_num_alloc,
    double* PinholePrincipalPoint_w,
    unsigned int PinholePrincipalPoint_w_num_alloc,
    double* const PinholePrincipalPoint_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholePrincipalPoint_alpha_denumerator_or_beta_nummerator_kernel<<<n_blocks,
                                                                      1024>>>(
      PinholePrincipalPoint_p_kp1,
      PinholePrincipalPoint_p_kp1_num_alloc,
      PinholePrincipalPoint_w,
      PinholePrincipalPoint_w_num_alloc,
      PinholePrincipalPoint_out,
      problem_size);
}

}  // namespace caspar