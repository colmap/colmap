#include "kernel_SimpleRadialFocal_update_r_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialFocal_update_r_first_kernel(
        double* SimpleRadialFocal_r_k,
        unsigned int SimpleRadialFocal_r_k_num_alloc,
        double* SimpleRadialFocal_w,
        unsigned int SimpleRadialFocal_w_num_alloc,
        const double* const negalpha,
        double* out_SimpleRadialFocal_r_kp1,
        unsigned int out_SimpleRadialFocal_r_kp1_num_alloc,
        double* const out_SimpleRadialFocal_r_0_norm2_tot,
        double* const out_SimpleRadialFocal_r_kp1_norm2_tot,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_SimpleRadialFocal_r_0_norm2_tot_local[1];

  __shared__ double out_SimpleRadialFocal_r_kp1_norm2_tot_local[1];

  double r0, r1, r2;

  if (global_thread_idx < problem_size) {
    read_idx_1<1024, double, double, double>(
        SimpleRadialFocal_r_k,
        0 * SimpleRadialFocal_r_k_num_alloc,
        global_thread_idx,
        r0);
    read_idx_1<1024, double, double, double>(SimpleRadialFocal_w,
                                             0 * SimpleRadialFocal_w_num_alloc,
                                             global_thread_idx,
                                             r1);
  };
  load_unique<1, double, double>(negalpha, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r1, r2, r0);
    write_idx_1<1024, double, double, double>(
        out_SimpleRadialFocal_r_kp1,
        0 * out_SimpleRadialFocal_r_kp1_num_alloc,
        global_thread_idx,
        r2);
    r0 = r0 * r0;
  };
  sum_store<double>(out_SimpleRadialFocal_r_0_norm2_tot_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r0);
  if (global_thread_idx < problem_size) {
    r2 = r2 * r2;
  };
  sum_store<double>(out_SimpleRadialFocal_r_kp1_norm2_tot_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r2);
  sum_flush_final<double>(out_SimpleRadialFocal_r_0_norm2_tot_local,
                          out_SimpleRadialFocal_r_0_norm2_tot,
                          1);
  sum_flush_final<double>(out_SimpleRadialFocal_r_kp1_norm2_tot_local,
                          out_SimpleRadialFocal_r_kp1_norm2_tot,
                          1);
}

void SimpleRadialFocal_update_r_first(
    double* SimpleRadialFocal_r_k,
    unsigned int SimpleRadialFocal_r_k_num_alloc,
    double* SimpleRadialFocal_w,
    unsigned int SimpleRadialFocal_w_num_alloc,
    const double* const negalpha,
    double* out_SimpleRadialFocal_r_kp1,
    unsigned int out_SimpleRadialFocal_r_kp1_num_alloc,
    double* const out_SimpleRadialFocal_r_0_norm2_tot,
    double* const out_SimpleRadialFocal_r_kp1_norm2_tot,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFocal_update_r_first_kernel<<<n_blocks, 1024>>>(
      SimpleRadialFocal_r_k,
      SimpleRadialFocal_r_k_num_alloc,
      SimpleRadialFocal_w,
      SimpleRadialFocal_w_num_alloc,
      negalpha,
      out_SimpleRadialFocal_r_kp1,
      out_SimpleRadialFocal_r_kp1_num_alloc,
      out_SimpleRadialFocal_r_0_norm2_tot,
      out_SimpleRadialFocal_r_kp1_norm2_tot,
      problem_size);
}

}  // namespace caspar