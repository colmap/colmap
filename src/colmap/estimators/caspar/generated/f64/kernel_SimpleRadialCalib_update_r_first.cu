#include "kernel_SimpleRadialCalib_update_r_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialCalib_update_r_first_kernel(
        double* SimpleRadialCalib_r_k,
        unsigned int SimpleRadialCalib_r_k_num_alloc,
        double* SimpleRadialCalib_w,
        unsigned int SimpleRadialCalib_w_num_alloc,
        const double* const negalpha,
        double* out_SimpleRadialCalib_r_kp1,
        unsigned int out_SimpleRadialCalib_r_kp1_num_alloc,
        double* const out_SimpleRadialCalib_r_0_norm2_tot,
        double* const out_SimpleRadialCalib_r_kp1_norm2_tot,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_SimpleRadialCalib_r_0_norm2_tot_local[1];

  __shared__ double out_SimpleRadialCalib_r_kp1_norm2_tot_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        SimpleRadialCalib_r_k,
        0 * SimpleRadialCalib_r_k_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(SimpleRadialCalib_w,
                                              0 * SimpleRadialCalib_w_num_alloc,
                                              global_thread_idx,
                                              r2,
                                              r3);
  };
  load_unique<1, double, double>(negalpha, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r2, r4, r0);
    r3 = fma(r3, r4, r1);
    write_idx_2<1024, double, double, double2>(
        out_SimpleRadialCalib_r_kp1,
        0 * out_SimpleRadialCalib_r_kp1_num_alloc,
        global_thread_idx,
        r2,
        r3);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialCalib_r_k,
        2 * SimpleRadialCalib_r_k_num_alloc,
        global_thread_idx,
        r5,
        r6);
    read_idx_2<1024, double, double, double2>(SimpleRadialCalib_w,
                                              2 * SimpleRadialCalib_w_num_alloc,
                                              global_thread_idx,
                                              r7,
                                              r8);
    r7 = fma(r7, r4, r5);
    r4 = fma(r8, r4, r6);
    write_idx_2<1024, double, double, double2>(
        out_SimpleRadialCalib_r_kp1,
        2 * out_SimpleRadialCalib_r_kp1_num_alloc,
        global_thread_idx,
        r7,
        r4);
    r0 = fma(r0, r0, r5 * r5);
    r0 = fma(r1, r1, r0);
    r0 = fma(r6, r6, r0);
  };
  sum_store<double>(out_SimpleRadialCalib_r_0_norm2_tot_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r0);
  if (global_thread_idx < problem_size) {
    r2 = fma(r2, r2, r4 * r4);
    r2 = fma(r7, r7, r2);
    r2 = fma(r3, r3, r2);
  };
  sum_store<double>(out_SimpleRadialCalib_r_kp1_norm2_tot_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r2);
  sum_flush_final<double>(out_SimpleRadialCalib_r_0_norm2_tot_local,
                          out_SimpleRadialCalib_r_0_norm2_tot,
                          1);
  sum_flush_final<double>(out_SimpleRadialCalib_r_kp1_norm2_tot_local,
                          out_SimpleRadialCalib_r_kp1_norm2_tot,
                          1);
}

void SimpleRadialCalib_update_r_first(
    double* SimpleRadialCalib_r_k,
    unsigned int SimpleRadialCalib_r_k_num_alloc,
    double* SimpleRadialCalib_w,
    unsigned int SimpleRadialCalib_w_num_alloc,
    const double* const negalpha,
    double* out_SimpleRadialCalib_r_kp1,
    unsigned int out_SimpleRadialCalib_r_kp1_num_alloc,
    double* const out_SimpleRadialCalib_r_0_norm2_tot,
    double* const out_SimpleRadialCalib_r_kp1_norm2_tot,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialCalib_update_r_first_kernel<<<n_blocks, 1024>>>(
      SimpleRadialCalib_r_k,
      SimpleRadialCalib_r_k_num_alloc,
      SimpleRadialCalib_w,
      SimpleRadialCalib_w_num_alloc,
      negalpha,
      out_SimpleRadialCalib_r_kp1,
      out_SimpleRadialCalib_r_kp1_num_alloc,
      out_SimpleRadialCalib_r_0_norm2_tot,
      out_SimpleRadialCalib_r_kp1_norm2_tot,
      problem_size);
}

}  // namespace caspar