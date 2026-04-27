#include "kernel_SimpleRadialPose_alpha_denumerator_or_beta_nummerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialPose_alpha_denumerator_or_beta_nummerator_kernel(
        double* SimpleRadialPose_p_kp1,
        unsigned int SimpleRadialPose_p_kp1_num_alloc,
        double* SimpleRadialPose_w,
        unsigned int SimpleRadialPose_w_num_alloc,
        double* const SimpleRadialPose_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double SimpleRadialPose_out_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        SimpleRadialPose_p_kp1,
        4 * SimpleRadialPose_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(SimpleRadialPose_w,
                                              4 * SimpleRadialPose_w_num_alloc,
                                              global_thread_idx,
                                              r2,
                                              r3);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialPose_p_kp1,
        0 * SimpleRadialPose_p_kp1_num_alloc,
        global_thread_idx,
        r4,
        r5);
    read_idx_2<1024, double, double, double2>(SimpleRadialPose_w,
                                              0 * SimpleRadialPose_w_num_alloc,
                                              global_thread_idx,
                                              r6,
                                              r7);
    r6 = fma(r4, r6, r1 * r3);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialPose_p_kp1,
        2 * SimpleRadialPose_p_kp1_num_alloc,
        global_thread_idx,
        r4,
        r3);
    read_idx_2<1024, double, double, double2>(SimpleRadialPose_w,
                                              2 * SimpleRadialPose_w_num_alloc,
                                              global_thread_idx,
                                              r1,
                                              r8);
    r6 = fma(r5, r7, r6);
    r6 = fma(r3, r8, r6);
    r6 = fma(r0, r2, r6);
    r6 = fma(r4, r1, r6);
  };
  sum_store<double>(SimpleRadialPose_out_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r6);
  sum_flush_final<double>(SimpleRadialPose_out_local, SimpleRadialPose_out, 1);
}

void SimpleRadialPose_alpha_denumerator_or_beta_nummerator(
    double* SimpleRadialPose_p_kp1,
    unsigned int SimpleRadialPose_p_kp1_num_alloc,
    double* SimpleRadialPose_w,
    unsigned int SimpleRadialPose_w_num_alloc,
    double* const SimpleRadialPose_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialPose_alpha_denumerator_or_beta_nummerator_kernel<<<n_blocks,
                                                                 1024>>>(
      SimpleRadialPose_p_kp1,
      SimpleRadialPose_p_kp1_num_alloc,
      SimpleRadialPose_w,
      SimpleRadialPose_w_num_alloc,
      SimpleRadialPose_out,
      problem_size);
}

}  // namespace caspar