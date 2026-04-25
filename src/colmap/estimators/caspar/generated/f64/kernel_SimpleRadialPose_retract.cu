#include "kernel_SimpleRadialPose_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SimpleRadialPose_retract_kernel(
    double* SimpleRadialPose,
    unsigned int SimpleRadialPose_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_SimpleRadialPose_retracted,
    unsigned int out_SimpleRadialPose_retracted_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(SimpleRadialPose,
                                              0 * SimpleRadialPose_num_alloc,
                                              global_thread_idx,
                                              r0,
                                              r1);
    r2 = 5.00000000000000000e-01;
    r3 = 1.00000000000000008e-30;
    read_idx_2<1024, double, double, double2>(
        delta, 0 * delta_num_alloc, global_thread_idx, r4, r5);
    r3 = fma(r5, r5, r3);
    read_idx_2<1024, double, double, double2>(
        delta, 2 * delta_num_alloc, global_thread_idx, r6, r7);
    r3 = fma(r6, r6, r3);
    r3 = fma(r4, r4, r3);
    r8 = sqrt(r3);
    r8 = r2 * r8;
    r2 = cos(r8);
    r8 = sin(r8);
    r3 = rsqrt(r3);
    r3 = r8 * r3;
    r6 = r6 * r3;
    r8 = fma(r1, r6, r0 * r2);
    read_idx_2<1024, double, double, double2>(SimpleRadialPose,
                                              2 * SimpleRadialPose_num_alloc,
                                              global_thread_idx,
                                              r9,
                                              r10);
    r11 = r10 * r4;
    r8 = fma(r3, r11, r8);
    r12 = r9 * r5;
    r13 = -1.00000000000000000e+00;
    r12 = r12 * r13;
    r8 = fma(r3, r12, r8);
    r12 = r0 * r13;
    r12 = fma(r6, r12, r1 * r2);
    r11 = r10 * r5;
    r12 = fma(r3, r11, r12);
    r14 = r9 * r4;
    r12 = fma(r3, r14, r12);
    write_idx_2<1024, double, double, double2>(
        out_SimpleRadialPose_retracted,
        0 * out_SimpleRadialPose_retracted_num_alloc,
        global_thread_idx,
        r8,
        r12);
    r12 = r0 * r4;
    r8 = r1 * r5;
    r8 = fma(r3, r8, r3 * r12);
    r8 = fma(r9, r6, r8);
    r8 = fma(r13, r8, r10 * r2);
    r12 = r0 * r5;
    r12 = fma(r3, r12, r9 * r2);
    r2 = r1 * r4;
    r2 = r2 * r13;
    r12 = fma(r3, r2, r12);
    r12 = fma(r10, r6, r12);
    write_idx_2<1024, double, double, double2>(
        out_SimpleRadialPose_retracted,
        2 * out_SimpleRadialPose_retracted_num_alloc,
        global_thread_idx,
        r12,
        r8);
    read_idx_2<1024, double, double, double2>(SimpleRadialPose,
                                              4 * SimpleRadialPose_num_alloc,
                                              global_thread_idx,
                                              r8,
                                              r12);
    r7 = r8 + r7;
    read_idx_2<1024, double, double, double2>(
        delta, 4 * delta_num_alloc, global_thread_idx, r8, r6);
    r8 = r12 + r8;
    write_idx_2<1024, double, double, double2>(
        out_SimpleRadialPose_retracted,
        4 * out_SimpleRadialPose_retracted_num_alloc,
        global_thread_idx,
        r7,
        r8);
    read_idx_1<1024, double, double, double>(SimpleRadialPose,
                                             6 * SimpleRadialPose_num_alloc,
                                             global_thread_idx,
                                             r8);
    r6 = r8 + r6;
    write_idx_1<1024, double, double, double>(
        out_SimpleRadialPose_retracted,
        6 * out_SimpleRadialPose_retracted_num_alloc,
        global_thread_idx,
        r6);
  };
}

void SimpleRadialPose_retract(
    double* SimpleRadialPose,
    unsigned int SimpleRadialPose_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_SimpleRadialPose_retracted,
    unsigned int out_SimpleRadialPose_retracted_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialPose_retract_kernel<<<n_blocks, 1024>>>(
      SimpleRadialPose,
      SimpleRadialPose_num_alloc,
      delta,
      delta_num_alloc,
      out_SimpleRadialPose_retracted,
      out_SimpleRadialPose_retracted_num_alloc,
      problem_size);
}

}  // namespace caspar