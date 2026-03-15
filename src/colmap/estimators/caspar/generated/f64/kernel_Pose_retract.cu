#include "kernel_Pose_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    Pose_retract_kernel(double* Pose,
                        unsigned int Pose_num_alloc,
                        double* delta,
                        unsigned int delta_num_alloc,
                        double* out_Pose_retracted,
                        unsigned int out_Pose_retracted_num_alloc,
                        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        Pose, 0 * Pose_num_alloc, global_thread_idx, r0, r1);
    r2 = 5.00000000000000000e-01;
    r3 = 1.00000000000000008e-30;
    read_idx_2<1024, double, double, double2>(
        delta, 2 * delta_num_alloc, global_thread_idx, r4, r5);
    r3 = fma(r4, r4, r3);
    read_idx_2<1024, double, double, double2>(
        delta, 0 * delta_num_alloc, global_thread_idx, r6, r7);
    r3 = fma(r7, r7, r3);
    r3 = fma(r6, r6, r3);
    r8 = sqrt(r3);
    r8 = r2 * r8;
    r2 = cos(r8);
    r8 = sin(r8);
    r3 = rsqrt(r3);
    r3 = r8 * r3;
    r4 = r4 * r3;
    r8 = fma(r1, r4, r0 * r2);
    read_idx_2<1024, double, double, double2>(
        Pose, 2 * Pose_num_alloc, global_thread_idx, r9, r10);
    r11 = r10 * r6;
    r8 = fma(r3, r11, r8);
    r12 = r9 * r7;
    r13 = -1.00000000000000000e+00;
    r12 = r12 * r13;
    r8 = fma(r3, r12, r8);
    r12 = r0 * r13;
    r12 = fma(r4, r12, r1 * r2);
    r11 = r9 * r6;
    r12 = fma(r3, r11, r12);
    r14 = r10 * r7;
    r12 = fma(r3, r14, r12);
    write_idx_2<1024, double, double, double2>(out_Pose_retracted,
                                               0 * out_Pose_retracted_num_alloc,
                                               global_thread_idx,
                                               r8,
                                               r12);
    r12 = r0 * r6;
    r12 = fma(r3, r12, r9 * r4);
    r8 = r1 * r7;
    r12 = fma(r3, r8, r12);
    r12 = fma(r10, r2, r13 * r12);
    r4 = fma(r10, r4, r9 * r2);
    r2 = r1 * r6;
    r2 = r2 * r13;
    r4 = fma(r3, r2, r4);
    r8 = r0 * r7;
    r4 = fma(r3, r8, r4);
    write_idx_2<1024, double, double, double2>(out_Pose_retracted,
                                               2 * out_Pose_retracted_num_alloc,
                                               global_thread_idx,
                                               r4,
                                               r12);
    read_idx_2<1024, double, double, double2>(
        Pose, 4 * Pose_num_alloc, global_thread_idx, r12, r4);
    r5 = r12 + r5;
    read_idx_2<1024, double, double, double2>(
        delta, 4 * delta_num_alloc, global_thread_idx, r12, r8);
    r12 = r4 + r12;
    write_idx_2<1024, double, double, double2>(out_Pose_retracted,
                                               4 * out_Pose_retracted_num_alloc,
                                               global_thread_idx,
                                               r5,
                                               r12);
    read_idx_1<1024, double, double, double>(
        Pose, 6 * Pose_num_alloc, global_thread_idx, r12);
    r8 = r12 + r8;
    write_idx_1<1024, double, double, double>(out_Pose_retracted,
                                              6 * out_Pose_retracted_num_alloc,
                                              global_thread_idx,
                                              r8);
  };
}

void Pose_retract(double* Pose,
                  unsigned int Pose_num_alloc,
                  double* delta,
                  unsigned int delta_num_alloc,
                  double* out_Pose_retracted,
                  unsigned int out_Pose_retracted_num_alloc,
                  size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Pose_retract_kernel<<<n_blocks, 1024>>>(Pose,
                                          Pose_num_alloc,
                                          delta,
                                          delta_num_alloc,
                                          out_Pose_retracted,
                                          out_Pose_retracted_num_alloc,
                                          problem_size);
}

}  // namespace caspar