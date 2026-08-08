#include "kernel_SphericalPose_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SphericalPoseRetractKernel(
    double* SphericalPose,
    unsigned int SphericalPose_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_SphericalPose_retracted,
    unsigned int out_SphericalPose_retracted_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        SphericalPose, 0 * SphericalPose_num_alloc, global_thread_idx, r0, r1);
    r2 = 5.00000000000000000e-01;
    r3 = 1.00000000000000008e-30;
    ReadIdx2<1024, double, double, double2>(
        delta, 0 * delta_num_alloc, global_thread_idx, r4, r5);
    r3 = fma(r5, r5, r3);
    ReadIdx2<1024, double, double, double2>(
        delta, 2 * delta_num_alloc, global_thread_idx, r6, r7);
    r3 = fma(r4, r4, r3);
    r3 = fma(r6, r6, r3);
    r8 = sqrt(r3);
    r8 = r2 * r8;
    r2 = cos(r8);
    ReadIdx2<1024, double, double, double2>(
        SphericalPose, 2 * SphericalPose_num_alloc, global_thread_idx, r9, r10);
    r11 = -1.00000000000000000e+00;
    r12 = r9 * r11;
    r8 = sin(r8);
    r3 = rsqrt(r3);
    r3 = r8 * r3;
    r5 = r5 * r3;
    r12 = fma(r5, r12, r0 * r2);
    r8 = r1 * r6;
    r12 = fma(r3, r8, r12);
    r13 = r10 * r4;
    r12 = fma(r3, r13, r12);
    r13 = r0 * r6;
    r13 = r13 * r11;
    r13 = fma(r3, r13, r1 * r2);
    r8 = r9 * r4;
    r13 = fma(r3, r8, r13);
    r13 = fma(r10, r5, r13);
    WriteIdx2<1024, double, double, double2>(
        out_SphericalPose_retracted,
        0 * out_SphericalPose_retracted_num_alloc,
        global_thread_idx,
        r12,
        r13);
    r13 = r0 * r4;
    r12 = r9 * r6;
    r12 = fma(r3, r12, r3 * r13);
    r12 = fma(r1, r5, r12);
    r12 = fma(r10, r2, r11 * r12);
    r5 = fma(r0, r5, r9 * r2);
    r2 = r1 * r4;
    r2 = r2 * r11;
    r5 = fma(r3, r2, r5);
    r13 = r10 * r6;
    r5 = fma(r3, r13, r5);
    WriteIdx2<1024, double, double, double2>(
        out_SphericalPose_retracted,
        2 * out_SphericalPose_retracted_num_alloc,
        global_thread_idx,
        r5,
        r12);
    ReadIdx2<1024, double, double, double2>(
        SphericalPose, 4 * SphericalPose_num_alloc, global_thread_idx, r12, r5);
    r7 = r12 + r7;
    ReadIdx2<1024, double, double, double2>(
        delta, 4 * delta_num_alloc, global_thread_idx, r12, r13);
    r12 = r5 + r12;
    WriteIdx2<1024, double, double, double2>(
        out_SphericalPose_retracted,
        4 * out_SphericalPose_retracted_num_alloc,
        global_thread_idx,
        r7,
        r12);
    ReadIdx1<1024, double, double, double>(
        SphericalPose, 6 * SphericalPose_num_alloc, global_thread_idx, r12);
    r13 = r12 + r13;
    WriteIdx1<1024, double, double, double>(
        out_SphericalPose_retracted,
        6 * out_SphericalPose_retracted_num_alloc,
        global_thread_idx,
        r13);
  };
}

void SphericalPoseRetract(double* SphericalPose,
                          unsigned int SphericalPose_num_alloc,
                          double* delta,
                          unsigned int delta_num_alloc,
                          double* out_SphericalPose_retracted,
                          unsigned int out_SphericalPose_retracted_num_alloc,
                          size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalPoseRetractKernel<<<n_blocks, 1024>>>(
      SphericalPose,
      SphericalPose_num_alloc,
      delta,
      delta_num_alloc,
      out_SphericalPose_retracted,
      out_SphericalPose_retracted_num_alloc,
      problem_size);
}

}  // namespace caspar