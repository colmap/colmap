#include "kernel_SphericalPose_update_r_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SphericalPoseUpdateRFirstKernel(
    double* SphericalPose_r_k,
    unsigned int SphericalPose_r_k_num_alloc,
    double* SphericalPose_w,
    unsigned int SphericalPose_w_num_alloc,
    const double* const negalpha,
    double* out_SphericalPose_r_kp1,
    unsigned int out_SphericalPose_r_kp1_num_alloc,
    double* const out_SphericalPose_r_0_norm2_tot,
    double* const out_SphericalPose_r_kp1_norm2_tot,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_SphericalPose_r_0_norm2_tot_local[1];

  __shared__ double out_SphericalPose_r_kp1_norm2_tot_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(SphericalPose_r_k,
                                            0 * SphericalPose_r_k_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
    ReadIdx2<1024, double, double, double2>(SphericalPose_w,
                                            0 * SphericalPose_w_num_alloc,
                                            global_thread_idx,
                                            r2,
                                            r3);
  };
  LoadUnique<1, double, double>(negalpha, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r2, r4, r0);
    r3 = fma(r3, r4, r1);
    WriteIdx2<1024, double, double, double2>(
        out_SphericalPose_r_kp1,
        0 * out_SphericalPose_r_kp1_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(SphericalPose_r_k,
                                            2 * SphericalPose_r_k_num_alloc,
                                            global_thread_idx,
                                            r5,
                                            r6);
    ReadIdx2<1024, double, double, double2>(SphericalPose_w,
                                            2 * SphericalPose_w_num_alloc,
                                            global_thread_idx,
                                            r7,
                                            r8);
    r7 = fma(r7, r4, r5);
    r8 = fma(r8, r4, r6);
    WriteIdx2<1024, double, double, double2>(
        out_SphericalPose_r_kp1,
        2 * out_SphericalPose_r_kp1_num_alloc,
        global_thread_idx,
        r7,
        r8);
    ReadIdx2<1024, double, double, double2>(SphericalPose_r_k,
                                            4 * SphericalPose_r_k_num_alloc,
                                            global_thread_idx,
                                            r9,
                                            r10);
    ReadIdx2<1024, double, double, double2>(SphericalPose_w,
                                            4 * SphericalPose_w_num_alloc,
                                            global_thread_idx,
                                            r11,
                                            r12);
    r11 = fma(r11, r4, r9);
    r4 = fma(r12, r4, r10);
    WriteIdx2<1024, double, double, double2>(
        out_SphericalPose_r_kp1,
        4 * out_SphericalPose_r_kp1_num_alloc,
        global_thread_idx,
        r11,
        r4);
    r5 = fma(r5, r5, r1 * r1);
    r5 = fma(r0, r0, r5);
    r5 = fma(r10, r10, r5);
    r5 = fma(r9, r9, r5);
    r5 = fma(r6, r6, r5);
  };
  SumStore<double>(out_SphericalPose_r_0_norm2_tot_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r5);
  if (global_thread_idx < problem_size) {
    r11 = fma(r11, r11, r3 * r3);
    r11 = fma(r2, r2, r11);
    r11 = fma(r7, r7, r11);
    r11 = fma(r8, r8, r11);
    r11 = fma(r4, r4, r11);
  };
  SumStore<double>(out_SphericalPose_r_kp1_norm2_tot_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r11);
  SumFlushFinal<double>(out_SphericalPose_r_0_norm2_tot_local,
                        out_SphericalPose_r_0_norm2_tot,
                        1);
  SumFlushFinal<double>(out_SphericalPose_r_kp1_norm2_tot_local,
                        out_SphericalPose_r_kp1_norm2_tot,
                        1);
}

void SphericalPoseUpdateRFirst(double* SphericalPose_r_k,
                               unsigned int SphericalPose_r_k_num_alloc,
                               double* SphericalPose_w,
                               unsigned int SphericalPose_w_num_alloc,
                               const double* const negalpha,
                               double* out_SphericalPose_r_kp1,
                               unsigned int out_SphericalPose_r_kp1_num_alloc,
                               double* const out_SphericalPose_r_0_norm2_tot,
                               double* const out_SphericalPose_r_kp1_norm2_tot,
                               size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalPoseUpdateRFirstKernel<<<n_blocks, 1024>>>(
      SphericalPose_r_k,
      SphericalPose_r_k_num_alloc,
      SphericalPose_w,
      SphericalPose_w_num_alloc,
      negalpha,
      out_SphericalPose_r_kp1,
      out_SphericalPose_r_kp1_num_alloc,
      out_SphericalPose_r_0_norm2_tot,
      out_SphericalPose_r_kp1_norm2_tot,
      problem_size);
}

}  // namespace caspar