#include "kernel_SphericalPose_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SphericalPosePredDecreaseTimesTwoKernel(
        double* SphericalPose_step,
        unsigned int SphericalPose_step_num_alloc,
        double* SphericalPose_precond_diag,
        unsigned int SphericalPose_precond_diag_num_alloc,
        const double* const diag,
        double* SphericalPose_njtr,
        unsigned int SphericalPose_njtr_num_alloc,
        double* const out_SphericalPose_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_SphericalPose_pred_dec_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(SphericalPose_step,
                                            0 * SphericalPose_step_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
    ReadIdx2<1024, double, double, double2>(SphericalPose_njtr,
                                            0 * SphericalPose_njtr_num_alloc,
                                            global_thread_idx,
                                            r2,
                                            r3);
    ReadIdx2<1024, double, double, double2>(
        SphericalPose_precond_diag,
        0 * SphericalPose_precond_diag_num_alloc,
        global_thread_idx,
        r4,
        r5);
    r6 = r1 * r5;
  };
  LoadUnique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = fma(r7, r6, r3);
    ReadIdx2<1024, double, double, double2>(SphericalPose_step,
                                            2 * SphericalPose_step_num_alloc,
                                            global_thread_idx,
                                            r3,
                                            r8);
    ReadIdx2<1024, double, double, double2>(SphericalPose_njtr,
                                            2 * SphericalPose_njtr_num_alloc,
                                            global_thread_idx,
                                            r9,
                                            r10);
    ReadIdx2<1024, double, double, double2>(
        SphericalPose_precond_diag,
        2 * SphericalPose_precond_diag_num_alloc,
        global_thread_idx,
        r11,
        r12);
    r13 = r3 * r11;
    r13 = fma(r7, r13, r9);
    r13 = fma(r3, r13, r1 * r6);
    r6 = r0 * r4;
    r6 = fma(r7, r6, r2);
    ReadIdx2<1024, double, double, double2>(SphericalPose_step,
                                            4 * SphericalPose_step_num_alloc,
                                            global_thread_idx,
                                            r2,
                                            r9);
    ReadIdx2<1024, double, double, double2>(SphericalPose_njtr,
                                            4 * SphericalPose_njtr_num_alloc,
                                            global_thread_idx,
                                            r14,
                                            r15);
    ReadIdx2<1024, double, double, double2>(
        SphericalPose_precond_diag,
        4 * SphericalPose_precond_diag_num_alloc,
        global_thread_idx,
        r16,
        r17);
    r18 = r2 * r16;
    r18 = fma(r7, r18, r14);
    r14 = r8 * r12;
    r14 = fma(r7, r14, r10);
    r10 = r9 * r17;
    r10 = fma(r7, r10, r15);
    r13 = fma(r0, r6, r13);
    r13 = fma(r2, r18, r13);
    r13 = fma(r8, r14, r13);
    r13 = fma(r9, r10, r13);
  };
  SumStore<double>(out_SphericalPose_pred_dec_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r13);
  SumFlushFinal<double>(
      out_SphericalPose_pred_dec_local, out_SphericalPose_pred_dec, 1);
}

void SphericalPosePredDecreaseTimesTwo(
    double* SphericalPose_step,
    unsigned int SphericalPose_step_num_alloc,
    double* SphericalPose_precond_diag,
    unsigned int SphericalPose_precond_diag_num_alloc,
    const double* const diag,
    double* SphericalPose_njtr,
    unsigned int SphericalPose_njtr_num_alloc,
    double* const out_SphericalPose_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalPosePredDecreaseTimesTwoKernel<<<n_blocks, 1024>>>(
      SphericalPose_step,
      SphericalPose_step_num_alloc,
      SphericalPose_precond_diag,
      SphericalPose_precond_diag_num_alloc,
      diag,
      SphericalPose_njtr,
      SphericalPose_njtr_num_alloc,
      out_SphericalPose_pred_dec,
      problem_size);
}

}  // namespace caspar