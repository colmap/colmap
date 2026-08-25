#include "kernel_SphericalPose_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SphericalPoseAlphaDenominatorOrBetaNumeratorKernel(
        double* SphericalPose_p_kp1,
        unsigned int SphericalPose_p_kp1_num_alloc,
        double* SphericalPose_w,
        unsigned int SphericalPose_w_num_alloc,
        double* const SphericalPose_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double SphericalPose_out_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(SphericalPose_p_kp1,
                                            2 * SphericalPose_p_kp1_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
    ReadIdx2<1024, double, double, double2>(SphericalPose_w,
                                            2 * SphericalPose_w_num_alloc,
                                            global_thread_idx,
                                            r2,
                                            r3);
    ReadIdx2<1024, double, double, double2>(SphericalPose_p_kp1,
                                            4 * SphericalPose_p_kp1_num_alloc,
                                            global_thread_idx,
                                            r4,
                                            r5);
    ReadIdx2<1024, double, double, double2>(SphericalPose_w,
                                            4 * SphericalPose_w_num_alloc,
                                            global_thread_idx,
                                            r6,
                                            r7);
    r6 = fma(r4, r6, r1 * r3);
    ReadIdx2<1024, double, double, double2>(SphericalPose_p_kp1,
                                            0 * SphericalPose_p_kp1_num_alloc,
                                            global_thread_idx,
                                            r4,
                                            r3);
    ReadIdx2<1024, double, double, double2>(SphericalPose_w,
                                            0 * SphericalPose_w_num_alloc,
                                            global_thread_idx,
                                            r1,
                                            r8);
    r6 = fma(r5, r7, r6);
    r6 = fma(r4, r1, r6);
    r6 = fma(r0, r2, r6);
    r6 = fma(r3, r8, r6);
  };
  SumStore<double>(SphericalPose_out_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r6);
  SumFlushFinal<double>(SphericalPose_out_local, SphericalPose_out, 1);
}

void SphericalPoseAlphaDenominatorOrBetaNumerator(
    double* SphericalPose_p_kp1,
    unsigned int SphericalPose_p_kp1_num_alloc,
    double* SphericalPose_w,
    unsigned int SphericalPose_w_num_alloc,
    double* const SphericalPose_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalPoseAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks, 1024>>>(
      SphericalPose_p_kp1,
      SphericalPose_p_kp1_num_alloc,
      SphericalPose_w,
      SphericalPose_w_num_alloc,
      SphericalPose_out,
      problem_size);
}

}  // namespace caspar