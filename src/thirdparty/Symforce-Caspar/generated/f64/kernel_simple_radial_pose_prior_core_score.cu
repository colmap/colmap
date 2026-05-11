#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_pose_prior_core_score.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SimpleRadialPosePriorCoreScoreKernel(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *prior_position, unsigned int prior_position_num_alloc,
    double *sqrt_info, unsigned int sqrt_info_num_alloc, double *const out_rTr,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sqrt_info, 0 * sqrt_info_num_alloc,
                                            global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(prior_position,
                                            0 * prior_position_num_alloc,
                                            global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
  };
  LoadShared<2, double, double>(pose, 4 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r5, r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r7 = -2.00000000000000000e+00;
  };
  LoadShared<2, double, double>(pose, 2 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = r8 * r8;
    r10 = r7 * r10;
    r11 = 1.00000000000000000e+00;
  };
  LoadShared<2, double, double>(pose, 0 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = r12 * r12;
    r14 = fma(r7, r14, r11);
    r15 = r10 + r14;
  };
  LoadShared<1, double, double>(pose, 6 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r17 = r12 * r9;
    r18 = 2.00000000000000000e+00;
    r19 = r13 * r18;
    r20 = r8 * r19;
    r17 = fma(r18, r17, r20);
    r17 = fma(r16, r17, r6 * r15);
    r15 = r9 * r7;
    r21 = r12 * r19;
    r22 = fma(r8, r15, r21);
    r17 = fma(r5, r22, r17);
    r17 = fma(r4, r17, r3 * r4);
    ReadIdx2<1024, double, double, double2>(sqrt_info, 2 * sqrt_info_num_alloc,
                                            global_thread_idx, r3, r22);
    ReadIdx1<1024, double, double, double>(
        prior_position, 2 * prior_position_num_alloc, global_thread_idx, r23);
    r24 = r13 * r13;
    r24 = r24 * r7;
    r14 = r24 + r14;
    r20 = fma(r12, r15, r20);
    r20 = fma(r6, r20, r16 * r14);
    r14 = r12 * r8;
    r14 = r14 * r18;
    r19 = fma(r9, r19, r14);
    r20 = fma(r5, r19, r20);
    r20 = fma(r4, r20, r23 * r4);
    r3 = fma(r3, r20, r1 * r17);
    r24 = r11 + r24;
    r24 = r24 + r10;
    r10 = r8 * r9;
    r10 = fma(r18, r10, r21);
    r10 = fma(r6, r10, r5 * r24);
    r15 = fma(r13, r15, r14);
    r10 = fma(r16, r15, r10);
    r10 = fma(r4, r10, r2 * r4);
    r3 = fma(r0, r10, r3);
    ReadIdx2<1024, double, double, double2>(sqrt_info, 4 * sqrt_info_num_alloc,
                                            global_thread_idx, r10, r0);
    r10 = fma(r10, r20, r22 * r17);
    r10 = fma(r10, r10, r3 * r3);
    r20 = r20 * r20;
    r0 = r0 * r0;
    r10 = fma(r20, r0, r10);
  };
  SumStore<double>(out_rTr_local, (double *)inout_shared, 0,
                   global_thread_idx < problem_size, r10);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialPosePriorCoreScore(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *prior_position, unsigned int prior_position_num_alloc,
    double *sqrt_info, unsigned int sqrt_info_num_alloc, double *const out_rTr,
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialPosePriorCoreScoreKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, prior_position,
      prior_position_num_alloc, sqrt_info, sqrt_info_num_alloc, out_rTr,
      problem_size);
}

} // namespace caspar