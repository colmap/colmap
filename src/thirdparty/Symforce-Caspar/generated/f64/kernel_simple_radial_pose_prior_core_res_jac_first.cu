#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_pose_prior_core_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialPosePriorCoreResJacFirstKernel(
        double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        double *prior_position, unsigned int prior_position_num_alloc,
        double *sqrt_info, unsigned int sqrt_info_num_alloc, double *out_res,
        unsigned int out_res_num_alloc, double *const out_rTr,
        double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
        double *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45;

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
    r11 = r7 * r10;
    r12 = 1.00000000000000000e+00;
  };
  LoadShared<2, double, double>(pose, 0 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r13, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = r13 * r13;
    r16 = fma(r7, r15, r12);
    r17 = r11 + r16;
  };
  LoadShared<1, double, double>(pose, 6 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r19 = r13 * r9;
    r20 = 2.00000000000000000e+00;
    r19 = r19 * r20;
    r21 = r14 * r20;
    r22 = r8 * r21;
    r23 = r19 + r22;
    r23 = r18 * r23;
    r17 = fma(r6, r17, r23);
    r24 = r9 * r7;
    r25 = r8 * r24;
    r26 = r13 * r21;
    r27 = r25 + r26;
    r27 = r5 * r27;
    r17 = r17 + r27;
    r17 = fma(r4, r17, r3 * r4);
    ReadIdx2<1024, double, double, double2>(sqrt_info, 2 * sqrt_info_num_alloc,
                                            global_thread_idx, r3, r28);
    ReadIdx1<1024, double, double, double>(
        prior_position, 2 * prior_position_num_alloc, global_thread_idx, r29);
    r30 = r14 * r14;
    r30 = r30 * r7;
    r16 = r30 + r16;
    r31 = r13 * r24;
    r22 = r22 + r31;
    r16 = fma(r6, r22, r18 * r16);
    r32 = r13 * r8;
    r32 = r32 * r20;
    r33 = r9 * r21;
    r34 = r32 + r33;
    r16 = fma(r5, r34, r16);
    r16 = fma(r4, r16, r29 * r4);
    r29 = fma(r3, r16, r1 * r17);
    r11 = r12 + r11;
    r11 = r11 + r30;
    r30 = r8 * r9;
    r30 = r30 * r20;
    r26 = r30 + r26;
    r11 = fma(r6, r26, r5 * r11);
    r24 = r14 * r24;
    r32 = r32 + r24;
    r11 = fma(r18, r32, r11);
    r11 = fma(r4, r11, r2 * r4);
    r29 = fma(r0, r11, r29);
    ReadIdx2<1024, double, double, double2>(sqrt_info, 4 * sqrt_info_num_alloc,
                                            global_thread_idx, r11, r2);
    r17 = fma(r11, r16, r28 * r17);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc,
                                             global_thread_idx, r29, r17);
    r12 = r2 * r16;
    WriteIdx1<1024, double, double, double>(out_res, 2 * out_res_num_alloc,
                                            global_thread_idx, r12);
    r12 = fma(r17, r17, r29 * r29);
    r35 = r16 * r16;
    r2 = r2 * r2;
    r12 = fma(r2, r35, r12);
  };
  SumStore<double>(out_rTr_local, (double *)inout_shared, 0,
                   global_thread_idx < problem_size, r12);
  if (global_thread_idx < problem_size) {
    r12 = r4 * r17;
    r35 = r18 * r4;
    r36 = r14 * r8;
    r36 = r36 * r7;
    r31 = r36 + r31;
    r37 = r5 * r4;
    r38 = r13 * r14;
    r38 = r38 * r7;
    r30 = r30 + r38;
    r37 = fma(r30, r37, r31 * r35);
    r6 = r6 * r4;
    r35 = r9 * r9;
    r35 = r35 * r4;
    r39 = r15 + r35;
    r40 = r14 * r14;
    r40 = r40 * r4;
    r41 = r10 + r40;
    r42 = r39 + r41;
    r37 = fma(r42, r6, r37);
    r42 = r18 * r4;
    r9 = r9 * r9;
    r43 = r4 * r15;
    r44 = r9 + r43;
    r41 = r41 + r44;
    r45 = r5 * r4;
    r45 = fma(r34, r45, r41 * r42);
    r45 = fma(r22, r6, r45);
    r22 = fma(r28, r45, r11 * r37);
    r42 = r4 * r29;
    r45 = fma(r1, r45, r3 * r37);
    r42 = fma(r45, r42, r22 * r12);
    r12 = r4 * r16;
    r12 = r12 * r37;
    r42 = fma(r2, r12, r42);
    r12 = r4 * r29;
    r34 = r5 * r4;
    r9 = r15 + r9;
    r41 = r4 * r10;
    r9 = r9 + r40;
    r9 = r9 + r41;
    r40 = r18 * r4;
    r40 = fma(r32, r40, r9 * r34);
    r40 = fma(r26, r6, r40);
    r26 = r18 * r4;
    r34 = r14 * r14;
    r41 = r34 + r41;
    r39 = r39 + r41;
    r32 = r5 * r4;
    r8 = r13 * r8;
    r8 = r8 * r7;
    r24 = r8 + r24;
    r32 = fma(r24, r32, r39 * r26);
    r36 = r19 + r36;
    r32 = fma(r36, r6, r32);
    r32 = fma(r0, r32, r3 * r40);
    r19 = r4 * r17;
    r26 = r11 * r40;
    r19 = fma(r26, r19, r32 * r12);
    r12 = r4 * r16;
    r12 = r12 * r40;
    r19 = fma(r2, r12, r19);
    WriteSum2<double, double>((double *)inout_shared, r42, r19);
  };
  FlushSumShared<2, double>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r4 * r29;
    r41 = r44 + r41;
    r23 = fma(r4, r23, r41 * r6);
    r23 = fma(r4, r27, r23);
    r27 = r5 * r4;
    r34 = r10 + r34;
    r34 = r34 + r35;
    r34 = r34 + r43;
    r43 = r18 * r4;
    r33 = r8 + r33;
    r43 = fma(r33, r43, r34 * r27);
    r25 = r38 + r25;
    r43 = fma(r25, r6, r43);
    r23 = fma(r1, r43, r0 * r23);
    r6 = r4 * r17;
    r38 = r28 * r43;
    r6 = fma(r38, r6, r23 * r19);
    r19 = r4 * r17;
    r27 = fma(r11, r24, r28 * r30);
    r34 = r4 * r29;
    r10 = r20 * r10;
    r21 = fma(r14, r21, r4);
    r14 = r10 + r21;
    r30 = fma(r1, r30, r0 * r14);
    r30 = fma(r3, r24, r30);
    r34 = fma(r30, r34, r27 * r19);
    r19 = r4 * r16;
    r19 = r19 * r24;
    r34 = fma(r2, r19, r34);
    WriteSum2<double, double>((double *)inout_shared, r6, r34);
  };
  FlushSumShared<2, double>(out_pose_njtr, 2 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r4 * r17;
    r10 = r4 + r10;
    r15 = r20 * r15;
    r10 = r10 + r15;
    r20 = fma(r11, r36, r28 * r10);
    r6 = r4 * r29;
    r10 = fma(r3, r36, r1 * r10);
    r10 = fma(r0, r25, r10);
    r6 = fma(r10, r6, r20 * r34);
    r34 = r4 * r16;
    r25 = r36 * r2;
    r6 = fma(r25, r34, r6);
    r34 = r4 * r17;
    r21 = r15 + r21;
    r15 = fma(r28, r31, r11 * r21);
    r19 = r4 * r29;
    r31 = fma(r1, r31, r3 * r21);
    r31 = fma(r0, r33, r31);
    r19 = fma(r31, r19, r15 * r34);
    r34 = r4 * r16;
    r34 = r34 * r21;
    r19 = fma(r2, r34, r19);
    WriteSum2<double, double>((double *)inout_shared, r6, r19);
  };
  FlushSumShared<2, double>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fma(r45, r45, r22 * r22);
    r6 = r37 * r37;
    r19 = fma(r2, r6, r19);
    r6 = r40 * r40;
    r6 = fma(r2, r6, r32 * r32);
    r34 = r11 * r40;
    r6 = fma(r26, r34, r6);
    WriteSum2<double, double>((double *)inout_shared, r19, r6);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r28 * r43;
    r6 = fma(r38, r6, r23 * r23);
    r19 = fma(r27, r27, r30 * r30);
    r34 = r24 * r24;
    r19 = fma(r2, r34, r19);
    WriteSum2<double, double>((double *)inout_shared, r6, r19);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fma(r10, r10, r20 * r20);
    r19 = fma(r36, r25, r19);
    r36 = fma(r15, r15, r31 * r31);
    r6 = r21 * r21;
    r36 = fma(r2, r6, r36);
    WriteSum2<double, double>((double *)inout_shared, r19, r36);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fma(r22, r26, r45 * r32);
    r19 = r37 * r40;
    r36 = fma(r2, r19, r36);
    r19 = fma(r22, r38, r45 * r23);
    WriteSum2<double, double>((double *)inout_shared, r36, r19);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fma(r45, r30, r22 * r27);
    r36 = r37 * r24;
    r19 = fma(r2, r36, r19);
    r36 = fma(r45, r10, r22 * r20);
    r36 = fma(r37, r25, r36);
    WriteSum2<double, double>((double *)inout_shared, r19, r36);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r45, r31, r22 * r15);
    r22 = r37 * r21;
    r45 = fma(r2, r22, r45);
    r22 = fma(r26, r38, r32 * r23);
    WriteSum2<double, double>((double *)inout_shared, r45, r22);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = r40 * r24;
    r22 = fma(r2, r22, r32 * r30);
    r22 = fma(r27, r26, r22);
    r45 = fma(r20, r26, r32 * r10);
    r45 = fma(r40, r25, r45);
    WriteSum2<double, double>((double *)inout_shared, r22, r45);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fma(r15, r26, r32 * r31);
    r32 = r40 * r21;
    r26 = fma(r2, r32, r26);
    r32 = fma(r27, r38, r23 * r30);
    WriteSum2<double, double>((double *)inout_shared, r26, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r20, r38, r23 * r10);
    r38 = fma(r15, r38, r23 * r31);
    WriteSum2<double, double>((double *)inout_shared, r32, r38);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fma(r27, r20, r30 * r10);
    r38 = fma(r24, r25, r38);
    r30 = fma(r30, r31, r27 * r15);
    r27 = r24 * r21;
    r30 = fma(r2, r27, r30);
    WriteSum2<double, double>((double *)inout_shared, r38, r30);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fma(r10, r31, r20 * r15);
    r31 = fma(r21, r25, r31);
    WriteSum1<double, double>((double *)inout_shared, r31);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialPosePriorCoreResJacFirst(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *prior_position, unsigned int prior_position_num_alloc,
    double *sqrt_info, unsigned int sqrt_info_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr,
    double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
    double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialPosePriorCoreResJacFirstKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, prior_position,
      prior_position_num_alloc, sqrt_info, sqrt_info_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_pose_njtr, out_pose_njtr_num_alloc,
      out_pose_precond_diag, out_pose_precond_diag_num_alloc,
      out_pose_precond_tril, out_pose_precond_tril_num_alloc, problem_size);
}

} // namespace caspar