#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_split_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPrincipalPointFixedPointResJacFirstKernel(
        double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        double *focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
        SharedIndex *focal_and_distortion_indices, double *pixel,
        unsigned int pixel_num_alloc, double *principal_point,
        unsigned int principal_point_num_alloc, double *point,
        unsigned int point_num_alloc, double *out_res,
        unsigned int out_res_num_alloc, double *const out_rTr,
        double *out_pose_jac, unsigned int out_pose_jac_num_alloc,
        double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
        double *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        double *out_focal_and_distortion_jac,
        unsigned int out_focal_and_distortion_jac_num_alloc,
        double *const out_focal_and_distortion_njtr,
        unsigned int out_focal_and_distortion_njtr_num_alloc,
        double *const out_focal_and_distortion_precond_diag,
        unsigned int out_focal_and_distortion_precond_diag_num_alloc,
        double *const out_focal_and_distortion_precond_tril,
        unsigned int out_focal_and_distortion_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex focal_and_distortion_indices_loc[1024];
  focal_and_distortion_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_distortion_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(pixel, 0 * pixel_num_alloc,
                                            global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
  };
  LoadShared<2, double, double>(
      focal_and_distortion, 0 * focal_and_distortion_num_alloc,
      focal_and_distortion_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        focal_and_distortion_indices_loc[threadIdx.x].target,
                        r0, r5);
  };
  __syncthreads();
  LoadShared<2, double, double>(pose, 4 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(point, 0 * point_num_alloc,
                                            global_thread_idx, r8, r9);
  };
  LoadShared<2, double, double>(pose, 0 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = r10 * r11;
    r13 = 2.00000000000000000e+00;
    r12 = r12 * r13;
  };
  LoadShared<2, double, double>(pose, 2 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r14, r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = -2.00000000000000000e+00;
    r17 = r14 * r16;
    r18 = r15 * r17;
    r19 = r12 + r18;
    r6 = fma(r9, r19, r6);
    ReadIdx1<1024, double, double, double>(point, 2 * point_num_alloc,
                                           global_thread_idx, r20);
    r21 = r10 * r14;
    r21 = r21 * r13;
    r22 = r11 * r15;
    r22 = r22 * r13;
    r23 = r21 + r22;
    r24 = r14 * r17;
    r25 = 1.00000000000000000e+00;
    r26 = r11 * r11;
    r27 = fma(r16, r26, r25);
    r28 = r24 + r27;
    r6 = fma(r20, r23, r6);
    r6 = fma(r8, r28, r6);
    r28 = r0 * r6;
    r29 = r6 * r6;
    r30 = 1.00000000000000008e-15;
  };
  LoadShared<1, double, double>(pose, 6 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r32 = r11 * r14;
    r32 = r32 * r13;
    r33 = r10 * r15;
    r33 = r33 * r13;
    r34 = r32 + r33;
    r31 = fma(r9, r34, r31);
    r35 = r11 * r15;
    r35 = r35 * r16;
    r21 = r21 + r35;
    r36 = r10 * r10;
    r37 = r16 * r36;
    r27 = r37 + r27;
    r31 = fma(r8, r21, r31);
    r31 = fma(r20, r27, r31);
    r27 = copysign(1.0, r31);
    r27 = fma(r30, r27, r31);
    r30 = r27 * r27;
    r31 = 1.0 / r30;
    r38 = r14 * r15;
    r38 = r38 * r13;
    r12 = r12 + r38;
    r7 = fma(r8, r12, r7);
    r39 = r10 * r15;
    r39 = r39 * r16;
    r32 = r32 + r39;
    r24 = r25 + r24;
    r24 = r24 + r37;
    r7 = fma(r20, r32, r7);
    r7 = fma(r9, r24, r7);
    r24 = r7 * r7;
    r24 = fma(r31, r24, r31 * r29);
    r25 = fma(r5, r24, r25);
    r29 = 1.0 / r27;
    r37 = r25 * r29;
    r2 = fma(r28, r37, r2);
    r3 = fma(r3, r4, r1);
    r1 = r0 * r25;
    r1 = r1 * r29;
    r3 = fma(r7, r1, r3);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc,
                                             global_thread_idx, r2, r3);
    r40 = fma(r3, r3, r2 * r2);
  };
  SumStore<double>(out_rTr_local, (double *)inout_shared, 0,
                   global_thread_idx < problem_size, r40);
  if (global_thread_idx < problem_size) {
    r40 = r16 * r6;
    r41 = r11 * r17;
    r39 = r39 + r41;
    r42 = r14 * r14;
    r43 = r4 * r26;
    r44 = r42 + r43;
    r15 = r15 * r15;
    r45 = r4 * r36;
    r46 = r15 + r45;
    r47 = r44 + r46;
    r47 = fma(r9, r47, r20 * r39);
    r30 = r27 * r30;
    r27 = 1.0 / r30;
    r40 = r40 * r6;
    r40 = r40 * r47;
    r39 = r16 * r7;
    r39 = r39 * r7;
    r39 = r39 * r47;
    r39 = fma(r27, r39, r27 * r40);
    r40 = r13 * r7;
    r48 = r4 * r15;
    r49 = r36 + r48;
    r44 = r44 + r49;
    r44 = fma(r20, r44, r9 * r32);
    r40 = r40 * r44;
    r39 = fma(r31, r40, r39);
    r32 = r13 * r6;
    r11 = r10 * r11;
    r11 = r11 * r16;
    r38 = r38 + r11;
    r23 = fma(r9, r23, r20 * r38);
    r32 = r32 * r23;
    r39 = fma(r31, r32, r39);
    r32 = r5 * r39;
    r32 = r32 * r29;
    r23 = fma(r23, r1, r28 * r32);
    r32 = r47 * r28;
    r40 = r25 * r4;
    r40 = r40 * r31;
    r23 = fma(r40, r32, r23);
    r32 = r0 * r31;
    r32 = r32 * r7;
    r32 = r32 * r25;
    r32 = r32 * r4;
    r44 = fma(r47, r32, r44 * r1);
    r38 = r0 * r39;
    r50 = r5 * r7;
    r38 = r38 * r29;
    r44 = fma(r50, r38, r44);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r23, r44);
    r17 = r10 * r17;
    r35 = r35 + r17;
    r15 = r36 + r15;
    r14 = r14 * r14;
    r14 = r14 * r4;
    r15 = r15 + r43;
    r15 = r15 + r14;
    r15 = fma(r20, r15, r8 * r35);
    r14 = r26 + r14;
    r49 = r49 + r14;
    r49 = fma(r8, r49, r20 * r21);
    r21 = r49 * r28;
    r21 = fma(r40, r21, r15 * r1);
    r35 = r16 * r7;
    r35 = r35 * r7;
    r35 = r35 * r49;
    r43 = r13 * r7;
    r41 = r33 + r41;
    r41 = fma(r8, r41, r20 * r12);
    r43 = r43 * r41;
    r43 = fma(r31, r43, r27 * r35);
    r35 = r16 * r6;
    r35 = r35 * r6;
    r35 = r35 * r49;
    r43 = fma(r27, r35, r43);
    r12 = r13 * r6;
    r12 = r12 * r15;
    r43 = fma(r31, r12, r43);
    r12 = r5 * r43;
    r12 = r12 * r29;
    r21 = fma(r28, r12, r21);
    r12 = r0 * r43;
    r12 = r12 * r29;
    r12 = fma(r50, r12, r41 * r1);
    r12 = fma(r49, r32, r12);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r21, r12);
    r42 = r26 + r42;
    r42 = r42 + r45;
    r42 = r42 + r48;
    r42 = fma(r9, r42, r8 * r19);
    r17 = r22 + r17;
    r17 = fma(r9, r17, r8 * r34);
    r34 = r17 * r28;
    r34 = fma(r40, r34, r42 * r1);
    r22 = r13 * r7;
    r18 = r11 + r18;
    r14 = r46 + r14;
    r14 = fma(r8, r14, r9 * r18);
    r22 = r22 * r14;
    r8 = r16 * r6;
    r8 = r8 * r6;
    r8 = r8 * r17;
    r8 = fma(r27, r8, r31 * r22);
    r22 = r16 * r7;
    r22 = r22 * r7;
    r22 = r22 * r17;
    r8 = fma(r27, r22, r8);
    r18 = r13 * r6;
    r18 = r18 * r42;
    r8 = fma(r31, r18, r8);
    r18 = r5 * r8;
    r18 = r18 * r29;
    r34 = fma(r28, r18, r34);
    r18 = r0 * r8;
    r18 = r18 * r29;
    r18 = fma(r17, r32, r50 * r18);
    r18 = fma(r14, r1, r18);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r34, r18);
    r14 = r0 * r5;
    r14 = r14 * r13;
    r14 = r14 * r6;
    r14 = r14 * r7;
    r14 = r14 * r27;
    r22 = r27 * r28;
    r42 = r13 * r22;
    r9 = r5 * r6;
    r42 = fma(r9, r42, r1);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r42, r14);
    r46 = r0 * r13;
    r46 = r46 * r7;
    r46 = r46 * r27;
    r46 = fma(r50, r46, r1);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r14, r46);
    r1 = r16 * r6;
    r1 = r1 * r6;
    r11 = r16 * r7;
    r11 = r11 * r7;
    r11 = fma(r27, r11, r27 * r1);
    r1 = r5 * r11;
    r1 = r1 * r29;
    r1 = fma(r28, r1, r28 * r40);
    r40 = r0 * r11;
    r40 = r40 * r29;
    r40 = fma(r50, r40, r32);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r1, r40);
    r32 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r23, r27, r44 * r32);
    r32 = r4 * r3;
    r19 = r4 * r2;
    r19 = fma(r21, r19, r12 * r32);
    WriteSum2<double, double>((double *)inout_shared, r27, r19);
  };
  FlushSumShared<2, double>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r4 * r2;
    r27 = r4 * r3;
    r27 = fma(r18, r27, r34 * r19);
    r19 = r4 * r2;
    r32 = r16 * r3;
    r48 = r50 * r22;
    r32 = fma(r48, r32, r42 * r19);
    WriteSum2<double, double>((double *)inout_shared, r27, r32);
  };
  FlushSumShared<2, double>(out_pose_njtr, 2 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r1, r27, r40 * r32);
    r32 = r4 * r3;
    r19 = r16 * r2;
    r19 = fma(r48, r19, r46 * r32);
    WriteSum2<double, double>((double *)inout_shared, r19, r27);
  };
  FlushSumShared<2, double>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r44, r44, r23 * r23);
    r19 = fma(r12, r12, r21 * r21);
    WriteSum2<double, double>((double *)inout_shared, r27, r19);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fma(r34, r34, r18 * r18);
    r27 = 4.00000000000000000e+00;
    r30 = r30 * r30;
    r30 = 1.0 / r30;
    r30 = r27 * r30;
    r27 = r0 * r7;
    r30 = r30 * r28;
    r30 = r30 * r50;
    r30 = r30 * r9;
    r30 = r30 * r27;
    r9 = fma(r42, r42, r30);
    WriteSum2<double, double>((double *)inout_shared, r19, r9);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r46, r46, r30);
    r9 = fma(r40, r40, r1 * r1);
    WriteSum2<double, double>((double *)inout_shared, r30, r9);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fma(r23, r21, r44 * r12);
    r30 = fma(r44, r18, r23 * r34);
    WriteSum2<double, double>((double *)inout_shared, r9, r30);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r44, r14, r23 * r42);
    r9 = fma(r23, r14, r44 * r46);
    WriteSum2<double, double>((double *)inout_shared, r30, r9);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r23, r1, r44 * r40);
    r44 = fma(r12, r18, r21 * r34);
    WriteSum2<double, double>((double *)inout_shared, r23, r44);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fma(r12, r14, r21 * r42);
    r23 = fma(r21, r14, r12 * r46);
    WriteSum2<double, double>((double *)inout_shared, r44, r23);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = fma(r21, r1, r12 * r40);
    r12 = fma(r18, r14, r34 * r42);
    WriteSum2<double, double>((double *)inout_shared, r21, r12);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r18, r40, r34 * r1);
    r34 = fma(r34, r14, r18 * r46);
    WriteSum2<double, double>((double *)inout_shared, r34, r12);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r46, r14, r42 * r14);
    r42 = fma(r40, r14, r42 * r1);
    WriteSum2<double, double>((double *)inout_shared, r12, r42);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = fma(r1, r14, r46 * r40);
    WriteSum1<double, double>((double *)inout_shared, r14);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = r6 * r37;
    r1 = r7 * r37;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_distortion_jac,
        0 * out_focal_and_distortion_jac_num_alloc, global_thread_idx, r14, r1);
    r1 = r24 * r29;
    r1 = r1 * r28;
    r14 = r0 * r7;
    r14 = r14 * r24;
    r14 = r14 * r29;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_distortion_jac,
        2 * out_focal_and_distortion_jac_num_alloc, global_thread_idx, r1, r14);
    r14 = r6 * r4;
    r14 = r14 * r2;
    r1 = r7 * r4;
    r1 = r1 * r3;
    r1 = fma(r37, r1, r37 * r14);
    r14 = r24 * r4;
    r14 = r14 * r2;
    r14 = r14 * r29;
    r37 = r0 * r7;
    r37 = r37 * r24;
    r37 = r37 * r4;
    r37 = r37 * r3;
    r37 = fma(r29, r37, r28 * r14);
    WriteSum2<double, double>((double *)inout_shared, r1, r37);
  };
  FlushSumShared<2, double>(out_focal_and_distortion_njtr,
                            0 * out_focal_and_distortion_njtr_num_alloc,
                            focal_and_distortion_indices_loc,
                            (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r6 * r6;
    r37 = r37 * r25;
    r37 = r37 * r25;
    r1 = r7 * r7;
    r1 = r1 * r25;
    r1 = r1 * r25;
    r1 = fma(r31, r1, r31 * r37);
    r37 = r6 * r28;
    r14 = r0 * r31;
    r29 = r24 * r24;
    r14 = r14 * r29;
    r29 = r7 * r14;
    r29 = fma(r27, r29, r14 * r37);
    WriteSum2<double, double>((double *)inout_shared, r1, r29);
  };
  FlushSumShared<2, double>(out_focal_and_distortion_precond_diag,
                            0 * out_focal_and_distortion_precond_diag_num_alloc,
                            focal_and_distortion_indices_loc,
                            (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = r0 * r7;
    r29 = r29 * r7;
    r29 = r29 * r24;
    r29 = r29 * r25;
    r1 = r6 * r24;
    r1 = r1 * r25;
    r1 = r1 * r31;
    r1 = fma(r28, r1, r31 * r29);
    WriteSum1<double, double>((double *)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_focal_and_distortion_precond_tril,
                            0 * out_focal_and_distortion_precond_tril_num_alloc,
                            focal_and_distortion_indices_loc,
                            (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedPrincipalPointFixedPointResJacFirst(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *focal_and_distortion, unsigned int focal_and_distortion_num_alloc,
    SharedIndex *focal_and_distortion_indices, double *pixel,
    unsigned int pixel_num_alloc, double *principal_point,
    unsigned int principal_point_num_alloc, double *point,
    unsigned int point_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr, double *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, double *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double *out_focal_and_distortion_jac,
    unsigned int out_focal_and_distortion_jac_num_alloc,
    double *const out_focal_and_distortion_njtr,
    unsigned int out_focal_and_distortion_njtr_num_alloc,
    double *const out_focal_and_distortion_precond_diag,
    unsigned int out_focal_and_distortion_precond_diag_num_alloc,
    double *const out_focal_and_distortion_precond_tril,
    unsigned int out_focal_and_distortion_precond_tril_num_alloc,
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedPrincipalPointFixedPointResJacFirstKernel<<<n_blocks,
                                                                    1024>>>(
      pose, pose_num_alloc, pose_indices, focal_and_distortion,
      focal_and_distortion_num_alloc, focal_and_distortion_indices, pixel,
      pixel_num_alloc, principal_point, principal_point_num_alloc, point,
      point_num_alloc, out_res, out_res_num_alloc, out_rTr, out_pose_jac,
      out_pose_jac_num_alloc, out_pose_njtr, out_pose_njtr_num_alloc,
      out_pose_precond_diag, out_pose_precond_diag_num_alloc,
      out_pose_precond_tril, out_pose_precond_tril_num_alloc,
      out_focal_and_distortion_jac, out_focal_and_distortion_jac_num_alloc,
      out_focal_and_distortion_njtr, out_focal_and_distortion_njtr_num_alloc,
      out_focal_and_distortion_precond_diag,
      out_focal_and_distortion_precond_diag_num_alloc,
      out_focal_and_distortion_precond_tril,
      out_focal_and_distortion_precond_tril_num_alloc, problem_size);
}

} // namespace caspar