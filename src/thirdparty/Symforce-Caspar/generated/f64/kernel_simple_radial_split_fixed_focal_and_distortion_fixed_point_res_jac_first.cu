#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_split_fixed_focal_and_distortion_fixed_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndDistortionFixedPointResJacFirstKernel(
        double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        double *principal_point, unsigned int principal_point_num_alloc,
        SharedIndex *principal_point_indices, double *pixel,
        unsigned int pixel_num_alloc, double *focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc, double *point,
        unsigned int point_num_alloc, double *out_res,
        unsigned int out_res_num_alloc, double *const out_rTr,
        double *out_pose_jac, unsigned int out_pose_jac_num_alloc,
        double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
        double *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        double *out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        double *const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        double *const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        double *const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48;
  LoadShared<2, double, double>(principal_point, 0 * principal_point_num_alloc,
                                principal_point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        principal_point_indices_loc[threadIdx.x].target, r0,
                        r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(pixel, 0 * pixel_num_alloc,
                                            global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    ReadIdx2<1024, double, double, double2>(focal_and_distortion,
                                            0 * focal_and_distortion_num_alloc,
                                            global_thread_idx, r0, r5);
  };
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
    r24 = fma(r5, r24, r25);
    r29 = 1.0 / r27;
    r37 = r24 * r29;
    r2 = fma(r28, r37, r2);
    r3 = fma(r3, r4, r1);
    r1 = r0 * r24;
    r1 = r1 * r29;
    r3 = fma(r7, r1, r3);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc,
                                             global_thread_idx, r2, r3);
    r37 = fma(r3, r3, r2 * r2);
  };
  SumStore<double>(out_rTr_local, (double *)inout_shared, 0,
                   global_thread_idx < problem_size, r37);
  if (global_thread_idx < problem_size) {
    r37 = r16 * r6;
    r40 = r11 * r17;
    r39 = r39 + r40;
    r41 = r14 * r14;
    r42 = r4 * r26;
    r43 = r41 + r42;
    r15 = r15 * r15;
    r44 = r4 * r36;
    r45 = r15 + r44;
    r46 = r43 + r45;
    r46 = fma(r9, r46, r20 * r39);
    r30 = r27 * r30;
    r27 = 1.0 / r30;
    r37 = r37 * r6;
    r37 = r37 * r46;
    r39 = r16 * r7;
    r39 = r39 * r7;
    r39 = r39 * r46;
    r39 = fma(r27, r39, r27 * r37);
    r37 = r13 * r7;
    r47 = r4 * r15;
    r48 = r36 + r47;
    r43 = r43 + r48;
    r43 = fma(r20, r43, r9 * r32);
    r37 = r37 * r43;
    r39 = fma(r31, r37, r39);
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
    r32 = r46 * r28;
    r37 = r24 * r4;
    r37 = r37 * r31;
    r23 = fma(r37, r32, r23);
    r32 = r0 * r31;
    r32 = r32 * r7;
    r32 = r32 * r24;
    r32 = r32 * r4;
    r43 = fma(r46, r32, r43 * r1);
    r24 = r0 * r39;
    r38 = r5 * r7;
    r24 = r24 * r29;
    r43 = fma(r38, r24, r43);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r23, r43);
    r17 = r10 * r17;
    r35 = r35 + r17;
    r15 = r36 + r15;
    r14 = r14 * r14;
    r14 = r14 * r4;
    r15 = r15 + r42;
    r15 = r15 + r14;
    r15 = fma(r20, r15, r8 * r35);
    r14 = r26 + r14;
    r48 = r48 + r14;
    r48 = fma(r8, r48, r20 * r21);
    r21 = r48 * r28;
    r21 = fma(r37, r21, r15 * r1);
    r35 = r16 * r7;
    r35 = r35 * r7;
    r35 = r35 * r48;
    r42 = r13 * r7;
    r40 = r33 + r40;
    r40 = fma(r8, r40, r20 * r12);
    r42 = r42 * r40;
    r42 = fma(r31, r42, r27 * r35);
    r35 = r16 * r6;
    r35 = r35 * r6;
    r35 = r35 * r48;
    r42 = fma(r27, r35, r42);
    r12 = r13 * r6;
    r12 = r12 * r15;
    r42 = fma(r31, r12, r42);
    r12 = r5 * r42;
    r12 = r12 * r29;
    r21 = fma(r28, r12, r21);
    r12 = r0 * r42;
    r12 = r12 * r29;
    r12 = fma(r38, r12, r40 * r1);
    r12 = fma(r48, r32, r12);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r21, r12);
    r41 = r26 + r41;
    r41 = r41 + r44;
    r41 = r41 + r47;
    r41 = fma(r9, r41, r8 * r19);
    r17 = r22 + r17;
    r17 = fma(r9, r17, r8 * r34);
    r34 = r17 * r28;
    r34 = fma(r37, r34, r41 * r1);
    r22 = r13 * r7;
    r18 = r11 + r18;
    r14 = r45 + r14;
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
    r18 = r18 * r41;
    r8 = fma(r31, r18, r8);
    r18 = r5 * r8;
    r18 = r18 * r29;
    r34 = fma(r28, r18, r34);
    r18 = r0 * r8;
    r18 = r18 * r29;
    r18 = fma(r17, r32, r38 * r18);
    r18 = fma(r14, r1, r18);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r34, r18);
    r14 = r0 * r5;
    r14 = r14 * r13;
    r14 = r14 * r6;
    r14 = r14 * r7;
    r14 = r14 * r27;
    r22 = r27 * r28;
    r31 = r13 * r22;
    r41 = r5 * r6;
    r31 = fma(r41, r31, r1);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r31, r14);
    r9 = r0 * r13;
    r9 = r9 * r7;
    r9 = r9 * r27;
    r9 = fma(r38, r9, r1);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r14, r9);
    r1 = r16 * r6;
    r1 = r1 * r6;
    r45 = r16 * r7;
    r45 = r45 * r7;
    r45 = fma(r27, r45, r27 * r1);
    r1 = r5 * r45;
    r1 = r1 * r29;
    r1 = fma(r28, r1, r28 * r37);
    r37 = r0 * r45;
    r37 = r37 * r29;
    r37 = fma(r38, r37, r32);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r1, r37);
    r32 = r4 * r3;
    r29 = r4 * r2;
    r29 = fma(r23, r29, r43 * r32);
    r32 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r21, r27, r12 * r32);
    WriteSum2<double, double>((double *)inout_shared, r29, r27);
  };
  FlushSumShared<2, double>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = r4 * r2;
    r29 = r4 * r3;
    r29 = fma(r18, r29, r34 * r27);
    r27 = r4 * r2;
    r32 = r16 * r3;
    r11 = r38 * r22;
    r32 = fma(r11, r32, r31 * r27);
    WriteSum2<double, double>((double *)inout_shared, r29, r32);
  };
  FlushSumShared<2, double>(out_pose_njtr, 2 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r4 * r3;
    r29 = r4 * r2;
    r29 = fma(r1, r29, r37 * r32);
    r32 = r4 * r3;
    r27 = r16 * r2;
    r27 = fma(r11, r27, r9 * r32);
    WriteSum2<double, double>((double *)inout_shared, r27, r29);
  };
  FlushSumShared<2, double>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r43, r43, r23 * r23);
    r27 = fma(r12, r12, r21 * r21);
    WriteSum2<double, double>((double *)inout_shared, r29, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r34, r34, r18 * r18);
    r29 = r0 * r7;
    r32 = 4.00000000000000000e+00;
    r30 = r30 * r30;
    r30 = 1.0 / r30;
    r29 = r29 * r32;
    r29 = r29 * r30;
    r29 = r29 * r28;
    r29 = r29 * r38;
    r29 = r29 * r41;
    r41 = fma(r31, r31, r29);
    WriteSum2<double, double>((double *)inout_shared, r27, r41);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r9, r9, r29);
    r41 = fma(r37, r37, r1 * r1);
    WriteSum2<double, double>((double *)inout_shared, r29, r41);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r23, r21, r43 * r12);
    r29 = fma(r43, r18, r23 * r34);
    WriteSum2<double, double>((double *)inout_shared, r41, r29);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r43, r14, r23 * r31);
    r41 = fma(r23, r14, r43 * r9);
    WriteSum2<double, double>((double *)inout_shared, r29, r41);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r23, r1, r43 * r37);
    r43 = fma(r12, r18, r21 * r34);
    WriteSum2<double, double>((double *)inout_shared, r23, r43);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = fma(r12, r14, r21 * r31);
    r23 = fma(r21, r14, r12 * r9);
    WriteSum2<double, double>((double *)inout_shared, r43, r23);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = fma(r21, r1, r12 * r37);
    r12 = fma(r18, r14, r34 * r31);
    WriteSum2<double, double>((double *)inout_shared, r21, r12);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r18, r37, r34 * r1);
    r34 = fma(r34, r14, r18 * r9);
    WriteSum2<double, double>((double *)inout_shared, r34, r12);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r9, r14, r31 * r14);
    r31 = fma(r37, r14, r31 * r1);
    WriteSum2<double, double>((double *)inout_shared, r12, r31);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = fma(r1, r14, r9 * r37);
    WriteSum1<double, double>((double *)inout_shared, r14);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = r4 * r2;
    r1 = r4 * r3;
    WriteSum2<double, double>((double *)inout_shared, r14, r1);
  };
  FlushSumShared<2, double>(
      out_principal_point_njtr, 0 * out_principal_point_njtr_num_alloc,
      principal_point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double *)inout_shared, r25, r25);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedFocalAndDistortionFixedPointResJacFirst(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *principal_point, unsigned int principal_point_num_alloc,
    SharedIndex *principal_point_indices, double *pixel,
    unsigned int pixel_num_alloc, double *focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc, double *point,
    unsigned int point_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr, double *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, double *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double *out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    double *const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    double *const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    double *const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedFocalAndDistortionFixedPointResJacFirstKernel<<<
      n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, principal_point,
      principal_point_num_alloc, principal_point_indices, pixel,
      pixel_num_alloc, focal_and_distortion, focal_and_distortion_num_alloc,
      point, point_num_alloc, out_res, out_res_num_alloc, out_rTr, out_pose_jac,
      out_pose_jac_num_alloc, out_pose_njtr, out_pose_njtr_num_alloc,
      out_pose_precond_diag, out_pose_precond_diag_num_alloc,
      out_pose_precond_tril, out_pose_precond_tril_num_alloc,
      out_principal_point_jac, out_principal_point_jac_num_alloc,
      out_principal_point_njtr, out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar