#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_split_fixed_principal_point_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void
__launch_bounds__(1024, 1) SimpleRadialSplitFixedPrincipalPointResJacKernel(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *focal_and_distortion, unsigned int focal_and_distortion_num_alloc,
    SharedIndex *focal_and_distortion_indices, double *point,
    unsigned int point_num_alloc, SharedIndex *point_indices, double *pixel,
    unsigned int pixel_num_alloc, double *principal_point,
    unsigned int principal_point_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *out_pose_jac,
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
    double *out_point_jac, unsigned int out_point_jac_num_alloc,
    double *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
    double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
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
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55;

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
  LoadShared<2, double, double>(point, 0 * point_num_alloc, point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        point_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  LoadShared<2, double, double>(pose, 2 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = r10 * r11;
    r13 = -2.00000000000000000e+00;
    r12 = r12 * r13;
  };
  LoadShared<2, double, double>(pose, 0 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r14, r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = r14 * r15;
    r17 = 2.00000000000000000e+00;
    r16 = r16 * r17;
    r18 = r12 + r16;
    r6 = fma(r9, r18, r6);
  };
  LoadShared<1, double, double>(point, 2 * point_num_alloc, point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        point_indices_loc[threadIdx.x].target, r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r20 = r14 * r10;
    r20 = r20 * r17;
    r21 = r15 * r11;
    r21 = r21 * r17;
    r22 = r20 + r21;
    r23 = r10 * r10;
    r24 = r13 * r23;
    r25 = 1.00000000000000000e+00;
    r26 = r15 * r15;
    r27 = fma(r13, r26, r25);
    r28 = r24 + r27;
    r6 = fma(r19, r22, r6);
    r6 = fma(r8, r28, r6);
    r29 = r0 * r6;
    r30 = r6 * r6;
    r31 = 1.00000000000000008e-15;
  };
  LoadShared<1, double, double>(pose, 6 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = r15 * r10;
    r33 = r33 * r17;
    r34 = r14 * r11;
    r34 = r34 * r17;
    r35 = r33 + r34;
    r32 = fma(r9, r35, r32);
    r36 = r15 * r11;
    r36 = r36 * r13;
    r20 = r20 + r36;
    r37 = r14 * r14;
    r38 = r13 * r37;
    r27 = r38 + r27;
    r32 = fma(r8, r20, r32);
    r32 = fma(r19, r27, r32);
    r39 = copysign(1.0, r32);
    r39 = fma(r31, r39, r32);
    r31 = r39 * r39;
    r32 = 1.0 / r31;
    r40 = r10 * r11;
    r40 = r40 * r17;
    r16 = r16 + r40;
    r7 = fma(r8, r16, r7);
    r41 = r14 * r11;
    r41 = r41 * r13;
    r33 = r33 + r41;
    r24 = r25 + r24;
    r24 = r24 + r38;
    r7 = fma(r19, r33, r7);
    r7 = fma(r9, r24, r7);
    r38 = r7 * r7;
    r38 = fma(r32, r38, r32 * r30);
    r25 = fma(r5, r38, r25);
    r30 = 1.0 / r39;
    r42 = r25 * r30;
    r2 = fma(r29, r42, r2);
    r3 = fma(r3, r4, r1);
    r1 = r0 * r25;
    r1 = r1 * r30;
    r3 = fma(r7, r1, r3);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc,
                                             global_thread_idx, r2, r3);
    r43 = r15 * r10;
    r43 = r43 * r13;
    r41 = r41 + r43;
    r44 = r4 * r26;
    r45 = r23 + r44;
    r11 = r11 * r11;
    r46 = r4 * r37;
    r47 = r11 + r46;
    r48 = r45 + r47;
    r48 = fma(r9, r48, r19 * r41);
    r41 = r6 * r6;
    r41 = r13 * r41;
    r31 = r39 * r31;
    r39 = 1.0 / r31;
    r41 = r41 * r39;
    r49 = r7 * r7;
    r49 = r13 * r49;
    r49 = r49 * r39;
    r50 = fma(r48, r49, r48 * r41);
    r51 = r17 * r7;
    r52 = r4 * r11;
    r53 = r37 + r52;
    r45 = r45 + r53;
    r45 = fma(r19, r45, r9 * r33);
    r51 = r51 * r45;
    r50 = fma(r32, r51, r50);
    r54 = r17 * r6;
    r15 = r14 * r15;
    r15 = r15 * r13;
    r40 = r40 + r15;
    r40 = fma(r9, r22, r19 * r40);
    r54 = r54 * r40;
    r50 = fma(r32, r54, r50);
    r54 = r5 * r50;
    r54 = r54 * r30;
    r40 = fma(r40, r1, r29 * r54);
    r54 = r48 * r29;
    r51 = r25 * r4;
    r51 = r51 * r32;
    r40 = fma(r51, r54, r40);
    r54 = r0 * r7;
    r54 = r54 * r48;
    r54 = fma(r51, r54, r45 * r1);
    r45 = r0 * r50;
    r55 = r5 * r7;
    r45 = r45 * r30;
    r54 = fma(r55, r45, r54);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r40, r54);
    r10 = r14 * r10;
    r10 = r10 * r13;
    r36 = r36 + r10;
    r11 = r37 + r11;
    r37 = r4 * r23;
    r11 = r11 + r44;
    r11 = r11 + r37;
    r11 = fma(r19, r11, r8 * r36);
    r37 = r26 + r37;
    r53 = r53 + r37;
    r53 = fma(r8, r53, r19 * r20);
    r36 = r53 * r51;
    r44 = fma(r29, r36, r11 * r1);
    r14 = r17 * r7;
    r43 = r34 + r43;
    r43 = fma(r8, r43, r19 * r16);
    r14 = r14 * r43;
    r14 = fma(r32, r14, r53 * r49);
    r19 = r17 * r6;
    r19 = r19 * r11;
    r14 = fma(r32, r19, r14);
    r14 = fma(r53, r41, r14);
    r19 = r5 * r14;
    r19 = r19 * r30;
    r44 = fma(r29, r19, r44);
    r19 = r0 * r14;
    r19 = r19 * r30;
    r19 = fma(r55, r19, r43 * r1);
    r43 = r0 * r7;
    r19 = fma(r36, r43, r19);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r44, r19);
    r23 = r26 + r23;
    r23 = r23 + r46;
    r23 = r23 + r52;
    r23 = fma(r9, r23, r8 * r18);
    r10 = r21 + r10;
    r10 = fma(r9, r10, r8 * r35);
    r21 = r10 * r29;
    r21 = fma(r51, r21, r23 * r1);
    r52 = r17 * r7;
    r15 = r12 + r15;
    r37 = r47 + r37;
    r37 = fma(r8, r37, r9 * r15);
    r52 = r52 * r37;
    r52 = fma(r10, r41, r32 * r52);
    r8 = r17 * r6;
    r8 = r8 * r23;
    r52 = fma(r32, r8, r52);
    r52 = fma(r10, r49, r52);
    r8 = r5 * r52;
    r8 = r8 * r30;
    r21 = fma(r29, r8, r21);
    r8 = r0 * r52;
    r8 = r8 * r30;
    r23 = r0 * r7;
    r23 = r23 * r10;
    r23 = fma(r51, r23, r55 * r8);
    r23 = fma(r37, r1, r23);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r21, r23);
    r37 = r0 * r5;
    r37 = r37 * r17;
    r37 = r37 * r6;
    r37 = r37 * r7;
    r37 = r37 * r39;
    r8 = r5 * r17;
    r8 = r8 * r6;
    r8 = r8 * r39;
    r8 = fma(r29, r8, r1);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r8, r37);
    r15 = r0 * r17;
    r15 = r15 * r7;
    r15 = r15 * r39;
    r15 = fma(r55, r15, r1);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r37, r15);
    r9 = r41 + r49;
    r47 = r5 * r9;
    r47 = r47 * r30;
    r47 = fma(r29, r47, r29 * r51);
    r12 = r0 * r7;
    r46 = r0 * r9;
    r46 = r46 * r30;
    r46 = fma(r55, r46, r51 * r12);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r47, r46);
    r12 = r4 * r3;
    r26 = r4 * r2;
    r26 = fma(r40, r26, r54 * r12);
    r12 = r4 * r3;
    r36 = r4 * r2;
    r36 = fma(r44, r36, r19 * r12);
    WriteSum2<double, double>((double *)inout_shared, r26, r36);
  };
  FlushSumShared<2, double>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r4 * r2;
    r26 = r4 * r3;
    r26 = fma(r23, r26, r21 * r36);
    r36 = r4 * r2;
    r39 = r13 * r39;
    r13 = r3 * r39;
    r12 = r29 * r55;
    r13 = fma(r12, r13, r8 * r36);
    WriteSum2<double, double>((double *)inout_shared, r26, r13);
  };
  FlushSumShared<2, double>(out_pose_njtr, 2 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r4 * r3;
    r26 = r4 * r2;
    r26 = fma(r47, r26, r46 * r13);
    r13 = r4 * r3;
    r36 = r2 * r39;
    r36 = fma(r12, r36, r15 * r13);
    WriteSum2<double, double>((double *)inout_shared, r36, r26);
  };
  FlushSumShared<2, double>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fma(r54, r54, r40 * r40);
    r36 = fma(r19, r19, r44 * r44);
    WriteSum2<double, double>((double *)inout_shared, r26, r36);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fma(r21, r21, r23 * r23);
    r26 = r0 * r5;
    r13 = 4.00000000000000000e+00;
    r31 = r31 * r31;
    r31 = 1.0 / r31;
    r26 = r26 * r6;
    r26 = r26 * r7;
    r26 = r26 * r13;
    r26 = r26 * r31;
    r26 = r26 * r12;
    r12 = fma(r8, r8, r26);
    WriteSum2<double, double>((double *)inout_shared, r36, r12);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fma(r15, r15, r26);
    r12 = fma(r46, r46, r47 * r47);
    WriteSum2<double, double>((double *)inout_shared, r26, r12);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r40, r44, r54 * r19);
    r26 = fma(r54, r23, r40 * r21);
    WriteSum2<double, double>((double *)inout_shared, r12, r26);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fma(r54, r37, r40 * r8);
    r12 = fma(r40, r37, r54 * r15);
    WriteSum2<double, double>((double *)inout_shared, r26, r12);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fma(r40, r47, r54 * r46);
    r54 = fma(r19, r23, r44 * r21);
    WriteSum2<double, double>((double *)inout_shared, r40, r54);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fma(r19, r37, r44 * r8);
    r40 = fma(r44, r37, r19 * r15);
    WriteSum2<double, double>((double *)inout_shared, r54, r40);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fma(r44, r47, r19 * r46);
    r19 = fma(r23, r37, r21 * r8);
    WriteSum2<double, double>((double *)inout_shared, r44, r19);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fma(r23, r46, r21 * r47);
    r21 = fma(r21, r37, r23 * r15);
    WriteSum2<double, double>((double *)inout_shared, r21, r19);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fma(r15, r37, r8 * r37);
    r8 = fma(r46, r37, r8 * r47);
    WriteSum2<double, double>((double *)inout_shared, r19, r8);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r47, r37, r15 * r46);
    WriteSum1<double, double>((double *)inout_shared, r37);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r6 * r42;
    r47 = r7 * r42;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_distortion_jac,
        0 * out_focal_and_distortion_jac_num_alloc, global_thread_idx, r37,
        r47);
    r47 = r38 * r30;
    r47 = r47 * r29;
    r37 = r0 * r7;
    r37 = r37 * r38;
    r37 = r37 * r30;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_distortion_jac,
        2 * out_focal_and_distortion_jac_num_alloc, global_thread_idx, r47,
        r37);
    r37 = r6 * r4;
    r37 = r37 * r2;
    r47 = r7 * r4;
    r47 = r47 * r3;
    r47 = fma(r42, r47, r42 * r37);
    r37 = r38 * r4;
    r37 = r37 * r2;
    r37 = r37 * r30;
    r42 = r0 * r7;
    r42 = r42 * r38;
    r42 = r42 * r4;
    r42 = r42 * r3;
    r42 = fma(r30, r42, r29 * r37);
    WriteSum2<double, double>((double *)inout_shared, r47, r42);
  };
  FlushSumShared<2, double>(out_focal_and_distortion_njtr,
                            0 * out_focal_and_distortion_njtr_num_alloc,
                            focal_and_distortion_indices_loc,
                            (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r6 * r6;
    r42 = r42 * r25;
    r42 = r42 * r25;
    r47 = r7 * r7;
    r47 = r47 * r25;
    r47 = r47 * r25;
    r47 = fma(r32, r47, r32 * r42);
    r42 = r6 * r29;
    r37 = r0 * r32;
    r46 = r38 * r38;
    r37 = r37 * r46;
    r46 = r7 * r37;
    r46 = fma(r43, r46, r37 * r42);
    WriteSum2<double, double>((double *)inout_shared, r47, r46);
  };
  FlushSumShared<2, double>(out_focal_and_distortion_precond_diag,
                            0 * out_focal_and_distortion_precond_diag_num_alloc,
                            focal_and_distortion_indices_loc,
                            (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r0 * r7;
    r46 = r46 * r7;
    r46 = r46 * r38;
    r46 = r46 * r25;
    r47 = r6 * r38;
    r47 = r47 * r25;
    r47 = r47 * r32;
    r47 = fma(r29, r47, r32 * r46);
    WriteSum1<double, double>((double *)inout_shared, r47);
  };
  FlushSumShared<1, double>(out_focal_and_distortion_precond_tril,
                            0 * out_focal_and_distortion_precond_tril_num_alloc,
                            focal_and_distortion_indices_loc,
                            (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r20 * r29;
    r47 = fma(r28, r1, r51 * r47);
    r46 = r17 * r28;
    r46 = r46 * r6;
    r46 = fma(r32, r46, r20 * r41);
    r25 = r17 * r16;
    r25 = r25 * r7;
    r46 = fma(r32, r25, r46);
    r46 = fma(r20, r49, r46);
    r25 = r5 * r46;
    r25 = r25 * r30;
    r47 = fma(r29, r25, r47);
    r25 = r0 * r20;
    r25 = r25 * r7;
    r25 = fma(r51, r25, r16 * r1);
    r42 = r0 * r46;
    r42 = r42 * r30;
    r25 = fma(r55, r42, r25);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx, r47, r25);
    r42 = r17 * r18;
    r42 = r42 * r6;
    r42 = fma(r35, r41, r32 * r42);
    r43 = r17 * r24;
    r43 = r43 * r7;
    r42 = fma(r32, r43, r42);
    r42 = fma(r35, r49, r42);
    r43 = r5 * r42;
    r43 = r43 * r30;
    r43 = fma(r18, r1, r29 * r43);
    r15 = r35 * r29;
    r43 = fma(r51, r15, r43);
    r15 = r0 * r42;
    r15 = r15 * r30;
    r15 = fma(r24, r1, r55 * r15);
    r8 = r0 * r35;
    r8 = r8 * r7;
    r15 = fma(r51, r8, r15);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx, r43, r15);
    r8 = r17 * r22;
    r8 = r8 * r6;
    r41 = fma(r27, r41, r32 * r8);
    r8 = r17 * r33;
    r8 = r8 * r7;
    r41 = fma(r32, r8, r41);
    r41 = fma(r27, r49, r41);
    r49 = r5 * r41;
    r49 = r49 * r30;
    r49 = fma(r29, r49, r22 * r1);
    r8 = r27 * r29;
    r49 = fma(r51, r8, r49);
    r8 = r0 * r27;
    r8 = r8 * r7;
    r1 = fma(r33, r1, r51 * r8);
    r8 = r0 * r41;
    r8 = r8 * r30;
    r1 = fma(r55, r8, r1);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r49, r1);
    r8 = r4 * r2;
    r55 = r4 * r3;
    r55 = fma(r25, r55, r47 * r8);
    r8 = r4 * r3;
    r30 = r4 * r2;
    r30 = fma(r43, r30, r15 * r8);
    WriteSum2<double, double>((double *)inout_shared, r55, r30);
  };
  FlushSumShared<2, double>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = r4 * r3;
    r55 = r4 * r2;
    r55 = fma(r49, r55, r1 * r30);
    WriteSum1<double, double>((double *)inout_shared, r55);
  };
  FlushSumShared<1, double>(out_point_njtr, 2 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fma(r25, r25, r47 * r47);
    r30 = fma(r15, r15, r43 * r43);
    WriteSum2<double, double>((double *)inout_shared, r55, r30);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r49, r49, r1 * r1);
    WriteSum1<double, double>((double *)inout_shared, r30);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r47, r43, r25 * r15);
    r47 = fma(r47, r49, r25 * r1);
    WriteSum2<double, double>((double *)inout_shared, r30, r47);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r15, r1, r43 * r49);
    WriteSum1<double, double>((double *)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
}

void SimpleRadialSplitFixedPrincipalPointResJac(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *focal_and_distortion, unsigned int focal_and_distortion_num_alloc,
    SharedIndex *focal_and_distortion_indices, double *point,
    unsigned int point_num_alloc, SharedIndex *point_indices, double *pixel,
    unsigned int pixel_num_alloc, double *principal_point,
    unsigned int principal_point_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *out_pose_jac,
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
    double *out_point_jac, unsigned int out_point_jac_num_alloc,
    double *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
    double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedPrincipalPointResJacKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, focal_and_distortion,
      focal_and_distortion_num_alloc, focal_and_distortion_indices, point,
      point_num_alloc, point_indices, pixel, pixel_num_alloc, principal_point,
      principal_point_num_alloc, out_res, out_res_num_alloc, out_pose_jac,
      out_pose_jac_num_alloc, out_pose_njtr, out_pose_njtr_num_alloc,
      out_pose_precond_diag, out_pose_precond_diag_num_alloc,
      out_pose_precond_tril, out_pose_precond_tril_num_alloc,
      out_focal_and_distortion_jac, out_focal_and_distortion_jac_num_alloc,
      out_focal_and_distortion_njtr, out_focal_and_distortion_njtr_num_alloc,
      out_focal_and_distortion_precond_diag,
      out_focal_and_distortion_precond_diag_num_alloc,
      out_focal_and_distortion_precond_tril,
      out_focal_and_distortion_precond_tril_num_alloc, out_point_jac,
      out_point_jac_num_alloc, out_point_njtr, out_point_njtr_num_alloc,
      out_point_precond_diag, out_point_precond_diag_num_alloc,
      out_point_precond_tril, out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar