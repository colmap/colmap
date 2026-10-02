#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_split_fixed_pose_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPoseFixedPrincipalPointResJacFirstKernel(
        double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        double *focal_and_extra, unsigned int focal_and_extra_num_alloc,
        SharedIndex *focal_and_extra_indices, double *point,
        unsigned int point_num_alloc, SharedIndex *point_indices, double *pixel,
        unsigned int pixel_num_alloc, double *pose, unsigned int pose_num_alloc,
        double *principal_point, unsigned int principal_point_num_alloc,
        double *out_res, unsigned int out_res_num_alloc, double *const out_rTr,
        double *out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        double *const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        double *const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        double *const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
        double *out_point_jac, unsigned int out_point_jac_num_alloc,
        double *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
        double *const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        double *const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0,
         r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0,
         r17 = 0, r18 = 0, r19 = 0, r20 = 0, r21 = 0, r22 = 0, r23 = 0, r24 = 0,
         r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0, r31 = 0, r32 = 0,
         r33 = 0, r34 = 0, r35 = 0, r36 = 0, r37 = 0, r38 = 0, r39 = 0, r40 = 0,
         r41 = 0, r42 = 0, r43 = 0, r44 = 0, r45 = 0, r46 = 0, r47 = 0, r48 = 0,
         r49 = 0, r50 = 0, r51 = 0, r52 = 0, r53 = 0, r54 = 0, r55 = 0, r56 = 0,
         r57 = 0, r58 = 0, r59 = 0, r60 = 0, r61 = 0, r62 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(pixel, 0 * pixel_num_alloc,
                                            global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r5 = fma(r2, r4, r0);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r6, r7);
  };
  LoadShared<2, double, double>(point, 0 * point_num_alloc, point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        point_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = -2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r11, r12);
    ReadIdx2<1024, double, double, double2>(pose, 2 * pose_num_alloc,
                                            global_thread_idx, r13, r14);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r15, r16);
    ReadIdx2<1024, double, double, double2>(pose, 0 * pose_num_alloc,
                                            global_thread_idx, r17, r18);
    r19 = fma(r15, r18, r12 * r13);
    r20 = r16 * r17;
    r19 = fma(r4, r20, r19);
    r19 = fma(r11, r14, r19);
    r20 = r19 * r19;
    r20 = r10 * r20;
    r21 = 1.00000000000000000e+00;
    r22 = r15 * r13;
    r22 = fma(r4, r22, r12 * r18);
    r22 = fma(r16, r14, r22);
    r22 = fma(r11, r17, r22);
    r23 = r22 * r22;
    r23 = fma(r10, r23, r21);
    r24 = r20 + r23;
    r24 = fma(r8, r24, r6);
    r25 = 2.00000000000000000e+00;
    r26 = fma(r15, r14, r12 * r17);
    r27 = r11 * r18;
    r26 = fma(r4, r27, r26);
    r26 = fma(r16, r13, r26);
    r27 = r25 * r26;
    r28 = r22 * r27;
    r29 = fma(r16, r18, r15 * r17);
    r29 = fma(r11, r13, r29);
    r29 = fma(r4, r29, r12 * r14);
    r30 = r10 * r29;
    r31 = fma(r19, r30, r28);
  };
  LoadShared<1, double, double>(point, 2 * point_num_alloc, point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        point_indices_loc[threadIdx.x].target, r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = r25 * r22;
    r34 = r19 * r27;
    r33 = fma(r29, r33, r34);
    ReadIdx1<1024, double, double, double>(pose, 6 * pose_num_alloc,
                                           global_thread_idx, r35);
    r36 = r15 * r11;
    r36 = r36 * r25;
    r37 = r16 * r12;
    r38 = fma(r25, r37, r36);
    ReadIdx2<1024, double, double, double2>(pose, 4 * pose_num_alloc,
                                            global_thread_idx, r39, r40);
    r41 = r11 * r12;
    r42 = r15 * r16;
    r42 = r42 * r25;
    r41 = fma(r10, r41, r42);
    r43 = r16 * r16;
    r43 = r43 * r10;
    r44 = r21 + r43;
    r45 = r11 * r11;
    r45 = r10 * r45;
    r44 = r44 + r45;
    r24 = fma(r9, r31, r24);
    r24 = fma(r32, r33, r24);
    r24 = fma(r35, r38, r24);
    r24 = fma(r40, r41, r24);
    r24 = fma(r39, r44, r24);
  };
  LoadShared<2, double, double>(focal_and_extra, 0 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target, r44,
                        r41);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r33);
    r34 = fma(r22, r30, r34);
    r34 = fma(r8, r34, r33);
    r37 = fma(r10, r37, r36);
    r43 = r21 + r43;
    r36 = r15 * r15;
    r36 = r10 * r36;
    r43 = r43 + r36;
    r31 = r16 * r11;
    r31 = r31 * r25;
    r46 = r15 * r12;
    r46 = fma(r25, r46, r31);
    r47 = r25 * r19;
    r47 = r47 * r22;
    r27 = fma(r29, r27, r47);
    r48 = r26 * r26;
    r48 = r48 * r10;
    r23 = r48 + r23;
    r34 = fma(r39, r37, r34);
    r34 = fma(r35, r43, r34);
    r34 = fma(r40, r46, r34);
    r34 = fma(r9, r27, r34);
    r34 = fma(r32, r23, r34);
    r23 = copysign(1.0, r34);
    r23 = fma(r38, r23, r34);
    r38 = r23 * r23;
    r38 = 1.0 / r38;
    r34 = r24 * r24;
    r27 = r25 * r19;
    r27 = fma(r29, r27, r28);
    r27 = fma(r8, r27, r7);
    r28 = r11 * r12;
    r28 = fma(r25, r28, r42);
    r45 = r21 + r45;
    r45 = r45 + r36;
    r36 = r15 * r12;
    r36 = fma(r10, r36, r31);
    r30 = fma(r26, r30, r47);
    r20 = r21 + r20;
    r20 = r20 + r48;
    r27 = fma(r39, r28, r27);
    r27 = fma(r40, r45, r27);
    r27 = fma(r35, r36, r27);
    r27 = fma(r32, r30, r27);
    r27 = fma(r9, r20, r27);
    r20 = r27 * r27;
    r20 = fma(r38, r20, r38 * r34);
    r20 = fma(r41, r20, r21);
    r20 = r44 * r20;
    r23 = 1.0 / r23;
    r20 = r20 * r23;
    r5 = fma(r24, r20, r5);
    r4 = fma(r3, r4, r1);
    r4 = fma(r27, r20, r4);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc,
                                             global_thread_idx, r5, r4);
    r4 = fma(r4, r4, r5 * r5);
  };
  SumStore<double>(out_rTr_local, (double *)inout_shared, 0,
                   global_thread_idx < problem_size, r4);
  if (global_thread_idx < problem_size) {
    r4 = -2.00000000000000000e+00;
    r5 = fma(r15, r18, r12 * r13);
    r20 = r16 * r17;
    r27 = -1.00000000000000000e+00;
    r5 = fma(r27, r20, r5);
    r5 = fma(r11, r14, r5);
    r20 = r5 * r5;
    r20 = r4 * r20;
    r24 = 1.00000000000000000e+00;
    r23 = r15 * r13;
    r23 = fma(r27, r23, r12 * r18);
    r23 = fma(r16, r14, r23);
    r23 = fma(r11, r17, r23);
    r21 = r23 * r23;
    r21 = fma(r4, r21, r24);
    r38 = r20 + r21;
    r38 = fma(r8, r38, r6);
    r34 = 2.00000000000000000e+00;
    r30 = fma(r15, r14, r12 * r17);
    r36 = r11 * r18;
    r30 = fma(r27, r36, r30);
    r30 = fma(r16, r13, r30);
    r36 = r34 * r30;
    r45 = r23 * r36;
    r28 = fma(r16, r18, r15 * r17);
    r28 = fma(r11, r13, r28);
    r28 = fma(r27, r28, r12 * r14);
    r48 = r4 * r28;
    r26 = fma(r5, r48, r45);
    r47 = r34 * r23;
    r31 = r5 * r36;
    r47 = fma(r28, r47, r31);
    r10 = r15 * r11;
    r10 = r10 * r34;
    r42 = r16 * r12;
    r29 = fma(r34, r42, r10);
    r46 = r11 * r12;
    r43 = r15 * r16;
    r43 = r43 * r34;
    r46 = fma(r4, r46, r43);
    r37 = r16 * r16;
    r37 = r37 * r4;
    r49 = r24 + r37;
    r50 = r11 * r11;
    r50 = r4 * r50;
    r49 = r49 + r50;
    r38 = fma(r9, r26, r38);
    r38 = fma(r32, r47, r38);
    r38 = fma(r35, r29, r38);
    r38 = fma(r40, r46, r38);
    r38 = fma(r39, r49, r38);
    r49 = 1.00000000000000008e-15;
    r31 = fma(r23, r48, r31);
    r31 = fma(r8, r31, r33);
    r42 = fma(r4, r42, r10);
    r37 = r24 + r37;
    r10 = r15 * r15;
    r10 = r4 * r10;
    r37 = r37 + r10;
    r46 = r16 * r11;
    r46 = r46 * r34;
    r29 = r15 * r12;
    r29 = fma(r34, r29, r46);
    r47 = r34 * r5;
    r47 = r47 * r23;
    r36 = fma(r28, r36, r47);
    r26 = r30 * r30;
    r26 = r26 * r4;
    r21 = r26 + r21;
    r31 = fma(r39, r42, r31);
    r31 = fma(r35, r37, r31);
    r31 = fma(r40, r29, r31);
    r31 = fma(r9, r36, r31);
    r31 = fma(r32, r21, r31);
    r21 = copysign(1.0, r31);
    r21 = fma(r49, r21, r31);
    r49 = r21 * r21;
    r49 = 1.0 / r49;
    r31 = r38 * r38;
    r36 = r34 * r5;
    r36 = fma(r28, r36, r45);
    r36 = fma(r8, r36, r7);
    r45 = r11 * r12;
    r45 = fma(r34, r45, r43);
    r50 = r24 + r50;
    r50 = r50 + r10;
    r10 = r15 * r12;
    r10 = fma(r4, r10, r46);
    r48 = fma(r30, r48, r47);
    r20 = r24 + r20;
    r20 = r20 + r26;
    r36 = fma(r39, r45, r36);
    r36 = fma(r40, r50, r36);
    r36 = fma(r35, r10, r36);
    r36 = fma(r32, r48, r36);
    r36 = fma(r9, r20, r36);
    r20 = r36 * r36;
    r48 = fma(r49, r20, r49 * r31);
    r24 = fma(r41, r48, r24);
    r21 = 1.0 / r21;
    r10 = r24 * r21;
    r50 = r38 * r10;
    r45 = r36 * r10;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac, 0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx, r50, r45);
    r45 = r38 * r21;
    r50 = r44 * r48;
    r45 = r45 * r50;
    r26 = r36 * r21;
    r26 = r26 * r50;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac, 2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx, r45, r26);
    r26 = r27 * r38;
    r45 = fma(r2, r27, r0);
    r30 = r44 * r38;
    r45 = fma(r10, r30, r45);
    r26 = r26 * r45;
    r45 = r27 * r36;
    r30 = fma(r3, r27, r1);
    r47 = r44 * r36;
    r30 = fma(r10, r47, r30);
    r45 = r45 * r30;
    r45 = fma(r10, r45, r10 * r26);
    r10 = r21 * r50;
    r47 = r27 * r36;
    r47 = r47 * r30;
    r47 = r47 * r21;
    r47 = fma(r50, r47, r26 * r10);
    WriteSum2<double, double>((double *)inout_shared, r45, r47);
  };
  FlushSumShared<2, double>(
      out_focal_and_extra_njtr, 0 * out_focal_and_extra_njtr_num_alloc,
      focal_and_extra_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r24 * r24;
    r47 = r47 * r49;
    r47 = fma(r20, r47, r31 * r47);
    r48 = r44 * r48;
    r49 = r49 * r50;
    r48 = r48 * r49;
    r48 = fma(r20, r48, r31 * r48);
    WriteSum2<double, double>((double *)inout_shared, r47, r48);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            0 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r24 * r31;
    r47 = r24 * r20;
    r47 = fma(r49, r47, r49 * r48);
    WriteSum1<double, double>((double *)inout_shared, r47);
  };
  FlushSumShared<1, double>(out_focal_and_extra_precond_tril,
                            0 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fma(r15, r14, r12 * r17);
    r48 = r11 * r18;
    r49 = -1.00000000000000000e+00;
    r47 = fma(r49, r48, r47);
    r47 = fma(r16, r13, r47);
    r48 = 2.00000000000000000e+00;
    r45 = fma(r15, r18, r12 * r13);
    r10 = r16 * r17;
    r45 = fma(r49, r10, r45);
    r45 = fma(r11, r14, r45);
    r10 = r48 * r45;
    r26 = r47 * r10;
    r30 = -2.00000000000000000e+00;
    r46 = r15 * r13;
    r46 = fma(r49, r46, r12 * r18);
    r46 = fma(r16, r14, r46);
    r46 = fma(r11, r17, r46);
    r4 = fma(r16, r18, r15 * r17);
    r4 = fma(r11, r13, r4);
    r4 = fma(r49, r4, r12 * r14);
    r14 = r46 * r4;
    r43 = fma(r30, r14, r26);
    r28 = 1.00000000000000000e+00;
    r29 = r45 * r45;
    r29 = r29 * r30;
    r37 = r30 * r46;
    r37 = fma(r46, r37, r28);
    r42 = r29 + r37;
    r6 = fma(r8, r42, r6);
    r51 = r48 * r47;
    r51 = r51 * r46;
    r52 = r45 * r30;
    r52 = fma(r4, r52, r51);
    r14 = fma(r48, r14, r26);
    r26 = r15 * r11;
    r26 = r26 * r48;
    r53 = r16 * r12;
    r54 = fma(r48, r53, r26);
    r55 = r11 * r12;
    r56 = r15 * r16;
    r56 = r56 * r48;
    r55 = fma(r30, r55, r56);
    r57 = r16 * r16;
    r57 = r57 * r30;
    r58 = r28 + r57;
    r59 = r11 * r11;
    r59 = r30 * r59;
    r58 = r58 + r59;
    r6 = fma(r9, r52, r6);
    r6 = fma(r32, r14, r6);
    r6 = fma(r35, r54, r6);
    r6 = fma(r40, r55, r6);
    r6 = fma(r39, r58, r6);
    r58 = 1.00000000000000008e-15;
    r33 = fma(r8, r43, r33);
    r53 = fma(r30, r53, r26);
    r57 = r28 + r57;
    r26 = r15 * r15;
    r26 = r30 * r26;
    r57 = r57 + r26;
    r55 = r16 * r11;
    r55 = r55 * r48;
    r54 = r15 * r12;
    r54 = fma(r48, r54, r55);
    r60 = r48 * r47;
    r61 = r46 * r10;
    r60 = fma(r4, r60, r61);
    r62 = r47 * r47;
    r62 = r30 * r62;
    r37 = r62 + r37;
    r33 = fma(r39, r53, r33);
    r33 = fma(r35, r57, r33);
    r33 = fma(r40, r54, r33);
    r33 = fma(r9, r60, r33);
    r33 = fma(r32, r37, r33);
    r54 = copysign(1.0, r33);
    r54 = fma(r58, r54, r33);
    r58 = r54 * r54;
    r33 = 1.0 / r58;
    r57 = r6 * r33;
    r10 = fma(r4, r10, r51);
    r8 = fma(r8, r10, r7);
    r7 = r11 * r12;
    r7 = fma(r48, r7, r56);
    r59 = r28 + r59;
    r59 = r59 + r26;
    r26 = r15 * r12;
    r26 = fma(r30, r26, r55);
    r55 = r47 * r30;
    r55 = fma(r4, r55, r61);
    r29 = r28 + r29;
    r29 = r29 + r62;
    r8 = fma(r39, r7, r8);
    r8 = fma(r40, r59, r8);
    r8 = fma(r35, r26, r8);
    r8 = fma(r32, r55, r8);
    r8 = fma(r9, r29, r8);
    r9 = r8 * r8;
    r32 = fma(r33, r9, r6 * r57);
    r32 = fma(r41, r32, r28);
    r32 = r44 * r32;
    r28 = r49 * r32;
    r28 = r28 * r57;
    r26 = 1.0 / r54;
    r35 = r26 * r32;
    r59 = fma(r42, r35, r43 * r28);
    r40 = r44 * r41;
    r7 = r48 * r42;
    r58 = r54 * r58;
    r58 = 1.0 / r58;
    r58 = r30 * r58;
    r54 = r43 * r58;
    r54 = fma(r9, r54, r57 * r7);
    r7 = r6 * r6;
    r7 = r7 * r58;
    r39 = r48 * r10;
    r39 = r39 * r8;
    r54 = fma(r33, r39, r54);
    r54 = fma(r43, r7, r54);
    r40 = r40 * r54;
    r40 = r40 * r26;
    r59 = fma(r6, r40, r59);
    r54 = r49 * r43;
    r54 = r54 * r8;
    r54 = r54 * r33;
    r54 = fma(r10, r35, r32 * r54);
    r54 = fma(r8, r40, r54);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx, r59, r54);
    r40 = fma(r52, r35, r60 * r28);
    r39 = r44 * r41;
    r62 = r48 * r52;
    r62 = fma(r57, r62, r60 * r7);
    r61 = r48 * r29;
    r61 = r61 * r8;
    r62 = fma(r33, r61, r62);
    r4 = r60 * r58;
    r62 = fma(r9, r4, r62);
    r39 = r39 * r6;
    r39 = r39 * r62;
    r40 = fma(r26, r39, r40);
    r39 = r49 * r60;
    r39 = r39 * r8;
    r39 = r39 * r33;
    r39 = fma(r32, r39, r29 * r35);
    r4 = r44 * r41;
    r4 = r4 * r8;
    r4 = r4 * r62;
    r39 = fma(r26, r4, r39);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx, r40, r39);
    r4 = r44 * r41;
    r62 = r48 * r14;
    r61 = r37 * r58;
    r61 = fma(r9, r61, r57 * r62);
    r62 = r48 * r55;
    r62 = r62 * r8;
    r61 = fma(r33, r62, r61);
    r61 = fma(r37, r7, r61);
    r4 = r4 * r6;
    r4 = r4 * r61;
    r4 = fma(r26, r4, r37 * r28);
    r4 = fma(r14, r35, r4);
    r28 = r44 * r41;
    r28 = r28 * r8;
    r28 = r28 * r61;
    r61 = r49 * r37;
    r61 = r61 * r8;
    r61 = r61 * r33;
    r61 = fma(r32, r61, r26 * r28);
    r61 = fma(r55, r35, r61);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r4, r61);
    r28 = r49 * r59;
    r2 = fma(r2, r49, r0);
    r2 = fma(r6, r35, r2);
    r6 = r49 * r54;
    r3 = fma(r3, r49, r1);
    r3 = fma(r8, r35, r3);
    r6 = fma(r3, r6, r2 * r28);
    r28 = r49 * r39;
    r35 = r49 * r40;
    r35 = fma(r2, r35, r3 * r28);
    WriteSum2<double, double>((double *)inout_shared, r6, r35);
  };
  FlushSumShared<2, double>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r49 * r61;
    r6 = r49 * r4;
    r6 = fma(r2, r6, r3 * r35);
    WriteSum1<double, double>((double *)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_point_njtr, 2 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r54, r54, r59 * r59);
    r35 = fma(r39, r39, r40 * r40);
    WriteSum2<double, double>((double *)inout_shared, r6, r35);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r61, r61, r4 * r4);
    WriteSum1<double, double>((double *)inout_shared, r35);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r54, r39, r59 * r40);
    r6 = fma(r59, r4, r54 * r61);
    WriteSum2<double, double>((double *)inout_shared, r35, r6);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r40, r4, r39 * r61);
    WriteSum1<double, double>((double *)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedPoseFixedPrincipalPointResJacFirst(
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *focal_and_extra, unsigned int focal_and_extra_num_alloc,
    SharedIndex *focal_and_extra_indices, double *point,
    unsigned int point_num_alloc, SharedIndex *point_indices, double *pixel,
    unsigned int pixel_num_alloc, double *pose, unsigned int pose_num_alloc,
    double *principal_point, unsigned int principal_point_num_alloc,
    double *out_res, unsigned int out_res_num_alloc, double *const out_rTr,
    double *out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    double *const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double *const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    double *const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
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
  SimpleRadialSplitFixedPoseFixedPrincipalPointResJacFirstKernel<<<n_blocks,
                                                                   1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, focal_and_extra,
      focal_and_extra_num_alloc, focal_and_extra_indices, point,
      point_num_alloc, point_indices, pixel, pixel_num_alloc, pose,
      pose_num_alloc, principal_point, principal_point_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc, out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc, out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc, out_point_jac,
      out_point_jac_num_alloc, out_point_njtr, out_point_njtr_num_alloc,
      out_point_precond_diag, out_point_precond_diag_num_alloc,
      out_point_precond_tril, out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar