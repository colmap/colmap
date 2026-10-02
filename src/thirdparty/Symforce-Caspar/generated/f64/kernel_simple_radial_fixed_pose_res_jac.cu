#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_fixed_pose_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SimpleRadialFixedPoseResJacKernel(
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    double *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    double *pixel, unsigned int pixel_num_alloc, double *pose,
    unsigned int pose_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *out_calib_jac,
    unsigned int out_calib_jac_num_alloc, double *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, double *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, double *out_point_jac,
    unsigned int out_point_jac_num_alloc, double *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0,
         r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0,
         r17 = 0, r18 = 0, r19 = 0, r20 = 0, r21 = 0, r22 = 0, r23 = 0, r24 = 0,
         r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0, r31 = 0, r32 = 0,
         r33 = 0, r34 = 0, r35 = 0, r36 = 0, r37 = 0, r38 = 0, r39 = 0, r40 = 0,
         r41 = 0, r42 = 0, r43 = 0, r44 = 0, r45 = 0, r46 = 0, r47 = 0, r48 = 0,
         r49 = 0, r50 = 0, r51 = 0, r52 = 0, r53 = 0, r54 = 0, r55 = 0, r56 = 0,
         r57 = 0, r58 = 0, r59 = 0, r60 = 0, r61 = 0, r62 = 0, r63 = 0;
  LoadShared<2, double, double>(calib, 2 * calib_num_alloc, calib_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        calib_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
  LoadShared<2, double, double>(calib, 0 * calib_num_alloc, calib_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        calib_indices_loc[threadIdx.x].target, r44, r41);
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
    r10 = fma(r41, r48, r24);
    r21 = 1.0 / r21;
    r50 = r10 * r21;
    r45 = r38 * r50;
    r26 = r36 * r50;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             0 * out_calib_jac_num_alloc,
                                             global_thread_idx, r45, r26);
    r30 = r38 * r21;
    r47 = r44 * r48;
    r30 = r30 * r47;
    r46 = r36 * r21;
    r46 = r46 * r47;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             2 * out_calib_jac_num_alloc,
                                             global_thread_idx, r30, r46);
    r4 = fma(r3, r27, r1);
    r43 = r44 * r36;
    r4 = fma(r50, r43, r4);
    r4 = r27 * r4;
    r43 = r36 * r4;
    r28 = r27 * r38;
    r29 = fma(r2, r27, r0);
    r37 = r44 * r38;
    r29 = fma(r50, r37, r29);
    r28 = r28 * r29;
    r28 = fma(r50, r28, r50 * r43);
    r50 = r27 * r38;
    r50 = r50 * r29;
    r50 = r50 * r21;
    r37 = r21 * r47;
    r37 = fma(r43, r37, r47 * r50);
    WriteSum2<double, double>((double *)inout_shared, r28, r37);
  };
  FlushSumShared<2, double>(out_calib_njtr, 0 * out_calib_njtr_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = r27 * r29;
    WriteSum2<double, double>((double *)inout_shared, r29, r4);
  };
  FlushSumShared<2, double>(out_calib_njtr, 2 * out_calib_njtr_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = r10 * r10;
    r4 = r4 * r49;
    r4 = fma(r20, r4, r31 * r4);
    r48 = r44 * r48;
    r49 = r49 * r47;
    r48 = r48 * r49;
    r48 = fma(r31, r48, r20 * r48);
    WriteSum2<double, double>((double *)inout_shared, r4, r48);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double *)inout_shared, r24, r24);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r10 * r31;
    r48 = r10 * r20;
    r48 = fma(r49, r48, r49 * r24);
    WriteSum2<double, double>((double *)inout_shared, r48, r45);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double *)inout_shared, r26, r30);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = 0.00000000000000000e+00;
    WriteSum2<double, double>((double *)inout_shared, r46, r30);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = -2.00000000000000000e+00;
    r46 = fma(r15, r18, r12 * r13);
    r26 = r16 * r17;
    r45 = -1.00000000000000000e+00;
    r46 = fma(r45, r26, r46);
    r46 = fma(r11, r14, r46);
    r26 = r46 * r46;
    r26 = r30 * r26;
    r48 = 1.00000000000000000e+00;
    r24 = r15 * r13;
    r24 = fma(r45, r24, r12 * r18);
    r24 = fma(r16, r14, r24);
    r24 = fma(r11, r17, r24);
    r49 = r24 * r24;
    r49 = fma(r30, r49, r48);
    r4 = r26 + r49;
    r6 = fma(r8, r4, r6);
    r29 = r46 * r30;
    r37 = fma(r16, r18, r15 * r17);
    r37 = fma(r11, r13, r37);
    r37 = fma(r45, r37, r12 * r14);
    r28 = 2.00000000000000000e+00;
    r14 = fma(r15, r14, r12 * r17);
    r50 = r11 * r18;
    r14 = fma(r45, r50, r14);
    r14 = fma(r16, r13, r14);
    r50 = r28 * r14;
    r43 = r24 * r50;
    r29 = fma(r37, r29, r43);
    r42 = r28 * r24;
    r51 = r46 * r50;
    r42 = fma(r37, r42, r51);
    r52 = r15 * r11;
    r52 = r52 * r28;
    r53 = r16 * r12;
    r54 = fma(r28, r53, r52);
    r55 = r11 * r12;
    r56 = r15 * r16;
    r56 = r56 * r28;
    r55 = fma(r30, r55, r56);
    r57 = r16 * r16;
    r57 = r57 * r30;
    r58 = r48 + r57;
    r59 = r11 * r11;
    r59 = r30 * r59;
    r58 = r58 + r59;
    r6 = fma(r9, r29, r6);
    r6 = fma(r32, r42, r6);
    r6 = fma(r35, r54, r6);
    r6 = fma(r40, r55, r6);
    r6 = fma(r39, r58, r6);
    r58 = r28 * r4;
    r55 = 1.00000000000000008e-15;
    r54 = r30 * r24;
    r54 = fma(r37, r54, r51);
    r33 = fma(r8, r54, r33);
    r53 = fma(r30, r53, r52);
    r57 = r48 + r57;
    r52 = r15 * r15;
    r52 = r30 * r52;
    r57 = r57 + r52;
    r51 = r16 * r11;
    r51 = r51 * r28;
    r60 = r15 * r12;
    r60 = fma(r28, r60, r51);
    r61 = r28 * r46;
    r61 = r61 * r24;
    r50 = fma(r37, r50, r61);
    r62 = r14 * r14;
    r62 = r62 * r30;
    r49 = r62 + r49;
    r33 = fma(r39, r53, r33);
    r33 = fma(r35, r57, r33);
    r33 = fma(r40, r60, r33);
    r33 = fma(r9, r50, r33);
    r33 = fma(r32, r49, r33);
    r60 = copysign(1.0, r33);
    r60 = fma(r55, r60, r33);
    r55 = r60 * r60;
    r33 = 1.0 / r55;
    r57 = r6 * r33;
    r55 = r60 * r55;
    r55 = 1.0 / r55;
    r55 = r30 * r55;
    r53 = r54 * r55;
    r63 = r28 * r46;
    r63 = fma(r37, r63, r43);
    r8 = fma(r8, r63, r7);
    r7 = r11 * r12;
    r7 = fma(r28, r7, r56);
    r59 = r48 + r59;
    r59 = r59 + r52;
    r52 = r15 * r12;
    r52 = fma(r30, r52, r51);
    r51 = r14 * r30;
    r51 = fma(r37, r51, r61);
    r26 = r48 + r26;
    r26 = r26 + r62;
    r8 = fma(r39, r7, r8);
    r8 = fma(r40, r59, r8);
    r8 = fma(r35, r52, r8);
    r8 = fma(r32, r51, r8);
    r8 = fma(r9, r26, r8);
    r9 = r8 * r8;
    r53 = fma(r9, r53, r57 * r58);
    r58 = r6 * r6;
    r58 = r58 * r55;
    r32 = r28 * r63;
    r32 = r32 * r8;
    r53 = fma(r33, r32, r53);
    r53 = fma(r54, r58, r53);
    r53 = r41 * r53;
    r60 = 1.0 / r60;
    r60 = r44 * r60;
    r53 = r53 * r60;
    r32 = r44 * r45;
    r52 = fma(r33, r9, r6 * r57);
    r52 = fma(r41, r52, r48);
    r32 = r32 * r52;
    r32 = r32 * r57;
    r48 = fma(r54, r32, r6 * r53);
    r35 = r52 * r60;
    r48 = fma(r4, r35, r48);
    r59 = r44 * r45;
    r59 = r59 * r54;
    r59 = r59 * r8;
    r59 = r59 * r52;
    r59 = fma(r33, r59, r63 * r35);
    r59 = fma(r8, r53, r59);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx, r48, r59);
    r53 = fma(r50, r32, r29 * r35);
    r40 = r41 * r6;
    r7 = r28 * r29;
    r7 = fma(r57, r7, r50 * r58);
    r39 = r28 * r26;
    r39 = r39 * r8;
    r7 = fma(r33, r39, r7);
    r62 = r50 * r55;
    r7 = fma(r9, r62, r7);
    r40 = r40 * r7;
    r53 = fma(r60, r40, r53);
    r40 = r41 * r8;
    r40 = r40 * r7;
    r7 = r44 * r45;
    r7 = r7 * r50;
    r7 = r7 * r8;
    r7 = r7 * r52;
    r7 = fma(r33, r7, r60 * r40);
    r7 = fma(r26, r35, r7);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r53, r7);
    r40 = r41 * r6;
    r62 = r28 * r42;
    r39 = r49 * r55;
    r39 = fma(r9, r39, r57 * r62);
    r62 = r28 * r51;
    r62 = r62 * r8;
    r39 = fma(r33, r62, r39);
    r39 = fma(r49, r58, r39);
    r40 = r40 * r39;
    r32 = fma(r49, r32, r60 * r40);
    r32 = fma(r42, r35, r32);
    r40 = r41 * r8;
    r40 = r40 * r39;
    r40 = fma(r51, r35, r60 * r40);
    r60 = r44 * r45;
    r60 = r60 * r49;
    r60 = r60 * r8;
    r60 = r60 * r52;
    r40 = fma(r33, r60, r40);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx, r32, r40);
    r60 = r45 * r48;
    r2 = fma(r2, r45, r0);
    r2 = fma(r6, r35, r2);
    r0 = r45 * r59;
    r3 = fma(r3, r45, r1);
    r3 = fma(r8, r35, r3);
    r0 = fma(r3, r0, r2 * r60);
    r60 = r45 * r7;
    r35 = r45 * r53;
    r35 = fma(r2, r35, r3 * r60);
    WriteSum2<double, double>((double *)inout_shared, r0, r35);
  };
  FlushSumShared<2, double>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r45 * r40;
    r0 = r45 * r32;
    r0 = fma(r2, r0, r3 * r35);
    WriteSum1<double, double>((double *)inout_shared, r0);
  };
  FlushSumShared<1, double>(out_point_njtr, 2 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r59, r59, r48 * r48);
    r35 = fma(r7, r7, r53 * r53);
    WriteSum2<double, double>((double *)inout_shared, r0, r35);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r32, r32, r40 * r40);
    WriteSum1<double, double>((double *)inout_shared, r35);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r59, r7, r48 * r53);
    r0 = fma(r48, r32, r59 * r40);
    WriteSum2<double, double>((double *)inout_shared, r35, r0);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r53, r32, r7 * r40);
    WriteSum1<double, double>((double *)inout_shared, r0);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
}

void SimpleRadialFixedPoseResJac(
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    double *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    double *pixel, unsigned int pixel_num_alloc, double *pose,
    unsigned int pose_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *out_calib_jac,
    unsigned int out_calib_jac_num_alloc, double *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, double *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, double *out_point_jac,
    unsigned int out_point_jac_num_alloc, double *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFixedPoseResJacKernel<<<n_blocks, 1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, calib, calib_num_alloc,
      calib_indices, point, point_num_alloc, point_indices, pixel,
      pixel_num_alloc, pose, pose_num_alloc, out_res, out_res_num_alloc,
      out_calib_jac, out_calib_jac_num_alloc, out_calib_njtr,
      out_calib_njtr_num_alloc, out_calib_precond_diag,
      out_calib_precond_diag_num_alloc, out_calib_precond_tril,
      out_calib_precond_tril_num_alloc, out_point_jac, out_point_jac_num_alloc,
      out_point_njtr, out_point_njtr_num_alloc, out_point_precond_diag,
      out_point_precond_diag_num_alloc, out_point_precond_tril,
      out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar