#include "kernel_thin_prism_fisheye_fixed_pose_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFixedPoseFixedPointResJacFirstKernel(
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        double* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        double* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56;
  LoadShared<2, double, double>(
      calib, 2 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r2, r3);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      calib, 6 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r4, r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r6,
                                            r7);
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9);
    r10 = 2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r11,
                                            r12);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r13, r14);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r15,
                                            r16);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r17, r18);
    r19 = fma(r15, r18, r12 * r13);
    r20 = r11 * r14;
    r21 = -1.00000000000000000e+00;
    r19 = fma(r21, r20, r19);
    r19 = fma(r16, r17, r19);
    r20 = r10 * r19;
    r22 = r15 * r17;
    r22 = fma(r21, r22, r12 * r14);
    r22 = fma(r16, r18, r22);
    r22 = fma(r11, r13, r22);
    r20 = r20 * r22;
    r23 = fma(r15, r14, r12 * r17);
    r24 = r16 * r13;
    r23 = fma(r21, r24, r23);
    r23 = fma(r11, r18, r23);
    r24 = r10 * r23;
    r25 = fma(r16, r14, r15 * r13);
    r25 = fma(r11, r17, r25);
    r25 = fma(r21, r25, r12 * r18);
    r24 = fma(r25, r24, r20);
    r24 = fma(r8, r24, r7);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r7, r18);
    r26 = r15 * r16;
    r26 = r26 * r10;
    r27 = r11 * r12;
    r27 = fma(r10, r27, r26);
    r28 = -2.00000000000000000e+00;
    r29 = r15 * r15;
    r29 = r28 * r29;
    r30 = 1.00000000000000000e+00;
    r31 = r11 * r11;
    r31 = fma(r28, r31, r30);
    r32 = r29 + r31;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r33);
    r34 = r16 * r11;
    r34 = r34 * r10;
    r35 = r15 * r12;
    r35 = fma(r28, r35, r34);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r36);
    r37 = r10 * r23;
    r37 = r37 * r22;
    r38 = r28 * r25;
    r39 = fma(r19, r38, r37);
    r40 = r23 * r23;
    r40 = r28 * r40;
    r41 = r30 + r40;
    r42 = r19 * r19;
    r42 = r28 * r42;
    r41 = r41 + r42;
    r24 = fma(r7, r27, r24);
    r24 = fma(r18, r32, r24);
    r24 = fma(r33, r35, r24);
    r24 = fma(r36, r39, r24);
    r24 = fma(r9, r41, r24);
    r41 = r22 * r22;
    r41 = r28 * r41;
    r39 = r30 + r41;
    r39 = r39 + r40;
    r39 = fma(r8, r39, r6);
    r20 = fma(r23, r38, r20);
    r6 = r10 * r23;
    r6 = r6 * r19;
    r40 = r10 * r22;
    r40 = fma(r25, r40, r6);
    r35 = r15 * r11;
    r35 = r35 * r10;
    r32 = r16 * r12;
    r27 = fma(r10, r32, r35);
    r43 = r11 * r12;
    r43 = fma(r28, r43, r26);
    r26 = r16 * r16;
    r26 = r26 * r28;
    r31 = r26 + r31;
    r39 = fma(r9, r20, r39);
    r39 = fma(r36, r40, r39);
    r39 = fma(r33, r27, r39);
    r39 = fma(r18, r43, r39);
    r39 = fma(r7, r31, r39);
    r31 = r39 * r39;
    r43 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r27);
    r38 = fma(r22, r38, r6);
    r38 = fma(r8, r38, r27);
    r32 = fma(r28, r32, r35);
    r26 = r30 + r26;
    r26 = r26 + r29;
    r29 = r15 * r12;
    r29 = fma(r10, r29, r34);
    r34 = r10 * r19;
    r34 = fma(r25, r34, r37);
    r41 = r30 + r41;
    r41 = r41 + r42;
    r38 = fma(r7, r32, r38);
    r38 = fma(r33, r26, r38);
    r38 = fma(r18, r29, r38);
    r38 = fma(r9, r34, r38);
    r38 = fma(r36, r41, r38);
    r41 = copysign(1.0, r38);
    r41 = fma(r43, r41, r38);
    r38 = r41 * r41;
    r36 = 1.0 / r38;
    r34 = r24 * r24;
    r34 = fma(r36, r34, r36 * r31);
    r34 = sqrt(r34);
    r31 = atan(r34);
    r9 = r24 * r31;
    r29 = r24 * r9;
    r18 = copysign(1.0, r34);
    r18 = fma(r43, r18, r34);
    r43 = r18 * r18;
    r34 = 1.0 / r43;
    r34 = r36 * r34;
    r36 = r31 * r34;
    r29 = r29 * r36;
    r26 = r39 * r39;
    r33 = 3.00000000000000000e+00;
    r26 = r26 * r31;
    r26 = r26 * r33;
    r26 = fma(r36, r26, r29);
  };
  LoadShared<2, double, double>(
      calib, 10 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r32, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r42 = r39 * r39;
    r42 = r42 * r31;
    r42 = r42 * r36;
    r29 = r29 + r42;
    r32 = fma(r32, r29, r5 * r26);
  };
  LoadShared<2, double, double>(
      calib, 4 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r37, r25);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r35 = r29 * r29;
    r25 = fma(r25, r35, r37 * r29);
  };
  LoadShared<2, double, double>(
      calib, 8 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r37, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r27 = r35 * r35;
    r6 = r29 * r35;
    r25 = fma(r8, r27, r25);
    r25 = fma(r37, r6, r25);
    r37 = 1.0 / r41;
    r8 = 1.0 / r18;
    r40 = r37 * r8;
    r20 = r25 * r40;
    r44 = r39 * r31;
    r32 = fma(r44, r20, r32);
    r45 = r4 * r44;
    r46 = r10 * r9;
    r47 = r34 * r46;
    r32 = fma(r47, r45, r32);
    r32 = fma(r40, r44, r32);
    r0 = fma(r2, r32, r0);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r45, r20);
    r0 = fma(r45, r21, r0);
    r45 = r24 * r33;
    r45 = r45 * r9;
    r45 = fma(r36, r45, r42);
    r7 = fma(r7, r29, r4 * r45);
    r42 = r25 * r9;
    r7 = fma(r40, r42, r7);
    r48 = r5 * r44;
    r7 = fma(r47, r48, r7);
    r7 = fma(r9, r40, r7);
    r1 = fma(r3, r7, r1);
    r1 = fma(r20, r21, r1);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r0, r1);
    r20 = fma(r1, r1, r0 * r0);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r20);
  if (global_thread_idx < problem_size) {
    r20 = r21 * r7;
    r20 = r20 * r1;
    r48 = r21 * r0;
    r42 = r32 * r48;
    WriteSum2<double, double>((double*)inout_shared, r42, r20);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r21 * r1;
    WriteSum2<double, double>((double*)inout_shared, r48, r20);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r2 * r44;
    r42 = r20 * r48;
    r49 = r40 * r42;
    r50 = r21 * r29;
    r51 = r3 * r9;
    r50 = r50 * r1;
    r50 = r50 * r40;
    r50 = fma(r51, r50, r29 * r49);
    r52 = r21 * r1;
    r53 = r3 * r24;
    r53 = r53 * r31;
    r53 = r53 * r35;
    r53 = r53 * r37;
    r53 = r53 * r8;
    r8 = r35 * r40;
    r42 = fma(r8, r42, r53 * r52);
    WriteSum2<double, double>((double*)inout_shared, r50, r42);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            4 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r3 * r21;
    r42 = r42 * r45;
    r50 = r28 * r0;
    r50 = r50 * r9;
    r50 = r50 * r20;
    r50 = fma(r34, r50, r1 * r42);
    r42 = r2 * r26;
    r52 = r28 * r1;
    r52 = r52 * r44;
    r52 = r52 * r51;
    r52 = fma(r34, r52, r48 * r42);
    WriteSum2<double, double>((double*)inout_shared, r50, r52);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            6 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = r21 * r1;
    r52 = r52 * r6;
    r52 = r52 * r40;
    r52 = fma(r6, r49, r51 * r52);
    r50 = r21 * r1;
    r50 = r50 * r40;
    r50 = r50 * r51;
    r49 = fma(r27, r49, r27 * r50);
    WriteSum2<double, double>((double*)inout_shared, r52, r49);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            8 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r3 * r21;
    r49 = r49 * r29;
    r49 = r49 * r1;
    r52 = r2 * r29;
    r52 = r52 * r48;
    WriteSum2<double, double>((double*)inout_shared, r52, r49);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            10 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r32 * r32;
    r52 = r7 * r7;
    WriteSum2<double, double>((double*)inout_shared, r49, r52);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r30, r30);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = r3 * r51;
    r52 = r24 * r30;
    r49 = r36 * r52;
    r48 = r39 * r39;
    r50 = r2 * r2;
    r48 = r48 * r31;
    r48 = r48 * r35;
    r48 = r48 * r36;
    r48 = fma(r50, r48, r35 * r49);
    r42 = r27 * r36;
    r37 = r2 * r20;
    r54 = r39 * r37;
    r42 = fma(r54, r42, r27 * r49);
    WriteSum2<double, double>((double*)inout_shared, r48, r42);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            4 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r24 * r9;
    r55 = r31 * r31;
    r56 = 4.00000000000000000e+00;
    r38 = r41 * r38;
    r41 = r41 * r38;
    r41 = 1.0 / r41;
    r43 = r18 * r43;
    r18 = r18 * r43;
    r18 = 1.0 / r18;
    r55 = r55 * r56;
    r55 = r55 * r41;
    r55 = r55 * r18;
    r48 = r48 * r54;
    r18 = r3 * r3;
    r41 = r45 * r45;
    r18 = fma(r41, r18, r55 * r48);
    r48 = r39 * r44;
    r48 = r48 * r52;
    r41 = r26 * r26;
    r50 = fma(r41, r50, r55 * r48);
    WriteSum2<double, double>((double*)inout_shared, r18, r50);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            6 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r6 * r6;
    r18 = r36 * r54;
    r18 = fma(r50, r18, r50 * r49);
    r41 = r27 * r27;
    r48 = r36 * r54;
    r48 = fma(r41, r48, r49 * r41);
    WriteSum2<double, double>((double*)inout_shared, r18, r48);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            8 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r2 * r2;
    r48 = r48 * r35;
    r41 = r3 * r3;
    r41 = r41 * r35;
    WriteSum2<double, double>((double*)inout_shared, r48, r41);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            10 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = 0.00000000000000000e+00;
    WriteSum2<double, double>((double*)inout_shared, r41, r32);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r29 * r32;
    r48 = r48 * r40;
    r48 = r48 * r20;
    WriteSum2<double, double>((double*)inout_shared, r41, r48);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r32 * r20;
    r48 = r48 * r8;
    r55 = r32 * r20;
    r55 = r55 * r47;
    WriteSum2<double, double>((double*)inout_shared, r48, r55);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = r2 * r26;
    r55 = r55 * r32;
    r48 = r32 * r6;
    r48 = r48 * r40;
    r48 = r48 * r20;
    WriteSum2<double, double>((double*)inout_shared, r55, r48);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            6 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r2 * r29;
    r48 = r48 * r32;
    r32 = r32 * r40;
    r32 = r32 * r20;
    r32 = r32 * r27;
    WriteSum2<double, double>((double*)inout_shared, r32, r48);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            8 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r29 * r7;
    r48 = r48 * r40;
    r48 = r48 * r51;
    WriteSum2<double, double>((double*)inout_shared, r7, r48);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            12 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r3 * r45;
    r48 = r48 * r7;
    r32 = r7 * r53;
    WriteSum2<double, double>((double*)inout_shared, r32, r48);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            14 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r10 * r7;
    r48 = r48 * r44;
    r48 = r48 * r51;
    r48 = r48 * r34;
    r32 = r7 * r6;
    r32 = r32 * r40;
    r32 = r32 * r51;
    WriteSum2<double, double>((double*)inout_shared, r48, r32);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            16 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r7 * r40;
    r32 = r32 * r51;
    r32 = r32 * r27;
    WriteSum2<double, double>((double*)inout_shared, r32, r41);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            18 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r3 * r29;
    r32 = r32 * r7;
    WriteSum2<double, double>((double*)inout_shared, r32, r41);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            20 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r29 * r40;
    r32 = r32 * r20;
    r7 = r20 * r8;
    WriteSum2<double, double>((double*)inout_shared, r32, r7);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            22 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r2 * r26;
    r32 = r20 * r47;
    WriteSum2<double, double>((double*)inout_shared, r32, r7);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            24 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r6 * r40;
    r7 = r7 * r20;
    r20 = r40 * r20;
    r20 = r20 * r27;
    WriteSum2<double, double>((double*)inout_shared, r7, r20);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            26 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r2 * r29;
    WriteSum2<double, double>((double*)inout_shared, r20, r41);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            28 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r29 * r40;
    r20 = r20 * r51;
    WriteSum2<double, double>((double*)inout_shared, r20, r53);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            30 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r3 * r45;
    r20 = r10 * r44;
    r20 = r20 * r51;
    r20 = r20 * r34;
    WriteSum2<double, double>((double*)inout_shared, r53, r20);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            32 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r6 * r40;
    r20 = r20 * r51;
    r51 = r40 * r51;
    r51 = r51 * r27;
    WriteSum2<double, double>((double*)inout_shared, r20, r51);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            34 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r3 * r29;
    WriteSum2<double, double>((double*)inout_shared, r41, r51);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            36 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r6 * r54;
    r20 = fma(r36, r51, r6 * r49);
    r53 = r29 * r45;
    r53 = r53 * r40;
    r7 = r29 * r54;
    r38 = 1.0 / r38;
    r38 = r31 * r38;
    r43 = 1.0 / r43;
    r38 = r38 * r43;
    r7 = r7 * r46;
    r7 = fma(r38, r7, r30 * r53);
    WriteSum2<double, double>((double*)inout_shared, r20, r7);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            38 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r10 * r44;
    r7 = r7 * r52;
    r7 = r7 * r38;
    r52 = r29 * r26;
    r52 = r52 * r40;
    r52 = fma(r37, r52, r29 * r7);
    WriteSum2<double, double>((double*)inout_shared, r52, r42);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            40 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r37 * r8;
    r52 = r29 * r27;
    r20 = r36 * r52;
    r20 = fma(r54, r20, r52 * r49);
    WriteSum2<double, double>((double*)inout_shared, r20, r42);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            42 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r30 * r8;
    r53 = r45 * r30;
    r43 = r35 * r54;
    r43 = r43 * r46;
    r43 = fma(r38, r43, r8 * r53);
    WriteSum2<double, double>((double*)inout_shared, r42, r43);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            44 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r26 * r37;
    r43 = fma(r8, r43, r35 * r7);
    WriteSum2<double, double>((double*)inout_shared, r43, r20);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            46 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r6 * r40;
    r20 = r20 * r37;
    WriteSum2<double, double>((double*)inout_shared, r18, r20);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            48 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r6 * r40;
    r20 = r20 * r30;
    r47 = r37 * r47;
    r18 = r10 * r45;
    r18 = r18 * r44;
    r18 = r18 * r34;
    r18 = fma(r30, r18, r26 * r47);
    WriteSum2<double, double>((double*)inout_shared, r20, r18);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            50 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = r45 * r6;
    r18 = r18 * r40;
    r20 = r46 * r38;
    r20 = fma(r51, r20, r30 * r18);
    r18 = r45 * r40;
    r18 = r18 * r27;
    r51 = r27 * r54;
    r51 = r51 * r46;
    r51 = fma(r38, r51, r30 * r18);
    WriteSum2<double, double>((double*)inout_shared, r20, r51);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            52 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r29 * r47;
    r51 = r3 * r3;
    r51 = r51 * r29;
    r51 = r51 * r45;
    WriteSum2<double, double>((double*)inout_shared, r47, r51);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            54 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r26 * r6;
    r51 = r51 * r40;
    r51 = fma(r37, r51, r6 * r7);
    r47 = r26 * r40;
    r47 = r47 * r27;
    r47 = fma(r37, r47, r27 * r7);
    WriteSum2<double, double>((double*)inout_shared, r51, r47);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            56 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r2 * r2;
    r47 = r47 * r29;
    r47 = r47 * r26;
    r51 = r10 * r29;
    r51 = r51 * r44;
    r51 = r51 * r34;
    r51 = r51 * r30;
    WriteSum2<double, double>((double*)inout_shared, r47, r51);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            58 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r40 * r27;
    r51 = r51 * r37;
    r50 = r29 * r50;
    r47 = r36 * r54;
    r47 = fma(r50, r47, r49 * r50);
    WriteSum2<double, double>((double*)inout_shared, r47, r51);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            60 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r40 * r27;
    r51 = r51 * r30;
    r47 = r40 * r37;
    r47 = r47 * r52;
    WriteSum2<double, double>((double*)inout_shared, r51, r47);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            62 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r40 * r30;
    r47 = r47 * r52;
    WriteSum2<double, double>((double*)inout_shared, r47, r41);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            64 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeFixedPoseFixedPointResJacFirst(
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    double* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFixedPoseFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar