#include "kernel_pinhole_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) PinholeFixedPointResJacFirstKernel(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double* out_calib_jac,
    unsigned int out_calib_jac_num_alloc,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    double* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74;
  LoadShared<2, double, double>(
      calib, 2 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    r0 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r5);
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r6, r7);
  };
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r10,
                                            r11);
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r14,
                                            r15);
    r16 = fma(r13, r14, r8 * r11);
    r17 = r12 * r15;
    r16 = fma(r4, r17, r16);
    r16 = fma(r9, r10, r16);
    r17 = 2.00000000000000000e+00;
    r18 = fma(r9, r14, r12 * r11);
    r19 = r13 * r10;
    r18 = fma(r4, r19, r18);
    r18 = fma(r8, r15, r18);
    r19 = r17 * r18;
    r20 = r16 * r19;
    r21 = -2.00000000000000000e+00;
    r22 = fma(r13, r15, r12 * r14);
    r22 = fma(r8, r10, r22);
    r22 = fma(r4, r22, r9 * r11);
    r23 = r21 * r22;
    r24 = r9 * r15;
    r25 = fma(r13, r11, r24);
    r26 = r12 * r10;
    r27 = r8 * r14;
    r25 = r25 + r26;
    r25 = fma(r4, r27, r25);
    r28 = fma(r25, r23, r20);
    r28 = fma(r6, r28, r5);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r5, r29);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r30 = r14 * r10;
    r30 = r30 * r17;
    r31 = r15 * r11;
    r31 = fma(r21, r31, r30);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = r14 * r14;
    r33 = r33 * r21;
    r34 = 1.00000000000000000e+00;
    r35 = r15 * r15;
    r35 = fma(r21, r35, r34);
    r36 = r33 + r35;
    r37 = r15 * r10;
    r37 = r37 * r17;
    r38 = r14 * r11;
    r38 = fma(r17, r38, r37);
    r39 = r17 * r16;
    r39 = r39 * r25;
    r40 = fma(r22, r19, r39);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r41);
    r42 = r21 * r25;
    r42 = r42 * r25;
    r43 = r34 + r42;
    r44 = r18 * r18;
    r44 = r44 * r21;
    r43 = r43 + r44;
    r28 = fma(r5, r31, r28);
    r28 = fma(r32, r36, r28);
    r28 = fma(r29, r38, r28);
    r28 = fma(r7, r40, r28);
    r28 = fma(r41, r43, r28);
    r43 = copysign(1.0, r28);
    r43 = fma(r0, r43, r28);
    r0 = 1.0 / r43;
  };
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r28, r40);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r45,
                                            r46);
    r42 = r34 + r42;
    r47 = r16 * r16;
    r47 = r47 * r21;
    r42 = r42 + r47;
    r42 = fma(r6, r42, r45);
    r45 = r25 * r19;
    r48 = fma(r16, r23, r45);
    r49 = r17 * r25;
    r49 = fma(r22, r49, r20);
    r20 = r15 * r11;
    r20 = fma(r17, r20, r30);
    r30 = r10 * r11;
    r50 = r14 * r15;
    r50 = r50 * r17;
    r30 = fma(r21, r30, r50);
    r51 = r10 * r10;
    r51 = r51 * r21;
    r35 = r51 + r35;
    r42 = fma(r7, r48, r42);
    r42 = fma(r41, r49, r42);
    r42 = fma(r32, r20, r42);
    r42 = fma(r29, r30, r42);
    r42 = fma(r5, r35, r42);
    r49 = r28 * r42;
    r2 = fma(r0, r49, r2);
    r3 = fma(r3, r4, r1);
    r1 = r17 * r16;
    r1 = fma(r22, r1, r45);
    r1 = fma(r6, r1, r46);
    r46 = r10 * r11;
    r46 = fma(r17, r46, r50);
    r51 = r34 + r51;
    r51 = r51 + r33;
    r33 = r14 * r11;
    r33 = fma(r21, r33, r37);
    r39 = fma(r18, r23, r39);
    r47 = r34 + r47;
    r47 = r47 + r44;
    r1 = fma(r5, r46, r1);
    r1 = fma(r29, r51, r1);
    r1 = fma(r32, r33, r1);
    r1 = fma(r41, r39, r1);
    r1 = fma(r7, r47, r1);
    r47 = r40 * r1;
    r3 = fma(r0, r47, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r39 = fma(r3, r3, r2 * r2);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r39);
  if (global_thread_idx < problem_size) {
    r39 = r17 * r25;
    r32 = -5.00000000000000000e-01;
    r29 = r13 * r32;
    r5 = 5.00000000000000000e-01;
    r44 = fma(r5, r27, r11 * r29);
    r44 = fma(r32, r24, r44);
    r44 = fma(r32, r26, r44);
    r39 = r39 * r44;
    r37 = r17 * r16;
    r50 = r13 * r14;
    r45 = r12 * r15;
    r45 = fma(r32, r45, r5 * r50);
    r50 = r9 * r10;
    r45 = fma(r5, r50, r45);
    r48 = r11 * r5;
    r45 = fma(r8, r48, r45);
    r37 = fma(r45, r37, r39);
    r50 = r17 * r22;
    r52 = r12 * r14;
    r53 = r8 * r10;
    r53 = fma(r32, r53, r32 * r52);
    r53 = fma(r9, r48, r53);
    r53 = fma(r15, r29, r53);
    r50 = r50 * r53;
    r52 = r12 * r11;
    r54 = r9 * r14;
    r54 = fma(r32, r54, r32 * r52);
    r52 = r8 * r15;
    r54 = fma(r32, r52, r54);
    r55 = r13 * r10;
    r54 = fma(r5, r55, r54);
    r55 = r54 * r19;
    r52 = r50 + r55;
    r56 = r37 + r52;
    r57 = r25 * r54;
    r58 = fma(r45, r23, r21 * r57);
    r59 = r17 * r16;
    r59 = r59 * r53;
    r60 = fma(r44, r19, r59);
    r58 = r58 + r60;
    r58 = fma(r6, r58, r7 * r56);
    r56 = r25 * r45;
    r61 = -4.00000000000000000e+00;
    r56 = r56 * r61;
    r62 = r53 * r61;
    r63 = r18 * r62;
    r64 = r56 + r63;
    r58 = fma(r41, r64, r58);
    r43 = r43 * r43;
    r43 = 1.0 / r43;
    r64 = r4 * r43;
    r49 = r64 * r49;
    r65 = r17 * r22;
    r65 = fma(r17, r57, r45 * r65);
    r65 = r65 + r60;
    r66 = r17 * r25;
    r66 = r66 * r53;
    r67 = r16 * r21;
    r67 = fma(r54, r67, r66);
    r45 = r45 * r19;
    r67 = r67 + r45;
    r67 = fma(r44, r23, r67);
    r67 = fma(r7, r67, r41 * r65);
    r65 = r16 * r44;
    r68 = r61 * r65;
    r56 = r56 + r68;
    r67 = fma(r6, r56, r67);
    r56 = r28 * r67;
    r56 = fma(r0, r56, r58 * r49);
    r69 = r18 * r21;
    r70 = r53 * r23;
    r69 = fma(r54, r69, r70);
    r69 = r69 + r37;
    r68 = r63 + r68;
    r68 = fma(r7, r68, r41 * r69);
    r69 = r17 * r22;
    r69 = fma(r44, r69, r45);
    r45 = r17 * r16;
    r45 = fma(r54, r45, r66);
    r69 = r69 + r45;
    r68 = fma(r6, r69, r68);
    r69 = r40 * r68;
    r66 = r58 * r64;
    r66 = fma(r47, r66, r0 * r69);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r56, r66);
    r69 = r21 * r25;
    r69 = fma(r44, r69, r70);
    r63 = r17 * r16;
    r37 = r8 * r11;
    r71 = r12 * r15;
    r71 = fma(r5, r71, r32 * r37);
    r37 = r9 * r10;
    r71 = fma(r32, r37, r71);
    r71 = fma(r14, r29, r71);
    r63 = r63 * r71;
    r37 = r9 * r14;
    r72 = r8 * r15;
    r72 = fma(r5, r72, r5 * r37);
    r72 = fma(r12, r48, r72);
    r72 = fma(r10, r29, r72);
    r29 = fma(r72, r19, r63);
    r69 = r69 + r29;
    r37 = r17 * r25;
    r37 = r37 * r72;
    r73 = r17 * r22;
    r73 = fma(r71, r73, r37);
    r73 = r73 + r60;
    r73 = fma(r7, r73, r6 * r69);
    r69 = r18 * r61;
    r69 = r69 * r71;
    r60 = r25 * r62;
    r74 = r69 + r60;
    r73 = fma(r41, r74, r73);
    r50 = r39 + r50;
    r50 = r50 + r29;
    r29 = r16 * r61;
    r29 = r29 * r72;
    r60 = r29 + r60;
    r60 = fma(r6, r60, r41 * r50);
    r50 = fma(r72, r23, r21 * r65);
    r39 = r17 * r25;
    r53 = r53 * r19;
    r39 = fma(r71, r39, r53);
    r50 = r50 + r39;
    r60 = fma(r7, r50, r60);
    r50 = r28 * r60;
    r50 = fma(r0, r50, r73 * r49);
    r37 = r59 + r37;
    r59 = r18 * r21;
    r37 = fma(r44, r59, r37);
    r37 = fma(r71, r23, r37);
    r59 = r17 * r22;
    r65 = fma(r17, r65, r72 * r59);
    r65 = r65 + r39;
    r65 = fma(r6, r65, r41 * r37);
    r29 = r69 + r29;
    r65 = fma(r7, r29, r65);
    r29 = r40 * r65;
    r69 = r73 * r64;
    r69 = fma(r47, r69, r0 * r29);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r50, r69);
    r29 = r17 * r25;
    r27 = fma(r32, r27, r13 * r48);
    r27 = fma(r5, r24, r27);
    r27 = fma(r5, r26, r27);
    r29 = r29 * r27;
    r26 = r16 * r21;
    r26 = fma(r71, r26, r29);
    r26 = r26 + r55;
    r26 = r26 + r70;
    r62 = r16 * r62;
    r57 = r61 * r57;
    r70 = r62 + r57;
    r70 = fma(r6, r70, r7 * r26);
    r26 = r17 * r16;
    r26 = r26 * r27;
    r55 = r17 * r22;
    r55 = fma(r54, r55, r26);
    r55 = r55 + r39;
    r70 = fma(r41, r55, r70);
    r55 = r28 * r70;
    r61 = r18 * r61;
    r61 = r61 * r27;
    r57 = r61 + r57;
    r39 = r21 * r25;
    r39 = fma(r71, r39, r26);
    r39 = r39 + r53;
    r39 = fma(r54, r23, r39);
    r39 = fma(r6, r39, r41 * r57);
    r57 = r17 * r22;
    r19 = fma(r71, r19, r27 * r57);
    r19 = r19 + r45;
    r39 = fma(r7, r19, r39);
    r55 = fma(r39, r49, r0 * r55);
    r29 = r63 + r29;
    r29 = r29 + r52;
    r52 = r18 * r21;
    r23 = fma(r27, r23, r71 * r52);
    r23 = r23 + r45;
    r23 = fma(r41, r23, r6 * r29);
    r62 = r61 + r62;
    r23 = fma(r7, r62, r23);
    r62 = r40 * r23;
    r7 = r39 * r64;
    r7 = fma(r47, r7, r0 * r62);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r55, r7);
    r62 = r28 * r35;
    r62 = fma(r31, r49, r0 * r62);
    r61 = r40 * r46;
    r41 = r31 * r64;
    r41 = fma(r47, r41, r0 * r61);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r62, r41);
    r61 = r28 * r30;
    r61 = fma(r0, r61, r38 * r49);
    r29 = r38 * r64;
    r6 = r40 * r51;
    r6 = fma(r0, r6, r47 * r29);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r61, r6);
    r29 = r28 * r20;
    r29 = fma(r0, r29, r36 * r49);
    r49 = r36 * r64;
    r45 = r40 * r33;
    r45 = fma(r0, r45, r47 * r49);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r29, r45);
    r49 = r4 * r3;
    r47 = r4 * r2;
    r47 = fma(r56, r47, r66 * r49);
    r49 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r50, r27, r69 * r49);
    WriteSum2<double, double>((double*)inout_shared, r47, r27);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = r4 * r3;
    r47 = r4 * r2;
    r47 = fma(r55, r47, r7 * r27);
    r27 = r4 * r2;
    r49 = r4 * r3;
    r49 = fma(r41, r49, r62 * r27);
    WriteSum2<double, double>((double*)inout_shared, r47, r49);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r4 * r3;
    r47 = r4 * r2;
    r47 = fma(r61, r47, r6 * r49);
    r49 = r4 * r2;
    r27 = r4 * r3;
    r27 = fma(r45, r27, r29 * r49);
    WriteSum2<double, double>((double*)inout_shared, r47, r27);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r66, r66, r56 * r56);
    r47 = fma(r50, r50, r69 * r69);
    WriteSum2<double, double>((double*)inout_shared, r27, r47);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fma(r55, r55, r7 * r7);
    r27 = fma(r41, r41, r62 * r62);
    WriteSum2<double, double>((double*)inout_shared, r47, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r61, r61, r6 * r6);
    r47 = fma(r45, r45, r29 * r29);
    WriteSum2<double, double>((double*)inout_shared, r27, r47);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fma(r56, r50, r66 * r69);
    r27 = fma(r56, r55, r66 * r7);
    WriteSum2<double, double>((double*)inout_shared, r47, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r56, r62, r66 * r41);
    r47 = fma(r56, r61, r66 * r6);
    WriteSum2<double, double>((double*)inout_shared, r27, r47);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = fma(r66, r45, r56 * r29);
    r56 = fma(r50, r55, r69 * r7);
    WriteSum2<double, double>((double*)inout_shared, r66, r56);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = fma(r69, r41, r50 * r62);
    r66 = fma(r50, r61, r69 * r6);
    WriteSum2<double, double>((double*)inout_shared, r56, r66);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fma(r69, r45, r50 * r29);
    r50 = fma(r7, r41, r55 * r62);
    WriteSum2<double, double>((double*)inout_shared, r69, r50);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fma(r7, r6, r55 * r61);
    r55 = fma(r55, r29, r7 * r45);
    WriteSum2<double, double>((double*)inout_shared, r50, r55);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fma(r41, r6, r62 * r61);
    r41 = fma(r41, r45, r62 * r29);
    WriteSum2<double, double>((double*)inout_shared, r55, r41);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r61, r29, r6 * r45);
    WriteSum1<double, double>((double*)inout_shared, r29);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = r42 * r0;
    r61 = r1 * r0;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             0 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r29,
                                             r61);
    r45 = r4 * r42;
    r45 = r45 * r2;
    r45 = r45 * r0;
    r6 = r4 * r1;
    r6 = r6 * r3;
    r6 = r6 * r0;
    WriteSum2<double, double>((double*)inout_shared, r45, r6);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r4 * r2;
    r45 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r6, r45);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r42 * r42;
    r42 = r42 * r43;
    r1 = r1 * r1;
    r1 = r1 * r43;
    WriteSum2<double, double>((double*)inout_shared, r42, r1);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r34, r34);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = 0.00000000000000000e+00;
    WriteSum2<double, double>((double*)inout_shared, r34, r29);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r61, r34);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeFixedPointResJacFirst(double* pose,
                                  unsigned int pose_num_alloc,
                                  SharedIndex* pose_indices,
                                  double* sensor_from_rig,
                                  unsigned int sensor_from_rig_num_alloc,
                                  double* calib,
                                  unsigned int calib_num_alloc,
                                  SharedIndex* calib_indices,
                                  double* pixel,
                                  unsigned int pixel_num_alloc,
                                  double* point,
                                  unsigned int point_num_alloc,
                                  double* out_res,
                                  unsigned int out_res_num_alloc,
                                  double* const out_rTr,
                                  double* out_pose_jac,
                                  unsigned int out_pose_jac_num_alloc,
                                  double* const out_pose_njtr,
                                  unsigned int out_pose_njtr_num_alloc,
                                  double* const out_pose_precond_diag,
                                  unsigned int out_pose_precond_diag_num_alloc,
                                  double* const out_pose_precond_tril,
                                  unsigned int out_pose_precond_tril_num_alloc,
                                  double* out_calib_jac,
                                  unsigned int out_calib_jac_num_alloc,
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
  PinholeFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar