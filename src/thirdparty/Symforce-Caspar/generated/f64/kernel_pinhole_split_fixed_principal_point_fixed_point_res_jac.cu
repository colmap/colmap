#include "kernel_pinhole_split_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPrincipalPointFixedPointResJacKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        double* out_focal_jac,
        unsigned int out_focal_jac_num_alloc,
        double* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        double* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        double* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
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
      focal, 0 * focal_num_alloc, focal_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, focal_indices_loc[threadIdx.x].target, r28, r40);
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
    r39 = r17 * r25;
    r32 = -5.00000000000000000e-01;
    r29 = r13 * r32;
    r5 = 5.00000000000000000e-01;
    r44 = fma(r5, r27, r11 * r29);
    r44 = fma(r32, r24, r44);
    r44 = fma(r32, r26, r44);
    r39 = r39 * r44;
    r34 = r17 * r16;
    r37 = r13 * r14;
    r50 = r12 * r15;
    r50 = fma(r32, r50, r5 * r37);
    r37 = r9 * r10;
    r50 = fma(r5, r37, r50);
    r45 = r11 * r5;
    r50 = fma(r8, r45, r50);
    r34 = fma(r50, r34, r39);
    r37 = r17 * r22;
    r48 = r12 * r14;
    r52 = r8 * r10;
    r52 = fma(r32, r52, r32 * r48);
    r52 = fma(r9, r45, r52);
    r52 = fma(r15, r29, r52);
    r37 = r37 * r52;
    r48 = r12 * r11;
    r53 = r9 * r14;
    r53 = fma(r32, r53, r32 * r48);
    r48 = r8 * r15;
    r53 = fma(r32, r48, r53);
    r54 = r13 * r10;
    r53 = fma(r5, r54, r53);
    r54 = r53 * r19;
    r48 = r37 + r54;
    r55 = r34 + r48;
    r56 = r25 * r53;
    r57 = fma(r50, r23, r21 * r56);
    r58 = r17 * r16;
    r58 = r58 * r52;
    r59 = fma(r44, r19, r58);
    r57 = r57 + r59;
    r57 = fma(r6, r57, r7 * r55);
    r55 = r25 * r50;
    r60 = -4.00000000000000000e+00;
    r55 = r55 * r60;
    r61 = r52 * r60;
    r62 = r18 * r61;
    r63 = r55 + r62;
    r57 = fma(r41, r63, r57);
    r43 = r43 * r43;
    r43 = 1.0 / r43;
    r63 = r4 * r43;
    r49 = r63 * r49;
    r64 = r17 * r22;
    r64 = fma(r17, r56, r50 * r64);
    r64 = r64 + r59;
    r65 = r17 * r25;
    r65 = r65 * r52;
    r66 = r16 * r21;
    r66 = fma(r53, r66, r65);
    r50 = r50 * r19;
    r66 = r66 + r50;
    r66 = fma(r44, r23, r66);
    r66 = fma(r7, r66, r41 * r64);
    r64 = r16 * r44;
    r67 = r60 * r64;
    r55 = r55 + r67;
    r66 = fma(r6, r55, r66);
    r55 = r28 * r66;
    r55 = fma(r0, r55, r57 * r49);
    r68 = r18 * r21;
    r69 = r52 * r23;
    r68 = fma(r53, r68, r69);
    r68 = r68 + r34;
    r67 = r62 + r67;
    r67 = fma(r7, r67, r41 * r68);
    r68 = r17 * r22;
    r68 = fma(r44, r68, r50);
    r50 = r17 * r16;
    r50 = fma(r53, r50, r65);
    r68 = r68 + r50;
    r67 = fma(r6, r68, r67);
    r68 = r40 * r67;
    r65 = r57 * r63;
    r65 = fma(r47, r65, r0 * r68);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r55, r65);
    r68 = r21 * r25;
    r68 = fma(r44, r68, r69);
    r62 = r17 * r16;
    r34 = r8 * r11;
    r70 = r12 * r15;
    r70 = fma(r5, r70, r32 * r34);
    r34 = r9 * r10;
    r70 = fma(r32, r34, r70);
    r70 = fma(r14, r29, r70);
    r62 = r62 * r70;
    r34 = r9 * r14;
    r71 = r8 * r15;
    r71 = fma(r5, r71, r5 * r34);
    r71 = fma(r12, r45, r71);
    r71 = fma(r10, r29, r71);
    r29 = fma(r71, r19, r62);
    r68 = r68 + r29;
    r34 = r17 * r25;
    r34 = r34 * r71;
    r72 = r17 * r22;
    r72 = fma(r70, r72, r34);
    r72 = r72 + r59;
    r72 = fma(r7, r72, r6 * r68);
    r68 = r18 * r60;
    r68 = r68 * r70;
    r59 = r25 * r61;
    r73 = r68 + r59;
    r72 = fma(r41, r73, r72);
    r37 = r39 + r37;
    r37 = r37 + r29;
    r29 = r16 * r60;
    r29 = r29 * r71;
    r59 = r29 + r59;
    r59 = fma(r6, r59, r41 * r37);
    r37 = fma(r71, r23, r21 * r64);
    r39 = r17 * r25;
    r52 = r52 * r19;
    r39 = fma(r70, r39, r52);
    r37 = r37 + r39;
    r59 = fma(r7, r37, r59);
    r37 = r28 * r59;
    r37 = fma(r0, r37, r72 * r49);
    r73 = r72 * r63;
    r34 = r58 + r34;
    r58 = r18 * r21;
    r34 = fma(r44, r58, r34);
    r34 = fma(r70, r23, r34);
    r58 = r17 * r22;
    r64 = fma(r17, r64, r71 * r58);
    r64 = r64 + r39;
    r64 = fma(r6, r64, r41 * r34);
    r29 = r68 + r29;
    r64 = fma(r7, r29, r64);
    r29 = r40 * r64;
    r29 = fma(r0, r29, r47 * r73);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r37, r29);
    r73 = r18 * r60;
    r27 = fma(r32, r27, r13 * r45);
    r27 = fma(r5, r24, r27);
    r27 = fma(r5, r26, r27);
    r73 = r73 * r27;
    r56 = r60 * r56;
    r60 = r73 + r56;
    r26 = r17 * r16;
    r26 = r26 * r27;
    r5 = r21 * r25;
    r5 = fma(r70, r5, r26);
    r5 = r5 + r52;
    r5 = fma(r53, r23, r5);
    r5 = fma(r6, r5, r41 * r60);
    r60 = r17 * r22;
    r19 = fma(r70, r19, r27 * r60);
    r19 = r19 + r50;
    r5 = fma(r7, r19, r5);
    r19 = r17 * r25;
    r19 = r19 * r27;
    r60 = r16 * r21;
    r60 = fma(r70, r60, r19);
    r60 = r60 + r54;
    r60 = r60 + r69;
    r61 = r16 * r61;
    r56 = r56 + r61;
    r56 = fma(r6, r56, r7 * r60);
    r60 = r17 * r22;
    r60 = fma(r53, r60, r26);
    r60 = r60 + r39;
    r56 = fma(r41, r60, r56);
    r60 = r28 * r56;
    r60 = fma(r0, r60, r5 * r49);
    r19 = r62 + r19;
    r19 = r19 + r48;
    r48 = r18 * r21;
    r23 = fma(r27, r23, r70 * r48);
    r23 = r23 + r50;
    r23 = fma(r41, r23, r6 * r19);
    r61 = r73 + r61;
    r23 = fma(r7, r61, r23);
    r61 = r40 * r23;
    r7 = r5 * r63;
    r7 = fma(r47, r7, r0 * r61);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r60, r7);
    r61 = r28 * r35;
    r61 = fma(r31, r49, r0 * r61);
    r73 = r31 * r63;
    r41 = r40 * r46;
    r41 = fma(r0, r41, r47 * r73);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r61, r41);
    r73 = r28 * r30;
    r73 = fma(r0, r73, r38 * r49);
    r19 = r38 * r63;
    r6 = r40 * r51;
    r6 = fma(r0, r6, r47 * r19);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r73, r6);
    r19 = r28 * r20;
    r19 = fma(r0, r19, r36 * r49);
    r49 = r40 * r33;
    r50 = r36 * r63;
    r50 = fma(r47, r50, r0 * r49);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r19, r50);
    r49 = r4 * r2;
    r47 = r4 * r3;
    r47 = fma(r65, r47, r55 * r49);
    r49 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r37, r27, r29 * r49);
    WriteSum2<double, double>((double*)inout_shared, r47, r27);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = r4 * r2;
    r47 = r4 * r3;
    r47 = fma(r7, r47, r60 * r27);
    r27 = r4 * r3;
    r49 = r4 * r2;
    r49 = fma(r61, r49, r41 * r27);
    WriteSum2<double, double>((double*)inout_shared, r47, r49);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r4 * r3;
    r47 = r4 * r2;
    r47 = fma(r73, r47, r6 * r49);
    r49 = r4 * r2;
    r27 = r4 * r3;
    r27 = fma(r50, r27, r19 * r49);
    WriteSum2<double, double>((double*)inout_shared, r47, r27);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r65, r65, r55 * r55);
    r47 = fma(r37, r37, r29 * r29);
    WriteSum2<double, double>((double*)inout_shared, r27, r47);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fma(r60, r60, r7 * r7);
    r27 = fma(r61, r61, r41 * r41);
    WriteSum2<double, double>((double*)inout_shared, r47, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r6, r6, r73 * r73);
    r47 = fma(r50, r50, r19 * r19);
    WriteSum2<double, double>((double*)inout_shared, r27, r47);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fma(r55, r37, r65 * r29);
    r27 = fma(r55, r60, r65 * r7);
    WriteSum2<double, double>((double*)inout_shared, r47, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r65, r41, r55 * r61);
    r47 = fma(r55, r73, r65 * r6);
    WriteSum2<double, double>((double*)inout_shared, r27, r47);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = fma(r65, r50, r55 * r19);
    r55 = fma(r29, r7, r37 * r60);
    WriteSum2<double, double>((double*)inout_shared, r65, r55);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fma(r37, r61, r29 * r41);
    r65 = fma(r37, r73, r29 * r6);
    WriteSum2<double, double>((double*)inout_shared, r55, r65);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r37, r19, r29 * r50);
    r29 = fma(r7, r41, r60 * r61);
    WriteSum2<double, double>((double*)inout_shared, r37, r29);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r60, r73, r7 * r6);
    r7 = fma(r7, r50, r60 * r19);
    WriteSum2<double, double>((double*)inout_shared, r29, r7);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r61, r73, r41 * r6);
    r41 = fma(r41, r50, r61 * r19);
    WriteSum2<double, double>((double*)inout_shared, r7, r41);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fma(r6, r50, r73 * r19);
    WriteSum1<double, double>((double*)inout_shared, r50);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r42 * r0;
    r6 = r1 * r0;
    WriteIdx2<1024, double, double, double2>(
        out_focal_jac, 0 * out_focal_jac_num_alloc, global_thread_idx, r50, r6);
    r6 = r4 * r42;
    r6 = r6 * r2;
    r6 = r6 * r0;
    r50 = r4 * r1;
    r50 = r50 * r3;
    r50 = r50 * r0;
    WriteSum2<double, double>((double*)inout_shared, r6, r50);
  };
  FlushSumShared<2, double>(out_focal_njtr,
                            0 * out_focal_njtr_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r42 * r42;
    r42 = r42 * r43;
    r1 = r1 * r1;
    r1 = r1 * r43;
    WriteSum2<double, double>((double*)inout_shared, r42, r1);
  };
  FlushSumShared<2, double>(out_focal_precond_diag,
                            0 * out_focal_precond_diag_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
}

void PinholeSplitFixedPrincipalPointFixedPointResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
    double* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    double* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    double* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPrincipalPointFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal,
      focal_num_alloc,
      focal_indices,
      pixel,
      pixel_num_alloc,
      principal_point,
      principal_point_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_focal_jac,
      out_focal_jac_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_focal_precond_diag,
      out_focal_precond_diag_num_alloc,
      out_focal_precond_tril,
      out_focal_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar