#include "kernel_pinhole_split_fixed_focal_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedFocalFixedPointResJacFirstKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
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
        double* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        double* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        double* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        double* const out_principal_point_precond_tril,
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
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71;
  LoadShared<2, double, double>(principal_point,
                                0 * principal_point_num_alloc,
                                principal_point_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        principal_point_indices_loc[threadIdx.x].target,
                        r0,
                        r1);
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
    ReadIdx2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r28, r40);
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
    r42 = r28 * r42;
    r2 = fma(r0, r42, r2);
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
    r1 = r40 * r1;
    r3 = fma(r0, r1, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r47 = fma(r3, r3, r2 * r2);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r47);
  if (global_thread_idx < problem_size) {
    r47 = r17 * r25;
    r39 = -5.00000000000000000e-01;
    r32 = r13 * r39;
    r29 = 5.00000000000000000e-01;
    r5 = fma(r29, r27, r11 * r32);
    r5 = fma(r39, r24, r5);
    r5 = fma(r39, r26, r5);
    r47 = r47 * r5;
    r44 = r17 * r16;
    r37 = r13 * r14;
    r50 = r12 * r15;
    r50 = fma(r39, r50, r29 * r37);
    r37 = r9 * r10;
    r50 = fma(r29, r37, r50);
    r45 = r11 * r29;
    r50 = fma(r8, r45, r50);
    r44 = fma(r50, r44, r47);
    r37 = r17 * r22;
    r49 = r12 * r14;
    r48 = r8 * r10;
    r48 = fma(r39, r48, r39 * r49);
    r48 = fma(r9, r45, r48);
    r48 = fma(r15, r32, r48);
    r37 = r37 * r48;
    r49 = r12 * r11;
    r52 = r9 * r14;
    r52 = fma(r39, r52, r39 * r49);
    r49 = r8 * r15;
    r52 = fma(r39, r49, r52);
    r53 = r13 * r10;
    r52 = fma(r29, r53, r52);
    r53 = r52 * r19;
    r49 = r37 + r53;
    r54 = r44 + r49;
    r55 = r25 * r52;
    r56 = fma(r50, r23, r21 * r55);
    r57 = r17 * r16;
    r57 = r57 * r48;
    r58 = fma(r5, r19, r57);
    r56 = r56 + r58;
    r56 = fma(r6, r56, r7 * r54);
    r54 = r25 * r50;
    r59 = -4.00000000000000000e+00;
    r54 = r54 * r59;
    r60 = r48 * r59;
    r61 = r18 * r60;
    r62 = r54 + r61;
    r56 = fma(r41, r62, r56);
    r43 = r43 * r43;
    r43 = 1.0 / r43;
    r43 = r4 * r43;
    r42 = r43 * r42;
    r62 = r17 * r22;
    r62 = fma(r17, r55, r50 * r62);
    r62 = r62 + r58;
    r63 = r17 * r25;
    r63 = r63 * r48;
    r64 = r16 * r21;
    r64 = fma(r52, r64, r63);
    r50 = r50 * r19;
    r64 = r64 + r50;
    r64 = fma(r5, r23, r64);
    r64 = fma(r7, r64, r41 * r62);
    r62 = r16 * r5;
    r65 = r59 * r62;
    r54 = r54 + r65;
    r64 = fma(r6, r54, r64);
    r54 = r28 * r64;
    r54 = fma(r0, r54, r56 * r42);
    r66 = r18 * r21;
    r67 = r48 * r23;
    r66 = fma(r52, r66, r67);
    r66 = r66 + r44;
    r65 = r61 + r65;
    r65 = fma(r7, r65, r41 * r66);
    r66 = r17 * r22;
    r66 = fma(r5, r66, r50);
    r50 = r17 * r16;
    r50 = fma(r52, r50, r63);
    r66 = r66 + r50;
    r65 = fma(r6, r66, r65);
    r66 = r40 * r65;
    r63 = r56 * r43;
    r63 = fma(r1, r63, r0 * r66);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r54, r63);
    r66 = r21 * r25;
    r66 = fma(r5, r66, r67);
    r61 = r17 * r16;
    r44 = r8 * r11;
    r68 = r12 * r15;
    r68 = fma(r29, r68, r39 * r44);
    r44 = r9 * r10;
    r68 = fma(r39, r44, r68);
    r68 = fma(r14, r32, r68);
    r61 = r61 * r68;
    r44 = r9 * r14;
    r69 = r8 * r15;
    r69 = fma(r29, r69, r29 * r44);
    r69 = fma(r12, r45, r69);
    r69 = fma(r10, r32, r69);
    r32 = fma(r69, r19, r61);
    r66 = r66 + r32;
    r44 = r17 * r25;
    r44 = r44 * r69;
    r70 = r17 * r22;
    r70 = fma(r68, r70, r44);
    r70 = r70 + r58;
    r70 = fma(r7, r70, r6 * r66);
    r66 = r18 * r59;
    r66 = r66 * r68;
    r58 = r25 * r60;
    r71 = r66 + r58;
    r70 = fma(r41, r71, r70);
    r37 = r47 + r37;
    r37 = r37 + r32;
    r32 = r16 * r59;
    r32 = r32 * r69;
    r58 = r32 + r58;
    r58 = fma(r6, r58, r41 * r37);
    r37 = fma(r69, r23, r21 * r62);
    r47 = r17 * r25;
    r48 = r48 * r19;
    r47 = fma(r68, r47, r48);
    r37 = r37 + r47;
    r58 = fma(r7, r37, r58);
    r37 = r28 * r58;
    r37 = fma(r0, r37, r70 * r42);
    r71 = r70 * r43;
    r44 = r57 + r44;
    r57 = r18 * r21;
    r44 = fma(r5, r57, r44);
    r44 = fma(r68, r23, r44);
    r57 = r17 * r22;
    r62 = fma(r17, r62, r69 * r57);
    r62 = r62 + r47;
    r62 = fma(r6, r62, r41 * r44);
    r32 = r66 + r32;
    r62 = fma(r7, r32, r62);
    r32 = r40 * r62;
    r32 = fma(r0, r32, r1 * r71);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r37, r32);
    r71 = r18 * r59;
    r27 = fma(r39, r27, r13 * r45);
    r27 = fma(r29, r24, r27);
    r27 = fma(r29, r26, r27);
    r71 = r71 * r27;
    r55 = r59 * r55;
    r59 = r71 + r55;
    r26 = r17 * r16;
    r26 = r26 * r27;
    r29 = r21 * r25;
    r29 = fma(r68, r29, r26);
    r29 = r29 + r48;
    r29 = fma(r52, r23, r29);
    r29 = fma(r6, r29, r41 * r59);
    r59 = r17 * r22;
    r19 = fma(r68, r19, r27 * r59);
    r19 = r19 + r50;
    r29 = fma(r7, r19, r29);
    r19 = r17 * r25;
    r19 = r19 * r27;
    r59 = r16 * r21;
    r59 = fma(r68, r59, r19);
    r59 = r59 + r53;
    r59 = r59 + r67;
    r60 = r16 * r60;
    r55 = r55 + r60;
    r55 = fma(r6, r55, r7 * r59);
    r59 = r17 * r22;
    r59 = fma(r52, r59, r26);
    r59 = r59 + r47;
    r55 = fma(r41, r59, r55);
    r59 = r28 * r55;
    r59 = fma(r0, r59, r29 * r42);
    r19 = r61 + r19;
    r19 = r19 + r49;
    r49 = r18 * r21;
    r23 = fma(r27, r23, r68 * r49);
    r23 = r23 + r50;
    r23 = fma(r41, r23, r6 * r19);
    r60 = r71 + r60;
    r23 = fma(r7, r60, r23);
    r60 = r40 * r23;
    r7 = r29 * r43;
    r7 = fma(r1, r7, r0 * r60);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r59, r7);
    r60 = r28 * r35;
    r60 = fma(r31, r42, r0 * r60);
    r71 = r31 * r43;
    r41 = r40 * r46;
    r41 = fma(r0, r41, r1 * r71);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r60, r41);
    r71 = r28 * r30;
    r71 = fma(r0, r71, r38 * r42);
    r19 = r38 * r43;
    r6 = r40 * r51;
    r6 = fma(r0, r6, r1 * r19);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r71, r6);
    r19 = r28 * r20;
    r19 = fma(r0, r19, r36 * r42);
    r42 = r40 * r33;
    r50 = r36 * r43;
    r50 = fma(r1, r50, r0 * r42);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r19, r50);
    r42 = r4 * r2;
    r1 = r4 * r3;
    r1 = fma(r63, r1, r54 * r42);
    r42 = r4 * r3;
    r0 = r4 * r2;
    r0 = fma(r37, r0, r32 * r42);
    WriteSum2<double, double>((double*)inout_shared, r1, r0);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r4 * r2;
    r1 = r4 * r3;
    r1 = fma(r7, r1, r59 * r0);
    r0 = r4 * r3;
    r42 = r4 * r2;
    r42 = fma(r60, r42, r41 * r0);
    WriteSum2<double, double>((double*)inout_shared, r1, r42);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r4 * r3;
    r1 = r4 * r2;
    r1 = fma(r71, r1, r6 * r42);
    r42 = r4 * r2;
    r0 = r4 * r3;
    r0 = fma(r50, r0, r19 * r42);
    WriteSum2<double, double>((double*)inout_shared, r1, r0);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r63, r63, r54 * r54);
    r1 = fma(r37, r37, r32 * r32);
    WriteSum2<double, double>((double*)inout_shared, r0, r1);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r59, r59, r7 * r7);
    r0 = fma(r60, r60, r41 * r41);
    WriteSum2<double, double>((double*)inout_shared, r1, r0);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r6, r6, r71 * r71);
    r1 = fma(r50, r50, r19 * r19);
    WriteSum2<double, double>((double*)inout_shared, r0, r1);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r54, r37, r63 * r32);
    r0 = fma(r54, r59, r63 * r7);
    WriteSum2<double, double>((double*)inout_shared, r1, r0);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r63, r41, r54 * r60);
    r1 = fma(r54, r71, r63 * r6);
    WriteSum2<double, double>((double*)inout_shared, r0, r1);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fma(r63, r50, r54 * r19);
    r54 = fma(r32, r7, r37 * r59);
    WriteSum2<double, double>((double*)inout_shared, r63, r54);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fma(r37, r60, r32 * r41);
    r63 = fma(r37, r71, r32 * r6);
    WriteSum2<double, double>((double*)inout_shared, r54, r63);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r37, r19, r32 * r50);
    r32 = fma(r7, r41, r59 * r60);
    WriteSum2<double, double>((double*)inout_shared, r37, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r59, r71, r7 * r6);
    r7 = fma(r7, r50, r59 * r19);
    WriteSum2<double, double>((double*)inout_shared, r32, r7);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r60, r71, r41 * r6);
    r41 = fma(r41, r50, r60 * r19);
    WriteSum2<double, double>((double*)inout_shared, r7, r41);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fma(r6, r50, r71 * r19);
    WriteSum1<double, double>((double*)inout_shared, r50);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r4 * r2;
    r6 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r50, r6);
  };
  FlushSumShared<2, double>(out_principal_point_njtr,
                            0 * out_principal_point_njtr_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r34, r34);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedFocalFixedPointResJacFirst(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
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
    double* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    double* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    double* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedFocalFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      pixel,
      pixel_num_alloc,
      focal,
      focal_num_alloc,
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
      out_principal_point_jac,
      out_principal_point_jac_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar