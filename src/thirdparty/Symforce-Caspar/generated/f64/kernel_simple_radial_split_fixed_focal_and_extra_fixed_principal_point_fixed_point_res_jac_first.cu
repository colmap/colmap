#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacFirstKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        size_t problem_size) {
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
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80;

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
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r5);
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
    r17 = r16 * r16;
    r18 = -2.00000000000000000e+00;
    r17 = r17 * r18;
    r19 = 1.00000000000000000e+00;
    r20 = fma(r9, r15, r13 * r11);
    r21 = r12 * r10;
    r22 = r8 * r14;
    r20 = r20 + r21;
    r20 = fma(r4, r22, r20);
    r23 = r18 * r20;
    r23 = fma(r20, r23, r19);
    r24 = r17 + r23;
    r24 = fma(r6, r24, r0);
    r0 = 2.00000000000000000e+00;
    r25 = fma(r9, r14, r12 * r11);
    r26 = r13 * r10;
    r25 = fma(r4, r26, r25);
    r25 = fma(r8, r15, r25);
    r26 = r0 * r25;
    r26 = r26 * r20;
    r27 = r16 * r18;
    r28 = fma(r13, r15, r12 * r14);
    r28 = fma(r8, r10, r28);
    r28 = fma(r4, r28, r9 * r11);
    r27 = fma(r28, r27, r26);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r29);
    r30 = r0 * r16;
    r30 = r30 * r25;
    r31 = r0 * r28;
    r32 = fma(r20, r31, r30);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r33);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r34 = r14 * r10;
    r34 = r34 * r0;
    r35 = r15 * r11;
    r36 = fma(r0, r35, r34);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r37, r38);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r39 = r10 * r11;
    r40 = r14 * r15;
    r40 = r40 * r0;
    r39 = fma(r18, r39, r40);
    r41 = r15 * r15;
    r41 = r41 * r18;
    r42 = r19 + r41;
    r43 = r10 * r10;
    r43 = r43 * r18;
    r42 = r42 + r43;
    r24 = fma(r7, r27, r24);
    r24 = fma(r29, r32, r24);
    r24 = fma(r33, r36, r24);
    r24 = fma(r38, r39, r24);
    r24 = fma(r37, r42, r24);
    r32 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r27);
    r44 = r18 * r20;
    r44 = fma(r28, r44, r30);
    r44 = fma(r6, r44, r27);
    r35 = fma(r18, r35, r34);
    r41 = r19 + r41;
    r34 = r14 * r14;
    r34 = r34 * r18;
    r41 = r41 + r34;
    r27 = r15 * r10;
    r27 = r27 * r0;
    r30 = r14 * r11;
    r30 = fma(r0, r30, r27);
    r45 = r0 * r16;
    r45 = r45 * r20;
    r46 = fma(r25, r31, r45);
    r47 = r25 * r25;
    r47 = r47 * r18;
    r23 = r47 + r23;
    r44 = fma(r37, r35, r44);
    r44 = fma(r33, r41, r44);
    r44 = fma(r38, r30, r44);
    r44 = fma(r7, r46, r44);
    r44 = fma(r29, r23, r44);
    r23 = copysign(1.0, r44);
    r23 = fma(r32, r23, r44);
    r32 = 1.0 / r23;
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            0 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r44,
                                            r46);
    r48 = r23 * r23;
    r49 = 1.0 / r48;
    r50 = r24 * r49;
    r26 = fma(r16, r31, r26);
    r26 = fma(r6, r26, r5);
    r5 = r10 * r11;
    r5 = fma(r0, r5, r40);
    r43 = r19 + r43;
    r43 = r43 + r34;
    r34 = r14 * r11;
    r34 = fma(r18, r34, r27);
    r27 = r25 * r18;
    r27 = fma(r28, r27, r45);
    r17 = r19 + r17;
    r17 = r17 + r47;
    r26 = fma(r37, r5, r26);
    r26 = fma(r38, r43, r26);
    r26 = fma(r33, r34, r26);
    r26 = fma(r29, r27, r26);
    r26 = fma(r7, r17, r26);
    r17 = r26 * r26;
    r27 = fma(r49, r17, r24 * r50);
    r27 = fma(r46, r27, r19);
    r27 = r44 * r27;
    r19 = r32 * r27;
    r2 = fma(r24, r19, r2);
    r3 = fma(r3, r4, r1);
    r3 = fma(r26, r19, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = fma(r3, r3, r2 * r2);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  if (global_thread_idx < problem_size) {
    r1 = r4 * r3;
    r33 = r44 * r46;
    r38 = r12 * r11;
    r37 = -5.00000000000000000e-01;
    r47 = r9 * r14;
    r47 = fma(r37, r47, r37 * r38);
    r38 = r8 * r15;
    r47 = fma(r37, r38, r47);
    r45 = r13 * r10;
    r40 = 5.00000000000000000e-01;
    r47 = fma(r40, r45, r47);
    r45 = r20 * r47;
    r38 = r8 * r11;
    r51 = r13 * r14;
    r51 = fma(r40, r51, r40 * r38);
    r38 = r12 * r15;
    r51 = fma(r37, r38, r51);
    r52 = r9 * r40;
    r51 = fma(r10, r52, r51);
    r38 = fma(r51, r31, r0 * r45);
    r53 = r0 * r25;
    r54 = r9 * r15;
    r55 = r13 * r37;
    r54 = fma(r11, r55, r37 * r54);
    r54 = fma(r40, r22, r54);
    r54 = fma(r37, r21, r54);
    r56 = r0 * r16;
    r57 = r12 * r14;
    r58 = r8 * r10;
    r58 = fma(r37, r58, r37 * r57);
    r58 = fma(r11, r52, r58);
    r58 = fma(r15, r55, r58);
    r56 = r56 * r58;
    r53 = fma(r54, r53, r56);
    r38 = r38 + r53;
    r57 = r0 * r20;
    r57 = r57 * r58;
    r59 = r0 * r25;
    r59 = r59 * r51;
    r60 = r57 + r59;
    r61 = r16 * r18;
    r60 = fma(r47, r61, r60);
    r62 = r18 * r28;
    r60 = fma(r54, r62, r60);
    r60 = fma(r7, r60, r29 * r38);
    r38 = r20 * r51;
    r62 = -4.00000000000000000e+00;
    r38 = r38 * r62;
    r61 = r16 * r54;
    r63 = r62 * r61;
    r64 = r38 + r63;
    r60 = fma(r6, r64, r60);
    r64 = r0 * r60;
    r65 = r0 * r20;
    r65 = r65 * r54;
    r66 = r0 * r16;
    r66 = fma(r51, r66, r65);
    r67 = r0 * r25;
    r67 = r67 * r47;
    r68 = r58 * r31;
    r69 = r67 + r68;
    r70 = r66 + r69;
    r71 = r18 * r28;
    r71 = fma(r18, r45, r51 * r71);
    r71 = r71 + r53;
    r71 = fma(r6, r71, r7 * r70);
    r70 = r25 * r62;
    r51 = r58 * r70;
    r38 = r38 + r51;
    r71 = fma(r29, r38, r71);
    r38 = r24 * r24;
    r48 = r23 * r48;
    r48 = 1.0 / r48;
    r48 = r18 * r48;
    r38 = r38 * r48;
    r64 = fma(r71, r38, r50 * r64);
    r23 = r71 * r48;
    r64 = fma(r17, r23, r64);
    r72 = r0 * r26;
    r73 = r25 * r18;
    r74 = r18 * r28;
    r74 = r74 * r58;
    r73 = fma(r47, r73, r74);
    r73 = r73 + r66;
    r51 = r63 + r51;
    r51 = fma(r7, r51, r29 * r73);
    r59 = fma(r54, r31, r59);
    r73 = r0 * r16;
    r73 = fma(r47, r73, r57);
    r59 = r59 + r73;
    r51 = fma(r6, r59, r51);
    r72 = r72 * r51;
    r64 = fma(r49, r72, r64);
    r33 = r33 * r64;
    r33 = r33 * r32;
    r51 = fma(r51, r19, r26 * r33);
    r64 = r4 * r26;
    r64 = r64 * r71;
    r64 = r64 * r49;
    r51 = fma(r27, r64, r51);
    r64 = r4 * r2;
    r33 = fma(r60, r19, r24 * r33);
    r72 = r4 * r27;
    r72 = r72 * r50;
    r33 = fma(r71, r72, r33);
    r64 = fma(r33, r64, r51 * r1);
    r1 = r4 * r2;
    r23 = r44 * r46;
    r68 = r65 + r68;
    r65 = r0 * r16;
    r59 = r8 * r11;
    r57 = r12 * r15;
    r57 = fma(r40, r57, r37 * r59);
    r59 = r9 * r10;
    r57 = fma(r37, r59, r57);
    r57 = fma(r14, r55, r57);
    r65 = r65 * r57;
    r59 = r0 * r25;
    r63 = r12 * r11;
    r66 = r8 * r15;
    r66 = fma(r40, r66, r40 * r63);
    r66 = fma(r14, r52, r66);
    r66 = fma(r10, r55, r66);
    r59 = fma(r66, r59, r65);
    r68 = r68 + r59;
    r55 = r20 * r58;
    r55 = r55 * r62;
    r63 = r16 * r62;
    r63 = r63 * r66;
    r75 = r55 + r63;
    r75 = fma(r6, r75, r29 * r68);
    r68 = r18 * r28;
    r68 = fma(r18, r61, r66 * r68);
    r76 = r0 * r25;
    r76 = r76 * r58;
    r77 = r0 * r20;
    r77 = fma(r57, r77, r76);
    r68 = r68 + r77;
    r75 = fma(r7, r68, r75);
    r68 = r0 * r75;
    r78 = r0 * r26;
    r79 = r25 * r18;
    r79 = fma(r54, r79, r56);
    r56 = r0 * r20;
    r56 = r56 * r66;
    r80 = r18 * r28;
    r79 = fma(r57, r80, r79);
    r79 = r79 + r56;
    r66 = fma(r66, r31, r0 * r61);
    r66 = r66 + r77;
    r66 = fma(r6, r66, r29 * r79);
    r79 = r57 * r70;
    r63 = r63 + r79;
    r66 = fma(r7, r63, r66);
    r78 = r78 * r66;
    r78 = fma(r49, r78, r50 * r68);
    r68 = r18 * r20;
    r68 = fma(r54, r68, r74);
    r68 = r68 + r59;
    r56 = fma(r57, r31, r56);
    r56 = r56 + r53;
    r56 = fma(r7, r56, r6 * r68);
    r79 = r55 + r79;
    r56 = fma(r29, r79, r56);
    r79 = r56 * r48;
    r78 = fma(r17, r79, r78);
    r78 = fma(r56, r38, r78);
    r23 = r23 * r24;
    r23 = r23 * r78;
    r23 = fma(r56, r72, r32 * r23);
    r23 = fma(r75, r19, r23);
    r79 = r4 * r3;
    r55 = r44 * r46;
    r55 = r55 * r26;
    r55 = r55 * r78;
    r55 = fma(r32, r55, r66 * r19);
    r66 = r4 * r26;
    r66 = r66 * r56;
    r66 = r66 * r49;
    r55 = fma(r27, r66, r55);
    r79 = fma(r55, r79, r23 * r1);
    WriteSum2<double, double>((double*)inout_shared, r64, r79);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = r4 * r3;
    r64 = r44 * r46;
    r1 = r0 * r26;
    r66 = r0 * r20;
    r78 = r13 * r11;
    r22 = fma(r37, r22, r40 * r78);
    r22 = fma(r15, r52, r22);
    r22 = fma(r40, r21, r22);
    r66 = r66 * r22;
    r65 = r65 + r66;
    r65 = r65 + r69;
    r69 = r25 * r18;
    r21 = r18 * r28;
    r21 = fma(r22, r21, r57 * r69);
    r21 = r21 + r73;
    r21 = fma(r29, r21, r6 * r65);
    r58 = r16 * r58;
    r58 = r58 * r62;
    r70 = r22 * r70;
    r65 = r58 + r70;
    r21 = fma(r7, r65, r21);
    r1 = r1 * r21;
    r45 = r62 * r45;
    r70 = r70 + r45;
    r62 = r0 * r16;
    r62 = r62 * r22;
    r76 = r76 + r62;
    r65 = r18 * r20;
    r76 = fma(r57, r65, r76);
    r69 = r18 * r28;
    r76 = fma(r47, r69, r76);
    r76 = fma(r6, r76, r29 * r70);
    r70 = r0 * r25;
    r22 = fma(r22, r31, r57 * r70);
    r22 = r22 + r73;
    r76 = fma(r7, r22, r76);
    r1 = fma(r76, r38, r49 * r1);
    r74 = r67 + r74;
    r67 = r16 * r18;
    r74 = fma(r57, r67, r74);
    r74 = r74 + r66;
    r45 = r58 + r45;
    r45 = fma(r6, r45, r7 * r74);
    r31 = fma(r47, r31, r62);
    r31 = r31 + r77;
    r45 = fma(r29, r31, r45);
    r31 = r0 * r45;
    r1 = fma(r50, r31, r1);
    r29 = r76 * r48;
    r1 = fma(r17, r29, r1);
    r64 = r64 * r26;
    r64 = r64 * r1;
    r29 = r4 * r26;
    r29 = r29 * r76;
    r29 = r29 * r49;
    r29 = fma(r27, r29, r32 * r64);
    r29 = fma(r21, r19, r29);
    r21 = r4 * r2;
    r64 = fma(r45, r19, r76 * r72);
    r31 = r44 * r46;
    r31 = r31 * r24;
    r31 = r31 * r1;
    r64 = fma(r32, r31, r64);
    r21 = fma(r64, r21, r29 * r79);
    r79 = r4 * r2;
    r31 = r44 * r46;
    r1 = r0 * r5;
    r1 = r1 * r26;
    r77 = r35 * r48;
    r77 = fma(r17, r77, r49 * r1);
    r1 = r0 * r42;
    r77 = fma(r50, r1, r77);
    r77 = fma(r35, r38, r77);
    r31 = r31 * r24;
    r31 = r31 * r77;
    r31 = fma(r32, r31, r35 * r72);
    r31 = fma(r42, r19, r31);
    r1 = r4 * r3;
    r47 = r44 * r46;
    r47 = r47 * r26;
    r47 = r47 * r77;
    r47 = fma(r32, r47, r5 * r19);
    r77 = r4 * r35;
    r77 = r77 * r26;
    r77 = r77 * r49;
    r47 = fma(r27, r77, r47);
    r1 = fma(r47, r1, r31 * r79);
    WriteSum2<double, double>((double*)inout_shared, r21, r1);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r4 * r3;
    r21 = r44 * r46;
    r79 = r30 * r48;
    r77 = r0 * r43;
    r77 = r77 * r26;
    r77 = fma(r49, r77, r17 * r79);
    r79 = r0 * r39;
    r77 = fma(r50, r79, r77);
    r77 = fma(r30, r38, r77);
    r21 = r21 * r26;
    r21 = r21 * r77;
    r21 = fma(r43, r19, r32 * r21);
    r79 = r4 * r30;
    r79 = r79 * r26;
    r79 = r79 * r49;
    r21 = fma(r27, r79, r21);
    r79 = r4 * r2;
    r62 = fma(r30, r72, r39 * r19);
    r6 = r44 * r46;
    r6 = r6 * r24;
    r6 = r6 * r77;
    r62 = fma(r32, r6, r62);
    r79 = fma(r62, r79, r21 * r1);
    r1 = r4 * r2;
    r6 = r44 * r46;
    r77 = r0 * r34;
    r77 = r77 * r26;
    r74 = r41 * r48;
    r74 = fma(r17, r74, r49 * r77);
    r77 = r0 * r36;
    r74 = fma(r50, r77, r74);
    r74 = fma(r41, r38, r74);
    r6 = r6 * r24;
    r6 = r6 * r74;
    r6 = fma(r36, r19, r32 * r6);
    r6 = fma(r41, r72, r6);
    r72 = r4 * r3;
    r24 = r4 * r41;
    r24 = r24 * r26;
    r24 = r24 * r49;
    r49 = r44 * r46;
    r49 = r49 * r26;
    r49 = r49 * r74;
    r49 = fma(r32, r49, r27 * r24);
    r49 = fma(r34, r19, r49);
    r72 = fma(r49, r72, r6 * r1);
    WriteSum2<double, double>((double*)inout_shared, r79, r72);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r33, r33, r51 * r51);
    r79 = fma(r55, r55, r23 * r23);
    WriteSum2<double, double>((double*)inout_shared, r72, r79);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = fma(r64, r64, r29 * r29);
    r72 = fma(r31, r31, r47 * r47);
    WriteSum2<double, double>((double*)inout_shared, r79, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r62, r62, r21 * r21);
    r79 = fma(r6, r6, r49 * r49);
    WriteSum2<double, double>((double*)inout_shared, r72, r79);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = fma(r33, r23, r51 * r55);
    r72 = fma(r51, r29, r33 * r64);
    WriteSum2<double, double>((double*)inout_shared, r79, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r51, r47, r33 * r31);
    r79 = fma(r33, r62, r51 * r21);
    WriteSum2<double, double>((double*)inout_shared, r72, r79);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r33, r6, r51 * r49);
    r51 = fma(r55, r29, r23 * r64);
    WriteSum2<double, double>((double*)inout_shared, r33, r51);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r55, r47, r23 * r31);
    r33 = fma(r55, r21, r23 * r62);
    WriteSum2<double, double>((double*)inout_shared, r51, r33);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fma(r55, r49, r23 * r6);
    r23 = fma(r64, r31, r29 * r47);
    WriteSum2<double, double>((double*)inout_shared, r55, r23);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r64, r62, r29 * r21);
    r29 = fma(r29, r49, r64 * r6);
    WriteSum2<double, double>((double*)inout_shared, r23, r29);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r47, r21, r31 * r62);
    r47 = fma(r47, r49, r31 * r6);
    WriteSum2<double, double>((double*)inout_shared, r29, r47);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fma(r21, r49, r62 * r6);
    WriteSum1<double, double>((double*)inout_shared, r49);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacFirst(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              sensor_from_rig,
              sensor_from_rig_num_alloc,
              pixel,
              pixel_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_pose_njtr,
              out_pose_njtr_num_alloc,
              out_pose_precond_diag,
              out_pose_precond_diag_num_alloc,
              out_pose_precond_tril,
              out_pose_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar