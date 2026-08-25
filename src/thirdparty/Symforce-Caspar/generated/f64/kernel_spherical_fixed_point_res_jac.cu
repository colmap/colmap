#include "kernel_spherical_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SphericalFixedPointResJacKernel(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* wh,
    unsigned int wh_num_alloc,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        wh, 0 * wh_num_alloc, global_thread_idx, r0, r1);
    r2 = 5.00000000000000000e-01;
    r3 = 1.59154943091895346e-01;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r4,
                                            r5);
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r6, r7);
    r8 = -2.00000000000000000e+00;
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r9, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r11,
                                            r12);
  };
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r13, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r15,
                                            r16);
    r17 = r13 * r15;
    r18 = -1.00000000000000000e+00;
    r17 = fma(r18, r17, r10 * r12);
    r17 = fma(r14, r16, r17);
    r17 = fma(r9, r11, r17);
    r19 = r8 * r17;
    r19 = r19 * r17;
    r20 = 1.00000000000000000e+00;
    r21 = r13 * r12;
    r22 = fma(r10, r15, r21);
    r23 = r14 * r11;
    r24 = r9 * r16;
    r22 = r22 + r23;
    r22 = fma(r18, r24, r22);
    r25 = r8 * r22;
    r25 = fma(r22, r25, r20);
    r26 = r19 + r25;
    r26 = fma(r6, r26, r4);
    r4 = fma(r14, r15, r9 * r12);
    r27 = r10 * r11;
    r4 = fma(r18, r27, r4);
    r4 = fma(r13, r16, r4);
    r27 = 2.00000000000000000e+00;
    r28 = r27 * r17;
    r29 = r4 * r28;
    r30 = fma(r10, r16, r9 * r15);
    r30 = fma(r13, r11, r30);
    r30 = fma(r18, r30, r14 * r12);
    r31 = r8 * r30;
    r32 = fma(r22, r31, r29);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r33);
    r34 = r22 * r27;
    r34 = r34 * r4;
    r35 = fma(r30, r28, r34);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r36);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r37 = r15 * r11;
    r37 = r37 * r27;
    r38 = r16 * r12;
    r38 = fma(r27, r38, r37);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r39, r40);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r41 = r11 * r12;
    r42 = r15 * r16;
    r42 = r42 * r27;
    r41 = fma(r8, r41, r42);
    r43 = r11 * r11;
    r43 = r43 * r8;
    r44 = r20 + r43;
    r45 = r16 * r16;
    r45 = r45 * r8;
    r44 = r44 + r45;
    r26 = fma(r7, r32, r26);
    r26 = fma(r33, r35, r26);
    r26 = fma(r36, r38, r26);
    r26 = fma(r40, r41, r26);
    r26 = fma(r39, r44, r26);
    r35 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r32);
    r34 = fma(r17, r31, r34);
    r34 = fma(r6, r34, r32);
    r32 = r16 * r12;
    r32 = fma(r8, r32, r37);
    r45 = r20 + r45;
    r37 = r15 * r15;
    r37 = r8 * r37;
    r45 = r45 + r37;
    r46 = r16 * r11;
    r46 = r46 * r27;
    r47 = r15 * r12;
    r47 = fma(r27, r47, r46);
    r48 = r27 * r4;
    r49 = r22 * r28;
    r48 = fma(r30, r48, r49);
    r19 = r20 + r19;
    r50 = r8 * r4;
    r50 = r50 * r4;
    r19 = r19 + r50;
    r34 = fma(r39, r32, r34);
    r34 = fma(r36, r45, r34);
    r34 = fma(r40, r47, r34);
    r34 = fma(r7, r48, r34);
    r34 = fma(r33, r19, r34);
    r19 = copysign(r35, r34);
    r19 = r19 + r34;
    r48 = atan2(r26, r19);
    r48 = fma(r3, r48, r2);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r51);
    r3 = fma(r3, r18, r0 * r48);
    r48 = -3.18309886183790691e-01;
    r52 = r22 * r27;
    r52 = fma(r30, r52, r29);
    r52 = fma(r6, r52, r5);
    r5 = r11 * r12;
    r5 = fma(r27, r5, r42);
    r43 = r20 + r43;
    r43 = r43 + r37;
    r37 = r15 * r12;
    r37 = fma(r8, r37, r46);
    r49 = fma(r4, r31, r49);
    r25 = r50 + r25;
    r52 = fma(r39, r5, r52);
    r52 = fma(r40, r43, r52);
    r52 = fma(r36, r37, r52);
    r52 = fma(r33, r49, r52);
    r52 = fma(r7, r25, r52);
    r25 = r18 * r52;
    r49 = r26 * r26;
    r36 = r35 + r49;
    r36 = fma(r34, r34, r36);
    r40 = sqrt(r36);
    r35 = copysign(r35, r40);
    r40 = r35 + r40;
    r25 = atan2(r25, r40);
    r25 = fma(r48, r25, r2);
    r51 = fma(r51, r18, r1 * r25);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3, r51);
    r25 = 3.18309886183790691e-01;
    r25 = r51 * r25;
    r51 = r40 * r40;
    r48 = fma(r52, r52, r51);
    r35 = 1.0 / r48;
    r39 = r1 * r51;
    r25 = r25 * r35;
    r25 = r25 * r39;
    r35 = r27 * r26;
    r50 = r27 * r30;
    r46 = r10 * r15;
    r46 = fma(r2, r21, r2 * r46);
    r20 = -5.00000000000000000e-01;
    r46 = fma(r20, r24, r46);
    r46 = fma(r2, r23, r46);
    r42 = r9 * r12;
    r29 = r14 * r15;
    r29 = fma(r20, r29, r20 * r42);
    r42 = r13 * r16;
    r29 = fma(r20, r42, r29);
    r53 = r10 * r11;
    r29 = fma(r2, r53, r29);
    r50 = fma(r29, r28, r46 * r50);
    r53 = r27 * r4;
    r42 = r13 * r15;
    r54 = r14 * r16;
    r54 = fma(r20, r54, r2 * r42);
    r42 = r9 * r11;
    r54 = fma(r20, r42, r54);
    r55 = r10 * r20;
    r54 = fma(r12, r55, r54);
    r42 = r22 * r27;
    r56 = r14 * r12;
    r57 = r9 * r15;
    r57 = fma(r20, r57, r2 * r56);
    r56 = r13 * r11;
    r57 = fma(r20, r56, r57);
    r57 = fma(r16, r55, r57);
    r42 = r42 * r57;
    r53 = fma(r54, r53, r42);
    r50 = r50 + r53;
    r56 = r27 * r4;
    r56 = r56 * r46;
    r58 = r8 * r22;
    r58 = fma(r29, r58, r56);
    r59 = r57 * r28;
    r58 = r58 + r59;
    r58 = fma(r54, r31, r58);
    r58 = fma(r7, r58, r33 * r50);
    r50 = r17 * r46;
    r60 = -4.00000000000000000e+00;
    r50 = r50 * r60;
    r61 = r22 * r54;
    r62 = r60 * r61;
    r63 = r50 + r62;
    r58 = fma(r6, r63, r58);
    r63 = r27 * r34;
    r64 = r22 * r27;
    r65 = r54 * r28;
    r64 = fma(r46, r64, r65);
    r66 = r27 * r4;
    r66 = r66 * r29;
    r67 = r27 * r30;
    r67 = r67 * r57;
    r68 = r66 + r67;
    r69 = r64 + r68;
    r70 = r8 * r17;
    r46 = fma(r46, r31, r29 * r70);
    r46 = r46 + r53;
    r46 = fma(r6, r46, r7 * r69);
    r69 = r4 * r60;
    r70 = r57 * r69;
    r50 = r50 + r70;
    r46 = fma(r33, r50, r46);
    r63 = fma(r46, r63, r58 * r35);
    r52 = r2 * r52;
    r35 = 1.0 / r51;
    r36 = rsqrt(r36);
    r52 = r52 * r35;
    r52 = r52 * r36;
    r36 = r8 * r4;
    r35 = r57 * r31;
    r36 = fma(r29, r36, r35);
    r36 = r36 + r64;
    r70 = r62 + r70;
    r70 = fma(r7, r70, r33 * r36);
    r36 = r27 * r30;
    r36 = fma(r54, r36, r56);
    r56 = r22 * r27;
    r56 = fma(r29, r56, r59);
    r36 = r36 + r56;
    r70 = fma(r6, r36, r70);
    r36 = r18 * r70;
    r40 = 1.0 / r40;
    r36 = fma(r40, r36, r63 * r52);
    r63 = -1.59154943091895346e-01;
    r63 = r3 * r63;
    r3 = r19 * r19;
    r49 = r49 + r3;
    r59 = 1.0 / r49;
    r62 = r0 * r3;
    r63 = r63 * r59;
    r63 = r63 * r62;
    r19 = 1.0 / r19;
    r59 = r18 * r26;
    r64 = 1.0 / r3;
    r59 = r59 * r64;
    r46 = fma(r46, r59, r58 * r19);
    r58 = fma(r46, r63, r36 * r25);
    r64 = r8 * r17;
    r64 = fma(r54, r64, r35);
    r50 = r22 * r27;
    r21 = fma(r15, r55, r20 * r21);
    r21 = fma(r2, r24, r21);
    r21 = fma(r20, r23, r21);
    r50 = r50 * r21;
    r23 = r27 * r4;
    r24 = r9 * r12;
    r71 = r14 * r15;
    r71 = fma(r2, r71, r2 * r24);
    r24 = r13 * r16;
    r71 = fma(r2, r24, r71);
    r71 = fma(r11, r55, r71);
    r23 = fma(r71, r23, r50);
    r64 = r64 + r23;
    r55 = r27 * r30;
    r24 = r71 * r28;
    r55 = fma(r21, r55, r24);
    r55 = r55 + r53;
    r55 = fma(r7, r55, r6 * r64);
    r64 = r17 * r57;
    r64 = r64 * r60;
    r53 = r21 * r69;
    r72 = r64 + r53;
    r55 = fma(r33, r72, r55);
    r65 = r67 + r65;
    r65 = r65 + r23;
    r23 = r22 * r60;
    r23 = r23 * r71;
    r64 = r64 + r23;
    r64 = fma(r6, r64, r33 * r65);
    r65 = fma(r71, r31, r8 * r61);
    r67 = r27 * r4;
    r67 = r67 * r57;
    r72 = fma(r21, r28, r67);
    r65 = r65 + r72;
    r64 = fma(r7, r65, r64);
    r65 = fma(r64, r19, r55 * r59);
    r73 = r8 * r4;
    r73 = fma(r54, r73, r42);
    r73 = r73 + r24;
    r73 = fma(r21, r31, r73);
    r24 = r27 * r30;
    r61 = fma(r27, r61, r71 * r24);
    r61 = r61 + r72;
    r61 = fma(r6, r61, r33 * r73);
    r53 = r23 + r53;
    r61 = fma(r7, r53, r61);
    r53 = r18 * r61;
    r23 = r27 * r26;
    r73 = r27 * r34;
    r73 = fma(r55, r73, r64 * r23);
    r73 = fma(r73, r52, r40 * r53);
    r53 = fma(r73, r25, r65 * r63);
    WriteSum2<double, double>((double*)inout_shared, r58, r53);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r17 * r29;
    r53 = r53 * r60;
    r58 = r10 * r12;
    r23 = r13 * r15;
    r23 = fma(r20, r23, r2 * r58);
    r58 = r14 * r16;
    r23 = fma(r2, r58, r23);
    r20 = r9 * r11;
    r23 = fma(r2, r20, r23);
    r69 = r23 * r69;
    r20 = r53 + r69;
    r58 = r22 * r27;
    r58 = r58 * r23;
    r67 = r67 + r58;
    r2 = r8 * r17;
    r67 = fma(r21, r2, r67);
    r67 = fma(r29, r31, r67);
    r67 = fma(r6, r67, r33 * r20);
    r20 = r27 * r4;
    r2 = r27 * r30;
    r2 = fma(r23, r2, r21 * r20);
    r2 = r2 + r56;
    r67 = fma(r7, r2, r67);
    r2 = r8 * r22;
    r2 = fma(r21, r2, r66);
    r28 = r23 * r28;
    r2 = r2 + r35;
    r2 = r2 + r28;
    r57 = r22 * r57;
    r57 = r57 * r60;
    r53 = r53 + r57;
    r53 = fma(r6, r53, r7 * r2);
    r2 = r27 * r30;
    r2 = fma(r29, r2, r58);
    r2 = r2 + r72;
    r53 = fma(r33, r2, r53);
    r2 = fma(r53, r19, r67 * r59);
    r72 = r27 * r34;
    r58 = r27 * r26;
    r58 = fma(r53, r58, r67 * r72);
    r28 = r50 + r28;
    r28 = r28 + r68;
    r68 = r8 * r4;
    r31 = fma(r23, r31, r21 * r68);
    r31 = r31 + r56;
    r31 = fma(r33, r31, r6 * r28);
    r69 = r57 + r69;
    r31 = fma(r7, r69, r31);
    r69 = r18 * r31;
    r69 = fma(r40, r69, r58 * r52);
    r58 = fma(r69, r25, r2 * r63);
    r7 = r27 * r44;
    r57 = r27 * r32;
    r57 = fma(r34, r57, r26 * r7);
    r7 = r18 * r5;
    r7 = fma(r40, r7, r57 * r52);
    r57 = fma(r44, r19, r32 * r59);
    r33 = fma(r57, r63, r7 * r25);
    WriteSum2<double, double>((double*)inout_shared, r58, r33);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r27 * r41;
    r58 = r27 * r47;
    r58 = fma(r34, r58, r26 * r33);
    r33 = r18 * r43;
    r33 = fma(r40, r33, r58 * r52);
    r58 = fma(r41, r19, r47 * r59);
    r28 = fma(r58, r63, r33 * r25);
    r6 = r27 * r38;
    r56 = r27 * r45;
    r56 = fma(r34, r56, r26 * r6);
    r6 = r18 * r37;
    r6 = fma(r40, r6, r56 * r52);
    r19 = fma(r38, r19, r45 * r59);
    r63 = fma(r19, r63, r6 * r25);
    WriteSum2<double, double>((double*)inout_shared, r28, r63);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = r46 * r46;
    r28 = 2.53302959105844473e-02;
    r28 = r0 * r28;
    r49 = r49 * r49;
    r49 = 1.0 / r49;
    r28 = r28 * r49;
    r28 = r28 * r62;
    r28 = r28 * r3;
    r3 = r36 * r36;
    r62 = 1.01321183642337789e-01;
    r62 = r1 * r62;
    r48 = r48 * r48;
    r48 = 1.0 / r48;
    r62 = r62 * r48;
    r62 = r62 * r39;
    r62 = r62 * r51;
    r3 = fma(r62, r3, r28 * r63);
    r63 = r73 * r73;
    r51 = r65 * r65;
    r51 = fma(r28, r51, r62 * r63);
    WriteSum2<double, double>((double*)inout_shared, r3, r51);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r69 * r69;
    r3 = r2 * r2;
    r3 = fma(r28, r3, r62 * r51);
    r51 = r7 * r7;
    r63 = r57 * r57;
    r63 = fma(r28, r63, r62 * r51);
    WriteSum2<double, double>((double*)inout_shared, r3, r63);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = r58 * r58;
    r3 = r33 * r33;
    r3 = fma(r62, r3, r28 * r63);
    r63 = r19 * r28;
    r51 = r6 * r62;
    r6 = fma(r6, r51, r19 * r63);
    WriteSum2<double, double>((double*)inout_shared, r3, r6);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r46 * r65;
    r3 = r36 * r73;
    r3 = fma(r62, r3, r28 * r6);
    r6 = r46 * r2;
    r19 = r36 * r69;
    r19 = fma(r62, r19, r28 * r6);
    WriteSum2<double, double>((double*)inout_shared, r3, r19);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r46 * r57;
    r3 = r36 * r7;
    r3 = fma(r62, r3, r28 * r19);
    r19 = r36 * r33;
    r6 = r46 * r58;
    r6 = fma(r28, r6, r62 * r19);
    WriteSum2<double, double>((double*)inout_shared, r3, r6);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r36, r51, r46 * r63);
    r3 = r65 * r2;
    r19 = r73 * r69;
    r19 = fma(r62, r19, r28 * r3);
    WriteSum2<double, double>((double*)inout_shared, r6, r19);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r73 * r7;
    r6 = r65 * r57;
    r6 = fma(r28, r6, r62 * r19);
    r19 = r73 * r33;
    r3 = r65 * r58;
    r3 = fma(r28, r3, r62 * r19);
    WriteSum2<double, double>((double*)inout_shared, r6, r3);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = fma(r73, r51, r65 * r63);
    r6 = r69 * r7;
    r19 = r2 * r57;
    r19 = fma(r28, r19, r62 * r6);
    WriteSum2<double, double>((double*)inout_shared, r3, r19);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r2 * r58;
    r3 = r69 * r33;
    r3 = fma(r62, r3, r28 * r19);
    r19 = fma(r69, r51, r2 * r63);
    WriteSum2<double, double>((double*)inout_shared, r3, r19);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r57 * r58;
    r3 = r7 * r33;
    r3 = fma(r62, r3, r28 * r19);
    r19 = fma(r57, r63, r7 * r51);
    WriteSum2<double, double>((double*)inout_shared, r3, r19);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r33, r51, r58 * r63);
    WriteSum1<double, double>((double*)inout_shared, r51);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
}

void SphericalFixedPointResJac(double* pose,
                               unsigned int pose_num_alloc,
                               SharedIndex* pose_indices,
                               double* sensor_from_rig,
                               unsigned int sensor_from_rig_num_alloc,
                               double* wh,
                               unsigned int wh_num_alloc,
                               double* pixel,
                               unsigned int pixel_num_alloc,
                               double* point,
                               unsigned int point_num_alloc,
                               double* out_res,
                               unsigned int out_res_num_alloc,
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
  SphericalFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      wh,
      wh_num_alloc,
      pixel,
      pixel_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar