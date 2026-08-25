#include "kernel_spherical_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SphericalFixedPointResJacFirstKernel(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* wh,
    unsigned int wh_num_alloc,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        wh, 0 * wh_num_alloc, global_thread_idx, r0, r1);
    r2 = 5.00000000000000000e-01;
    r3 = 1.59154943091895346e-01;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r4,
                                         r5,
                                         r6);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r7, r8, r9);
    r10 = -2.00000000000000000e+00;
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r11,
                       r12,
                       r13,
                       r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r15,
                                         r16,
                                         r17,
                                         r18);
    r19 = r13 * r15;
    r20 = -1.00000000000000000e+00;
    r19 = fmaf(r20, r19, r12 * r18);
    r19 = fmaf(r14, r16, r19);
    r19 = fmaf(r11, r17, r19);
    r21 = r10 * r19;
    r21 = r21 * r19;
    r22 = 1.00000000000000000e+00;
    r23 = r13 * r18;
    r24 = fmaf(r12, r15, r23);
    r25 = r14 * r17;
    r26 = r11 * r16;
    r24 = r24 + r25;
    r24 = fmaf(r20, r26, r24);
    r27 = r10 * r24;
    r27 = fmaf(r24, r27, r22);
    r28 = r21 + r27;
    r28 = fmaf(r7, r28, r4);
    r4 = fmaf(r14, r15, r11 * r18);
    r29 = r12 * r17;
    r4 = fmaf(r20, r29, r4);
    r4 = fmaf(r13, r16, r4);
    r29 = 2.00000000000000000e+00;
    r30 = r29 * r19;
    r31 = r4 * r30;
    r32 = fmaf(r12, r16, r11 * r15);
    r32 = fmaf(r13, r17, r32);
    r32 = fmaf(r20, r32, r14 * r18);
    r33 = r10 * r32;
    r34 = fmaf(r24, r33, r31);
    r35 = r24 * r29;
    r35 = r35 * r4;
    r36 = fmaf(r32, r30, r35);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r37,
                       r38,
                       r39);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r40 = r15 * r17;
    r40 = r40 * r29;
    r41 = r16 * r18;
    r41 = fmaf(r29, r41, r40);
    r42 = r17 * r18;
    r43 = r15 * r16;
    r43 = r43 * r29;
    r42 = fmaf(r10, r42, r43);
    r44 = r17 * r17;
    r44 = r44 * r10;
    r45 = r22 + r44;
    r46 = r16 * r16;
    r46 = r46 * r10;
    r45 = r45 + r46;
    r28 = fmaf(r8, r34, r28);
    r28 = fmaf(r9, r36, r28);
    r28 = fmaf(r39, r41, r28);
    r28 = fmaf(r38, r42, r28);
    r28 = fmaf(r37, r45, r28);
    r36 = 9.99999999999999955e-07;
    r35 = fmaf(r19, r33, r35);
    r35 = fmaf(r7, r35, r6);
    r6 = r16 * r18;
    r6 = fmaf(r10, r6, r40);
    r46 = r22 + r46;
    r40 = r15 * r15;
    r40 = r10 * r40;
    r46 = r46 + r40;
    r34 = r16 * r17;
    r34 = r34 * r29;
    r47 = r15 * r18;
    r47 = fmaf(r29, r47, r34);
    r48 = r29 * r4;
    r49 = r24 * r30;
    r48 = fmaf(r32, r48, r49);
    r21 = r22 + r21;
    r50 = r10 * r4;
    r50 = r50 * r4;
    r21 = r21 + r50;
    r35 = fmaf(r37, r6, r35);
    r35 = fmaf(r39, r46, r35);
    r35 = fmaf(r38, r47, r35);
    r35 = fmaf(r8, r48, r35);
    r35 = fmaf(r9, r21, r35);
    r21 = copysignf(r36, r35);
    r21 = r21 + r35;
    r48 = atan2f(r28, r21);
    r48 = fmaf(r3, r48, r2);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r51);
    r3 = fmaf(r3, r20, r0 * r48);
    r48 = -3.18309886183790691e-01;
    r52 = r24 * r29;
    r52 = fmaf(r32, r52, r31);
    r52 = fmaf(r7, r52, r5);
    r5 = r17 * r18;
    r5 = fmaf(r29, r5, r43);
    r44 = r22 + r44;
    r44 = r44 + r40;
    r40 = r15 * r18;
    r40 = fmaf(r10, r40, r34);
    r49 = fmaf(r4, r33, r49);
    r27 = r50 + r27;
    r52 = fmaf(r37, r5, r52);
    r52 = fmaf(r38, r44, r52);
    r52 = fmaf(r39, r40, r52);
    r52 = fmaf(r9, r49, r52);
    r52 = fmaf(r8, r27, r52);
    r27 = r20 * r52;
    r49 = r28 * r28;
    r39 = r36 + r49;
    r39 = fmaf(r35, r35, r39);
    r38 = sqrtf(r39);
    r36 = copysignf(r36, r38);
    r38 = r36 + r38;
    r27 = atan2f(r27, r38);
    r27 = fmaf(r48, r27, r2);
    r51 = fmaf(r51, r20, r1 * r27);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3, r51);
    r27 = fmaf(r51, r51, r3 * r3);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r27);
  if (global_thread_idx < problem_size) {
    r27 = -1.59154943091895346e-01;
    r27 = r3 * r27;
    r3 = r21 * r21;
    r49 = r49 + r3;
    r48 = 1.0 / r49;
    r36 = r0 * r3;
    r27 = r27 * r48;
    r27 = r27 * r36;
    r21 = 1.0 / r21;
    r48 = r29 * r32;
    r37 = r12 * r15;
    r37 = fmaf(r2, r23, r2 * r37);
    r50 = -5.00000000000000000e-01;
    r37 = fmaf(r50, r26, r37);
    r37 = fmaf(r2, r25, r37);
    r34 = r11 * r18;
    r22 = r14 * r15;
    r22 = fmaf(r50, r22, r50 * r34);
    r34 = r13 * r16;
    r22 = fmaf(r50, r34, r22);
    r43 = r12 * r17;
    r22 = fmaf(r2, r43, r22);
    r48 = fmaf(r22, r30, r37 * r48);
    r43 = r29 * r4;
    r34 = r13 * r15;
    r31 = r14 * r16;
    r31 = fmaf(r50, r31, r2 * r34);
    r34 = r11 * r17;
    r31 = fmaf(r50, r34, r31);
    r53 = r12 * r50;
    r31 = fmaf(r18, r53, r31);
    r34 = r24 * r29;
    r54 = r14 * r18;
    r55 = r11 * r15;
    r55 = fmaf(r50, r55, r2 * r54);
    r54 = r13 * r17;
    r55 = fmaf(r50, r54, r55);
    r55 = fmaf(r16, r53, r55);
    r34 = r34 * r55;
    r43 = fmaf(r31, r43, r34);
    r48 = r48 + r43;
    r54 = r29 * r4;
    r54 = r54 * r37;
    r56 = r10 * r24;
    r56 = fmaf(r22, r56, r54);
    r57 = r55 * r30;
    r56 = r56 + r57;
    r56 = fmaf(r31, r33, r56);
    r56 = fmaf(r8, r56, r9 * r48);
    r48 = r19 * r37;
    r58 = -4.00000000000000000e+00;
    r48 = r48 * r58;
    r59 = r24 * r31;
    r60 = r58 * r59;
    r61 = r48 + r60;
    r56 = fmaf(r7, r61, r56);
    r61 = r24 * r29;
    r62 = r31 * r30;
    r61 = fmaf(r37, r61, r62);
    r63 = r29 * r4;
    r63 = r63 * r22;
    r64 = r29 * r32;
    r64 = r64 * r55;
    r65 = r63 + r64;
    r66 = r61 + r65;
    r67 = r10 * r19;
    r37 = fmaf(r37, r33, r22 * r67);
    r37 = r37 + r43;
    r37 = fmaf(r7, r37, r8 * r66);
    r66 = r4 * r58;
    r67 = r55 * r66;
    r48 = r48 + r67;
    r37 = fmaf(r9, r48, r37);
    r48 = r20 * r28;
    r68 = 1.0 / r3;
    r48 = r48 * r68;
    r68 = fmaf(r37, r48, r56 * r21);
    r69 = 3.18309886183790691e-01;
    r69 = r51 * r69;
    r51 = r38 * r38;
    r70 = fmaf(r52, r52, r51);
    r71 = 1.0 / r70;
    r72 = r1 * r51;
    r69 = r69 * r71;
    r69 = r69 * r72;
    r71 = r29 * r28;
    r73 = r29 * r35;
    r73 = fmaf(r37, r73, r56 * r71);
    r52 = r2 * r52;
    r71 = 1.0 / r51;
    r39 = rsqrtf(r39);
    r52 = r52 * r71;
    r52 = r52 * r39;
    r39 = r10 * r4;
    r71 = r55 * r33;
    r39 = fmaf(r22, r39, r71);
    r39 = r39 + r61;
    r67 = r60 + r67;
    r67 = fmaf(r8, r67, r9 * r39);
    r39 = r29 * r32;
    r39 = fmaf(r31, r39, r54);
    r54 = r24 * r29;
    r54 = fmaf(r22, r54, r57);
    r39 = r39 + r54;
    r67 = fmaf(r7, r39, r67);
    r39 = r20 * r67;
    r38 = 1.0 / r38;
    r39 = fmaf(r38, r39, r73 * r52);
    r73 = fmaf(r39, r69, r68 * r27);
    r57 = r29 * r28;
    r62 = r64 + r62;
    r64 = r24 * r29;
    r23 = fmaf(r15, r53, r50 * r23);
    r23 = fmaf(r2, r26, r23);
    r23 = fmaf(r50, r25, r23);
    r64 = r64 * r23;
    r25 = r29 * r4;
    r26 = r11 * r18;
    r60 = r14 * r15;
    r60 = fmaf(r2, r60, r2 * r26);
    r26 = r13 * r16;
    r60 = fmaf(r2, r26, r60);
    r60 = fmaf(r17, r53, r60);
    r25 = fmaf(r60, r25, r64);
    r62 = r62 + r25;
    r53 = r19 * r55;
    r53 = r53 * r58;
    r26 = r24 * r58;
    r26 = r26 * r60;
    r61 = r53 + r26;
    r61 = fmaf(r7, r61, r9 * r62);
    r62 = fmaf(r60, r33, r10 * r59);
    r37 = r29 * r4;
    r37 = r37 * r55;
    r56 = fmaf(r23, r30, r37);
    r62 = r62 + r56;
    r61 = fmaf(r8, r62, r61);
    r62 = r29 * r35;
    r74 = r10 * r19;
    r74 = fmaf(r31, r74, r71);
    r74 = r74 + r25;
    r25 = r29 * r32;
    r75 = r60 * r30;
    r25 = fmaf(r23, r25, r75);
    r25 = r25 + r43;
    r25 = fmaf(r8, r25, r7 * r74);
    r74 = r23 * r66;
    r53 = r53 + r74;
    r25 = fmaf(r9, r53, r25);
    r62 = fmaf(r25, r62, r61 * r57);
    r57 = r10 * r4;
    r57 = fmaf(r31, r57, r34);
    r57 = r57 + r75;
    r57 = fmaf(r23, r33, r57);
    r75 = r29 * r32;
    r59 = fmaf(r29, r59, r60 * r75);
    r59 = r59 + r56;
    r59 = fmaf(r7, r59, r9 * r57);
    r74 = r26 + r74;
    r59 = fmaf(r8, r74, r59);
    r74 = r20 * r59;
    r74 = fmaf(r38, r74, r62 * r52);
    r25 = fmaf(r25, r48, r61 * r21);
    r61 = fmaf(r25, r27, r74 * r69);
    r62 = r29 * r35;
    r26 = r19 * r22;
    r26 = r26 * r58;
    r57 = r12 * r18;
    r75 = r13 * r15;
    r75 = fmaf(r50, r75, r2 * r57);
    r57 = r14 * r16;
    r75 = fmaf(r2, r57, r75);
    r50 = r11 * r17;
    r75 = fmaf(r2, r50, r75);
    r66 = r75 * r66;
    r50 = r26 + r66;
    r57 = r24 * r29;
    r57 = r57 * r75;
    r37 = r37 + r57;
    r2 = r10 * r19;
    r37 = fmaf(r23, r2, r37);
    r37 = fmaf(r22, r33, r37);
    r37 = fmaf(r7, r37, r9 * r50);
    r50 = r29 * r4;
    r2 = r29 * r32;
    r2 = fmaf(r75, r2, r23 * r50);
    r2 = r2 + r54;
    r37 = fmaf(r8, r2, r37);
    r2 = r29 * r28;
    r50 = r10 * r24;
    r50 = fmaf(r23, r50, r63);
    r30 = r75 * r30;
    r50 = r50 + r71;
    r50 = r50 + r30;
    r55 = r24 * r55;
    r55 = r55 * r58;
    r26 = r26 + r55;
    r26 = fmaf(r7, r26, r8 * r50);
    r50 = r29 * r32;
    r50 = fmaf(r22, r50, r57);
    r50 = r50 + r56;
    r26 = fmaf(r9, r50, r26);
    r2 = fmaf(r26, r2, r37 * r62);
    r30 = r64 + r30;
    r30 = r30 + r65;
    r65 = r10 * r4;
    r33 = fmaf(r75, r33, r23 * r65);
    r33 = r33 + r54;
    r33 = fmaf(r9, r33, r7 * r30);
    r66 = r55 + r66;
    r33 = fmaf(r8, r66, r33);
    r66 = r20 * r33;
    r66 = fmaf(r38, r66, r2 * r52);
    r26 = fmaf(r26, r21, r37 * r48);
    r37 = fmaf(r26, r27, r66 * r69);
    r2 = fmaf(r45, r21, r6 * r48);
    r8 = r20 * r5;
    r55 = r29 * r45;
    r9 = r29 * r6;
    r9 = fmaf(r35, r9, r28 * r55);
    r9 = fmaf(r9, r52, r38 * r8);
    r8 = fmaf(r9, r69, r2 * r27);
    WriteSum4<float, float>((float*)inout_shared, r73, r61, r37, r8);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r42, r21, r47 * r48);
    r37 = r20 * r44;
    r61 = r29 * r42;
    r73 = r29 * r47;
    r73 = fmaf(r35, r73, r28 * r61);
    r73 = fmaf(r73, r52, r38 * r37);
    r37 = fmaf(r73, r69, r8 * r27);
    r61 = r20 * r40;
    r55 = r29 * r41;
    r30 = r29 * r46;
    r30 = fmaf(r35, r30, r28 * r55);
    r52 = fmaf(r30, r52, r38 * r61);
    r21 = fmaf(r41, r21, r46 * r48);
    r27 = fmaf(r21, r27, r52 * r69);
    WriteSum2<float, float>((float*)inout_shared, r37, r27);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = r68 * r68;
    r37 = 2.53302959105844473e-02;
    r37 = r0 * r37;
    r49 = r49 * r49;
    r49 = 1.0 / r49;
    r37 = r37 * r49;
    r37 = r37 * r36;
    r37 = r37 * r3;
    r3 = r39 * r39;
    r36 = 1.01321183642337789e-01;
    r36 = r1 * r36;
    r70 = r70 * r70;
    r70 = 1.0 / r70;
    r36 = r36 * r70;
    r36 = r36 * r72;
    r36 = r36 * r51;
    r3 = fmaf(r36, r3, r37 * r27);
    r27 = r74 * r74;
    r51 = r25 * r25;
    r51 = fmaf(r37, r51, r36 * r27);
    r27 = r66 * r66;
    r72 = r26 * r26;
    r72 = fmaf(r37, r72, r36 * r27);
    r27 = r9 * r9;
    r70 = r2 * r2;
    r70 = fmaf(r37, r70, r36 * r27);
    WriteSum4<float, float>((float*)inout_shared, r3, r51, r72, r70);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = r8 * r8;
    r72 = r73 * r36;
    r73 = fmaf(r73, r72, r37 * r70);
    r70 = r52 * r52;
    r51 = r21 * r37;
    r21 = fmaf(r21, r51, r36 * r70);
    WriteSum2<float, float>((float*)inout_shared, r73, r21);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = r39 * r74;
    r73 = r68 * r25;
    r73 = fmaf(r37, r73, r36 * r21);
    r21 = r68 * r26;
    r70 = r39 * r66;
    r70 = fmaf(r36, r70, r37 * r21);
    r21 = r68 * r2;
    r3 = r39 * r9;
    r3 = fmaf(r36, r3, r37 * r21);
    r21 = r68 * r8;
    r21 = fmaf(r39, r72, r37 * r21);
    WriteSum4<float, float>((float*)inout_shared, r73, r70, r3, r21);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = r39 * r52;
    r21 = fmaf(r68, r51, r36 * r21);
    r3 = r74 * r66;
    r70 = r25 * r26;
    r70 = fmaf(r37, r70, r36 * r3);
    r3 = r25 * r2;
    r73 = r74 * r9;
    r73 = fmaf(r36, r73, r37 * r3);
    r3 = r25 * r8;
    r3 = fmaf(r37, r3, r74 * r72);
    WriteSum4<float, float>((float*)inout_shared, r21, r70, r73, r3);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r74 * r52;
    r3 = fmaf(r36, r3, r25 * r51);
    r73 = r66 * r9;
    r70 = r26 * r2;
    r70 = fmaf(r37, r70, r36 * r73);
    r73 = r26 * r8;
    r73 = fmaf(r37, r73, r66 * r72);
    r21 = r66 * r52;
    r21 = fmaf(r26, r51, r36 * r21);
    WriteSum4<float, float>((float*)inout_shared, r3, r70, r73, r21);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = r2 * r8;
    r21 = fmaf(r37, r21, r9 * r72);
    r37 = r9 * r52;
    r37 = fmaf(r2, r51, r36 * r37);
    r72 = fmaf(r52, r72, r8 * r51);
    WriteSum3<float, float>((float*)inout_shared, r21, r37, r72);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void SphericalFixedPointResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* wh,
    unsigned int wh_num_alloc,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
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